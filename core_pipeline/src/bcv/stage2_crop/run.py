"""Drive cropping over one split: detections.parquet + video -> crop.mp4 + manifest."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..common import viz
from ..common.config import PipelineConfig
from ..common.contracts import CROP_MANIFEST_SCHEMA, PER_FIGHTER_MANIFEST_SCHEMA, ArtifactMeta
from ..common.geometry import Box
from ..common.io import read_meta, validate_meta, write_meta, write_parquet
from ..common.video import VideoReader, VideoWriter
from .cropper import Cropper, CropResult, Stage2Config

STAGE = "stage2_crop"
STAGE1 = "stage1_detect"

_INTERP = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}


def _box(row: pd.Series, prefix: str) -> Box | None:
    if not bool(row[f"{prefix}_present"]):
        return None
    return (
        int(row[f"{prefix}_x1"]),
        int(row[f"{prefix}_y1"]),
        int(row[f"{prefix}_x2"]),
        int(row[f"{prefix}_y2"]),
    )


def _crop_region(frame, crop_box, cs: int, interp: int):
    x1, y1, x2, y2 = crop_box
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        region = frame
    return cv2.resize(region, (cs, cs), interpolation=interp)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return None


def run_stage2(
    pipeline: PipelineConfig,
    stage: Stage2Config,
    *,
    bout: int,
    split: int,
    debug_video: bool = True,
) -> Path:
    """Produce the fixed-square crop video + manifest from Stage-1 detections."""
    video_path = pipeline.split_video(bout, split)
    det_dir = pipeline.artifact_dir(bout, split, STAGE1)
    df = pd.read_parquet(det_dir / "detections.parquet")
    det_meta = read_meta(det_dir)
    validate_meta(det_meta, source_video=str(video_path), num_frames=len(df))

    out_dir = pipeline.artifact_dir(bout, split, STAGE)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_frame = int(df["frame"].iloc[0]) if len(df) else 0
    cropper = Cropper(stage)  # union crop: expands when only one fighter is found
    interp = _INTERP.get(stage.resize_mode, cv2.INTER_AREA)
    cs = stage.crop_size
    pf = stage.emit_per_fighter
    # Per-fighter crops intentionally isolate ONE fighter -> never expand.
    single_cfg = stage.model_copy(update={"single_fighter_scale": 1.0})
    red_cropper = Cropper(single_cfg) if pf else None
    blue_cropper = Cropper(single_cfg) if pf else None

    reader = VideoReader(video_path)
    info = reader.info
    if start_frame > 0:
        reader.seek(start_frame)
    crop_writer = VideoWriter(out_dir / "crop.mp4", info.fps, cs, cs)
    red_writer = VideoWriter(out_dir / "red_crop.mp4", info.fps, cs, cs) if pf else None
    blue_writer = VideoWriter(out_dir / "blue_crop.mp4", info.fps, cs, cs) if pf else None
    dbg_writer = (
        VideoWriter(out_dir / "crop_overlay.mp4", info.fps, info.width, info.height)
        if debug_video
        else None
    )

    rows: list[dict[str, object]] = []
    pf_rows: list[dict[str, object]] = []
    try:
        frame_iter = iter(reader)
        for _, det in df.iterrows():
            frame = next(frame_iter, None)
            if frame is None:
                break
            red_box, blue_box = _box(det, "red"), _box(det, "blue")
            res: CropResult = cropper.step(red_box, blue_box, info.width, info.height)
            crop_writer.write(_crop_region(frame, res.crop_box, cs, interp))
            if dbg_writer is not None:
                dbg_writer.write(viz.draw_union_crop(frame, res.crop_box, valid=res.crop_valid))
            x1, y1, x2, y2 = res.crop_box
            rows.append(
                {
                    "frame": int(det["frame"]),
                    "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
                    "crop_valid": res.crop_valid, "staleness": res.staleness,
                    "ema_cx": res.ema_cx, "ema_cy": res.ema_cy, "ema_half": res.ema_half,
                }
            )
            if pf:
                assert red_cropper is not None and blue_cropper is not None
                assert red_writer is not None and blue_writer is not None
                # each fighter cropped on its own (carry-forward when that fighter is absent)
                rr = red_cropper.step(red_box, None, info.width, info.height)
                br = blue_cropper.step(None, blue_box, info.width, info.height)
                red_writer.write(_crop_region(frame, rr.crop_box, cs, interp))
                blue_writer.write(_crop_region(frame, br.crop_box, cs, interp))
                rcb, bcb = rr.crop_box, br.crop_box
                pf_rows.append(
                    {
                        "frame": int(det["frame"]),
                        "red_crop_x1": rcb[0], "red_crop_y1": rcb[1],
                        "red_crop_x2": rcb[2], "red_crop_y2": rcb[3], "red_valid": rr.crop_valid,
                        "blue_crop_x1": bcb[0], "blue_crop_y1": bcb[1],
                        "blue_crop_x2": bcb[2], "blue_crop_y2": bcb[3], "blue_valid": br.crop_valid,
                    }
                )
    finally:
        reader.release()
        crop_writer.close()
        if red_writer is not None:
            red_writer.close()
        if blue_writer is not None:
            blue_writer.close()
        if dbg_writer is not None:
            dbg_writer.close()

    manifest = pd.DataFrame(rows, columns=list(CROP_MANIFEST_SCHEMA)).astype(CROP_MANIFEST_SCHEMA)
    write_parquet(out_dir / "crop_manifest.parquet", manifest)
    if pf:
        pf_manifest = pd.DataFrame(
            pf_rows, columns=list(PER_FIGHTER_MANIFEST_SCHEMA)
        ).astype(PER_FIGHTER_MANIFEST_SCHEMA)
        write_parquet(out_dir / "per_fighter_manifest.parquet", pf_manifest)

    meta = ArtifactMeta(
        stage=STAGE,
        source_video=str(video_path),
        fps=info.fps,
        width=cs,
        height=cs,
        num_frames=len(rows),
        git_sha=_git_sha(),
        created_utc=datetime.now(UTC).isoformat(),
        producer={
            "crop_size": cs,
            "pad_frac": stage.pad_frac,
            "ema_alpha": stage.ema_alpha,
            "scale_mode": stage.scale_mode,
            "max_staleness": stage.max_staleness,
            "fallback": stage.fallback,
            "start_frame": start_frame,
            "valid_frac": float(np.mean([r["crop_valid"] for r in rows])) if rows else 0.0,
        },
    )
    write_meta(out_dir, meta)
    return out_dir
