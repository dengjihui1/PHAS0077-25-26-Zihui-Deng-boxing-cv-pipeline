"""Drive a detector backend over one split video → detections.parquet + debug.mp4."""

from __future__ import annotations

import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..common import viz
from ..common.config import PipelineConfig
from ..common.contracts import ArtifactMeta
from ..common.io import write_meta, write_parquet
from ..common.video import VideoReader, VideoWriter
from .interface import FighterDetector, FrameDetection
from .writer import detections_to_df

STAGE = "stage1_detect"


class Stage1Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "chain"
    detector_weights: str = "/home/ubuntu/moughton/models/yolo26x.pt"
    classifier_weights: str = "/home/ubuntu/moughton/models/boxer_yolo26_classifier/weights/best.pt"
    finetuned_weights: str | None = None
    det_imgsz: int = 512
    cls_imgsz: int = 224
    max_det: int = 5
    iou: float = 0.95
    crop_pad: float = 0.05
    crop_min_x: int = 25
    crop_min_y: int = 50
    min_cls_conf: float = 0.5
    box_hold_frames: int = 15
    tracker: str = "botsort.yaml"
    device: int | str = 0
    half: bool = True


def build_detector(cfg: Stage1Config) -> FighterDetector:
    if cfg.backend == "chain":
        from .backend_chain import ChainDetector

        return ChainDetector(
            cfg.detector_weights,
            cfg.classifier_weights,
            det_imgsz=cfg.det_imgsz,
            cls_imgsz=cfg.cls_imgsz,
            max_det=cfg.max_det,
            iou=cfg.iou,
            crop_pad=cfg.crop_pad,
            crop_min_x=cfg.crop_min_x,
            crop_min_y=cfg.crop_min_y,
            min_cls_conf=cfg.min_cls_conf,
            box_hold_frames=cfg.box_hold_frames,
            tracker=cfg.tracker,
            device=cfg.device,
            half=cfg.half,
        )
    if cfg.backend == "finetuned":
        from .backend_finetuned import FinetunedDetector

        if not cfg.finetuned_weights:
            raise ValueError("backend=finetuned requires finetuned_weights in the config")
        return FinetunedDetector(
            cfg.finetuned_weights,
            imgsz=cfg.det_imgsz,
            min_cls_conf=cfg.min_cls_conf,
            device=cfg.device,
            half=cfg.half,
        )
    raise ValueError(f"unknown backend {cfg.backend!r} (use 'chain' or 'finetuned')")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return None


def _draw_debug(frame, fd: FrameDetection):
    out = frame
    if fd.red is not None:
        out = viz.draw_box(out, fd.red.bbox, viz.RED, label=f"red {fd.red.cls_conf:.2f}")
    if fd.blue is not None:
        out = viz.draw_box(out, fd.blue.bbox, viz.BLUE, label=f"blue {fd.blue.cls_conf:.2f}")
    return out


def run_stage1(
    pipeline: PipelineConfig,
    stage: Stage1Config,
    *,
    bout: int,
    split: int,
    debug_video: bool = True,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> Path:
    """Process one split video and publish the Stage-1 artifact directory.

    Frames are labelled with their ABSOLUTE index (``start_frame + i``) so detections
    stay aligned with the bout's annotations even when a sub-range is processed.
    """
    video_path = pipeline.split_video(bout, split)
    out_dir = pipeline.artifact_dir(bout, split, STAGE)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(stage)
    detector.reset()

    reader = VideoReader(video_path)
    info = reader.info
    if start_frame > 0:
        reader.seek(start_frame)
    writer = (
        VideoWriter(out_dir / "boxes.mp4", info.fps, info.width, info.height)
        if debug_video
        else None
    )

    total = min(info.num_frames, max_frames) if max_frames is not None else info.num_frames
    log_every = 500
    t0 = time.time()
    frame_dets: list[FrameDetection] = []
    try:
        for idx, frame in enumerate(reader):
            if max_frames is not None and idx >= max_frames:
                break
            fd = detector.detect(frame, start_frame + idx)
            frame_dets.append(fd)
            if writer is not None:
                writer.write(_draw_debug(frame, fd))
            if idx and idx % log_every == 0:
                fps = idx / (time.time() - t0)
                eta = (total - idx) / fps if fps and total else 0.0
                pct = f"{100 * idx / total:.0f}%" if total else "?"
                print(
                    f"  [stage1 {bout}/{split}] frame {idx}/{total or '?'} ({pct})  "
                    f"{fps:.1f} fps  ETA {eta / 60:.1f} min",
                    flush=True,
                )
    finally:
        reader.release()
        if writer is not None:
            writer.close()

    df = detections_to_df(frame_dets)
    write_parquet(out_dir / "detections.parquet", df)

    meta = ArtifactMeta(
        stage=STAGE,
        source_video=str(video_path),
        fps=info.fps,
        width=info.width,
        height=info.height,
        num_frames=len(frame_dets),
        git_sha=_git_sha(),
        created_utc=datetime.now(UTC).isoformat(),
        producer={
            "backend": stage.backend,
            "detector_weights": stage.detector_weights,
            "classifier_weights": stage.classifier_weights,
            "min_cls_conf": stage.min_cls_conf,
            "partial": max_frames is not None or start_frame > 0,
            "start_frame": start_frame,
        },
    )
    write_meta(out_dir, meta)
    return out_dir
