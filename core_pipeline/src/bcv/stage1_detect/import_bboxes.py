"""Import legacy ``split_N_fighter_bboxes.json`` into our Stage-1 ``detections.parquet``.

External pipelines (e.g. Melik's) produced per-frame red/blue boxes in the old JSONL
format — line ``i`` is frame ``i``, a list of ``{bbox, det_conf, cls_confs:{red,blue,
unlabeled}}``. This converts them into a Stage-1 artifact by feeding each frame's
detections through the same ``select.py`` red/blue resolver, so downstream stages treat
them identically to a live detector's output (and we skip the hours-long detection pass).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ..common import viz
from ..common.config import PipelineConfig, load_pipeline_config
from ..common.contracts import ArtifactMeta
from ..common.geometry import to_int_box, union
from ..common.io import write_meta, write_parquet
from ..common.video import VideoReader, VideoWriter
from .interface import Candidate, FrameDetection
from .select import select_fighters
from .writer import detections_to_df

STAGE = "stage1_detect"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return None


def import_split(
    pipeline: PipelineConfig, *, bout: int, split: int, bbox_json: str | Path,
    min_cls_conf: float = 0.5,
    debug_video: bool = False,
) -> Path:
    """Convert one legacy bbox JSONL file into this split's detections.parquet artifact."""
    video_path = pipeline.split_video(bout, split)
    info = VideoReader(video_path).info
    lines = Path(bbox_json).read_text().splitlines()
    if info.num_frames and len(lines) != info.num_frames:
        print(f"  [warn] bbox frames {len(lines)} != video frames {info.num_frames}")

    frame_dets: list[FrameDetection] = []
    for i, line in enumerate(lines):
        dets = json.loads(line) if line.strip() else []
        cands = [
            Candidate(
                bbox=tuple(int(v) for v in d["bbox"]),  # type: ignore[arg-type]
                det_conf=float(d.get("det_conf", 0.0)),
                cls_conf={str(k).lower(): float(v) for k, v in d.get("cls_confs", {}).items()},
            )
            for d in dets
        ]
        red, blue = select_fighters(cands, min_cls_conf=min_cls_conf)
        frame_dets.append(FrameDetection(frame=i, red=red, blue=blue, n_candidates=len(cands)))

    df = detections_to_df(frame_dets)
    out_dir = pipeline.artifact_dir(bout, split, STAGE)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(out_dir / "detections.parquet", df)
    write_meta(
        out_dir,
        ArtifactMeta(
            stage=STAGE, source_video=str(video_path), fps=info.fps,
            width=info.width, height=info.height, num_frames=len(frame_dets),
            git_sha=_git_sha(), created_utc=datetime.now(UTC).isoformat(),
            producer={
                "backend": "imported_chain", "source_bbox": str(bbox_json),
                "min_cls_conf": min_cls_conf,
                "both_present_frac": float((df["red_present"] & df["blue_present"]).mean()),
            },
        ),
    )

    if debug_video:
        # draw red/blue fighter boxes + the union "crop box" (green) on the source video
        reader = VideoReader(video_path)
        writer = VideoWriter(out_dir / "boxes.mp4", info.fps, info.width, info.height)
        try:
            for fd, frame in zip(frame_dets, reader, strict=False):
                out = frame
                if fd.red is not None:
                    out = viz.draw_box(out, fd.red.bbox, viz.RED, label=f"red {fd.red.cls_conf:.2f}")
                if fd.blue is not None:
                    out = viz.draw_box(out, fd.blue.bbox, viz.BLUE, label=f"blue {fd.blue.cls_conf:.2f}")
                if fd.red is not None and fd.blue is not None:
                    out = viz.draw_box(out, to_int_box(union(fd.red.bbox, fd.blue.bbox)), viz.GREEN, thickness=1)
                writer.write(out)
        finally:
            reader.release()
            writer.close()
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Import legacy fighter_bboxes.json -> detections.parquet")
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--bout", type=int, required=True)
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--bbox-json", required=True)
    p.add_argument("--min-cls-conf", type=float, default=0.5)
    p.add_argument("--debug-video", action="store_true", help="render red/blue+union boxes on source")
    args = p.parse_args()
    pipeline = load_pipeline_config(args.pipeline_config)
    out = import_split(
        pipeline, bout=args.bout, split=args.split,
        bbox_json=args.bbox_json, min_cls_conf=args.min_cls_conf,
        debug_video=args.debug_video,
    )
    print(f"[import] bout {args.bout} split {args.split} -> {out}")


if __name__ == "__main__":
    main()
