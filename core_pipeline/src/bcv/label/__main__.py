"""`bcv-label` — launch the browser labelling GUI for one bout/split.

  uv run bcv-label --bout 120 --split 0 --mode bbox        # then open the tunneled URL
  ssh -L 8000:localhost:8000 <host>  ->  http://localhost:8000

Writes the editable keyframe project under ``output/label/`` and exports the canonical
``split_S_fighter_bboxes.json`` next to the bout's video on save.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from bcv.common.config import load_pipeline_config
from bcv.common.rounds import load_rounds, rounds_path

from .app import Session, create_app
from .boxes import load_project, project_path
from .frames import FrameSource


def _maybe_stage1(config: str):
    """Load Stage-1 chain config for pre-fill; return None if deps/config unavailable."""
    try:
        from bcv.common.config import load_config
        from bcv.stage1_detect.run import Stage1Config
        return load_config(config, Stage1Config)
    except Exception as e:  # pre-fill is optional; degrade gracefully
        print(f"[bcv-label] pre-fill disabled ({type(e).__name__}: {e})")
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Browser labelling GUI (bbox placer)")
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--stage1-config", default="configs/stage1_detect.yaml")
    p.add_argument("--stage2-config", default="configs/stage2_crop.yaml")
    p.add_argument("--bout", type=int, required=True)
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--mode", default="bbox", choices=["bbox"])
    p.add_argument("--export-file", default=None,
                   help="canonical bbox GT to write (default: split_S_fighter_bboxes.json by the video)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    video = pipeline.split_video(args.bout, args.split)
    if not video.exists():
        raise SystemExit(f"no split video: {video}")
    frames = FrameSource(video)

    proj_file = project_path(pipeline.output_root / "label", args.bout, args.split)
    project = load_project(proj_file, bout=args.bout, split=args.split, num_frames=frames.num_frames)
    export_file = Path(args.export_file) if args.export_file else (
        pipeline.bout_dir(args.bout) / f"split_{args.split}_fighter_bboxes.json"
    )
    stage_cfg = _maybe_stage1(args.stage1_config)
    rounds_file = rounds_path(pipeline.bout_dir(args.bout))
    rounds = load_rounds(pipeline.bout_dir(args.bout))
    try:
        from bcv.common.config import load_config
        from bcv.stage2_crop.cropper import Stage2Config
        crop_cfg = load_config(args.stage2_config, Stage2Config)
    except Exception:
        from bcv.stage2_crop.cropper import Stage2Config
        crop_cfg = Stage2Config()

    session = Session(
        bout=args.bout, split=args.split, mode=args.mode, frames=frames,
        project=project, project_file=proj_file, export_file=export_file,
        rounds_file=rounds_file, rounds=rounds, crop_cfg=crop_cfg,
        pipeline=pipeline if stage_cfg is not None else None, stage_cfg=stage_cfg,
    )
    print(f"[bcv-label] rounds: {len(rounds)} loaded from {rounds_file}")
    print(f"[bcv-label] bout {args.bout} split {args.split}: {frames.num_frames} frames "
          f"@ {frames.width}x{frames.height} {frames.fps:.1f}fps")
    print(f"[bcv-label] open http://{args.host}:{args.port}  (ssh -L {args.port}:localhost:{args.port} <host>)")
    print(f"[bcv-label] save -> {export_file}")
    uvicorn.run(create_app(session), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
