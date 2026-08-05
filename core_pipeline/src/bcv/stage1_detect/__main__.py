"""`bcv-detect` — run Stage 1 fighter detection on one split video."""

from __future__ import annotations

from ..common.cli import base_parser
from ..common.config import load_config, load_pipeline_config
from .run import Stage1Config, run_stage1


def main() -> None:
    parser = base_parser("Stage 1: fighter bounding-box detection")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Process only N frames from --start-frame"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="Absolute frame index to start at"
    )
    args = parser.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    if args.output_root:
        from pathlib import Path

        pipeline = pipeline.model_copy(update={"output_root": Path(args.output_root)})
    stage = load_config(args.config, Stage1Config)

    out_dir = run_stage1(
        pipeline,
        stage,
        bout=args.bout,
        split=args.split,
        debug_video=args.debug_video,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
    )
    print(f"[stage1_detect] wrote {out_dir}")


if __name__ == "__main__":
    main()
