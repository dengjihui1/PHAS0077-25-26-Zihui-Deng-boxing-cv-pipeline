"""`bcv-crop` — reframe a split video to the fixed-square fighter crop."""

from __future__ import annotations

from ..common.cli import base_parser
from ..common.config import load_config, load_pipeline_config
from .cropper import Stage2Config
from .run import run_stage2


def main() -> None:
    parser = base_parser("Stage 2: crop to the fighters (fixed square)")
    args = parser.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    if args.output_root:
        from pathlib import Path

        pipeline = pipeline.model_copy(update={"output_root": Path(args.output_root)})
    stage = load_config(args.config, Stage2Config)

    out_dir = run_stage2(
        pipeline, stage, bout=args.bout, split=args.split, debug_video=args.debug_video
    )
    print(f"[stage2_crop] wrote {out_dir}")


if __name__ == "__main__":
    main()
