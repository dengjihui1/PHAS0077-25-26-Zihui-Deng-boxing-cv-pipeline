"""`bcv-window` — convert per-frame probabilities into strike windows."""

from __future__ import annotations

from pathlib import Path

from ..common.cli import base_parser
from ..common.config import load_config, load_pipeline_config
from .run import Stage4Config, run_stage4


def main() -> None:
    parser = base_parser("Stage 4: per-frame probs -> strike windows")
    args = parser.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    if args.output_root:
        pipeline = pipeline.model_copy(update={"output_root": Path(args.output_root)})
    cfg = load_config(args.config, Stage4Config)

    out = run_stage4(pipeline, cfg, bout=args.bout, split=args.split, debug=args.debug_video)
    print(f"[stage4_windowing] wrote {out}")


if __name__ == "__main__":
    main()
