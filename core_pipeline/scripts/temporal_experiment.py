"""Within-fight experiment: train on the first half of one split, eval the held-out tail.

Thin shim — logic now lives in ``bcv.stage3_frame_classifier.run.run_temporal_eval``
(reachable as ``bcv-frame-clf temporal``).
"""
from __future__ import annotations

import argparse

import torch

from bcv.common.config import load_config, load_pipeline_config
from bcv.stage3_frame_classifier.run import Stage3Config, run_temporal_eval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage3_frame_classifier.yaml")
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--bout", type=int, default=116)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--gap-frac", type=float, default=0.01)
    p.add_argument("--max-epochs", type=int, default=8)
    p.add_argument("--img-size", type=int, default=112)
    p.add_argument("--backbone", default="channel_stack_2d")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    pipeline = load_pipeline_config(args.pipeline_config)
    cfg = load_config(args.config, Stage3Config).model_copy(update={
        "img_size": args.img_size, "backbone": args.backbone,
        "batch_size": args.batch_size, "max_epochs": args.max_epochs,
    })
    run_temporal_eval(
        pipeline, cfg, bout=args.bout, split=args.split,
        train_frac=args.train_frac, gap_frac=args.gap_frac,
    )


if __name__ == "__main__":
    main()
