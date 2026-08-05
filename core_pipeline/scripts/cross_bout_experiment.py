"""Cross-bout experiment: train Stage 3 on N fights, evaluate a held-out fight.

Thin shim — the logic now lives in ``bcv.stage3_frame_classifier.run.run_crossbout_eval``
(reachable as ``bcv-frame-clf crossbout``). Kept so the documented command and the other
scripts that call it keep working.
"""
from __future__ import annotations

import argparse

import torch

from bcv.common.config import load_config, load_pipeline_config
from bcv.stage3_frame_classifier.run import Stage3Config, run_crossbout_eval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage3_frame_classifier.yaml")
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--train-bouts", type=int, nargs="+", default=[116, 117])
    p.add_argument("--eval-bout", type=int, default=115)
    p.add_argument("--eval-split", type=int, default=1)
    p.add_argument("--img-size", type=int, default=112)
    p.add_argument("--backbone", default="channel_stack_2d")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=8)
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    pipeline = load_pipeline_config(args.pipeline_config)
    cfg = load_config(args.config, Stage3Config).model_copy(update={
        "img_size": args.img_size, "backbone": args.backbone,
        "batch_size": args.batch_size, "max_epochs": args.max_epochs,
    })
    run_crossbout_eval(
        pipeline, cfg, train_bouts=args.train_bouts,
        eval_bout=args.eval_bout, eval_split=args.eval_split,
    )


if __name__ == "__main__":
    main()
