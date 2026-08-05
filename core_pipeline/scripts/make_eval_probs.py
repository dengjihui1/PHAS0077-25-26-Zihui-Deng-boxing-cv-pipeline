"""Regenerate the held-out frame_probs.parquet from a saved Stage-3 checkpoint.

Thin shim — logic now lives in ``bcv.stage3_frame_classifier.run.eval_from_checkpoint``
(reachable as ``bcv-frame-clf eval-ckpt``). Predict-only: loads ONE eval split + a
checkpoint (no training, no train-data load -> no OOM), temperature=1.0.
"""
from __future__ import annotations

import argparse

import torch

from bcv.common.config import load_config, load_pipeline_config
from bcv.stage3_frame_classifier.run import Stage3Config, eval_from_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage3_frame_classifier.yaml")
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument(
        "--ckpt",
        default=".cometml-runs/boxing-stage3-frame/"
        "0f0def58ba61412ab2e4937c3b1ef645/checkpoints/epoch=7-step=11136.ckpt",
    )
    p.add_argument("--eval-bout", type=int, default=115)
    p.add_argument("--eval-split", type=int, default=1)
    p.add_argument("--img-size", type=int, default=112)
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    pipeline = load_pipeline_config(args.pipeline_config)
    cfg = load_config(args.config, Stage3Config).model_copy(update={"img_size": args.img_size})
    eval_from_checkpoint(
        pipeline, cfg, args.ckpt, eval_bout=args.eval_bout, eval_split=args.eval_split,
    )


if __name__ == "__main__":
    main()
