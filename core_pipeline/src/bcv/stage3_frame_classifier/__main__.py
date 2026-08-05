"""`bcv-frame-clf` — the per-frame punch classifier.

Subcommands:
  fit        production train (held-out val only if val bouts != train bouts)
  predict    score one split from a checkpoint -> frame_probs.parquet (+ debug video)
  crossbout  train on N fights, evaluate the held-out fight  (the 0.887 reproducer)
  temporal   train on the first half of one split, evaluate the held-out tail
  eval-ckpt  predict-only held-out eval from a saved checkpoint (no training, no OOM)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..common.config import load_config, load_pipeline_config
from .run import (
    Stage3Config,
    eval_from_checkpoint,
    run_crossbout_eval,
    run_fit,
    run_predict,
    run_temporal_eval,
)


def _add_experiment_overrides(sp: argparse.ArgumentParser) -> None:
    """Flags that override the config for the held-out experiments. Defaults match the
    0.887 run (2D backbone @ 112px, big batch, 8 fixed epochs)."""
    sp.add_argument("--img-size", type=int, default=112)
    sp.add_argument("--backbone", default="channel_stack_2d")
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--max-epochs", type=int, default=8)
    sp.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default=None)
    sp.add_argument("--devices", default=None, help="Lightning devices value, e.g. 1")
    sp.add_argument("--precision", default=None, help="Lightning precision, e.g. 32-true or 16-mixed")


def _apply_overrides(cfg: Stage3Config, args: argparse.Namespace) -> Stage3Config:
    updates: dict = {}
    for attr, field in (
        ("img_size", "img_size"), ("backbone", "backbone"),
        ("batch_size", "batch_size"), ("max_epochs", "max_epochs"),
        ("accelerator", "accelerator"), ("devices", "devices"), ("precision", "precision"),
    ):
        val = getattr(args, attr, None)
        if val is not None:
            if field == "devices":
                try:
                    val = int(val)
                except (TypeError, ValueError):
                    pass
            updates[field] = val
    return cfg.model_copy(update=updates) if updates else cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 3: per-frame punch classifier")
    p.add_argument("--config", required=True)
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--output-root", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--train-bouts", type=int, nargs="*", default=None)
    f.add_argument("--val-bouts", type=int, nargs="*", default=None)
    f.add_argument("--train-splits", type=int, nargs="*", default=None)
    f.add_argument("--resume-from", default=None)
    f.add_argument("--max-epochs", type=int, default=None)
    f.add_argument("--early-stopping-patience", type=int, default=None)

    pr = sub.add_parser("predict")
    pr.add_argument("--ckpt", required=True)
    pr.add_argument("--bout", type=int, required=True)
    pr.add_argument("--split", type=int, default=0)
    pr.add_argument("--debug-video", action=argparse.BooleanOptionalAction, default=True)

    cb = sub.add_parser("crossbout", help="train on N fights, evaluate the held-out fight")
    cb.add_argument("--train-bouts", type=int, nargs="+", default=[116, 117])
    cb.add_argument("--train-splits", type=int, nargs="*", default=None)
    cb.add_argument("--resume-from", default=None)
    cb.add_argument("--eval-bout", type=int, default=115)
    cb.add_argument("--eval-split", type=int, default=1)
    _add_experiment_overrides(cb)

    tp = sub.add_parser("temporal", help="train first half of one split, eval the tail")
    tp.add_argument("--bout", type=int, default=116)
    tp.add_argument("--split", type=int, default=0)
    tp.add_argument("--train-frac", type=float, default=0.5)
    tp.add_argument("--gap-frac", type=float, default=0.01)
    _add_experiment_overrides(tp)

    ec = sub.add_parser("eval-ckpt", help="predict-only held-out eval from a checkpoint")
    ec.add_argument("--ckpt", required=True)
    ec.add_argument("--eval-bout", type=int, default=115)
    ec.add_argument("--eval-split", type=int, default=1)
    ec.add_argument("--img-size", type=int, default=112)

    args = p.parse_args()
    pipeline = load_pipeline_config(args.pipeline_config)
    if args.output_root:
        pipeline = pipeline.model_copy(update={"output_root": Path(args.output_root)})
    cfg = load_config(args.config, Stage3Config)

    if args.cmd == "fit":
        updates = {}
        if args.max_epochs is not None:
            updates["max_epochs"] = args.max_epochs
        if args.early_stopping_patience is not None:
            updates["early_stopping_patience"] = args.early_stopping_patience
        if updates:
            cfg = cfg.model_copy(update=updates)
        ckpt = run_fit(
            pipeline, cfg, train_bouts=args.train_bouts, val_bouts=args.val_bouts,
            train_splits=args.train_splits, resume_from=args.resume_from,
        )
        print(f"[stage3] best checkpoint: {ckpt}")
    elif args.cmd == "predict":
        out = run_predict(
            pipeline, cfg, args.ckpt, bout=args.bout, split=args.split,
            debug_video=args.debug_video,
        )
        print(f"[stage3] wrote {out}")
    elif args.cmd == "crossbout":
        cfg = _apply_overrides(cfg, args)
        run_crossbout_eval(
            pipeline, cfg, train_bouts=args.train_bouts,
            train_splits=args.train_splits,
            resume_from=args.resume_from,
            eval_bout=args.eval_bout, eval_split=args.eval_split,
        )
    elif args.cmd == "temporal":
        cfg = _apply_overrides(cfg, args)
        run_temporal_eval(
            pipeline, cfg, bout=args.bout, split=args.split,
            train_frac=args.train_frac, gap_frac=args.gap_frac,
        )
    elif args.cmd == "eval-ckpt":
        cfg = cfg.model_copy(update={"img_size": args.img_size})
        eval_from_checkpoint(
            pipeline, cfg, args.ckpt, eval_bout=args.eval_bout, eval_split=args.eval_split,
        )


if __name__ == "__main__":
    main()
