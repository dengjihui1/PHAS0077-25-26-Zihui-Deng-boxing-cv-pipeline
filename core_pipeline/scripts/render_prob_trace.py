"""Render Stage-3 punch-classifier overlay videos (prob_trace.mp4) from saved probs.

Takes an already-computed frame_probs.parquet + the Stage-2 crop.mp4 and draws the
P(punch) trace (p_smooth) along the bottom with a threshold line and the GT punch band —
no model re-run. Output: prob_trace.mp4 next to the frame_probs (the eval_crossbout dir).
"""
from __future__ import annotations

import argparse

import pandas as pd

from bcv.common.config import load_config, load_pipeline_config
from bcv.common.io import read_meta
from bcv.stage3_frame_classifier.run import _write_debug
from bcv.stage4_windowing.run import Stage4Config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--bout", type=int, required=True)
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--threshold", type=float, default=None,
                   help="trace threshold line (default: Stage-4 t_high)")
    p.add_argument("--window-seconds", type=float, default=10.0,
                   help="rolling window for the bottom strip (0 = whole fight)")
    args = p.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    thr = args.threshold if args.threshold is not None else \
        load_config("configs/stage4_windowing.yaml", Stage4Config).t_high

    ev = pipeline.artifact_dir(args.bout, args.split, "eval_crossbout")
    crop = pipeline.artifact_dir(args.bout, args.split, "stage2_crop")
    fp = ev / "frame_probs.parquet"
    if not fp.exists() or not (crop / "crop.mp4").exists():
        raise SystemExit(f"missing frame_probs or crop for {args.bout}/{args.split}")

    df = pd.read_parquet(fp)
    fps = read_meta(crop).fps
    labels = df["label"].to_numpy() if "label" in df else None
    _write_debug(crop, ev, df, fps, thr, labels=labels, window_seconds=args.window_seconds)
    print(f"{args.bout}/{args.split} -> {ev / 'prob_trace.mp4'}  (thr={thr}, win={args.window_seconds}s)")


if __name__ == "__main__":
    main()
