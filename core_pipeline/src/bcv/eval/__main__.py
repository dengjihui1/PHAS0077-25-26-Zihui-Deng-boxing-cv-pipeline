"""`bcv-eval` — detection / frame (ROC/PR) / window (event) evaluation + plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..common.annotations import load_runs
from ..common.config import load_pipeline_config
from ..common.io import read_meta
from .detection import detection_metrics, plot_detection_eval
from .frame import frame_metrics, plot_frame_eval
from .summary import build_summary
from .window import plot_window_eval, window_metrics


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate Stage 1 (detection) / Stage 3 (frame) / Stage 4 (window) / summary"
    )
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--bout", type=int, default=None)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--output-root", default=None)
    p.add_argument("--pred-parquet", default=None,
                   help="detection: predicted detections.parquet (default: stage1 output for bout/split)")
    p.add_argument("--gt-parquet", default=None,
                   help="detection: reference (GT) detections.parquet to compare against")
    p.add_argument("level", choices=["detection", "frame", "window", "summary"])
    args = p.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    if args.output_root:
        pipeline = pipeline.model_copy(update={"output_root": Path(args.output_root)})

    if args.level == "summary":
        build_summary(pipeline)
        return
    if args.bout is None:
        p.error("--bout is required for detection/frame/window eval")

    eval_dir = pipeline.artifact_dir(args.bout, args.split, "eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.level == "detection":
        if args.gt_parquet is None:
            p.error("--gt-parquet is required for detection eval")
        pred_path = Path(args.pred_parquet) if args.pred_parquet else (
            pipeline.artifact_dir(args.bout, args.split, "stage1_detect") / "detections.parquet"
        )
        pred_df = pd.read_parquet(pred_path)
        gt_df = pd.read_parquet(args.gt_parquet)
        metrics = detection_metrics(pred_df, gt_df)
        out_png = eval_dir / "detection_eval.png"
        plot_detection_eval(metrics, pred_df, gt_df, str(out_png))
    elif args.level == "frame":
        df = pd.read_parquet(
            pipeline.artifact_dir(args.bout, args.split, "stage3_frame_classifier")
            / "frame_probs.parquet"
        )
        metrics = frame_metrics(df)
        out_png = eval_dir / "frame_roc_pr.png"
        plot_frame_eval(df, str(out_png))
    else:
        s4_dir = pipeline.artifact_dir(args.bout, args.split, "stage4_windowing")
        windows = json.loads((s4_dir / "windows.json").read_text())["windows"]
        pred = [(w["start_frame"], w["end_frame"]) for w in windows]
        # Restrict GT to the frames Stage 4 actually analyzed — otherwise events outside
        # the processed segment count as misses and recall is meaningless.
        prod = read_meta(s4_dir).producer
        fmin, fmax = int(prod.get("frame_min", 0)), int(prod.get("frame_max", 10**12))
        gt = [
            (r.start_frame, r.end_frame)
            for r in load_runs(pipeline.bout_dir(args.bout))
            if not (r.end_frame < fmin or r.start_frame > fmax)
        ]
        metrics = window_metrics(pred, gt)
        metrics["eval_frame_range"] = [fmin, fmax]
        out_png = eval_dir / "window_eval.png"
        plot_window_eval(metrics, str(out_png))

    (eval_dir / f"{args.level}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[eval:{args.level}] {json.dumps(metrics, indent=2)}")
    print(f"[eval:{args.level}] plot -> {out_png}")


if __name__ == "__main__":
    main()
