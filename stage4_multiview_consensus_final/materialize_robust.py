"""Materialize the modal leave-one-bout-out consensus configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bcv.common.config import load_pipeline_config
from bcv.common.rounds import load_rounds

from sweep_consensus import (
    TEST_BOUTS,
    TRAIN_BOUTS,
    VAL_BOUTS,
    Params,
    fuse_views,
    gt_for_bout,
    in_round_mask,
    load_bout_views,
    make_all_predictions,
    moving_average,
    normalize_views,
    score_predictions,
)


ROBUST_PARAMS = Params(
    normalization="rank",
    fusion="mean",
    smooth=1,
    threshold=0.80,
    min_distance=6,
    radius=6,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--probs-root",
        type=Path,
        default=Path("Zihui/stage5_latest_20260720/output/stage3_frame_classifier"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    all_bouts = TRAIN_BOUTS + VAL_BOUTS + TEST_BOUTS
    raw = {}
    cache = {}
    gt = {}
    for bout in all_bouts:
        frames, probs = load_bout_views(pipeline, args.probs_root, bout)
        valid = in_round_mask(frames, load_rounds(pipeline.bout_dir(bout)))
        raw[bout] = (frames, probs, valid)
        fused = fuse_views(normalize_views(probs, valid, "rank"), "mean")
        fused[~valid] = 0.0
        cache[(bout, "rank", "mean", 1)] = moving_average(fused, 1)
        gt[bout] = gt_for_bout(pipeline, bout)

    predictions = make_all_predictions(ROBUST_PARAMS, raw, cache)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = args.output_dir / "robust_windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    for bout, windows in predictions.items():
        payload = {
            "bout": bout,
            "selection": "modal parameters from five development leave-one-bout-out folds",
            "params": asdict(ROBUST_PARAMS),
            "windows": [
                {
                    "window_id": index,
                    "start_frame": start,
                    "end_frame": end,
                    "peak_frame": peak,
                    "peak_prob": score,
                }
                for index, (start, end, peak, score) in enumerate(windows)
            ],
        }
        (windows_dir / f"bout_{bout}_consensus_windows.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    result = {
        "selection": "modal parameters from five development leave-one-bout-out folds; Bout 115 not used",
        "params": asdict(ROBUST_PARAMS),
        "train": score_predictions(predictions, gt, TRAIN_BOUTS, exact=True),
        "validation": score_predictions(predictions, gt, VAL_BOUTS, exact=True),
        "test": score_predictions(predictions, gt, TEST_BOUTS, exact=True),
    }
    (args.output_dir / "robust_cv_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
