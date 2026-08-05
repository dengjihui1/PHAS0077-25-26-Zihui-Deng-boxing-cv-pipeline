"""Cross-bout sweep for synchronized multi-view punch proposals.

The existing pipeline detects windows independently in each camera split. This script
aligns the per-frame Stage-3 probabilities on the shared bout timeline, normalizes each
view, fuses the available views, and emits short peak-centred event proposals.

Selection protocol:
  train diagnostics: bouts 116, 117, 120, 121
  parameter selection: bout 122
  held-out test: bout 115 (evaluated once after selection)
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from bcv.common.annotations import load_runs
from bcv.common.config import load_pipeline_config
from bcv.common.rounds import load_rounds
from bcv.eval.window import window_metrics


TRAIN_BOUTS = (116, 117, 120, 121)
VAL_BOUTS = (122,)
TEST_BOUTS = (115,)


@dataclass(frozen=True)
class Params:
    normalization: str
    fusion: str
    smooth: int
    threshold: float
    min_distance: int
    radius: int


def _safe_f1(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _prob_path(probs_root: Path, bout_name: str, split: int) -> Path:
    return probs_root / bout_name / f"split_{split}" / "frame_probs.parquet"


def load_bout_views(pipeline, probs_root: Path, bout: int) -> tuple[np.ndarray, np.ndarray]:
    """Return shared frames and [views, frames] probabilities with NaN for invalid crops."""
    series: list[pd.Series] = []
    bout_name = pipeline.bouts[bout]
    for split in range(4):
        path = _prob_path(probs_root, bout_name, split)
        if not path.exists():
            continue
        frame_probs = pd.read_parquet(path, columns=["frame", "p_punch", "crop_valid"])
        values = frame_probs["p_punch"].astype(float).where(frame_probs["crop_valid"].astype(bool))
        series.append(pd.Series(values.to_numpy(), index=frame_probs["frame"].astype(int), name=f"split_{split}"))
    if not series:
        raise FileNotFoundError(f"no frame probabilities for bout {bout} below {probs_root}")
    aligned = pd.concat(series, axis=1).sort_index()
    frames = aligned.index.to_numpy(dtype=np.int64)
    probs = aligned.to_numpy(dtype=np.float64).T
    return frames, probs


def in_round_mask(frames: np.ndarray, rounds: list[tuple[int, int]]) -> np.ndarray:
    if not rounds:
        return np.ones(len(frames), dtype=bool)
    mask = np.zeros(len(frames), dtype=bool)
    for start, end in rounds:
        mask |= (frames >= int(start)) & (frames <= int(end))
    return mask


def normalize_views(probs: np.ndarray, valid_time: np.ndarray, mode: str) -> np.ndarray:
    out = probs.copy()
    if mode == "raw":
        return out
    if mode != "rank":
        raise ValueError(f"unknown normalization: {mode}")
    for view in range(out.shape[0]):
        valid = np.isfinite(out[view]) & valid_time
        if not valid.any():
            continue
        values = out[view, valid]
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
        out[view, valid] = ranks
    return out


def fuse_views(probs: np.ndarray, mode: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if mode == "max":
            fused = np.nanmax(probs, axis=0)
        elif mode == "mean":
            fused = np.nanmean(probs, axis=0)
        elif mode == "median":
            fused = np.nanmedian(probs, axis=0)
        elif mode == "top2_mean":
            ordered = np.sort(np.where(np.isfinite(probs), probs, -np.inf), axis=0)
            top = ordered[-2:]
            top[top == -np.inf] = np.nan
            fused = np.nanmean(top, axis=0)
        else:
            raise ValueError(f"unknown fusion: {mode}")
    return np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    kernel = np.ones(int(width), dtype=np.float64) / int(width)
    return np.convolve(values, kernel, mode="same")


def peak_windows(
    frames: np.ndarray,
    score: np.ndarray,
    valid_time: np.ndarray,
    *,
    threshold: float,
    min_distance: int,
    radius: int,
) -> list[tuple[int, int, int, float]]:
    """Threshold local maxima, then apply score-ordered temporal NMS."""
    usable = score.copy()
    usable[~valid_time] = 0.0
    local = np.zeros(len(usable), dtype=bool)
    if len(usable) >= 3:
        local[1:-1] = (usable[1:-1] >= usable[:-2]) & (usable[1:-1] > usable[2:])
    candidates = np.flatnonzero(local & (usable >= float(threshold)))
    selected: list[int] = []
    for index in candidates[np.argsort(usable[candidates])[::-1]]:
        if all(abs(int(index) - kept) >= int(min_distance) for kept in selected):
            selected.append(int(index))
    selected.sort()
    return [
        (
            int(frames[max(0, index - radius)]),
            int(frames[min(len(frames) - 1, index + radius)]),
            int(frames[index]),
            float(usable[index]),
        )
        for index in selected
    ]


def gt_for_bout(pipeline, bout: int) -> list[tuple[int, int, str]]:
    return [(r.start_frame, r.end_frame, r.label) for r in load_runs(pipeline.bout_dir(bout))]


def fast_bout_metrics(pred: list[tuple[int, int]], gt: list[tuple[int, int, str]]) -> dict:
    """Linear-time chronological matching used only inside the broad parameter sweep."""
    pred = sorted(pred)
    gt = sorted(gt)
    pred_index = gt_index = matched = 0
    while pred_index < len(pred) and gt_index < len(gt):
        ps, pe = pred[pred_index]
        gs, ge, _label = gt[gt_index]
        if pe < gs:
            pred_index += 1
        elif ge < ps:
            gt_index += 1
        else:
            matched += 1
            pred_index += 1
            gt_index += 1

    clean = multi = empty = 0
    left = 0
    for ps, pe in pred:
        while left < len(gt) and gt[left][1] < ps:
            left += 1
        labels: set[str] = set()
        cursor = left
        while cursor < len(gt) and gt[cursor][0] <= pe:
            labels.add(gt[cursor][2])
            cursor += 1
        if not labels:
            empty += 1
        elif len(labels) == 1:
            clean += 1
        else:
            multi += 1
    precision = matched / len(pred) if pred else 0.0
    recall = matched / len(gt) if gt else 0.0
    return {
        "n_gt_events": len(gt),
        "n_pred_windows": len(pred),
        "n_matched": matched,
        "n_missed_events": len(gt) - matched,
        "n_false_alarms": len(pred) - matched,
        "precision": precision,
        "recall": recall,
        "f1": _safe_f1(precision, recall),
        "clean_windows": clean,
        "multi_windows": multi,
        "empty_windows": empty,
    }


def score_predictions(
    predictions: dict[int, list[tuple[int, int, int, float]]],
    gt_by_bout: dict[int, list[tuple[int, int, str]]],
    bouts: tuple[int, ...],
    *,
    exact: bool,
) -> dict:
    clean = multi = empty = 0
    per_bout: dict[str, dict] = {}
    totals = {
        "n_gt_events": 0,
        "n_pred_windows": 0,
        "n_matched": 0,
        "n_missed_events": 0,
        "n_false_alarms": 0,
        "n_missed": 0,
        "n_fake": 0,
        "n_detected": 0,
        "n_merged": 0,
        "n_oversplit": 0,
    }
    for bout in bouts:
        pred = predictions[bout]
        gt = gt_by_bout[bout]
        spans = [(start, end) for start, end, _peak, _score in pred]
        if exact:
            metrics = window_metrics(spans, [(start, end) for start, end, _label in gt])
            metrics["f1"] = _safe_f1(metrics["precision"], metrics["recall"])
        else:
            metrics = fast_bout_metrics(spans, gt)
        per_bout[str(bout)] = metrics
        totals["n_gt_events"] += int(metrics["n_gt_events"])
        totals["n_pred_windows"] += int(metrics["n_pred_windows"])
        totals["n_matched"] += int(metrics["n_matched"])
        totals["n_missed_events"] += int(metrics["n_missed_events"])
        totals["n_false_alarms"] += int(metrics["n_false_alarms"])
        if exact:
            for key in ("n_missed", "n_fake", "n_detected", "n_merged", "n_oversplit"):
                totals[key] += int(metrics[key])
        for start, end in spans:
            labels = {label for gs, ge, label in gt if not (ge < start or gs > end)}
            if len(labels) == 0:
                empty += 1
            elif len(labels) == 1:
                clean += 1
            else:
                multi += 1
    precision = totals["n_matched"] / totals["n_pred_windows"] if totals["n_pred_windows"] else 0.0
    recall = totals["n_matched"] / totals["n_gt_events"] if totals["n_gt_events"] else 0.0
    if exact:
        detection_precision = totals["n_pred_windows"] - totals["n_fake"]
        detection_precision = detection_precision / totals["n_pred_windows"] if totals["n_pred_windows"] else 0.0
        detection_recall = totals["n_detected"] / totals["n_gt_events"] if totals["n_gt_events"] else 0.0
    else:
        totals["n_missed"] = totals["n_missed_events"]
        totals["n_fake"] = totals["n_false_alarms"]
        totals["n_detected"] = totals["n_matched"]
        detection_precision = precision
        detection_recall = recall
    metrics = {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": _safe_f1(precision, recall),
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "miss_rate": totals["n_missed"] / totals["n_gt_events"] if totals["n_gt_events"] else 0.0,
        "fake_rate": totals["n_fake"] / totals["n_pred_windows"] if totals["n_pred_windows"] else 0.0,
    }
    total = clean + multi + empty
    metrics.update(
        {
            "clean_windows": clean,
            "multi_windows": multi,
            "empty_windows": empty,
            "clean_fraction": clean / total if total else 0.0,
            "multi_fraction": multi / total if total else 0.0,
            "per_bout": per_bout,
        }
    )
    return metrics


def load_existing_baseline(pipeline, windows_root: Path, bout: int) -> dict:
    gt = gt_for_bout(pipeline, bout)
    rows = []
    for split in range(4):
        path = windows_root / pipeline.bouts[bout] / f"split_{split}" / "windows.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        windows = payload.get("windows", payload) if isinstance(payload, dict) else payload
        pred = [(int(row["start_frame"]), int(row["end_frame"])) for row in windows]
        metrics = window_metrics(pred, [(s, e) for s, e, _label in gt])
        metrics["f1"] = _safe_f1(metrics["precision"], metrics["recall"])
        metrics["split"] = split
        rows.append(metrics)
    return {
        "splits": rows,
        "mean_f1": float(np.mean([row["f1"] for row in rows])) if rows else 0.0,
        "best_f1": float(max((row["f1"] for row in rows), default=0.0)),
    }


def parameter_grid() -> list[Params]:
    rows: list[Params] = []
    for normalization, fusion, smooth, min_distance, radius in product(
        ("raw", "rank"),
        ("max", "top2_mean", "mean", "median"),
        (1, 3, 5),
        (4, 6, 8, 10, 12),
        (2, 4, 6),
    ):
        thresholds = (0.45, 0.55, 0.65, 0.75, 0.85) if normalization == "raw" else (0.80, 0.85, 0.90, 0.93, 0.96)
        rows.extend(
            Params(normalization, fusion, smooth, threshold, min_distance, radius)
            for threshold in thresholds
        )
    return rows


def params_from_row(row: pd.Series) -> Params:
    return Params(
        normalization=str(row.normalization),
        fusion=str(row.fusion),
        smooth=int(row.smooth),
        threshold=float(row.threshold),
        min_distance=int(row.min_distance),
        radius=int(row.radius),
    )


def make_all_predictions(
    params: Params,
    raw: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    cache: dict[tuple[int, str, str, int], np.ndarray],
) -> dict[int, list[tuple[int, int, int, float]]]:
    predictions = {}
    for bout, (frames, _probs, valid_time) in raw.items():
        score = cache[(bout, params.normalization, params.fusion, params.smooth)]
        predictions[bout] = peak_windows(
            frames,
            score,
            valid_time,
            threshold=params.threshold,
            min_distance=params.min_distance,
            radius=params.radius,
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--probs-root",
        type=Path,
        default=Path("Zihui/stage5_latest_20260720/output/stage3_frame_classifier"),
    )
    parser.add_argument(
        "--baseline-windows-root",
        type=Path,
        default=Path("Zihui/stage4_stage5_clean_windows_20260724/output/stage4_windowing"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    all_bouts = TRAIN_BOUTS + VAL_BOUTS + TEST_BOUTS
    raw: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    gt_by_bout = {bout: gt_for_bout(pipeline, bout) for bout in all_bouts}
    for bout in all_bouts:
        frames, probs = load_bout_views(pipeline, args.probs_root, bout)
        rounds = load_rounds(pipeline.bout_dir(bout))
        raw[bout] = (frames, probs, in_round_mask(frames, rounds))
        print(
            f"loaded bout {bout}: {probs.shape[0]} views x {probs.shape[1]} frames, "
            f"{len(gt_by_bout[bout])} GT",
            flush=True,
        )

    cache: dict[tuple[int, str, str, int], np.ndarray] = {}
    for bout, (frames, probs, valid_time) in raw.items():
        for normalization, fusion, smooth in product(
            ("raw", "rank"), ("max", "top2_mean", "mean", "median"), (1, 3, 5)
        ):
            normalized = normalize_views(probs, valid_time, normalization)
            fused = fuse_views(normalized, fusion)
            fused[~valid_time] = 0.0
            cache[(bout, normalization, fusion, smooth)] = moving_average(fused, smooth)

    sweep_rows: list[dict] = []
    for index, params in enumerate(parameter_grid(), start=1):
        predictions = {}
        for bout, (frames, _probs, valid_time) in raw.items():
            score = cache[(bout, params.normalization, params.fusion, params.smooth)]
            predictions[bout] = peak_windows(
                frames,
                score,
                valid_time,
                threshold=params.threshold,
                min_distance=params.min_distance,
                radius=params.radius,
            )
        train = score_predictions(predictions, gt_by_bout, TRAIN_BOUTS, exact=False)
        val = score_predictions(predictions, gt_by_bout, VAL_BOUTS, exact=False)
        sweep_rows.append(
            {
                **asdict(params),
                "train_f1": train["f1"],
                "train_precision": train["precision"],
                "train_recall": train["recall"],
                "train_clean_fraction": train["clean_fraction"],
                "val_f1": val["f1"],
                "val_precision": val["precision"],
                "val_recall": val["recall"],
                "val_clean_fraction": val["clean_fraction"],
                "val_n_pred": val["n_pred_windows"],
                **{
                    f"bout_{bout}_f1": train["per_bout"][str(bout)]["f1"]
                    for bout in TRAIN_BOUTS
                },
                "bout_122_f1": val["per_bout"]["122"]["f1"],
            }
        )
        if index % 300 == 0:
            print(f"swept {index} parameter sets", flush=True)

    sweep = pd.DataFrame(sweep_rows).sort_values(
        ["val_f1", "train_f1", "val_clean_fraction"], ascending=False
    )
    best_row = sweep.iloc[0]
    best = params_from_row(best_row)
    predictions = make_all_predictions(best, raw, cache)
    train_metrics = score_predictions(predictions, gt_by_bout, TRAIN_BOUTS, exact=True)
    val_metrics = score_predictions(predictions, gt_by_bout, VAL_BOUTS, exact=True)
    test_metrics = score_predictions(predictions, gt_by_bout, TEST_BOUTS, exact=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_dir / "sweep.csv", index=False)
    for bout in all_bouts:
        payload = {
            "bout": bout,
            "params": asdict(best),
            "windows": [
                {
                    "window_id": i,
                    "start_frame": start,
                    "end_frame": end,
                    "peak_frame": peak,
                    "peak_prob": score,
                }
                for i, (start, end, peak, score) in enumerate(predictions[bout])
            ],
        }
        (args.output_dir / f"bout_{bout}_consensus_windows.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    baseline = {str(bout): load_existing_baseline(pipeline, args.baseline_windows_root, bout) for bout in all_bouts}
    development_bouts = TRAIN_BOUTS + VAL_BOUTS
    cross_validation = {}
    for held_out in development_bouts:
        selection_columns = [f"bout_{bout}_f1" for bout in development_bouts if bout != held_out]
        selection_score = sweep[selection_columns].mean(axis=1)
        cv_row = sweep.loc[selection_score.idxmax()]
        cv_params = params_from_row(cv_row)
        cv_predictions = make_all_predictions(cv_params, raw, cache)
        held_metrics = score_predictions(cv_predictions, gt_by_bout, (held_out,), exact=True)
        cross_validation[str(held_out)] = {
            "selected_params": asdict(cv_params),
            "selection_mean_f1": float(selection_score.loc[cv_row.name]),
            "held_out": held_metrics,
        }
    result = {
        "protocol": {
            "train_bouts": list(TRAIN_BOUTS),
            "validation_bouts": list(VAL_BOUTS),
            "test_bouts": list(TEST_BOUTS),
            "test_used_for_selection": False,
        },
        "selected_params": asdict(best),
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "development_leave_one_bout_out": cross_validation,
        "existing_independent_view_baseline": baseline,
    }
    (args.output_dir / "selected_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
