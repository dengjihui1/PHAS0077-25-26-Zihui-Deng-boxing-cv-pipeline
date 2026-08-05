"""Sweep Stage-4 hysteresis thresholds (+ temperature) against saved frame probs.

The 0.887 model is fine; Stage-4 was emitting a few giant blobs purely because the
default operating point (t_high=0.5 / t_low=0.35) sits on the wrong part of the
over-confident probability curve. This loads a frozen frame_probs.parquet (no model,
no retrain), recovers logits from the uncalibrated p_raw, and grid-searches
t_high x t_low x temperature for the best window-detection F1, then writes the winner
into configs/stage4_windowing.yaml.

Temperature acts before smoothing: p = sigmoid(logit / T) -> rolling mean -> hysteresis.
It is fit on the eval labels here (in-sample, like the rest of the pipeline today); the
threshold sweep is the dominant lever, so this is a pragmatic operating-point search,
not a calibration guarantee. See HANDOFF step #5 for the in-sample-fit caveat.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bcv.common.annotations import load_runs
from bcv.common.config import load_config, load_pipeline_config
from bcv.eval.window import plot_window_eval, window_metrics
from bcv.stage3_frame_classifier.calibrate import fit_temperature
from bcv.stage4_windowing.hysteresis import make_windows
from bcv.stage4_windowing.run import Stage4Config

ROLL_W = 11  # must match build_frame_probs / stage3 cfg roll_w


def _f1(recall: float, precision: float) -> float:
    if not np.isfinite(recall) or not np.isfinite(precision) or recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def _probs_at_temperature(logits: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    p_punch = 1.0 / (1.0 + np.exp(-logits / t))
    p_smooth = pd.Series(p_punch).rolling(ROLL_W, center=True, min_periods=1).mean().to_numpy()
    return p_punch, p_smooth


def _plot_window_curves(
    frames: np.ndarray,
    p_punch: np.ndarray,
    p_smooth: np.ndarray,
    gt: list[tuple[int, int]],
    best: pd.Series,
    out_png: str,
) -> None:
    """Window-detection operating curves by sweeping a single threshold theta.

    A literal ROC needs a true-negative count, which is undefined for event detection
    (the negative class = every window you could emit). The detection analogs are the
    Precision-Recall curve and the FROC (recall vs absolute false-alarm count). Sweep
    theta over p_smooth (t_high=t_low=theta) with min_duration/merge_gap fixed at the
    chosen operating point; mark that point on each curve.
    """
    md, mg = int(best["min_dur"]), int(best["merge_gap"])
    split_valley = None if pd.isna(best["split_valley"]) else float(best["split_valley"])
    split_min_gap = int(best["split_min_gap"])
    split_peak_min_prob = (
        None if pd.isna(best["split_peak_min_prob"]) else float(best["split_peak_min_prob"])
    )
    split_peak_min_distance = int(best["split_peak_min_distance"])
    split_peak_min_drop = float(best["split_peak_min_drop"])
    thetas = np.round(np.arange(0.50, 0.995, 0.01), 3)
    rec, prec, falses = [], [], []
    for th in thetas:
        ws = make_windows(
            frames,
            p_smooth,
            p_punch,
            t_high=float(th),
            t_low=float(th),
            min_duration=md,
            merge_gap=mg,
            split_valley=split_valley,
            split_min_gap=split_min_gap,
            split_peak_min_prob=split_peak_min_prob,
            split_peak_min_distance=split_peak_min_distance,
            split_peak_min_drop=split_peak_min_drop,
        )
        wm = window_metrics([(w.start_frame, w.end_frame) for w in ws], gt)
        rec.append(wm["recall"])
        prec.append(wm["precision"])
        falses.append(wm["n_false_alarms"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot(rec, prec, "-o", ms=2, color="tab:blue")
    a1.scatter(
        [best["recall"]],
        [best["precision"]],
        color="tab:red",
        zorder=5,
        label=f"chosen (theta~{best['t_high']:.2f})",
    )
    a1.set(xlabel="recall", ylabel="precision", title="Window PR curve", xlim=(0, 1), ylim=(0, 1))
    a1.legend(loc="lower left", fontsize=8)
    a2.plot(falses, rec, "-o", ms=2, color="tab:green")
    a2.scatter([best["false"]], [best["recall"]], color="tab:red", zorder=5)
    a2.set(xlabel="false alarms (count)", ylabel="recall (sensitivity)",
           title="Window FROC", ylim=(0, 1))
    fig.suptitle("Stage 4 - window-detection operating curves (threshold sweep)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _peak_param_grid() -> list[tuple[float | None, int, float]]:
    """Candidate peak-splitting settings.

    The None row preserves the non-peak-split behavior. Numeric rows are intentionally
    compact: this script is meant to be rerun often while diagnosing Stage 4, so we keep
    the peak sweep targeted instead of multiplying every threshold row by a huge grid.
    """
    out: list[tuple[float | None, int, float]] = [(None, 0, 0.0)]
    for prob in (0.75, 0.80, 0.85, 0.90):
        for distance in (6, 8, 12, 16):
            for drop in (0.05, 0.10, 0.20):
                out.append((prob, distance, drop))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--eval-bout", type=int, default=115)
    p.add_argument("--eval-split", type=int, default=1)
    p.add_argument("--out-config", default="configs/stage4_windowing.yaml")
    p.add_argument(
        "--base-candidates",
        type=int,
        default=80,
        help="number of non-peak operating points to refine with peak splitting",
    )
    p.add_argument("--write", action="store_true", help="write the winner into --out-config")
    args = p.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    s4 = load_config(args.out_config, Stage4Config)
    out_dir = pipeline.artifact_dir(args.eval_bout, args.eval_split, "eval_crossbout")
    pq = out_dir / "frame_probs.parquet"
    if not pq.exists():
        raise SystemExit(f"no {pq} — run scripts/make_eval_probs.py first")
    df = pd.read_parquet(pq)
    frames = df["frame"].to_numpy()

    # Recover logits from the calibration-free raw sigmoid (build_frame_probs wrote p_raw at T=1).
    p_raw = np.clip(df["p_raw"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(p_raw / (1 - p_raw))
    labels = df["label"].to_numpy()

    fmin, fmax = int(frames.min()), int(frames.max())
    gt = [(r.start_frame, r.end_frame) for r in load_runs(pipeline.bout_dir(args.eval_bout))
          if not (r.end_frame < fmin or r.start_frame > fmax)]
    print(f"loaded {len(df)} frames, {len(gt)} GT events in [{fmin},{fmax}]", flush=True)

    fitted_t = fit_temperature(logits, labels)
    temps = sorted({1.0, round(fitted_t, 3)})
    print(f"temperatures: {temps}  (fitted T={fitted_t:.3f})", flush=True)

    highs = np.round(np.arange(0.55, 0.96, 0.05), 2)
    lows = np.round(np.arange(0.35, 0.91, 0.05), 2)
    # Smoothing (roll_w=11) + merge_gap fuse adjacent punches into one window, and greedy
    # matching scores only one GT per window -> a structural recall ceiling. Sweep the
    # split/merge knobs too so the search can trade fewer-merges for higher recall.
    min_durs = [1, 2, 3]
    merge_gaps = [0, 1, 2]
    # Optional valley splitting cuts long hysteresis spans at sustained low-probability
    # dips. None preserves the old behavior; numeric values are candidate split levels.
    split_valleys: list[float | None] = [None, 0.50, 0.60, 0.70, 0.80]
    split_min_gaps = [2, 3, 5]
    # First do the original threshold/valley search without peak splitting. Then refine
    # only the most promising rows with peak parameters. A full Cartesian product is too
    # slow for interactive iteration and mostly repeats unpromising threshold settings.
    base_results = []
    for t in temps:
        p_punch, p_smooth = _probs_at_temperature(logits, t)
        for th in highs:
            for tl in lows:
                if tl > th:
                    continue
                for md in min_durs:
                    for mg in merge_gaps:
                        for sv in split_valleys:
                            gap_candidates = [0] if sv is None else split_min_gaps
                            for smg in gap_candidates:
                                windows = make_windows(
                                    frames,
                                    p_smooth,
                                    p_punch,
                                    t_high=float(th),
                                    t_low=float(tl),
                                    min_duration=md,
                                    merge_gap=mg,
                                    split_valley=sv,
                                    split_min_gap=smg,
                                )
                                pred = [(w.start_frame, w.end_frame) for w in windows]
                                wm = window_metrics(pred, gt)
                                base_results.append(
                                    {
                                        "T": t,
                                        "t_high": float(th),
                                        "t_low": float(tl),
                                        "min_dur": md,
                                        "merge_gap": mg,
                                        "split_valley": sv,
                                        "split_min_gap": smg,
                                        "split_peak_min_prob": None,
                                        "split_peak_min_distance": 0,
                                        "split_peak_min_drop": 0.0,
                                        "f1": _f1(wm["recall"], wm["precision"]),
                                        "recall": wm["recall"],
                                        "precision": wm["precision"],
                                        "n_pred": wm["n_pred_windows"],
                                        "false": wm["n_false_alarms"],
                                        "merged": wm["n_merged"],
                                    }
                                )

    base_res = pd.DataFrame(base_results).sort_values("f1", ascending=False).reset_index(drop=True)
    print(f"base sweep rows: {len(base_res)}", flush=True)

    peak_results = []
    n_candidates = max(1, int(args.base_candidates))
    for row in base_res[base_res["T"] == 1.0].head(n_candidates).itertuples(index=False):
        p_punch, p_smooth = _probs_at_temperature(logits, float(row.T))
        split_valley = None if pd.isna(row.split_valley) else float(row.split_valley)
        for spp, spd, spdrop in _peak_param_grid():
            windows = make_windows(
                frames,
                p_smooth,
                p_punch,
                t_high=float(row.t_high),
                t_low=float(row.t_low),
                min_duration=int(row.min_dur),
                merge_gap=int(row.merge_gap),
                split_valley=split_valley,
                split_min_gap=int(row.split_min_gap),
                split_peak_min_prob=spp,
                split_peak_min_distance=spd,
                split_peak_min_drop=spdrop,
            )
            pred = [(w.start_frame, w.end_frame) for w in windows]
            wm = window_metrics(pred, gt)
            peak_results.append(
                {
                    "T": float(row.T),
                    "t_high": float(row.t_high),
                    "t_low": float(row.t_low),
                    "min_dur": int(row.min_dur),
                    "merge_gap": int(row.merge_gap),
                    "split_valley": split_valley,
                    "split_min_gap": int(row.split_min_gap),
                    "split_peak_min_prob": spp,
                    "split_peak_min_distance": spd,
                    "split_peak_min_drop": spdrop,
                    "f1": _f1(wm["recall"], wm["precision"]),
                    "recall": wm["recall"],
                    "precision": wm["precision"],
                    "n_pred": wm["n_pred_windows"],
                    "false": wm["n_false_alarms"],
                    "merged": wm["n_merged"],
                }
            )

    res = (
        pd.concat([base_res, pd.DataFrame(peak_results)], ignore_index=True)
        .drop_duplicates(
            subset=[
                "T",
                "t_high",
                "t_low",
                "min_dur",
                "merge_gap",
                "split_valley",
                "split_min_gap",
                "split_peak_min_prob",
                "split_peak_min_distance",
                "split_peak_min_drop",
            ]
        )
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    print(f"peak-refined rows: {len(peak_results)} | total rows: {len(res)}", flush=True)

    # Baseline at the current config operating point, for reference.
    base_p, base_s = _probs_at_temperature(logits, 1.0)
    base_w = make_windows(frames, base_s, base_p, t_high=s4.t_high, t_low=s4.t_low,
                          min_duration=s4.min_duration, merge_gap=s4.merge_gap,
                          split_valley=s4.split_valley, split_min_gap=s4.split_min_gap,
                          split_peak_min_prob=s4.split_peak_min_prob,
                          split_peak_min_distance=s4.split_peak_min_distance,
                          split_peak_min_drop=s4.split_peak_min_drop)
    base_wm = window_metrics([(w.start_frame, w.end_frame) for w in base_w], gt)
    print(f"\nBASELINE (T=1, t_high={s4.t_high}, t_low={s4.t_low}): "
          f"F1 {_f1(base_wm['recall'], base_wm['precision']):.3f}  "
          f"recall {base_wm['recall']:.3f}  precision {base_wm['precision']:.3f}  "
          f"n_pred {base_wm['n_pred_windows']}  false {base_wm['n_false_alarms']}")

    print("\n=== TOP 15 by event-F1 (all temperatures) ===")
    print(res.head(15).to_string(index=False,
          formatters={
              c: "{:.3f}".format
              for c in (
                  "T", "t_high", "t_low", "split_valley", "split_peak_min_prob",
                  "split_peak_min_drop", "f1", "recall", "precision"
              )
          }))

    # The config carries thresholds but NOT temperature, and the pipeline applies T=1
    # (temperature is not wired in yet — HANDOFF #5). So the operating point we persist
    # must be chosen in the SAME regime the pipeline runs: restrict to T=1 rows. (The
    # fitted T hit the degenerate clamp here and does not beat T=1 anyway.)
    best = res[res["T"] == 1.0].iloc[0]
    print(f"\nBEST @ T=1 (the persisted regime): t_high={best['t_high']:.2f} "
          f"t_low={best['t_low']:.2f} min_dur={int(best['min_dur'])} "
          f"merge_gap={int(best['merge_gap'])} split_valley={best['split_valley']} "
          f"split_min_gap={int(best['split_min_gap'])} "
          f"split_peak_min_prob={best['split_peak_min_prob']} "
          f"split_peak_min_distance={int(best['split_peak_min_distance'])} "
          f"split_peak_min_drop={best['split_peak_min_drop']} -> F1 {best['f1']:.3f} "
          f"(recall {best['recall']:.3f}, precision {best['precision']:.3f})")

    # Re-derive the windows at the chosen T=1 operating point and persist the eval
    # artifacts next to the probs, so output/ shows window performance at the TUNED
    # thresholds (the per-frame plot is written by make_eval_probs.py).
    win_p, win_s = _probs_at_temperature(logits, 1.0)
    best_w = make_windows(frames, win_s, win_p, t_high=float(best["t_high"]),
                          t_low=float(best["t_low"]), min_duration=int(best["min_dur"]),
                          merge_gap=int(best["merge_gap"]),
                          split_valley=None if pd.isna(best["split_valley"]) else float(best["split_valley"]),
                          split_min_gap=int(best["split_min_gap"]),
                          split_peak_min_prob=(
                              None if pd.isna(best["split_peak_min_prob"])
                              else float(best["split_peak_min_prob"])
                          ),
                          split_peak_min_distance=int(best["split_peak_min_distance"]),
                          split_peak_min_drop=float(best["split_peak_min_drop"]))
    best_wm = window_metrics([(w.start_frame, w.end_frame) for w in best_w], gt)
    plot_window_eval(best_wm, str(out_dir / "window_eval.png"))
    (out_dir / "window_metrics.json").write_text(json.dumps({"window": best_wm}, indent=2))
    _plot_window_curves(frames, win_p, win_s, gt, best, str(out_dir / "window_pr_froc.png"))
    print(f"\nwrote -> {out_dir / 'window_eval.png'}")
    print(f"wrote -> {out_dir / 'window_metrics.json'}")
    print(f"wrote -> {out_dir / 'window_pr_froc.png'}")

    if args.write:
        with open(args.out_config) as f:
            cfg = yaml.safe_load(f)
        cfg["t_high"] = float(best["t_high"])
        cfg["t_low"] = float(best["t_low"])
        cfg["min_duration"] = int(best["min_dur"])
        cfg["merge_gap"] = int(best["merge_gap"])
        cfg["split_valley"] = None if pd.isna(best["split_valley"]) else float(best["split_valley"])
        cfg["split_min_gap"] = int(best["split_min_gap"])
        cfg["split_peak_min_prob"] = (
            None if pd.isna(best["split_peak_min_prob"]) else float(best["split_peak_min_prob"])
        )
        cfg["split_peak_min_distance"] = int(best["split_peak_min_distance"])
        cfg["split_peak_min_drop"] = float(best["split_peak_min_drop"])
        with open(args.out_config, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"\nwrote t_high/t_low -> {args.out_config} (T={best['T']:.3f} not persisted; "
              f"see HANDOFF #5 to wire temperature into the pipeline)")


if __name__ == "__main__":
    main()
