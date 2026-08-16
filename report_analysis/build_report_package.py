"""Build lightweight report/PPT support analyses for the boxing CV project.

This script is intentionally read-only with respect to existing experiments. It
uses saved Stage-4/Stage-5 artifacts and writes only to the requested output
folder.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bcv.common.annotations import load_runs
from bcv.common.config import load_pipeline_config
from bcv.common.rounds import load_rounds
from bcv.eval.window import match_events

ROOT = Path("Zihui")
STAGE4 = ROOT / "stage4_multiview_consensus_20260727"
STAGE5 = ROOT / "stage5_multiview_structured_20260728"
PROBS_ROOT = ROOT / "stage5_latest_20260720" / "output" / "stage3_frame_classifier"
BOUTS = (115, 116, 117, 120, 121, 122)
TEST_BOUT = 115
BASELINE_STAGE4 = {"precision": 0.925, "recall": 0.209, "f1": 0.341}
CONSENSUS_STAGE4 = {"precision": 0.808, "recall": 0.754, "f1": 0.780}
LABEL_ORDER = (
    "blue_body_landed",
    "blue_head_landed",
    "blue_strike_blocked",
    "blue_strike_missed",
    "red_body_landed",
    "red_head_landed",
    "red_strike_blocked",
    "red_strike_missed",
)


def setup_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 120,
    })


def savefig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".jpg"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def import_stage4_helpers() -> object:
    sys.path.insert(0, str(STAGE4.resolve()))
    import sweep_consensus as sc  # type: ignore
    return sc


def load_bout_views_for_splits(pipeline, bout: int, view_subset: tuple[int, ...] | None) -> tuple[np.ndarray, np.ndarray]:
    """Load probabilities for explicit camera split ids, preserving missing-view behaviour."""
    selected = set(range(4) if view_subset is None else view_subset)
    series: list[pd.Series] = []
    bout_name = pipeline.bouts[bout]
    for split in range(4):
        if split not in selected:
            continue
        path = PROBS_ROOT / bout_name / f"split_{split}" / "frame_probs.parquet"
        if not path.exists():
            continue
        frame_probs = pd.read_parquet(path, columns=["frame", "p_punch", "crop_valid"])
        values = frame_probs["p_punch"].astype(float).where(frame_probs["crop_valid"].astype(bool))
        series.append(pd.Series(values.to_numpy(), index=frame_probs["frame"].astype(int), name=f"split_{split}"))
    if not series:
        raise FileNotFoundError(f"no selected frame probabilities for bout {bout}, views={view_subset}")
    aligned = pd.concat(series, axis=1).sort_index()
    return aligned.index.to_numpy(dtype=np.int64), aligned.to_numpy(dtype=np.float64).T


def load_gt(pipeline, bout: int) -> list[tuple[int, int, str]]:
    return [(r.start_frame, r.end_frame, r.label) for r in load_runs(pipeline.bout_dir(bout))]


def robust_predictions(sc, pipeline, bouts: tuple[int, ...], view_subset: tuple[int, ...] | None = None):
    params = sc.Params(normalization="rank", fusion="mean", smooth=1, threshold=0.80, min_distance=6, radius=6)
    raw, cache, gt = {}, {}, {}
    for bout in bouts:
        frames, probs = load_bout_views_for_splits(pipeline, bout, view_subset)
        valid = sc.in_round_mask(frames, load_rounds(pipeline.bout_dir(bout)))
        raw[bout] = (frames, probs, valid)
        fused = sc.fuse_views(sc.normalize_views(probs, valid, "rank"), "mean")
        fused[~valid] = 0.0
        cache[(bout, "rank", "mean", 1)] = sc.moving_average(fused, 1)
        gt[bout] = load_gt(pipeline, bout)
    return params, sc.make_all_predictions(params, raw, cache), gt, raw, cache


def analyse_view_ablation(sc, pipeline, tables: Path, figures: Path) -> list[dict]:
    rows = []
    for size in range(1, 5):
        for subset in itertools.combinations(range(4), size):
            try:
                _params, pred, gt, raw, _cache = robust_predictions(sc, pipeline, (TEST_BOUT,), subset)
            except FileNotFoundError:
                continue
            metrics = sc.score_predictions(pred, gt, (TEST_BOUT,), exact=True)
            n_loaded = int(raw[TEST_BOUT][1].shape[0])
            rows.append({
                "views": "".join(str(v) for v in subset),
                "n_requested_views": size,
                "n_views": n_loaded,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "n_pred": metrics["n_pred_windows"],
                "n_gt": metrics["n_gt_events"],
            })
    write_csv(tables / "stage4_view_ablation.csv", rows)

    df = pd.DataFrame(rows)
    summary = df.groupby("n_views").agg(
        f1_mean=("f1", "mean"), f1_min=("f1", "min"), f1_max=("f1", "max"),
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
    ).reset_index()
    summary.to_csv(tables / "stage4_view_count_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.1, 2.7))
    ax.errorbar(
        summary["n_views"], summary["f1_mean"],
        yerr=[summary["f1_mean"] - summary["f1_min"], summary["f1_max"] - summary["f1_mean"]],
        fmt="o-", color="#2f6f9f", lw=1.8, capsize=3,
    )
    ax.axhline(BASELINE_STAGE4["f1"], color="#999999", lw=1.0, ls="--", label="independent-view ref.")
    ax.set_xlabel("Number of synchronized views")
    ax.set_ylabel("Strict event F1")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, max(0.85, float(summary["f1_max"].max()) + 0.05))
    ax.legend(loc="lower right")
    ax.set_title("Multi-view evidence improves strike localisation")
    savefig(fig, figures / "fig_stage4_view_count_ablation")

    max_views = int(df["n_views"].max())
    leave_one = df[df["n_requested_views"] == 3].copy()
    full_f1 = float(df.loc[df["n_views"] == max_views, "f1"].iloc[0])
    leave_one["removed_view"] = leave_one["views"].apply(lambda s: next(str(v) for v in range(4) if str(v) not in s))
    leave_one["delta_vs_all4"] = leave_one["f1"] - full_f1
    leave_one.to_csv(tables / "stage4_leave_one_view_out.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.1, 2.4))
    leave_one = leave_one.sort_values("removed_view")
    colors = ["#b95450" if x < 0 else "#6aa36f" for x in leave_one["delta_vs_all4"]]
    ax.barh("remove split " + leave_one["removed_view"], leave_one["delta_vs_all4"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Delta strict F1 versus all four views")
    ax.set_title("Leave-one-view-out contribution")
    savefig(fig, figures / "fig_stage4_leave_one_view_out")
    return rows


def analyse_fusion_ablation(tables: Path, figures: Path) -> None:
    sweep = pd.read_csv(STAGE4 / "results_cv" / "sweep.csv")
    best = (
        sweep.groupby(["normalization", "fusion"], as_index=False)
        .agg(best_val_f1=("val_f1", "max"), best_train_f1=("train_f1", "max"))
        .sort_values("best_val_f1", ascending=False)
    )
    best.to_csv(tables / "stage4_fusion_component_ablation.csv", index=False)
    pivot = best.pivot(index="normalization", columns="fusion", values="best_val_f1")
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", vmin=max(0.0, np.nanmin(pivot.to_numpy()) - 0.02), vmax=np.nanmax(pivot.to_numpy()))
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("Stage 4 selection F1 by normalization and fusion")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Best validation F1")
    savefig(fig, figures / "fig_stage4_fusion_heatmap")


def score_radius(pred_peaks: list[int], gt_spans: list[tuple[int, int]], radius: int) -> dict:
    pred = [(peak - radius, peak + radius) for peak in pred_peaks]
    matches, missed, false = match_events(pred, gt_spans)
    precision = len(matches) / len(pred) if pred else 0.0
    recall = len(matches) / len(gt_spans) if gt_spans else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"radius": radius, "precision": precision, "recall": recall, "f1": f1, "matched": len(matches), "missed": len(missed), "false": len(false)}


def analyse_temporal_quality(pipeline, tables: Path, figures: Path) -> None:
    payload = json.loads((STAGE4 / "results_cv" / "robust_windows" / f"bout_{TEST_BOUT}_consensus_windows.json").read_text())
    peaks = [int(row["peak_frame"]) for row in payload["windows"]]
    gt_runs = load_runs(pipeline.bout_dir(TEST_BOUT))
    gt_spans = [(r.start_frame, r.end_frame) for r in gt_runs]
    tolerance_rows = [score_radius(peaks, gt_spans, radius) for radius in [2, 4, 6, 8, 10, 12, 15, 20]]
    write_csv(tables / "stage4_temporal_tolerance_curve.csv", tolerance_rows)

    fig, ax = plt.subplots(figsize=(4.1, 2.6))
    tdf = pd.DataFrame(tolerance_rows)
    ax.plot(tdf["radius"], tdf["f1"], "o-", color="#2f6f9f", label="F1")
    ax.plot(tdf["radius"], tdf["recall"], "o-", color="#6aa36f", label="Recall")
    ax.set_xlabel("Matching radius (frames)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.set_title("Temporal tolerance of consensus peaks")
    savefig(fig, figures / "fig_stage4_temporal_tolerance")

    pred = [(peak - 6, peak + 6) for peak in peaks]
    matches, _missed, _false = match_events(pred, gt_spans)
    error_rows = []
    for gi, pi in matches:
        run = gt_runs[gi]
        gt_center = 0.5 * (run.start_frame + run.end_frame)
        peak = peaks[pi]
        error_rows.append({
            "bout": TEST_BOUT,
            "label": run.label,
            "gt_start": run.start_frame,
            "gt_end": run.end_frame,
            "gt_center": gt_center,
            "peak_frame": peak,
            "error_frames": peak - gt_center,
            "error_ms_at_30fps": (peak - gt_center) / 30.0 * 1000.0,
        })
    write_csv(tables / "stage4_peak_timing_error.csv", error_rows)
    errors = np.asarray([row["error_frames"] for row in error_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.hist(errors, bins=np.arange(math.floor(errors.min()) - 0.5, math.ceil(errors.max()) + 1.5, 1), color="#7aa6c2", edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(np.median(errors), color="#b95450", lw=1.2, label=f"median {np.median(errors):.1f} frames")
    ax.set_xlabel("Predicted peak minus GT centre (frames)")
    ax.set_ylabel("Matched events")
    ax.set_title("Signed peak timing error")
    ax.legend()
    savefig(fig, figures / "fig_stage4_peak_timing_error")


def analyse_proposal_density(sc, pipeline, tables: Path, figures: Path) -> None:
    _params, pred, gt, raw, _cache = robust_predictions(sc, pipeline, BOUTS, None)
    rows = []
    for bout in BOUTS:
        frames, _probs, valid = raw[bout]
        duration_min = max(1, int(valid.sum())) / 30.0 / 60.0
        spans = [(start, end) for start, end, _label in gt[bout]]
        pred_spans = [(start, end) for start, end, _peak, _score in pred[bout]]
        matches, missed, false = match_events(pred_spans, spans)
        rows.append({
            "bout": bout,
            "duration_min_in_round": duration_min,
            "gt_events": len(spans),
            "proposals": len(pred_spans),
            "matched": len(matches),
            "false_proposals": len(false),
            "missed_gt": len(missed),
            "gt_per_min": len(spans) / duration_min,
            "proposals_per_min": len(pred_spans) / duration_min,
            "false_proposals_per_min": len(false) / duration_min,
        })
    write_csv(tables / "stage4_proposal_density_by_bout.csv", rows)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    x = np.arange(len(df))
    ax.plot(x, df["gt_per_min"], "o-", label="GT strikes/min", color="#4c78a8")
    ax.plot(x, df["proposals_per_min"], "o-", label="proposals/min", color="#72b7b2")
    ax.plot(x, df["false_proposals_per_min"], "o-", label="false proposals/min", color="#e45756")
    ax.set_xticks(x, df["bout"].astype(str))
    ax.set_xlabel("Bout")
    ax.set_ylabel("Events per minute")
    ax.set_title("Operational proposal density")
    ax.legend(ncol=1, loc="upper right")
    savefig(fig, figures / "fig_stage4_proposal_density")


def analyse_stage5(tables: Path, figures: Path) -> None:
    result = json.loads((STAGE5 / "models_matched" / "fighter_query_categorical" / "result.json").read_text())
    typed = result["typed_event_test"]["per_class"]
    rows = []
    for label in LABEL_ORDER:
        row = typed[label]
        rows.append({
            "label": label,
            "gt": row["gt"],
            "pred": row["pred"],
            "matched": row["matched"],
            "f1": row["f1"],
            "fighter": label.split("_", 1)[0],
            "outcome": label.split("_", 1)[1],
        })
    write_csv(tables / "stage5_class_support_vs_f1.csv", rows)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.9, 3.15))
    colors = df["outcome"].map({
        "body_landed": "#d98c4a",
        "head_landed": "#4c78a8",
        "strike_blocked": "#b95450",
        "strike_missed": "#6aa36f",
    })
    ax.scatter(df["gt"], df["f1"], s=58, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
    # Fixed point offsets are stable on the log x-axis.  These positions keep
    # the rare zero-F1 classes and the two missed classes separated after the
    # figure is reduced inside the manuscript.
    label_text = {
        "blue_body_landed": "blue body",
        "blue_head_landed": "blue head",
        "blue_strike_blocked": "blue blocked",
        "blue_strike_missed": "blue missed",
        "red_body_landed": "red body",
        "red_head_landed": "red head",
        "red_strike_blocked": "red blocked",
        "red_strike_missed": "red missed",
    }
    label_offsets = {
        "blue_body_landed": (18, 16),
        "red_body_landed": (-34, 30),
        "red_strike_blocked": (24, -8),
        "blue_strike_blocked": (10, -22),
        "red_head_landed": (-12, 18),
        "blue_head_landed": (12, -16),
        "red_strike_missed": (-52, 22),
        "blue_strike_missed": (14, -18),
    }
    for _, row in df.iterrows():
        label = row["label"]
        dx, dy = label_offsets[label]
        ax.annotate(
            label_text[label],
            xy=(row["gt"], row["f1"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.3,
            va="center",
            ha="left" if dx >= 0 else "right",
            arrowprops={
                "arrowstyle": "-",
                "color": "0.45",
                "lw": 0.45,
                "shrinkA": 1.5,
                "shrinkB": 4.5,
            },
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("GT support on Bout 115 (log scale)")
    ax.set_ylabel("Per-class typed F1")
    ax.set_xlim(0.75, 210)
    ax.set_ylim(-0.06, 0.64)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_title("Rare strike outcomes remain weak")
    savefig(fig, figures / "fig_stage5_support_vs_f1")

    family_rows = []
    for family, selector in {
        "body landed": "body_landed",
        "head landed": "head_landed",
        "blocked": "strike_blocked",
        "missed": "strike_missed",
    }.items():
        sub = df[df["outcome"] == selector]
        gt = int(sub["gt"].sum())
        pred = int(sub["pred"].sum())
        matched = int(sub["matched"].sum())
        family_rows.append({
            "family": family,
            "gt": gt,
            "pred": pred,
            "matched": matched,
            "precision": matched / pred if pred else 0.0,
            "recall": matched / gt if gt else 0.0,
            "f1": 2 * matched / (gt + pred) if gt + pred else 0.0,
        })
    write_csv(tables / "stage5_outcome_family_summary.csv", family_rows)
    fdf = pd.DataFrame(family_rows)
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.bar(fdf["family"], fdf["f1"], color=["#d98c4a", "#4c78a8", "#b95450", "#6aa36f"])
    ax.set_ylabel("Family-level typed F1")
    ax.set_ylim(0, 0.65)
    ax.set_title("Outcome families: missed/head easier than body/blocked")
    ax.tick_params(axis="x", rotation=20)
    savefig(fig, figures / "fig_stage5_outcome_family_f1")


def make_headline_figures(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.8))
    ax.scatter([BASELINE_STAGE4["recall"]], [BASELINE_STAGE4["precision"]], s=80, color="#999999", label="reference")
    ax.scatter([CONSENSUS_STAGE4["recall"]], [CONSENSUS_STAGE4["precision"]], s=95, color="#2f6f9f", label="multi-view consensus")
    ax.annotate("", xy=(CONSENSUS_STAGE4["recall"], CONSENSUS_STAGE4["precision"]), xytext=(BASELINE_STAGE4["recall"], BASELINE_STAGE4["precision"]), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2f6f9f"))
    ax.text(BASELINE_STAGE4["recall"], BASELINE_STAGE4["precision"] + 0.025, "F1 0.341", ha="center", fontsize=8)
    ax.text(CONSENSUS_STAGE4["recall"], CONSENSUS_STAGE4["precision"] + 0.025, "F1 0.780", ha="center", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Stage 4 moves from sparse recall to balanced detection")
    ax.legend(loc="lower left")
    savefig(fig, figures / "fig_stage4_precision_recall_movement")

    labels = ["localisation\nF1", "activity\nmacro-F1", "clean-GT type\nmacro-F1", "end-to-end\ntyped F1"]
    values = [0.780, 0.684, 0.274, 0.448]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(range(len(values)), values, "o-", color="#2f6f9f", lw=2)
    ax.fill_between(range(len(values)), values, color="#2f6f9f", alpha=0.12)
    for i, value in enumerate(values):
        ax.text(i, value + 0.035, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Metric value")
    ax.set_title("Pipeline bottleneck shifts to fine-grained strike type")
    savefig(fig, figures / "fig_pipeline_bottleneck_waterfall")


def write_summary(out_dir: Path) -> None:
    text = """# Report Support Package

Date: 2026-07-31

This package contains lightweight analyses for the report/PPT. No model training was run.
All outputs are derived from existing Stage-4 and Stage-5 artifacts.

## Core Claims Supported

1. Multi-view consensus is the main validated contribution: Stage-4 strict event F1 improves
   from 0.341 to 0.780, mainly by increasing recall from 0.209 to 0.754.
2. The robust Stage-4 configuration is not just a threshold accident: rank-normalisation,
   mean fusion, view-count behaviour, and cross-bout checks all support the design.
3. The remaining bottleneck is Stage-5 strike-type recognition, especially rare body/blocked
   outcomes and cross-bout outcome ambiguity.

## Useful Figures

- `figures/fig_stage4_precision_recall_movement.png`
- `figures/fig_stage4_view_count_ablation.png`
- `figures/fig_stage4_leave_one_view_out.png`
- `figures/fig_stage4_fusion_heatmap.png`
- `figures/fig_stage4_temporal_tolerance.png`
- `figures/fig_stage4_peak_timing_error.png`
- `figures/fig_stage4_proposal_density.png`
- `figures/fig_stage5_support_vs_f1.png`
- `figures/fig_stage5_outcome_family_f1.png`
- `figures/fig_pipeline_bottleneck_waterfall.png`

SVG copies are also provided for editing in Illustrator/PowerPoint.

## Useful Tables

- `tables/stage4_view_ablation.csv`
- `tables/stage4_view_count_summary.csv`
- `tables/stage4_leave_one_view_out.csv`
- `tables/stage4_fusion_component_ablation.csv`
- `tables/stage4_temporal_tolerance_curve.csv`
- `tables/stage4_peak_timing_error.csv`
- `tables/stage4_proposal_density_by_bout.csv`
- `tables/stage5_class_support_vs_f1.csv`
- `tables/stage5_outcome_family_summary.csv`

## Suggested Report Placement

Use the precision-recall movement and view-count ablation in the main Results section.
Use temporal tolerance and proposal density as supporting Stage-4 diagnostics. Use
class-support and outcome-family figures in the Discussion or Error Analysis section to
justify why Stage 5 has reached a data/supervision bottleneck.
"""
    (out_dir / "REPORT_PACKAGE_SUMMARY.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report_support_20260731" / "package")
    args = parser.parse_args()
    setup_matplotlib()
    out_dir = args.output_dir
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    sc = import_stage4_helpers()
    pipeline = load_pipeline_config(args.pipeline_config)
    make_headline_figures(figures)
    view_rows = analyse_view_ablation(sc, pipeline, tables, figures)
    analyse_fusion_ablation(tables, figures)
    analyse_temporal_quality(pipeline, tables, figures)
    analyse_proposal_density(sc, pipeline, tables, figures)
    analyse_stage5(tables, figures)
    manifest = {
        "status": "complete",
        "n_view_ablation_rows": len(view_rows),
        "outputs": {
            "tables": sorted(path.name for path in tables.glob("*.csv")),
            "figures_png": sorted(path.name for path in figures.glob("*.png")),
            "figures_svg": sorted(path.name for path in figures.glob("*.svg")),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(out_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
