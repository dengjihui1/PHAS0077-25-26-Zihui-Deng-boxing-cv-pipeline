"""Per-stage performance scorecard — walk every stage's artifacts into one figure + table.

For each pipeline stage it pulls the headline metric out of the per-split ``meta.json`` /
``metrics.json`` (Stage 1 detection coverage, Stage 2 crop validity, Stage 3 AUROC/AP,
Stage 4 window recall/precision) across all fights, so you can see the whole pipeline's
health at a glance instead of opening each stage's files.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common.config import PipelineConfig


def _read(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


# Each panel names the stage dir to WALK, the per-split metrics file to read, a display
# title + axis label, and an extractor(meta, metrics) -> float|None. Stage 1/2 read their
# producer stats from meta.json; Stage 3/4 read the metrics that `bcv-eval` writes into the
# separate `eval/` stage dir (frame_metrics.json / window_metrics.json), NOT a metrics.json
# under the stageN dir (which never exists — that was the silently-empty-panel bug).
# (title, walk_dir, metrics_file, label, extractor)
SPECS: list[tuple[str, str, str, str, Callable[[dict, dict], float | None]]] = [
    ("stage1_detect", "stage1_detect", "metrics.json", "both-fighters present",
     lambda m, _x: (m.get("producer") or {}).get("both_present_frac")),
    ("stage2_crop", "stage2_crop", "metrics.json", "valid crops",
     lambda m, _x: (m.get("producer") or {}).get("valid_frac")),
    ("stage3_frame_classifier", "eval", "frame_metrics.json", "frame AUROC",
     lambda _m, x: (x.get("frame") or x).get("auroc")),
    ("stage4_windowing", "eval", "window_metrics.json", "window recall",
     lambda _m, x: (x.get("window") or x).get("recall")),
]


def _stage_rows(
    pipeline: PipelineConfig, walk_dir: str, metrics_file: str, extract: Callable
) -> list[tuple[str, float]]:
    root = pipeline.output_root / walk_dir
    rows: list[tuple[str, float]] = []
    if not root.exists():
        return rows
    for fight in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "summary"):
        for split in sorted(fight.glob("split_*")):
            v = extract(_read(split / "meta.json"), _read(split / metrics_file))
            if v is not None:
                label = fight.name.replace("_Split 1-4", "") + "/" + split.name.replace("split_", "s")
                rows.append((label, float(v)))
    return rows


def build_summary(pipeline: PipelineConfig) -> Path:
    """Render output/summary/scorecard.png + return the dir. Also prints a table."""
    panels = [
        (title, label, _stage_rows(pipeline, walk_dir, metrics_file, fn))
        for title, walk_dir, metrics_file, label, fn in SPECS
    ]
    panels = [p for p in panels if p[2]]
    out_dir = pipeline.output_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not panels:
        print("[summary] no stage artifacts found yet")
        return out_dir

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.6 * len(panels)), squeeze=False)
    print("\n=== PIPELINE SCORECARD ===")
    for ax, (stage, label, rows) in zip(axes[:, 0], panels, strict=True):
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        mean = sum(vals) / len(vals)
        ax.bar(names, vals, color="tab:blue")
        ax.axhline(mean, ls="--", color="gray", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{stage} — {label}  (mean {mean:.2f}, n={len(vals)})", fontsize=10)
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        print(f"  {stage:26s} {label:22s} mean {mean:.3f}  over {len(vals)} splits")
    fig.tight_layout()
    out_png = out_dir / "scorecard.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[summary] -> {out_png}")
    return out_dir
