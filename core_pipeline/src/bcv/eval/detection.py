"""Detection eval: compare two per-frame red/blue box sets (pred vs reference GT).

Each frame has two FIXED fighter slots — red and blue — each either present (with a box)
or absent. This is NOT multi-object mAP: there is no assignment problem, red is scored
against red and blue against blue, frame by frame. Both inputs are Stage-1
``detections.parquet`` frames (``contracts.DETECTION_SCHEMA``); absent fighters carry the
``-1`` sentinel box and are gated out by the ``*_present`` flags (never fed to IoU).

Per fighter (and pooled over both) we report:
  * presence confusion tp/fp/fn/tn  -> recall / precision / f1 / accuracy,
  * IoU over the TP set (frames present in BOTH) -> mean / median,
  * box recall & precision at IoU 0.5 and 0.75 (a missed fighter AND a present-but-
    mislocalized box both fail box-recall),
  * a 4-way failure breakdown that partitions every present-or-mispredicted (frame,
    fighter) cell exactly once — missed_fighter (FN) / false_present (FP) /
    poorly_localized (TP, IoU<0.5) / well_localized (TP, IoU>=0.5); their sum is
    tp+fp+fn == n_frames - tn (correctly-absent TN cells are intentionally excluded),
and a red/blue swap check over frames where all four boxes are present.

Comparing a GT box-set against ITSELF yields a perfect score (recall=precision=1,
mean IoU=1, zero failures, zero swaps) — see ``tests/test_detection_eval.py``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGHTERS = ("red", "blue")
IOU_THRESHOLDS = (0.5, 0.75)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised IoU for two (N,4) xyxy box arrays (assumes valid, non-sentinel boxes)."""
    if len(a) == 0:
        return np.empty(0)
    ix1 = np.maximum(a[:, 0], b[:, 0])
    iy1 = np.maximum(a[:, 1], b[:, 1])
    ix2 = np.minimum(a[:, 2], b[:, 2])
    iy2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a + area_b - inter
    # np.divide(where=) skips the union==0 cells entirely (no divide-by-zero warning).
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def _slot_stats(pred_present: np.ndarray, gt_present: np.ndarray, ious: np.ndarray) -> dict:
    """Presence confusion + IoU + box recall/precision + failure breakdown for one slot.

    ``ious`` is aligned with the TP frames (pred_present & gt_present), in frame order.
    """
    tp_mask = pred_present & gt_present
    fp = int((pred_present & ~gt_present).sum())
    fn = int((~pred_present & gt_present).sum())
    tn = int((~pred_present & ~gt_present).sum())
    tp = int(tp_mask.sum())
    n = tp + fp + fn + tn
    gt_pos, pred_pos = tp + fn, tp + fp

    def _safe(num: float, den: float) -> float:
        return num / den if den else float("nan")

    well = int((ious >= 0.5).sum())
    out = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_frames": n,
        "presence_recall": _safe(tp, gt_pos),
        "presence_precision": _safe(tp, pred_pos),
        "presence_f1": _safe(2 * tp, 2 * tp + fp + fn),
        "presence_accuracy": _safe(tp + tn, n),
        "mean_iou": float(ious.mean()) if tp else float("nan"),
        "median_iou": float(np.median(ious)) if tp else float("nan"),
        "n_matched": tp,
        "failure": {
            "missed_fighter": fn,
            "false_present": fp,
            "poorly_localized": int((ious < 0.5).sum()),
            "well_localized": well,
        },
    }
    for t in IOU_THRESHOLDS:
        hit = int((ious >= t).sum())
        out[f"box_recall@{t}"] = _safe(hit, gt_pos)
        out[f"box_precision@{t}"] = _safe(hit, pred_pos)
    return out


def _pool(per: dict[str, dict], ious_by: dict[str, np.ndarray]) -> dict:
    tp = sum(per[c]["tp"] for c in FIGHTERS)
    fp = sum(per[c]["fp"] for c in FIGHTERS)
    fn = sum(per[c]["fn"] for c in FIGHTERS)
    tn = sum(per[c]["tn"] for c in FIGHTERS)
    ious = np.concatenate([ious_by[c] for c in FIGHTERS]) if ious_by else np.empty(0)
    gt_pos, pred_pos = tp + fn, tp + fp

    def _safe(num: float, den: float) -> float:
        return num / den if den else float("nan")

    out = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "presence_recall": _safe(tp, gt_pos),
        "presence_precision": _safe(tp, pred_pos),
        "presence_f1": _safe(2 * tp, 2 * tp + fp + fn),
        "mean_iou": float(ious.mean()) if tp else float("nan"),
        "median_iou": float(np.median(ious)) if tp else float("nan"),
    }
    for t in IOU_THRESHOLDS:
        hit = int((ious >= t).sum())
        out[f"box_recall@{t}"] = _safe(hit, gt_pos)
        out[f"box_precision@{t}"] = _safe(hit, pred_pos)
        out[f"box_f1@{t}"] = _safe(2 * hit, 2 * hit + (pred_pos - hit) + (gt_pos - hit))
    return out


def _swap_check(m: pd.DataFrame) -> dict:
    """Over frames where all four boxes are present, count FULL two-sided red/blue swaps.

    A swap requires BOTH pred-red to fit gt-blue better than gt-red AND pred-blue to fit
    gt-red better than gt-blue (strict ``>``, so identical self-vs-self never counts). This
    deliberately flags only complete identity crosses, not one-sided misassignments.
    """
    quad = (m["red_present_pred"].to_numpy() & m["blue_present_pred"].to_numpy()
            & m["red_present_gt"].to_numpy() & m["blue_present_gt"].to_numpy())
    if not quad.any():
        return {"n_quad_present": 0, "n_swapped": 0, "swap_rate": float("nan")}

    # Merged columns are red_x1_pred / red_x1_gt, so build the box arrays directly.
    def box(side: str, color: str) -> np.ndarray:
        cols = [f"{color}_{ax}_{side}" for ax in ("x1", "y1", "x2", "y2")]
        return m.loc[quad, cols].to_numpy(dtype=np.float64)
    pr_r, pr_b = box("pred", "red"), box("pred", "blue")
    gt_r, gt_b = box("gt", "red"), box("gt", "blue")
    iou_rr, iou_rb = _iou(pr_r, gt_r), _iou(pr_r, gt_b)
    iou_bb, iou_br = _iou(pr_b, gt_b), _iou(pr_b, gt_r)
    swapped = (iou_rb > iou_rr) & (iou_br > iou_bb)
    n_q = int(quad.sum())
    return {"n_quad_present": n_q, "n_swapped": int(swapped.sum()),
            "swap_rate": float(swapped.mean())}


def detection_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """Compare predicted vs reference red/blue boxes (both DETECTION_SCHEMA), aligned on frame.

    Returns ``{n_frames, per_fighter:{red,blue}, pooled, swap, headline}``. Frames present
    in only one input are dropped (inner-join on ``frame``) and reported in ``n_dropped``.
    """
    m = pd.merge(pred_df, gt_df, on="frame", suffixes=("_pred", "_gt"))
    n = len(m)
    per: dict[str, dict] = {}
    ious_by: dict[str, np.ndarray] = {}
    for c in FIGHTERS:
        # Coerce to bool defensively — ~mask on an int column would be a bit-flip, not negation.
        pp = m[f"{c}_present_pred"].to_numpy(dtype=bool)
        gp = m[f"{c}_present_gt"].to_numpy(dtype=bool)
        tp_mask = pp & gp
        cols = [f"{c}_{ax}" for ax in ("x1", "y1", "x2", "y2")]
        pred_b = m.loc[tp_mask, [f"{ax}_pred" for ax in cols]].to_numpy(dtype=np.float64)
        gt_b = m.loc[tp_mask, [f"{ax}_gt" for ax in cols]].to_numpy(dtype=np.float64)
        ious = _iou(pred_b, gt_b)
        ious_by[c] = ious
        per[c] = _slot_stats(pp, gp, ious)

    pooled = _pool(per, ious_by)
    swap = _swap_check(m)
    headline = {
        "mean_iou": pooled["mean_iou"],
        "presence_f1": pooled["presence_f1"],
        "box_recall@0.5": pooled["box_recall@0.5"],
        "box_precision@0.5": pooled["box_precision@0.5"],
    }
    return {
        "n_frames": n,
        "n_dropped": int((len(pred_df) - n) + (len(gt_df) - n)),  # pred-only + gt-only
        "per_fighter": per,
        "pooled": pooled,
        "swap": swap,
        "headline": headline,
    }


def plot_detection_eval(metrics: dict, pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                        out_png: str) -> None:
    """One figure: per-fighter IoU histogram + presence confusion + failure breakdown."""
    m = pd.merge(pred_df, gt_df, on="frame", suffixes=("_pred", "_gt"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    colors = {"red": "tab:red", "blue": "tab:blue"}
    for ax, c in zip(axes[0], FIGHTERS, strict=True):
        tp = m[f"{c}_present_pred"].to_numpy() & m[f"{c}_present_gt"].to_numpy()
        cols = [f"{c}_{ax2}" for ax2 in ("x1", "y1", "x2", "y2")]
        ious = _iou(m.loc[tp, [f"{x}_pred" for x in cols]].to_numpy(float),
                    m.loc[tp, [f"{x}_gt" for x in cols]].to_numpy(float))
        ax.hist(ious, bins=np.linspace(0, 1, 21), color=colors[c])
        s = metrics["per_fighter"][c]
        ax.axvline(0.5, ls="--", color="gray", lw=0.7)
        ax.set(title=f"{c} IoU (TP frames, n={s['tp']})  mean {s['mean_iou']:.2f}",
               xlabel="IoU", ylabel="# frames")
    # presence confusion + failure breakdown (pooled-ish: per fighter grouped)
    ax = axes[1, 0]
    keys = ["tp", "fp", "fn", "tn"]
    w = 0.38
    x = np.arange(len(keys))
    for i, c in enumerate(FIGHTERS):
        ax.bar(x + (i - 0.5) * w, [metrics["per_fighter"][c][k] for k in keys], w,
               label=c, color=colors[c])
    ax.set(title="Presence confusion", xticks=x, ylabel="# frames")
    ax.set_xticklabels(keys)
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    fk = ["missed_fighter", "false_present", "poorly_localized", "well_localized"]
    x = np.arange(len(fk))
    for i, c in enumerate(FIGHTERS):
        ax.bar(x + (i - 0.5) * w, [metrics["per_fighter"][c]["failure"][k] for k in fk], w,
               label=c, color=colors[c])
    ax.set(title="Failure breakdown", xticks=x, ylabel="# frames")
    ax.set_xticklabels(fk, rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    p = metrics["pooled"]
    fig.suptitle(
        f"Stage 1 — detection vs reference  |  pooled mean IoU {p['mean_iou']:.2f}  "
        f"presence F1 {p['presence_f1']:.2f}  box recall@0.5 {p['box_recall@0.5']:.2f}  "
        f"swap {metrics['swap']['n_swapped']}/{metrics['swap']['n_quad_present']}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
