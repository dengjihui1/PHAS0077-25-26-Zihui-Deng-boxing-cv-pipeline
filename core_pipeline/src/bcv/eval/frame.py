"""Frame-level evaluation: ROC + PR curves and AUROC/AP from frame_probs.

PR is reported alongside ROC because punch frames are rare (~6%), so ROC alone can look
flattering. Computed with numpy (no sklearn dep).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def roc_curve(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    y = y[order].astype(float)
    P, N = y.sum(), len(y) - y.sum()
    tp = np.concatenate([[0.0], np.cumsum(y)])
    fp = np.concatenate([[0.0], np.cumsum(1 - y)])
    tpr = tp / max(P, 1.0)
    fpr = fp / max(N, 1.0)
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def pr_curve(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    y = y[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / max(y.sum(), 1.0)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - recall_prev) * precision))
    return recall, precision, ap


def frame_metrics(
    df: pd.DataFrame, *, score_col: str = "p_punch", valid_only: bool = False
) -> dict:
    d = df[df["crop_valid"]] if valid_only else df
    y = d["label"].to_numpy().astype(int)
    s = d[score_col].to_numpy().astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return {
            "n": len(y),
            "n_pos": int(y.sum()),
            "auroc": float("nan"),
            "ap": float("nan"),
            "degenerate": True,
        }
    _, _, auroc = roc_curve(s, y)
    _, _, ap = pr_curve(s, y)
    return {
        "n": len(y),
        "n_pos": int(y.sum()),
        "pos_frac": float(y.mean()),
        "auroc": auroc,
        "ap": ap,
        "degenerate": False,
    }


def plot_frame_eval(df: pd.DataFrame, out_png: str, *, score_col: str = "p_punch") -> dict:
    y = df["label"].to_numpy().astype(int)
    s = df[score_col].to_numpy().astype(float)
    fpr, tpr, auroc = roc_curve(s, y)
    rec, prec, ap = pr_curve(s, y)
    base = float(y.mean())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.plot(fpr, tpr, color="tab:blue", lw=1.6, label=f"AUROC = {auroc:.3f}")
    a1.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    a1.set(xlabel="false positive rate", ylabel="true positive rate", title="Frame ROC")
    a1.legend(loc="lower right")
    a2.plot(rec, prec, color="tab:red", lw=1.6, label=f"AP = {ap:.3f}")
    a2.axhline(base, ls="--", color="gray", lw=0.8, label=f"base rate = {base:.3f}")
    a2.set(xlabel="recall", ylabel="precision", title="Frame PR", ylim=(0, 1.02))
    a2.legend(loc="upper right")
    fig.suptitle("Stage 3 — per-frame punch classifier")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return {"auroc": auroc, "ap": ap, "base_rate": base}
