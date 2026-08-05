"""Window-level (event) evaluation: match predicted strike windows to GT punch events.

Each GT event is greedily matched to the most-overlapping predicted window. For matched
pairs we report the boundary error as two counts: ``missed_frames`` (GT frames the
prediction failed to cover) and ``extra_frames`` (predicted frames outside the GT event).
"Found exactly" = both zero. Plus detection recall (GT events found), precision (windows
that hit a GT event), and false alarms (windows hitting nothing).
"""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Span = tuple[int, int]


def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def match_events(
    pred: list[Span], gt: list[Span]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy GT->best-overlapping-pred matching. Returns (matches, missed_gt, false_preds)."""
    used: set[int] = set()
    matches: list[tuple[int, int]] = []
    missed: list[int] = []
    for gi, g in enumerate(gt):
        best, best_ov = None, 0
        for pi, p in enumerate(pred):
            if pi in used:
                continue
            ov = _overlap(p, g)
            if ov > best_ov:
                best, best_ov = pi, ov
        if best is not None:
            used.add(best)
            matches.append((gi, best))
        else:
            missed.append(gi)
    false_preds = [pi for pi in range(len(pred)) if pi not in used]
    return matches, missed, false_preds


def _error_budget(pred: list[Span], gt: list[Span]) -> dict:
    """Overlap-based error budget, independent of the greedy 1-1 assignment.

    Greedy recall/precision blur two pairs of distinct failures, so we classify by raw
    overlap instead. Every GT event is one of: ``clean`` (covered 1-1 by a single window),
    ``merged`` (covered, but that window also covers another GT — under-segmentation),
    ``split`` (covered by >=2 windows — over-segmentation), or ``missed`` (no window
    touches it at all). Every predicted window is one of: ``clean``, ``merge`` (covers
    >=2 GTs), ``split`` (one of several windows on a single GT), or ``fake`` (covers no
    GT). ``missed`` answers "how often we miss a real punch"; ``fake`` answers "how often
    we emit a window for nothing" — neither inflated by merges/splits the way greedy is.
    """
    gt_hits = [[pi for pi, p in enumerate(pred) if _overlap(p, g) > 0] for g in gt]
    pred_hits = [[gi for gi, g in enumerate(gt) if _overlap(p, g) > 0] for p in pred]

    gt_cat = Counter[str]()
    for hits in gt_hits:
        if not hits:
            gt_cat["missed"] += 1
        elif len(hits) >= 2:
            gt_cat["split"] += 1
        elif len(pred_hits[hits[0]]) >= 2:
            gt_cat["merged"] += 1
        else:
            gt_cat["clean"] += 1

    pred_cat = Counter[str]()
    for hits in pred_hits:
        if not hits:
            pred_cat["fake"] += 1
        elif len(hits) >= 2:
            pred_cat["merge"] += 1
        elif len(gt_hits[hits[0]]) >= 2:
            pred_cat["split"] += 1
        else:
            pred_cat["clean"] += 1

    n_gt, n_pred = len(gt), len(pred)
    n_detected = n_gt - gt_cat["missed"]
    n_hit = n_pred - pred_cat["fake"]
    return {
        "gt_outcomes": {k: gt_cat[k] for k in ("clean", "merged", "split", "missed")},
        "window_outcomes": {k: pred_cat[k] for k in ("clean", "merge", "split", "fake")},
        "n_missed": gt_cat["missed"],        # real punches no window touches
        "n_fake": pred_cat["fake"],          # windows overlapping no real punch
        "n_detected": n_detected,
        "n_merged": gt_cat["merged"],        # detected punches fused with a neighbour
        "n_oversplit": gt_cat["split"],      # punches broken across >1 window
        "miss_rate": gt_cat["missed"] / n_gt if n_gt else float("nan"),
        "fake_rate": pred_cat["fake"] / n_pred if n_pred else float("nan"),
        "detection_recall": n_detected / n_gt if n_gt else float("nan"),
        "detection_precision": n_hit / n_pred if n_pred else float("nan"),
    }


def window_metrics(pred: list[Span], gt: list[Span]) -> dict:
    """Detection precision/recall, an overlap-based error budget, and boundary errors.

    Two views: the greedy 1-1 ``recall``/``precision``/``n_matched`` (strict — penalises
    merges and splits), and an overlap-based error budget (``miss_rate``/``fake_rate`` +
    merge/split diagnostics) that says plainly how often a real punch is missed vs merged,
    and how often a window is fake vs a split duplicate. Per matched pair we also report
    boundary error (``missed_frames``/``extra_frames`` histograms).
    """
    matches, missed, false_preds = match_events(pred, gt)
    missed_hist: Counter[int] = Counter()
    extra_hist: Counter[int] = Counter()
    exact = 0
    for gi, pi in matches:
        gs, ge = gt[gi]
        ps, pe = pred[pi]
        gframes = set(range(gs, ge + 1))
        pframes = set(range(ps, pe + 1))
        missed_n = len(gframes - pframes)
        extra_n = len(pframes - gframes)
        missed_hist[missed_n] += 1
        extra_hist[extra_n] += 1
        if missed_n == 0 and extra_n == 0:
            exact += 1
    n_gt, n_pred, n_match = len(gt), len(pred), len(matches)
    return {
        "n_gt_events": n_gt,
        "n_pred_windows": n_pred,
        "n_matched": n_match,
        "recall": n_match / n_gt if n_gt else float("nan"),
        "precision": n_match / n_pred if n_pred else float("nan"),
        "n_missed_events": len(missed),
        "n_false_alarms": len(false_preds),
        "exact": exact,
        "exact_frac_of_matched": exact / n_match if n_match else float("nan"),
        "missed_frames_hist": dict(sorted(missed_hist.items())),
        "extra_frames_hist": dict(sorted(extra_hist.items())),
        **_error_budget(pred, gt),
    }


def plot_window_eval(metrics: dict, out_png: str) -> None:
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(11, 7.6))

    # Top row: the overlap-based error budget — where every GT event and every window goes.
    gt_o = metrics["gt_outcomes"]
    gt_colors = {"clean": "tab:green", "merged": "tab:olive", "split": "tab:cyan", "missed": "tab:red"}
    a1.bar(list(gt_o), list(gt_o.values()), color=[gt_colors[k] for k in gt_o])
    a1.set(ylabel=f"# GT events (n={metrics['n_gt_events']})",
           title=f"Real-punch outcomes — miss rate {metrics['miss_rate']:.2f}")

    win_o = metrics["window_outcomes"]
    win_colors = {"clean": "tab:green", "merge": "tab:olive", "split": "tab:cyan", "fake": "tab:red"}
    a2.bar(list(win_o), list(win_o.values()), color=[win_colors[k] for k in win_o])
    a2.set(ylabel=f"# windows (n={metrics['n_pred_windows']})",
           title=f"Predicted-window outcomes — fake rate {metrics['fake_rate']:.2f}")

    # Bottom row: boundary error for the greedily-matched pairs.
    for ax, key, color, title in (
        (a3, "missed_frames_hist", "tab:orange", "Missed frames (GT not covered)"),
        (a4, "extra_frames_hist", "tab:purple", "Extra frames (predicted beyond GT)"),
    ):
        hist = metrics[key]
        xs = list(hist.keys()) or [0]
        ys = [hist.get(x, 0) for x in xs]
        ax.bar([str(x) for x in xs], ys, color=color)
        ax.set(xlabel="frames off", ylabel="# matched events", title=title)

    summ = (
        f"detection recall {metrics['detection_recall']:.2f}  "
        f"precision {metrics['detection_precision']:.2f}   |   "
        f"missed {metrics['n_missed']}  fake {metrics['n_fake']}  "
        f"merged {metrics['n_merged']}  split {metrics['n_oversplit']}   |   "
        f"greedy recall {metrics['recall']:.2f} precision {metrics['precision']:.2f}"
    )
    fig.suptitle("Stage 4 — window detection vs GT events\n" + summ, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
