"""Tests for the Stage-1 detection eval (per-fighter red/blue box comparison)."""
from __future__ import annotations

import math

import pandas as pd

from bcv.common.contracts import DETECTION_SCHEMA
from bcv.eval.detection import _iou, detection_metrics


def _row(frame, *, red=None, blue=None):
    """Build one DETECTION_SCHEMA row; red/blue are (x1,y1,x2,y2) or None (absent)."""
    r: dict = {"frame": frame, "n_candidates": 2}
    for color, box in (("red", red), ("blue", blue)):
        present = box is not None
        r[f"{color}_present"] = present
        x1, y1, x2, y2 = box if present else (-1, -1, -1, -1)
        r[f"{color}_x1"], r[f"{color}_y1"] = x1, y1
        r[f"{color}_x2"], r[f"{color}_y2"] = x2, y2
        r[f"{color}_det_conf"] = 1.0 if present else 0.0
        r[f"{color}_cls_conf"] = 1.0 if present else 0.0
    return r


def _df(rows):
    return pd.DataFrame(rows, columns=list(DETECTION_SCHEMA)).astype(DETECTION_SCHEMA)


def test_iou_basic():
    # identical box -> 1.0; half-overlap -> 1/3; disjoint -> 0
    import numpy as np
    a = np.array([[0, 0, 10, 10]], float)
    assert _iou(a, a)[0] == 1.0
    b = np.array([[5, 0, 15, 10]], float)  # inter=50, union=150
    assert abs(_iou(a, b)[0] - (50 / 150)) < 1e-9
    c = np.array([[100, 100, 110, 110]], float)
    assert _iou(a, c)[0] == 0.0


def test_self_vs_self_is_perfect():
    """The headline sanity check: comparing a box-set against itself scores perfectly."""
    df = _df([
        _row(0, red=(10, 10, 50, 80), blue=(100, 20, 140, 90)),
        _row(1, red=(12, 11, 52, 82), blue=None),          # blue absent both sides
        _row(2, red=None, blue=(200, 30, 240, 100)),
        _row(3, red=None, blue=None),                       # both absent
    ])
    m = detection_metrics(df, df)
    p = m["pooled"]
    assert p["presence_recall"] == 1.0 and p["presence_precision"] == 1.0
    assert p["presence_f1"] == 1.0
    assert p["mean_iou"] == 1.0 and p["median_iou"] == 1.0
    assert p["box_recall@0.5"] == 1.0 and p["box_recall@0.75"] == 1.0
    assert p["box_precision@0.5"] == 1.0
    assert m["swap"]["n_swapped"] == 0
    for c in ("red", "blue"):
        f = m["per_fighter"][c]["failure"]
        assert f["missed_fighter"] == 0 and f["false_present"] == 0 and f["poorly_localized"] == 0


def test_presence_confusion_and_failures():
    # frame0: red TP (good iou), blue FN (gt present, pred absent)
    # frame1: red FP (pred present, gt absent), blue TP poorly-localized (iou<0.5)
    gt = _df([
        _row(0, red=(0, 0, 10, 10), blue=(50, 50, 60, 60)),
        _row(1, red=None, blue=(0, 0, 10, 10)),
    ])
    pred = _df([
        _row(0, red=(0, 0, 10, 10), blue=None),
        _row(1, red=(0, 0, 10, 10), blue=(7, 0, 17, 10)),  # blue iou small
    ])
    m = detection_metrics(pred, gt)
    red, blue = m["per_fighter"]["red"], m["per_fighter"]["blue"]
    assert red["tp"] == 1 and red["fp"] == 1 and red["fn"] == 0
    assert blue["tp"] == 1 and blue["fn"] == 1 and blue["fp"] == 0
    assert red["failure"]["false_present"] == 1
    assert blue["failure"]["missed_fighter"] == 1
    assert blue["failure"]["poorly_localized"] == 1  # blue TP but iou<0.5
    assert blue["mean_iou"] < 0.5


def test_swap_detected():
    # all four present; pred has red/blue boxes swapped relative to gt
    gt = _df([_row(0, red=(0, 0, 10, 10), blue=(100, 100, 110, 110))])
    pred = _df([_row(0, red=(100, 100, 110, 110), blue=(0, 0, 10, 10))])
    m = detection_metrics(pred, gt)
    assert m["swap"]["n_quad_present"] == 1 and m["swap"]["n_swapped"] == 1
    assert m["swap"]["swap_rate"] == 1.0
    # presence is "correct" (both present both sides) but IoU is ~0 (boxes don't match by slot)
    assert m["pooled"]["mean_iou"] == 0.0


def test_frame_alignment_drops_unmatched():
    gt = _df([_row(0, red=(0, 0, 10, 10)), _row(1, red=(0, 0, 10, 10))])
    pred = _df([_row(0, red=(0, 0, 10, 10))])  # missing frame 1
    m = detection_metrics(pred, gt)
    assert m["n_frames"] == 1 and m["n_dropped"] == 1


def test_n_dropped_counts_both_sides():
    # partially-overlapping frame sets: pred {0,1,2}, gt {2,3,4} -> overlap {2}
    pred = _df([_row(0, red=(0, 0, 10, 10)), _row(1, red=(0, 0, 10, 10)), _row(2, red=(0, 0, 10, 10))])
    gt = _df([_row(2, red=(0, 0, 10, 10)), _row(3, red=(0, 0, 10, 10)), _row(4, red=(0, 0, 10, 10))])
    m = detection_metrics(pred, gt)
    assert m["n_frames"] == 1
    assert m["n_dropped"] == 4  # 2 pred-only + 2 gt-only


def test_empty_iou_is_nan_not_crash():
    # no overlapping-present frames -> mean_iou NaN, no exception
    gt = _df([_row(0, red=None, blue=None)])
    pred = _df([_row(0, red=None, blue=None)])
    m = detection_metrics(pred, gt)
    assert math.isnan(m["pooled"]["mean_iou"])
