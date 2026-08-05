from __future__ import annotations

from bcv.stage1_detect.interface import Candidate
from bcv.stage1_detect.select import select_fighters


def cand(box, det, red, blue, tid=None):
    return Candidate(bbox=box, det_conf=det, cls_conf={"red": red, "blue": blue}, track_id=tid)


def test_zero_candidates_both_absent():
    red, blue = select_fighters([])
    assert red is None and blue is None


def test_one_per_color_assigned():
    cs = [
        cand((0, 0, 10, 20), 0.9, red=0.95, blue=0.02),
        cand((30, 0, 40, 20), 0.9, red=0.01, blue=0.93),
    ]
    red, blue = select_fighters(cs)
    assert red is not None and red.bbox == (0, 0, 10, 20)
    assert blue is not None and blue.bbox == (30, 0, 40, 20)
    assert abs(red.cls_conf - 0.95) < 1e-6


def test_below_gate_dropped():
    red, blue = select_fighters([cand((0, 0, 5, 5), 0.9, red=0.40, blue=0.10)], min_cls_conf=0.5)
    assert red is None and blue is None


def test_multiple_same_color_takes_highest():
    cs = [
        cand((0, 0, 10, 10), 0.8, red=0.70, blue=0.0),
        cand((20, 0, 30, 10), 0.8, red=0.90, blue=0.0),  # stronger red
    ]
    red, blue = select_fighters(cs)
    assert red is not None and red.bbox == (20, 0, 30, 10)
    assert blue is None  # nobody clears the blue gate


def test_both_top_classify_same_color_second_falls_back():
    # A is strongly red; B is red-leaning but also clears the blue gate -> B should take blue
    cs = [
        cand((0, 0, 10, 10), 0.9, red=0.95, blue=0.10),
        cand((20, 0, 30, 10), 0.9, red=0.80, blue=0.60),
    ]
    red, blue = select_fighters(cs)
    assert red is not None and red.bbox == (0, 0, 10, 10)
    assert blue is not None and blue.bbox == (20, 0, 30, 10)


def test_distinct_candidates_one_cannot_fill_both():
    # A single candidate confident in both colors fills only one slot (its best).
    red, blue = select_fighters([cand((0, 0, 10, 10), 0.9, red=0.91, blue=0.88)])
    assert (red is None) ^ (blue is None)
    chosen = red or blue
    assert chosen.bbox == (0, 0, 10, 10)


def test_tie_broken_by_det_conf_then_area():
    cs = [
        cand((0, 0, 10, 10), 0.50, red=0.80, blue=0.0),  # smaller area, lower det
        cand((0, 0, 20, 20), 0.50, red=0.80, blue=0.0),  # larger area, same det/cls
    ]
    red, _ = select_fighters(cs)
    assert red is not None and red.bbox == (0, 0, 20, 20)  # area breaks the tie
