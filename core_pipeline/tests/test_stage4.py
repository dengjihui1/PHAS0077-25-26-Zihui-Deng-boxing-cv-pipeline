from __future__ import annotations

import numpy as np

from bcv.stage4_windowing.hysteresis import make_windows, tag_windows_with_gt

FR = np.arange(0, 20)


def _p(spans, val=0.9, n=20):
    p = np.full(n, 0.05)
    for s, e in spans:
        p[s : e + 1] = val
    return p


def test_single_window_boundaries():
    p = _p([(5, 9)])
    w = make_windows(FR, p, p, t_high=0.5, t_low=0.35, min_duration=3, merge_gap=0)
    assert len(w) == 1
    assert (w[0].start_frame, w[0].end_frame) == (5, 9)
    assert 5 <= w[0].peak_frame <= 9


def test_min_duration_drops_short_blip():
    p = _p([(5, 6)])  # length 2
    assert make_windows(FR, p, p, min_duration=3, merge_gap=0) == []


def test_merge_gap_joins_close_windows():
    p = _p([(5, 6), (9, 10)])  # gap of 2 frames (7,8)
    w = make_windows(FR, p, p, min_duration=1, merge_gap=2)
    assert len(w) == 1 and (w[0].start_frame, w[0].end_frame) == (5, 10)


def test_hysteresis_holds_through_dip_above_low():
    p = _p([(5, 9)])
    p[7] = 0.4  # dips below t_high but above t_low -> stays open
    w = make_windows(FR, p, p, t_high=0.5, t_low=0.35, min_duration=3, merge_gap=0)
    assert len(w) == 1 and (w[0].start_frame, w[0].end_frame) == (5, 9)


def test_optional_valley_split_cuts_long_window():
    p = _p([(3, 12)], val=0.9)
    p[7:9] = 0.45  # still above t_low, so plain hysteresis would keep one window
    w = make_windows(
        FR, p, p, t_high=0.5, t_low=0.35, min_duration=2, merge_gap=0,
        split_valley=0.5, split_min_gap=2,
    )
    assert [(x.start_frame, x.end_frame) for x in w] == [(3, 6), (9, 12)]


def test_optional_valley_split_ignores_short_dip():
    p = _p([(3, 12)], val=0.9)
    p[7] = 0.45
    w = make_windows(
        FR, p, p, t_high=0.5, t_low=0.35, min_duration=2, merge_gap=0,
        split_valley=0.5, split_min_gap=2,
    )
    assert [(x.start_frame, x.end_frame) for x in w] == [(3, 12)]


def test_optional_peak_split_cuts_multi_peak_plateau():
    p = _p([(2, 15)], val=0.88)
    p[4] = 0.95
    p[10] = 0.94
    p[7] = 0.76  # above t_low, so hysteresis keeps one segment without peak splitting
    w = make_windows(
        FR, p, p, t_high=0.7, t_low=0.6, min_duration=2, merge_gap=0,
        split_peak_min_prob=0.9, split_peak_min_distance=4, split_peak_min_drop=0.1,
    )
    assert [(x.start_frame, x.end_frame) for x in w] == [(2, 6), (8, 15)]


def test_optional_peak_split_requires_enough_drop():
    p = _p([(2, 15)], val=0.88)
    p[4] = 0.95
    p[10] = 0.94
    p[7] = 0.87
    w = make_windows(
        FR, p, p, t_high=0.7, t_low=0.6, min_duration=2, merge_gap=0,
        split_peak_min_prob=0.9, split_peak_min_distance=4, split_peak_min_drop=0.1,
    )
    assert [(x.start_frame, x.end_frame) for x in w] == [(2, 15)]


def test_tag_windows_with_gt():
    p = _p([(5, 9)])
    w = make_windows(FR, p, p, min_duration=3, merge_gap=0)
    tag_windows_with_gt(w, [(6, 8, "blue_strike_missed")])
    assert w[0].gt_labels == ["blue_strike_missed"] and not w[0].gt_multi
