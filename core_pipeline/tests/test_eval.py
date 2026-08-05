from __future__ import annotations

import numpy as np

from bcv.eval.frame import pr_curve, roc_curve
from bcv.eval.window import match_events, window_metrics


def test_roc_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    _, _, auroc = roc_curve(s, y)
    assert auroc == 1.0
    _, _, ap = pr_curve(s, y)
    assert ap == 1.0


def test_roc_reversed_is_zero():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.2, 0.1])  # scores anti-correlated with labels
    _, _, auroc = roc_curve(s, y)
    assert auroc == 0.0


def test_window_exact_match():
    m = window_metrics([(5, 9)], [(5, 9)])
    assert m["n_matched"] == 1 and m["exact"] == 1
    assert m["missed_frames_hist"] == {0: 1} and m["extra_frames_hist"] == {0: 1}
    assert m["recall"] == 1.0 and m["precision"] == 1.0


def test_window_missing_one_frame():
    m = window_metrics([(5, 8)], [(5, 9)])  # prediction 1 frame short
    assert m["exact"] == 0
    assert m["missed_frames_hist"] == {1: 1} and m["extra_frames_hist"] == {0: 1}


def test_window_one_extra_frame():
    m = window_metrics([(5, 10)], [(5, 9)])  # prediction 1 frame long
    assert m["missed_frames_hist"] == {0: 1} and m["extra_frames_hist"] == {1: 1}


def test_window_missed_event_and_false_alarm():
    m = window_metrics([(50, 55)], [(5, 9)])  # prediction nowhere near GT
    assert m["n_matched"] == 0 and m["n_missed_events"] == 1 and m["n_false_alarms"] == 1
    assert m["recall"] == 0.0 and m["precision"] == 0.0
    # error budget: one truly-missed punch, one truly-fake window
    assert m["n_missed"] == 1 and m["n_fake"] == 1
    assert m["miss_rate"] == 1.0 and m["fake_rate"] == 1.0
    assert m["n_merged"] == 0 and m["n_oversplit"] == 0


def test_window_merge_is_not_a_miss():
    # one window spans two adjacent punches -> both detected (merged), not missed.
    m = window_metrics([(5, 24)], [(5, 9), (20, 24)])
    assert m["n_missed"] == 0 and m["n_fake"] == 0        # nothing missed, nothing fake
    assert m["n_detected"] == 2 and m["n_merged"] == 2     # both detected, both merged
    assert m["detection_recall"] == 1.0
    assert m["gt_outcomes"]["missed"] == 0 and m["gt_outcomes"]["merged"] == 2
    assert m["window_outcomes"]["merge"] == 1
    # greedy view still penalises the merge (only one GT gets the window)
    assert m["recall"] == 0.5


def test_window_split_is_not_a_fake():
    # two windows both land on one punch -> a split, not a fake alarm.
    m = window_metrics([(5, 7), (9, 12)], [(5, 12)])
    assert m["n_fake"] == 0 and m["n_missed"] == 0
    assert m["n_oversplit"] == 1 and m["fake_rate"] == 0.0
    assert m["window_outcomes"]["split"] == 2 and m["window_outcomes"]["fake"] == 0
    assert m["detection_recall"] == 1.0
    # greedy view counts the second window as a false alarm
    assert m["n_false_alarms"] == 1


def test_match_events_greedy_assignment():
    matches, missed, fp = match_events([(5, 9), (20, 24)], [(5, 9), (20, 24)])
    assert len(matches) == 2 and not missed and not fp
