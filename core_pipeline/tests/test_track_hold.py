from __future__ import annotations

from bcv.stage1_detect.classify import TrackClassMemory


def test_confident_label_is_remembered_and_held():
    mem = TrackClassMemory(min_cls_conf=0.5, hold_frames=15)
    # frame 0: track 7 is confidently red
    out = mem.resolve(7, {"red": 0.9, "blue": 0.0, "unlabeled": 0.1}, 0)
    assert out["red"] == 0.9
    # frame 5: track 7 dips to unlabeled -> held red re-injected
    out = mem.resolve(7, {"red": 0.1, "blue": 0.0, "unlabeled": 0.9}, 5)
    assert out["red"] == 0.9


def test_hold_expires_after_window():
    mem = TrackClassMemory(min_cls_conf=0.5, hold_frames=10)
    mem.resolve(3, {"red": 0.0, "blue": 0.8}, 0)
    held = mem.resolve(3, {"red": 0.0, "blue": 0.2}, 8)  # within window
    assert held["blue"] == 0.8
    expired = mem.resolve(3, {"red": 0.0, "blue": 0.2}, 100)  # past window
    assert expired["blue"] == 0.2


def test_unknown_track_or_no_track_passes_through():
    mem = TrackClassMemory(min_cls_conf=0.5, hold_frames=15)
    assert mem.resolve(None, {"red": 0.2, "blue": 0.1}, 0) == {"red": 0.2, "blue": 0.1}
    assert mem.resolve(99, {"red": 0.2, "blue": 0.1}, 0) == {"red": 0.2, "blue": 0.1}


def test_reset_clears_memory():
    mem = TrackClassMemory(min_cls_conf=0.5, hold_frames=15)
    mem.resolve(1, {"red": 0.9, "blue": 0.0}, 0)
    mem.reset()
    out = mem.resolve(1, {"red": 0.1, "blue": 0.0}, 1)
    assert out["red"] == 0.1  # nothing held after reset
