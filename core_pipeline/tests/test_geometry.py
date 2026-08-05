from __future__ import annotations

import numpy as np

from bcv.common import geometry as g


def test_union_covers_both():
    a, b = (0, 0, 10, 10), (5, 5, 20, 30)
    assert g.union(a, b) == (0, 0, 20, 30)


def test_pad_box_expands_each_side_by_frac_of_dim():
    b = (0.0, 0.0, 10.0, 20.0)
    padded = g.pad_box(b, 0.1)  # +1 x each side, +2 y each side
    assert padded == (-1.0, -2.0, 11.0, 22.0)


def test_square_box_is_square_and_centered():
    b = (0.0, 0.0, 10.0, 4.0)  # wide
    sq = g.square_box(b)
    assert g.width(sq) == g.height(sq)
    assert g.center(sq) == g.center(b)
    assert g.width(sq) == 10.0


def test_square_box_min_half_floor():
    b = (5.0, 5.0, 6.0, 6.0)
    sq = g.square_box(b, min_half=4.0)
    assert g.width(sq) == 8.0


def test_clamp_shift_moves_inside_preserving_size():
    b = (-5.0, 2.0, 5.0, 12.0)  # 10x10, hangs off left edge
    c = g.clamp_shift(b, w=100, h=100)
    assert g.width(c) == 10.0 and g.height(c) == 10.0
    assert c[0] >= 0 and c[2] <= 100


def test_clamp_shift_clamps_when_larger_than_frame():
    b = (-10.0, -10.0, 200.0, 50.0)  # wider than 100
    c = g.clamp_shift(b, w=100, h=100)
    assert c[0] == 0.0 and c[2] == 100.0


def test_ema_first_value_passthrough_then_blends():
    assert g.ema(None, 5.0, 0.3) == 5.0
    assert g.ema(0.0, 10.0, 0.2) == 2.0


def test_iou_identity_and_disjoint():
    a = (0, 0, 10, 10)
    assert g.iou(a, a) == 1.0
    assert g.iou(a, (100, 100, 110, 110)) == 0.0


def test_crop_person_returns_region_and_none_when_small():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = g.crop_person(frame, (10, 10, 40, 60), pad_frac=0.0)
    assert crop is not None and crop.shape[0] == 50 and crop.shape[1] == 30
    assert g.crop_person(frame, (10, 10, 12, 12), min_w=50, min_h=50) is None
