from __future__ import annotations

import pytest

from bcv.preprocess.pov_split import crop_filter


def test_quadrant_offsets():
    # split_0 top-left (origin), split_1 top-right (x), split_3 bottom-right (x+y)
    assert crop_filter(0) == "crop=2*trunc(iw/4):2*trunc(ih/4):0:0"
    assert crop_filter(1) == "crop=2*trunc(iw/4):2*trunc(ih/4):2*trunc(iw/4):0"
    assert crop_filter(3) == "crop=2*trunc(iw/4):2*trunc(ih/4):2*trunc(iw/4):2*trunc(ih/4)"


def test_split_2_is_hflipped():
    # bottom-left camera is mirrored — the boxes were made against the flipped frame
    assert crop_filter(2).endswith(",hflip")
    assert crop_filter(2).startswith("crop=2*trunc(iw/4):2*trunc(ih/4):0:2*trunc(ih/4)")


def test_invalid_split_rejected():
    with pytest.raises(ValueError):
        crop_filter(4)
