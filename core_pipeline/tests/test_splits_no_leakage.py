from __future__ import annotations

import pytest

from bcv.common.splits import (
    BoutSplits,
    LeakageError,
    assert_no_leakage,
    assert_view_not_leaked,
)


def test_canonical_split_is_disjoint():
    s = BoutSplits(train=[115, 120, 121], val=[116], test=[122])
    assert s.all_bouts == [115, 116, 120, 121, 122]


def test_overlapping_bout_raises():
    # pydantic re-wraps the validator's LeakageError as a ValidationError (a ValueError).
    with pytest.raises(ValueError):
        BoutSplits(train=[115, 122], val=[116], test=[122])  # 122 in train & test


def test_assert_no_leakage_reports_pairs():
    with pytest.raises(LeakageError):
        assert_no_leakage([1, 2], [2, 3], [4])  # 2 in train & val


def test_view_leak_guard():
    # all four views of a bout move together; a held-out bout's view must not be in train
    train = ["122:0", "115:0", "115:1"]
    heldout = ["116:0"]
    assert_view_not_leaked(train, heldout)  # ok
    with pytest.raises(LeakageError):
        assert_view_not_leaked(["122:0", "122:1"], ["122:2"])  # same bout in train & heldout
