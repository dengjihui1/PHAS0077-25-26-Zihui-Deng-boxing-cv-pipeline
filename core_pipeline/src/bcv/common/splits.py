"""Bout-level train/val/test split resolution with a leakage-safety guard.

Each annotation JSON covers all four camera views of one physical bout at identical
timestamps, so the split UNIT is the bout: all four ``split_N`` views move together
and a bout must never appear in more than one of train/val/test. This guard is the
tested invariant that prevents the same punch leaking across the split boundary.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class LeakageError(ValueError):
    """Raised when train/val/test bout sets are not pairwise disjoint."""


class BoutSplits(BaseModel):
    train: list[int]
    val: list[int]
    test: list[int]

    @model_validator(mode="after")
    def _check_disjoint(self) -> BoutSplits:
        assert_no_leakage(self.train, self.val, self.test)
        return self

    @property
    def all_bouts(self) -> list[int]:
        return sorted({*self.train, *self.val, *self.test})


def assert_no_leakage(train: list[int], val: list[int], test: list[int]) -> None:
    """Raise ``LeakageError`` if any bout id appears in more than one split."""
    s_train, s_val, s_test = set(train), set(val), set(test)
    overlaps = {
        "train∩val": s_train & s_val,
        "train∩test": s_train & s_test,
        "val∩test": s_val & s_test,
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise LeakageError(f"bout(s) appear in multiple splits: {bad}")


def assert_view_not_leaked(train_keys: list[str], heldout_keys: list[str]) -> None:
    """Guard that no ``<bout>/<split_N>`` view of a held-out bout sits in the train set.

    ``keys`` are ``"<bout>:<split>"`` strings; held-out bouts must share no key with train.
    """
    train_bouts = {k.split(":", 1)[0] for k in train_keys}
    heldout_bouts = {k.split(":", 1)[0] for k in heldout_keys}
    leaked = train_bouts & heldout_bouts
    if leaked:
        raise LeakageError(f"held-out bout view(s) present in train: {sorted(leaked)}")
