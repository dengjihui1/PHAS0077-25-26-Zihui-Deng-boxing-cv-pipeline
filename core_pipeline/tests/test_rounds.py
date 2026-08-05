"""Tests for round inference from punch annotations."""
from __future__ import annotations

from dataclasses import dataclass

from bcv.common.rounds import between_round_fraction, in_rounds_mask, infer_rounds


@dataclass
class _Run:
    start_frame: int
    end_frame: int


def test_infer_three_rounds_from_gaps():
    fps = 30.0
    # 3 clusters of punches separated by ~70s gaps (2100 frames) -> 3 rounds.
    runs = []
    for base in (0, 5000, 10000):  # cluster starts; ~166s apart
        for off in range(0, 1000, 100):  # punches every ~3s within a ~33s window
            runs.append(_Run(base + off, base + off + 5))
    rounds = infer_rounds(runs, fps=fps, gap_s=30.0, pad_s=8.0)
    assert len(rounds) == 3
    # padded by 8s = 240 frames
    assert rounds[0][0] == 0 and rounds[0][1] == 905 + 240
    assert rounds[1][0] == 5000 - 240


def test_no_runs_no_rounds():
    assert infer_rounds([], fps=30.0) == []


def test_mask_and_fraction():
    rounds = [(0, 99), (200, 299)]
    mask = in_rounds_mask([50, 150, 250, 350], rounds)
    assert list(mask) == [True, False, True, False]
    # frames 100..199 (100) + 300..399 (100) are between-round of 400 total
    assert abs(between_round_fraction(400, rounds) - 0.5) < 1e-9


def test_within_round_lulls_dont_split():
    fps = 30.0
    # a 20s lull (< 30s gap) inside a round must NOT create a new round
    runs = [_Run(0, 5), _Run(600, 605), _Run(610, 615)]  # 20s gap then close
    assert len(infer_rounds(runs, fps=fps, gap_s=30.0)) == 1
