"""Infer fight rounds from punch annotations, to exclude between-round (rest) frames.

Between-round rest periods are mostly "fighters idle, no punches" — they inflate the
negative class and add little signal. Our fights show an unmistakable structure: punch
events cluster into rounds separated by ~60-75s rests, so a single gap threshold cleanly
splits them (verified: every labelled fight has exactly 2 gaps > 30s -> 3 rounds).

``infer_rounds`` turns punch ``runs`` into round ``[start, end]`` frame spans (padded to
catch the round edges before the first / after the last punch); ``in_rounds_mask`` flags
which frames are in-round. The old ``bout_1..14`` set also has explicit
``round_N_start`` offsets in ``raw_data/livecode-video-mapping_with_round_offsets.csv``,
but our 115-122 fights are not in it, so we infer instead.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

Span = tuple[int, int]
ROUNDS_FILENAME = "rounds.json"


def rounds_path(bout_dir: Path) -> Path:
    """Round spans are FIGHT-level (the 4 splits are frame-synced) -> one file per bout."""
    return bout_dir / ROUNDS_FILENAME


def load_rounds(bout_dir: Path) -> list[Span]:
    """Read fight round spans from ``bout_dir/rounds.json``; [] if absent."""
    p = rounds_path(bout_dir)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [(int(s), int(e)) for s, e in data.get("rounds", [])]


def save_rounds(bout_dir: Path, rounds: list[Span], *, fps: float, source: str) -> Path:
    p = rounds_path(bout_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"fps": fps, "source": source,
            "rounds": [[int(s), int(e)] for s, e in sorted(rounds)]}
    p.write_text(json.dumps(body, indent=1))
    return p


def infer_rounds(
    runs: Iterable, *, fps: float, gap_s: float = 30.0, pad_s: float = 8.0,
    max_frame: int | None = None,
) -> list[Span]:
    """Split punch-event ``runs`` into round ``[start, end]`` frame spans on gaps > ``gap_s``.

    Each round span is the first..last punch of its cluster, padded by ``pad_s`` on each
    side (to include the bell-to-first-punch and last-punch-to-bell edges). ``runs`` are
    objects with ``start_frame`` / ``end_frame``.
    """
    spans = sorted((int(r.start_frame), int(r.end_frame)) for r in runs)
    if not spans:
        return []
    gap_f = gap_s * fps
    clusters: list[list[Span]] = [[spans[0]]]
    for s, e in spans[1:]:
        if s - clusters[-1][-1][1] > gap_f:
            clusters.append([(s, e)])
        else:
            clusters[-1].append((s, e))
    pad = round(pad_s * fps)
    rounds: list[Span] = []
    for cl in clusters:
        lo = max(0, min(s for s, _ in cl) - pad)
        hi = max(e for _, e in cl) + pad
        if max_frame is not None:
            hi = min(max_frame, hi)
        rounds.append((lo, hi))
    return rounds


def in_rounds_mask(frame_idx: Sequence[int] | np.ndarray, rounds: list[Span]) -> np.ndarray:
    """Boolean mask: True where the frame index falls inside any round span (inclusive)."""
    fi = np.asarray(frame_idx)
    mask = np.zeros(len(fi), dtype=bool)
    for lo, hi in rounds:
        mask |= (fi >= lo) & (fi <= hi)
    return mask


def between_round_fraction(num_frames: int, rounds: list[Span]) -> float:
    """Fraction of [0, num_frames) frames that are NOT in any round (the rest periods)."""
    if num_frames <= 0:
        return 0.0
    in_round = in_rounds_mask(np.arange(num_frames), rounds).sum()
    return float(num_frames - in_round) / num_frames
