"""Shared fixtures: a tiny annotation doc (capitalized labels) and a mini video."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def anno_dict() -> dict:
    """Mixed-case labels + two distinct runs (one red, one blue) to exercise .lower()/grouping."""
    return {
        "video_path": "/some/Bout 999_Split 1-4.mp4",
        "fps": 30.0,
        "num_frames": 50,
        "annotations": [
            # blue run frames 10-13
            {"frame": 10, "time_sec": 0.33, "label": "Blue_strike_missed"},
            {"frame": 11, "time_sec": 0.36, "label": "Blue_strike_missed"},
            {"frame": 12, "time_sec": 0.40, "label": "Blue_strike_missed"},
            {"frame": 13, "time_sec": 0.43, "label": "Blue_strike_missed"},
            # red run frames 20-21
            {"frame": 20, "time_sec": 0.66, "label": "Red_head_landed"},
            {"frame": 21, "time_sec": 0.70, "label": "Red_head_landed"},
        ],
    }


@pytest.fixture
def bout_dir_legacy(tmp_path: Path, anno_dict: dict) -> Path:
    """A bout dir using the legacy ``annotations.json`` filename."""
    d = tmp_path / "Bout 115_Split 1-4"
    d.mkdir()
    (d / "annotations.json").write_text(json.dumps(anno_dict))
    # a bbox sidecar that discovery must ignore
    (d / "split_0_fighter_bboxes.json").write_text("[]\n")
    return d


@pytest.fixture
def bout_dir_split(tmp_path: Path, anno_dict: dict) -> Path:
    """A bout dir using the ``Bout <N>_Split 1-4.json`` filename."""
    d = tmp_path / "Bout 122_Split 1-4"
    d.mkdir()
    (d / "Bout 122_Split 1-4.json").write_text(json.dumps(anno_dict))
    return d


@pytest.fixture
def mini_video(tmp_path: Path) -> tuple[Path, int, int, int]:
    """Write a 12-frame 64x48 BGR video; returns (path, n, w, h). Skips if codec missing."""
    import cv2

    w, h, n = 64, 48, 12
    path = tmp_path / "mini.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    if not vw.isOpened():
        pytest.skip("mp4v VideoWriter unavailable in this environment")
    for i in range(n):
        frame = np.full((h, w, 3), i * 5 % 255, dtype=np.uint8)
        vw.write(frame)
    vw.release()
    return path, n, w, h
