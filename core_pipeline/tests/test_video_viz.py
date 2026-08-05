from __future__ import annotations

from pathlib import Path

import numpy as np

from bcv.common import viz
from bcv.common.video import VideoReader, VideoWriter


def test_draw_box_keeps_shape():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    out = viz.draw_box(frame, (5, 5, 30, 30), viz.RED, label="red")
    assert out.shape == frame.shape


def test_prob_trace_appends_strip():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    probs = np.linspace(0, 1, 20).astype(np.float32)
    out = viz.prob_trace(frame, probs, cur_idx=10, height=20, threshold=0.5)
    assert out.shape == (68, 64, 3)


def test_timeline_marks_windows():
    img = viz.draw_timeline(100, num_frames=50, windows=[(10, 20)])
    assert img.shape == (40, 100, 3)
    assert img.sum() > 0


def test_video_writer_reader_roundtrip(mini_video):
    src, n, w, h = mini_video
    info = VideoReader(src).info
    assert info.width == w and info.height == h

    dst = Path(src).with_name("copy.mp4")
    count = 0
    with VideoReader(src) as r, VideoWriter(dst, fps=10.0, width=w, height=h) as wr:
        for frame in r:
            wr.write(frame)
            count += 1
    assert count == n
    assert dst.exists()
    # atomic publish: no temp file remains
    assert not list(Path(src).parent.glob(".copy.mp4.tmp*"))
