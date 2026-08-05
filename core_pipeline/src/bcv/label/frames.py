"""Frame-accurate JPEG frame server for the labelling UI.

Wraps one video with a seek-and-decode cache so the browser can jump to any frame. cv2's
sequential reads are fast; backward seeks use ``CAP_PROP_POS_FRAMES``.

Concurrency: the FastAPI ``/frame`` route is a sync endpoint, so uvicorn serves it from a
threadpool — rapid scrubbing/playback (plus the background pre-fill thread) decode video
from several threads at once. A single ``VideoCapture`` is NOT thread-safe, and ffmpeg's
frame-threaded decoder trips ``Assertion fctx->async_lock failed`` under concurrent use. We
therefore (a) serialize every read with a lock and (b) force single-threaded ffmpeg decode.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

# Disable ffmpeg frame-level multithreading BEFORE any VideoCapture opens — this is what
# eliminates the libavcodec ``async_lock`` assertion under concurrent decode. setdefault so
# an explicit override is respected.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

import cv2
import numpy as np


class FrameSource:
    def __init__(self, video_path: str | Path) -> None:
        self.path = str(video_path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video: {self.path}")
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._pos = -1
        self._lock = threading.Lock()  # one decoder, many request threads -> serialize

    def read(self, idx: int) -> np.ndarray:
        idx = max(0, min(idx, self.num_frames - 1))
        with self._lock:
            if idx != self._pos + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            self._pos = idx
        if not ok:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

    def jpeg(self, idx: int, *, quality: int = 80) -> bytes:
        frame = self.read(idx)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else b""

    def release(self) -> None:
        with self._lock:
            self.cap.release()
