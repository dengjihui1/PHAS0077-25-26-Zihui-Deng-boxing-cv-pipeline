"""Thin, single-implementation video read/write wrappers over OpenCV.

One ``VideoReader`` and one ``VideoWriter`` for the whole repo (the old code had the
same logic copied across several modules). ``VideoWriter`` writes to a temp file and
``os.replace``s it into place on ``close()`` so a finished ``crop.mp4`` is atomic, in
keeping with the artifact-IO mandate.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    num_frames: int


class VideoReader:
    """Iterate decoded BGR frames of a video and expose its basic properties."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        self.info = VideoInfo(
            fps=float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            num_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )

    def seek(self, frame_idx: int) -> None:
        """Seek so the next read returns frame ``frame_idx`` (approximate for some codecs)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class VideoWriter:
    """Write BGR frames to an mp4; publishes atomically (tmp -> fsync -> replace)."""

    def __init__(
        self, path: str | Path, fps: float, width: int, height: int, *, fourcc: str = "mp4v"
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}.mp4")
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)  # type: ignore[attr-defined]
        self.writer = cv2.VideoWriter(
            str(self._tmp), fourcc_code, float(fps), (int(width), int(height))
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {self.path} (codec {fourcc})")
        self._n = 0

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)
        self._n += 1

    def close(self) -> int:
        self.writer.release()
        os.replace(self._tmp, self.path)
        dfd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return self._n

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type: object, *rest: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.writer.release()
            if self._tmp.exists():
                self._tmp.unlink()
