"""Stage 1 detector interface: one ``frame -> {red_box, blue_box}`` contract.

Both backends (the detect→classify chain and the future fine-tuned 2-class detector)
implement ``FighterDetector`` and emit the same ``FrameDetection`` with at most one
red and one blue box already resolved (see ``select.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

Bbox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Box:
    """A resolved fighter box in native pixel coords with its two confidences."""

    x1: int
    y1: int
    x2: int
    y2: int
    det_conf: float
    cls_conf: float

    @property
    def bbox(self) -> Bbox:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Candidate:
    """A pre-resolution detection: a person box plus per-color classifier confidences."""

    bbox: Bbox
    det_conf: float
    cls_conf: dict[str, float] = field(default_factory=dict)
    track_id: int | None = None


@dataclass(frozen=True)
class FrameDetection:
    """One frame's resolved fighters (either may be ``None`` if absent)."""

    frame: int
    red: Box | None
    blue: Box | None
    n_candidates: int


class FighterDetector(ABC):
    """Abstract ``frame -> FrameDetection`` detector; backends are interchangeable."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int) -> FrameDetection: ...

    def reset(self) -> None:  # noqa: B027  optional no-op hook; not all backends track
        """Clear any per-video tracker state (called between videos)."""

    @property
    @abstractmethod
    def native_resolution(self) -> tuple[int, int]:
        """(width, height) the boxes are expressed in."""
