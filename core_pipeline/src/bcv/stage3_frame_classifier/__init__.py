"""Stage 3 — per-frame punch classifier over a configurable centered window."""

from .dataset import CroppedWindowDataset
from .model import WindowPunchModule

__all__ = ["CroppedWindowDataset", "WindowPunchModule"]
