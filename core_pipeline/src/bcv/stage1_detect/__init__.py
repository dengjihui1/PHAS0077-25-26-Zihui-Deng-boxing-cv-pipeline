"""Stage 1 — fighter bounding-box detection (pluggable backend behind one interface)."""

from .interface import Box, Candidate, FighterDetector, FrameDetection
from .select import select_fighters

__all__ = ["Box", "Candidate", "FighterDetector", "FrameDetection", "select_fighters"]
