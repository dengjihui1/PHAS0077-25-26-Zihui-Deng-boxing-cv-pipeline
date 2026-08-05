"""Stage 4 — aggregate the per-frame probability curve into discrete strike windows."""

from .hysteresis import Window, make_windows, tag_windows_with_gt

__all__ = ["Window", "make_windows", "tag_windows_with_gt"]
