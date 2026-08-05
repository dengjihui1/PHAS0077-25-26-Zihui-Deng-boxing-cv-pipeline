"""Preprocessing: turn raw quad-view recordings into the per-POV ``new_splits`` layout."""

from .pov_split import crop_filter, split_quad_video

__all__ = ["crop_filter", "split_quad_video"]
