"""Stage 2 — reframe the video to a stabilized, fixed-size square fighter crop."""

from .cropper import Cropper, CropResult, Stage2Config

__all__ = ["CropResult", "Cropper", "Stage2Config"]
