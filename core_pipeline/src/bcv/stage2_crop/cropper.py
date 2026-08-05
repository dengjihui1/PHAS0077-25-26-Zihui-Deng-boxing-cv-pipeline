"""Stage 2 crop geometry: union of fighters → EMA-smoothed, padded, fixed square.

Per frame we take the union of the red/blue boxes (or the lone present box), turn it into
a padded square crop window, and EMA-smooth its center+size across frames so the output is
stable to watch and a fixed size for the CNN. When neither fighter is present we carry the
last window forward (``crop_valid=False``) up to ``max_staleness`` frames, then fall back to
a full-frame center crop so a long dropout never freezes the view on a stale region.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from ..common.geometry import Box, center, clamp_shift, ema, height, square_box, to_int_box, width


class Stage2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crop_size: int = 224
    pad_frac: float = 0.15
    ema_alpha: float = 0.2
    scale_mode: str = "union"  # "union" | "max_single_fighter"
    min_half_side: float = 0.0
    # When only ONE fighter is detected the union is just that fighter's box, so the
    # opponent falls outside the crop. Enlarge the window by this factor so a likely-
    # adjacent opponent is still covered (1.0 = off).
    single_fighter_scale: float = 1.8
    max_staleness: int = 30
    fallback: str = "full_frame"
    resize_mode: str = "area"  # cv2 interpolation: area|linear|cubic|nearest
    emit_per_fighter: bool = True  # also emit red_crop.mp4 + blue_crop.mp4 (Stage 5 / multi-view)


@dataclass
class CropResult:
    crop_box: tuple[int, int, int, int]
    crop_valid: bool
    staleness: int
    ema_cx: float
    ema_cy: float
    ema_half: float


def _target(
    red: Box | None, blue: Box | None, scale_mode: str, single_fighter_scale: float = 1.0
) -> tuple[float, float, float] | None:
    """Center + half-side for this frame's detections, or None if neither present.

    With a single detected fighter the window is enlarged by ``single_fighter_scale`` so a
    likely-adjacent opponent (whose box is missing) is still covered.
    """
    boxes = [b for b in (red, blue) if b is not None]
    if not boxes:
        return None
    if len(boxes) == 2:
        assert red is not None and blue is not None
        cx = 0.25 * (red[0] + red[2] + blue[0] + blue[2])
        cy = 0.25 * (red[1] + red[3] + blue[1] + blue[3])
        if scale_mode == "max_single_fighter":
            half = 0.5 * max(max(width(red), height(red)), max(width(blue), height(blue)))
        else:  # union span keeps both fighters in frame
            ux1, uy1 = min(red[0], blue[0]), min(red[1], blue[1])
            ux2, uy2 = max(red[2], blue[2]), max(red[3], blue[3])
            half = 0.5 * max(ux2 - ux1, uy2 - uy1)
    else:
        b = boxes[0]
        cx, cy = center(b)
        half = 0.5 * max(width(b), height(b)) * single_fighter_scale  # cover the missing opponent
    return cx, cy, half


def crop_box_for_frame(
    red: Box | None, blue: Box | None, w: int, h: int, cfg: Stage2Config
) -> tuple[int, int, int, int]:
    """Stateless per-frame crop window (no temporal EMA) — for previewing what Stage 2 crops.

    0 fighters -> full-frame center crop (nothing to anchor on, so cover everything);
    1 fighter -> enlarged by ``single_fighter_scale``; 2 -> their padded union.
    """
    tgt = _target(red, blue, cfg.scale_mode, cfg.single_fighter_scale)
    if tgt is None:
        cx, cy, half = w / 2.0, h / 2.0, 0.5 * min(w, h)
    else:
        cx, cy, half = tgt
        half = max(half, cfg.min_half_side) * (1.0 + cfg.pad_frac)
    sq = square_box((cx - half, cy - half, cx + half, cy + half))
    return to_int_box(clamp_shift(sq, w, h))


class Cropper:
    """Stateful per-video cropper; call ``step`` once per frame in order."""

    def __init__(self, cfg: Stage2Config) -> None:
        self.cfg = cfg
        self._cx: float | None = None
        self._cy: float | None = None
        self._half: float | None = None
        self._staleness = 0

    def _full_frame(self, w: int, h: int) -> tuple[float, float, float]:
        return (w / 2.0, h / 2.0, 0.5 * min(w, h))

    def step(self, red: Box | None, blue: Box | None, w: int, h: int) -> CropResult:
        tgt = _target(red, blue, self.cfg.scale_mode, self.cfg.single_fighter_scale)

        if tgt is not None:
            cx, cy, half = tgt
            half = max(half, self.cfg.min_half_side) * (1.0 + self.cfg.pad_frac)
            self._cx = ema(self._cx, cx, self.cfg.ema_alpha)
            self._cy = ema(self._cy, cy, self.cfg.ema_alpha)
            self._half = ema(self._half, half, self.cfg.ema_alpha)
            self._staleness = 0
            valid = True
        else:
            self._staleness += 1
            stale_out = self._staleness > self.cfg.max_staleness
            if self._half is None or (stale_out and self.cfg.fallback == "full_frame"):
                # never had a fix, or dropped out too long: snap to full-frame center crop
                self._cx, self._cy, self._half = self._full_frame(w, h)
            valid = False

        assert self._cx is not None and self._cy is not None and self._half is not None
        sq = square_box(
            (
                self._cx - self._half,
                self._cy - self._half,
                self._cx + self._half,
                self._cy + self._half,
            )
        )
        crop_box = to_int_box(clamp_shift(sq, w, h))
        return CropResult(
            crop_box=crop_box,
            crop_valid=valid,
            staleness=self._staleness,
            ema_cx=self._cx,
            ema_cy=self._cy,
            ema_half=self._half,
        )
