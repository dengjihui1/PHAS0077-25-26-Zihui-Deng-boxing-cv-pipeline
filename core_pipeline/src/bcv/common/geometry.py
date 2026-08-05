"""Pure box geometry: union, pad, square-ify, shift-clamp, EMA, person crop.

All boxes are ``(x1, y1, x2, y2)`` in pixel coordinates with ``x2 >= x1`` and
``y2 >= y1``. Functions are numpy-free where possible so they are trivially testable.
"""

from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]


def width(b: Box) -> float:
    return b[2] - b[0]


def height(b: Box) -> float:
    return b[3] - b[1]


def center(b: Box) -> tuple[float, float]:
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def area(b: Box) -> float:
    return max(0.0, width(b)) * max(0.0, height(b))


def union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    return inter / (area(a) + area(b) - inter)


def pad_box(b: Box, frac: float) -> Box:
    """Expand each side by ``frac`` of that dimension (width grows by ``2*frac*w``)."""
    px, py = width(b) * frac, height(b) * frac
    return (b[0] - px, b[1] - py, b[2] + px, b[3] + py)


def square_box(b: Box, *, min_half: float = 0.0) -> Box:
    """Smallest square centered on ``b`` that contains it (half-side >= ``min_half``)."""
    cx, cy = center(b)
    half = max(0.5 * width(b), 0.5 * height(b), min_half)
    return (cx - half, cy - half, cx + half, cy + half)


def clamp_shift(b: Box, w: int, h: int) -> Box:
    """Shift ``b`` to lie inside ``[0,w]x[0,h]`` preserving its size; clamp if it is larger."""
    bw, bh = width(b), height(b)
    x1, y1, x2, y2 = b
    if bw >= w:
        x1, x2 = 0.0, float(w)
    else:
        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        if x2 > w:
            x1 -= x2 - w
            x2 = float(w)
    if bh >= h:
        y1, y2 = 0.0, float(h)
    else:
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        if y2 > h:
            y1 -= y2 - h
            y2 = float(h)
    return (x1, y1, x2, y2)


def to_int_box(b: Box) -> tuple[int, int, int, int]:
    return (round(b[0]), round(b[1]), round(b[2]), round(b[3]))


def ema(prev: float | None, cur: float, alpha: float) -> float:
    """Exponential moving average; first observation passes through unchanged."""
    if prev is None:
        return float(cur)
    return float(alpha * cur + (1.0 - alpha) * prev)


def crop_person(
    frame: np.ndarray,
    b: Box,
    *,
    pad_frac: float = 0.05,
    min_w: int = 0,
    min_h: int = 0,
) -> np.ndarray | None:
    """Crop ``frame`` (H,W,C) to padded box ``b``; return ``None`` if below min size."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = to_int_box(clamp_shift(pad_box(b, pad_frac), w, h))
    if (x2 - x1) < min_w or (y2 - y1) < min_h or x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
