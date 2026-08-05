"""Debug-overlay primitives shared by every stage's ``debug.mp4``.

Pure OpenCV/numpy drawing helpers: fighter boxes, the crop rectangle, the
probability-vs-time strip drawn along the bottom of a frame, a window timeline, and a
per-clip label banner. Kept here so each stage stays thin and the five debug videos
look consistent.
"""

from __future__ import annotations

import cv2
import numpy as np

RED = (0, 0, 255)  # BGR
BLUE = (255, 0, 0)
GREEN = (0, 200, 0)
GREY = (128, 128, 128)


def draw_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str | None = None,
    thickness: int = 2,
) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def draw_union_crop(
    frame: np.ndarray, crop_box: tuple[int, int, int, int], *, valid: bool = True
) -> np.ndarray:
    color = GREEN if valid else GREY
    return draw_box(frame, crop_box, color, label=None if valid else "stale", thickness=2)


def prob_trace(
    frame: np.ndarray,
    probs: np.ndarray,
    cur_idx: int,
    *,
    height: int = 90,
    threshold: float | None = None,
    labels: np.ndarray | None = None,
    window_frames: int | None = None,
) -> np.ndarray:
    """Append a P(punch)-vs-time strip to the bottom of ``frame``, cursor at ``cur_idx``.

    If ``labels`` (a 0/1 ground-truth array aligned with ``probs``) is given, the
    frames where label==1 are shaded green behind the prediction trace, and a
    ``GT: PUNCH`` cue is drawn on the frame when the cursor is inside a truth span.

    With ``window_frames`` set, the strip shows only a rolling window of that many frames
    centered on the cursor (the playhead stays mid-strip and the trace scrolls), so local
    on/off detail is legible; unset, it shows the whole sequence squeezed to the width.
    """
    _h, w = frame.shape[:2]
    strip = np.full((height, w, 3), 30, dtype=np.uint8)

    def _y(p: float) -> int:
        # fixed axis: p in [0,1] -> [bottom, top] of the strip
        return round((height - 1) - min(max(p, 0.0), 1.0) * (height - 1))

    n = len(probs)
    if n > 0:
        # Visible window [lo, hi): a rolling span centered on the cursor, else the whole clip.
        if window_frames and window_frames > 1:
            lo = cur_idx - window_frames // 2
            hi = lo + window_frames
        else:
            lo, hi = 0, n
        span = max(1, hi - lo)

        def _x(f: int) -> int:
            return round((f - lo) * (w - 1) / span)

        i0, i1 = max(0, lo), min(n, hi)
        if labels is not None:
            gt = np.asarray(labels)
            punch = np.flatnonzero(gt[:n] > 0)
            punch = punch[(punch >= i0) & (punch < i1)]
            if punch.size:
                shade = strip.copy()
                for i in punch:
                    x0, x1 = _x(int(i)), _x(int(i) + 1)
                    cv2.rectangle(shade, (max(0, x0), 0), (min(w - 1, max(x0, x1)), height - 1),
                                  (0, 120, 0), -1)
                cv2.addWeighted(shade, 0.5, strip, 0.5, 0, strip)
        # fixed 0..1 y-axis: faint gridlines + labels so the scale is explicit
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            cv2.line(strip, (0, _y(p)), (w - 1, _y(p)), (70, 70, 70), 1)
        for p in (0.0, 0.5, 1.0):
            yy = min(max(_y(p), 9), height - 2)
            cv2.putText(
                strip, f"{p:.1f}", (2, yy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA,
            )
        # vertical time gridlines (quarters of the visible window) for temporal reference
        for q in (0.25, 0.5, 0.75):
            xq = round(q * (w - 1))
            cv2.line(strip, (xq, 0), (xq, height - 1), (55, 55, 55), 1)
        if threshold is not None:
            cv2.line(strip, (0, _y(threshold)), (w - 1, _y(threshold)), (60, 60, 160), 1)
        if i1 > i0:
            fidx = np.arange(i0, i1)
            xs = ((fidx - lo) * (w - 1) / span).astype(int)
            ys = (height - 1 - np.clip(probs[i0:i1], 0.0, 1.0) * (height - 1)).astype(int)
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.polylines(strip, [pts], isClosed=False, color=(0, 220, 220), thickness=1)
        cx = _x(cur_idx)  # playhead (centered when windowed)
        if 0 <= cx <= w - 1:
            cv2.line(strip, (cx, 0), (cx, height - 1), (255, 255, 255), 1)
    cv2.rectangle(strip, (0, 0), (w - 1, height - 1), (90, 90, 90), 1)  # box outline
    out = np.vstack([frame, strip])
    if labels is not None and 0 <= cur_idx < len(labels) and labels[cur_idx] > 0:
        cv2.putText(out, "GT: PUNCH", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    return out


def draw_timeline(
    width: int, num_frames: int, windows: list[tuple[int, int]], *, height: int = 40
) -> np.ndarray:
    """Render a horizontal timeline image with strike ``windows`` marked."""
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    if num_frames <= 1:
        return img
    for s, e in windows:
        x1 = int(s * (width - 1) / (num_frames - 1))
        x2 = int(e * (width - 1) / (num_frames - 1))
        cv2.rectangle(img, (x1, 4), (max(x1 + 1, x2), height - 5), GREEN, -1)
    return img


def clip_label(frame: np.ndarray, text: str, *, color: tuple[int, int, int] = GREEN) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out
