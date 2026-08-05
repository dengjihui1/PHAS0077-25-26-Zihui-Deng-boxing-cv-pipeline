"""Turn the per-frame probability curve into discrete strike windows.

Two-threshold hysteresis (rise above ``t_high`` to open, fall below ``t_low`` to close)
avoids flicker at a single threshold; short blips are dropped by ``min_duration`` and
near-touching windows are joined by ``merge_gap``. Optionally, long high-confidence
segments can be split at sustained probability valleys or around multiple local peaks to
avoid swallowing adjacent punches into one event. Operates on array positions, then maps
back to absolute frame indices.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict


class Window(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: int
    start_frame: int
    end_frame: int
    peak_frame: int
    peak_prob: float
    mean_prob: float
    gt_labels: list[str] = []
    gt_multi: bool = False

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1


def _hysteresis_segments(p: np.ndarray, t_high: float, t_low: float) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(p):
        if not in_seg and v >= t_high:
            in_seg, start = True, i
        elif in_seg and v < t_low:
            segs.append((start, i - 1))
            in_seg = False
    if in_seg:
        segs.append((start, len(p) - 1))
    return segs


def _merge_close(segs: list[tuple[int, int]], merge_gap: int) -> list[tuple[int, int]]:
    if not segs:
        return []
    out = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = out[-1]
        if s - pe - 1 <= merge_gap:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _split_on_valleys(
    segs: list[tuple[int, int]],
    p: np.ndarray,
    split_valley: float | None,
    split_min_gap: int,
) -> list[tuple[int, int]]:
    """Split existing segments at sustained dips in the probability curve.

    Hysteresis deliberately stays open through dips above ``t_low``. For boxing exchanges,
    that often merges several adjacent punches into one long window. ``split_valley`` adds
    an explicit under-segmentation correction: if a run inside a segment stays at or below
    this value for at least ``split_min_gap`` frames, the low run is treated as a separator.
    """
    if split_valley is None:
        return segs
    min_gap = max(1, int(split_min_gap))
    out: list[tuple[int, int]] = []
    for s, e in segs:
        valleys: list[tuple[int, int]] = []
        in_valley = False
        v_start = s
        for i in range(s, e + 1):
            if not in_valley and p[i] <= split_valley:
                in_valley = True
                v_start = i
            elif in_valley and p[i] > split_valley:
                v_end = i - 1
                if v_end - v_start + 1 >= min_gap:
                    valleys.append((v_start, v_end))
                in_valley = False
        if in_valley:
            v_end = e
            if v_end - v_start + 1 >= min_gap:
                valleys.append((v_start, v_end))

        cur = s
        for v_start, v_end in valleys:
            if v_start > cur:
                out.append((cur, v_start - 1))
            cur = v_end + 1
        if cur <= e:
            out.append((cur, e))
    return out


def _find_peaks(p: np.ndarray, s: int, e: int, min_prob: float, min_distance: int) -> list[int]:
    candidates: list[int] = []
    for i in range(s, e + 1):
        left = p[i - 1] if i > s else -np.inf
        right = p[i + 1] if i < e else -np.inf
        if p[i] >= min_prob and p[i] >= left and p[i] >= right:
            candidates.append(i)

    selected: list[int] = []
    for i in sorted(candidates, key=lambda j: float(p[j]), reverse=True):
        if all(abs(i - j) >= min_distance for j in selected):
            selected.append(i)
    return sorted(selected)


def _split_on_peaks(
    segs: list[tuple[int, int]],
    p: np.ndarray,
    split_peak_min_prob: float | None,
    split_peak_min_distance: int,
    split_peak_min_drop: float,
) -> list[tuple[int, int]]:
    """Split long segments by finding multiple strong local peaks.

    Valley splitting only works when a segment has a sustained low trough. In practice the
    punch probability often forms a high plateau with several local maxima. This splitter
    treats well-separated peaks as separate strike hypotheses and cuts at the local
    minimum between them, provided the drop from the weaker peak is large enough.
    """
    if split_peak_min_prob is None:
        return segs
    min_distance = max(1, int(split_peak_min_distance))
    min_drop = max(0.0, float(split_peak_min_drop))
    out: list[tuple[int, int]] = []
    for s, e in segs:
        peaks = _find_peaks(p, s, e, float(split_peak_min_prob), min_distance)
        if len(peaks) < 2:
            out.append((s, e))
            continue

        split_points: list[int] = []
        for a, b in zip(peaks, peaks[1:]):
            if b - a <= 1:
                continue
            between = p[a + 1 : b]
            valley = int(np.argmin(between)) + a + 1
            drop = min(float(p[a]), float(p[b])) - float(p[valley])
            if drop >= min_drop:
                split_points.append(valley)

        cur = s
        for valley in split_points:
            if valley > cur:
                out.append((cur, valley - 1))
            cur = valley + 1
        if cur <= e:
            out.append((cur, e))
    return out


def make_windows(
    frames: np.ndarray,
    p_smooth: np.ndarray,
    p_punch: np.ndarray,
    *,
    t_high: float = 0.5,
    t_low: float = 0.35,
    min_duration: int = 3,
    merge_gap: int = 2,
    split_valley: float | None = None,
    split_min_gap: int = 2,
    split_peak_min_prob: float | None = None,
    split_peak_min_distance: int = 8,
    split_peak_min_drop: float = 0.1,
) -> list[Window]:
    """Hysteresis + optional segment splitting over ``p_smooth`` -> strike windows."""
    p_smooth = np.asarray(p_smooth)
    segs = _hysteresis_segments(p_smooth, t_high, t_low)
    segs = [(s, e) for (s, e) in segs if (e - s + 1) >= min_duration]
    segs = _merge_close(segs, merge_gap)
    segs = _split_on_valleys(segs, p_smooth, split_valley, split_min_gap)
    segs = _split_on_peaks(
        segs, p_smooth, split_peak_min_prob, split_peak_min_distance, split_peak_min_drop
    )
    segs = [(s, e) for (s, e) in segs if (e - s + 1) >= min_duration]

    windows: list[Window] = []
    for wid, (s, e) in enumerate(segs):
        seg_p = np.asarray(p_punch[s : e + 1])
        peak_local = int(np.argmax(seg_p)) + s
        windows.append(
            Window(
                window_id=wid,
                start_frame=int(frames[s]),
                end_frame=int(frames[e]),
                peak_frame=int(frames[peak_local]),
                peak_prob=float(p_punch[peak_local]),
                mean_prob=float(seg_p.mean()),
            )
        )
    return windows


def tag_windows_with_gt(windows: list[Window], gt_events: list[tuple[int, int, str]]) -> None:
    """Annotate each window in-place with the labels of GT events it overlaps."""
    for w in windows:
        labels = [
            lab for (gs, ge, lab) in gt_events if not (ge < w.start_frame or gs > w.end_frame)
        ]
        w.gt_labels = labels
        w.gt_multi = len(set(labels)) > 1
