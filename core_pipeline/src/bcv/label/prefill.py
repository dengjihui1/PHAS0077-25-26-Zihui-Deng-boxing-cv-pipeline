"""Model-assisted pre-fill for the bbox placer: seed keyframes from the chain detector.

Runs the Stage-1 chain over a frame range and samples a box every ``stride`` frames as a
starting keyframe per fighter, so the human adjusts a skeleton instead of drawing from
scratch. Accumulates into LOCAL dicts and merges into the project atomically at the end, so
the server can keep serving frames/boxes while it runs (no mid-mutation races). Supports a
cancel check and progress callback; heavy deps are imported lazily.
"""
from __future__ import annotations

from collections.abc import Callable

from .boxes import BBoxProject, Box


def prefill_from_chain(
    proj: BBoxProject,
    pipeline,
    stage_cfg,
    *,
    stride: int = 15,
    start_frame: int = 0,
    max_frames: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    """Seed ``proj`` keyframes every ``stride`` frames from the chain detector over a range.

    Returns the number of sampled keyframe-frames added. Cancellation merges partial work.
    """
    from bcv.common.video import VideoReader
    from bcv.stage1_detect.run import build_detector

    detector = build_detector(stage_cfg)
    detector.reset()
    reader = VideoReader(pipeline.split_video(proj.bout, proj.split))
    if start_frame > 0:
        reader.seek(start_frame)
    total = max_frames if max_frames is not None else (proj.num_frames - start_frame)

    new_red: dict[str, Box | None] = {}
    new_blue: dict[str, Box | None] = {}
    n = 0
    try:
        for i, frame in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break
            if should_cancel is not None and should_cancel():
                break
            idx = start_frame + i
            fd = detector.detect(frame, idx)
            if i % stride == 0:
                # Record absent (None) too, so interpolation doesn't bridge across gaps.
                new_red[str(idx)] = [int(v) for v in fd.red.bbox] if fd.red is not None else None
                new_blue[str(idx)] = [int(v) for v in fd.blue.bbox] if fd.blue is not None else None
                n += 1
            if on_progress is not None and i % 50 == 0:
                on_progress(i, total)
    finally:
        reader.release()

    # Atomic merge: swap whole per-fighter dicts so concurrent readers never see a half-update.
    proj.keyframes = {
        "red": {**proj.keyframes.get("red", {}), **new_red},
        "blue": {**proj.keyframes.get("blue", {}), **new_blue},
    }
    if on_progress is not None:
        on_progress(total, total)
    return n
