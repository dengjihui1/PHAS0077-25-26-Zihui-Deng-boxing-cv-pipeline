"""Annotation INPUT contract — discovery, parsing, run-grouping, target derivation.

The raw annotations are dense multi-frame *runs* of a single label (verified on bout
122: median run length 4), NOT point instants. Stage 3's binary target is therefore
*span membership*, and Stage 5's type label is the run's label. Both annotation
filename patterns are supported; the frame-indexed labels apply to all four camera
splits of a bout (they are time-synced views of one fight).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import STRIKE_LABELS, Annotation, AnnotationDoc, StrikeRun
from .io import read_json

_KNOWN = set(STRIKE_LABELS)


def discover_annotation_file(bout_dir: str | Path) -> Path | None:
    """Locate a bout's annotation JSON across both naming conventions.

    Prefers ``annotations.json``; else the ``Bout <N>_Split 1-4.json`` form; else any
    ``*.json`` that is not a bbox/meta sidecar. Returns ``None`` if none found.
    """
    bout_dir = Path(bout_dir)
    preferred = bout_dir / "annotations.json"
    if preferred.is_file():
        return preferred

    def _is_sidecar(p: Path) -> bool:
        n = p.name.lower()
        return n.endswith("_fighter_bboxes.json") or n == "meta.json"

    split_pattern = sorted(p for p in bout_dir.glob("Bout *.json") if not _is_sidecar(p))
    if split_pattern:
        return split_pattern[0]
    others = sorted(p for p in bout_dir.glob("*.json") if not _is_sidecar(p))
    return others[0] if others else None


def load_annotations(path: str | Path) -> AnnotationDoc:
    """Parse an annotation file (single dict or list-of-dicts) into an ``AnnotationDoc``.

    Labels are lower-cased by the ``Annotation`` validator.
    """
    data = read_json(path)
    items = data if isinstance(data, list) else [data]
    anns: list[Annotation] = []
    doc_meta: dict = {}
    for item in items:
        if isinstance(item, dict) and "annotations" in item:
            if not doc_meta:
                doc_meta = {k: item.get(k) for k in ("video_path", "fps", "num_frames")}
            raw = item.get("annotations") or []
        else:
            raw = item
        if not isinstance(raw, list):
            raise ValueError(f"Expected a list of annotation dicts, got {type(raw)} in {path}")
        anns.extend(Annotation.model_validate(a) for a in raw)
    return AnnotationDoc(annotations=anns, **doc_meta)


def load_runs(bout_dir: str | Path) -> list[StrikeRun]:
    """Convenience: discover + parse + group a bout's annotations into strike runs (or [])."""
    path = discover_annotation_file(bout_dir)
    if path is None:
        return []
    return group_into_runs(load_annotations(path).annotations)


def group_into_runs(annotations: list[Annotation], *, validate: bool = False) -> list[StrikeRun]:
    """Group consecutive same-label frames into runs (no cross-label merge).

    Mirrors the per-label interval logic in ``labels.py`` (lines 333-347) but stops
    before its cross-label span merge, which neither training stage wants.
    """
    frames_by_label: dict[str, list[int]] = {}
    for a in annotations:
        if validate and a.label not in _KNOWN:
            raise ValueError(f"Unknown strike label {a.label!r}; expected one of {sorted(_KNOWN)}")
        frames_by_label.setdefault(a.label, []).append(int(a.frame))

    runs: list[StrikeRun] = []
    for label, frames in frames_by_label.items():
        ordered = sorted(set(frames))
        start = end = ordered[0]
        for fr in ordered[1:]:
            if fr <= end + 1:
                end = fr
                continue
            runs.append(StrikeRun(label=label, start_frame=start, end_frame=end))
            start = end = fr
        runs.append(StrikeRun(label=label, start_frame=start, end_frame=end))
    runs.sort(key=lambda r: (r.start_frame, r.end_frame, r.label))
    return runs


def frame_binary_target(runs: list[StrikeRun], num_frames: int, *, dilate: int = 0) -> np.ndarray:
    """Per-frame binary punch target: 1 if a frame lies within a run (optionally ±dilate)."""
    target = np.zeros(int(num_frames), dtype=np.int8)
    for r in runs:
        lo = max(0, r.start_frame - dilate)
        hi = min(num_frames - 1, r.end_frame + dilate)
        if hi >= lo:
            target[lo : hi + 1] = 1
    return target
