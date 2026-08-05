"""Typed contracts for every inter-stage artifact and the raw annotation input.

Each stage writes a *directory* containing a typed ``meta.json`` (``ArtifactMeta``)
plus one data file. ``meta.json`` is written last and atomically, so its presence
means "artifact complete". This module defines the schemas; ``io.py`` does the
atomic reading/writing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

SCHEMA_VERSION = "1.0"

# Canonical CustomLabels taxonomy (lower-cased, as stored on disk after .lower()).
STRIKE_TYPES = ("head_landed", "body_landed", "strike_blocked", "strike_missed")
FIGHTERS = ("red", "blue")
STRIKE_LABELS = tuple(f"{c}_{t}" for c in FIGHTERS for t in STRIKE_TYPES)


class ArtifactMeta(BaseModel):
    """Provenance + shape header written as ``meta.json`` for every stage artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    stage: str
    source_video: str
    fps: float
    width: int
    height: int
    num_frames: int
    source_mtime_ns: int | None = None
    source_sha1_short: str | None = None
    annotation_source: str | None = None
    annotation_sha1: str | None = None
    git_sha: str | None = None
    created_utc: str | None = None
    producer: dict = {}


class Annotation(BaseModel):
    """One raw per-frame strike annotation entry."""

    model_config = ConfigDict(extra="ignore")

    frame: int
    label: str
    time_sec: float | None = None

    @field_validator("label")
    @classmethod
    def _lower(cls, v: str) -> str:
        # On-disk labels are mixed-case ("Blue_strike_missed"); normalise like labels.py:325.
        return str(v).lower()


class AnnotationDoc(BaseModel):
    """A bout's annotation file (one JSON dict; frame indices apply to all 4 views)."""

    model_config = ConfigDict(extra="ignore")

    video_path: str | None = None
    fps: float | None = None
    num_frames: int | None = None
    annotations: list[Annotation] = []


class StrikeRun(BaseModel):
    """A maximal run of consecutive frames sharing one strike label (no cross-label merge)."""

    model_config = ConfigDict(extra="forbid")

    label: str
    start_frame: int
    end_frame: int

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1


# --- on-disk dataframe schemas (column -> pandas dtype) ---------------------

DETECTION_SCHEMA: dict[str, str] = {
    "frame": "int32",
    "red_present": "bool",
    "blue_present": "bool",
    "red_x1": "int32",
    "red_y1": "int32",
    "red_x2": "int32",
    "red_y2": "int32",
    "blue_x1": "int32",
    "blue_y1": "int32",
    "blue_x2": "int32",
    "blue_y2": "int32",
    "red_det_conf": "float32",
    "blue_det_conf": "float32",
    "red_cls_conf": "float32",
    "blue_cls_conf": "float32",
    "n_candidates": "int16",
}

CROP_MANIFEST_SCHEMA: dict[str, str] = {
    "frame": "int32",
    "crop_x1": "int32",
    "crop_y1": "int32",
    "crop_x2": "int32",
    "crop_y2": "int32",
    "crop_valid": "bool",
    "staleness": "int16",
    "ema_cx": "float32",
    "ema_cy": "float32",
    "ema_half": "float32",
}

PER_FIGHTER_MANIFEST_SCHEMA: dict[str, str] = {
    "frame": "int32",
    "red_crop_x1": "int32", "red_crop_y1": "int32", "red_crop_x2": "int32", "red_crop_y2": "int32",
    "red_valid": "bool",
    "blue_crop_x1": "int32", "blue_crop_y1": "int32", "blue_crop_x2": "int32", "blue_crop_y2": "int32",
    "blue_valid": "bool",
}

FRAME_PROBS_SCHEMA: dict[str, str] = {
    "frame": "int32",
    "p_punch": "float32",
    "p_raw": "float32",
    "coverage": "int16",
    "crop_valid": "bool",
    "p_smooth": "float32",
    "label": "int8",
}


def missing_columns(columns: list[str], schema: dict[str, str]) -> list[str]:
    """Return schema columns absent from ``columns`` (order-independent)."""
    return [c for c in schema if c not in columns]
