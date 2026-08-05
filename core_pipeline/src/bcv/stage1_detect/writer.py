"""Build the dense ``detections.parquet`` (one row per frame) from FrameDetections."""

from __future__ import annotations

import pandas as pd

from ..common.contracts import DETECTION_SCHEMA
from .interface import Box, FrameDetection


def _box_cols(prefix: str, box: Box | None) -> dict[str, object]:
    if box is None:
        return {
            f"{prefix}_present": False,
            f"{prefix}_x1": -1,
            f"{prefix}_y1": -1,
            f"{prefix}_x2": -1,
            f"{prefix}_y2": -1,
            f"{prefix}_det_conf": 0.0,
            f"{prefix}_cls_conf": 0.0,
        }
    return {
        f"{prefix}_present": True,
        f"{prefix}_x1": box.x1,
        f"{prefix}_y1": box.y1,
        f"{prefix}_x2": box.x2,
        f"{prefix}_y2": box.y2,
        f"{prefix}_det_conf": box.det_conf,
        f"{prefix}_cls_conf": box.cls_conf,
    }


def detections_to_df(frame_dets: list[FrameDetection]) -> pd.DataFrame:
    """Convert resolved per-frame detections into the typed, dense detections table."""
    rows: list[dict[str, object]] = []
    for fd in frame_dets:
        row: dict[str, object] = {"frame": fd.frame, "n_candidates": fd.n_candidates}
        row.update(_box_cols("red", fd.red))
        row.update(_box_cols("blue", fd.blue))
        rows.append(row)
    df = pd.DataFrame(rows, columns=list(DETECTION_SCHEMA))
    return df.astype(DETECTION_SCHEMA)
