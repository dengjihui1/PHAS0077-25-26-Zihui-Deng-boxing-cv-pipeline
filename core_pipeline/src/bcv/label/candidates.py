"""Raw single-frame person detections for the labeller — the UNFILTERED YOLO boxes.

The chain misses fighters not because YOLO fails (it produces a person box in ~100% of
frames) but because the red/blue classifier + 0.5 gate drops them. Surfacing the raw person
boxes lets the labeller CLICK an existing box to assign red/blue instead of redrawing it —
recovering exactly the detections the gate threw away. Loads only the YOLO detector (no
classifier, no tracker) and runs per-frame on demand.
"""
from __future__ import annotations

import numpy as np


class CandidateDetector:
    def __init__(self, weights: str, *, imgsz: int = 512, conf: float = 0.25,
                 device: int | str = 0, half: bool = True) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device, self.half = imgsz, conf, device, half

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """Return raw person boxes ``[x1, y1, x2, y2, conf]`` (lower conf gate than the chain)."""
        res = self.model.predict(
            frame, classes=[0], conf=self.conf, imgsz=self.imgsz,
            device=self.device, half=self.half, verbose=False,
        )[0]
        out: list[list[float]] = []
        if res.boxes is not None:
            for b in res.boxes:  # type: ignore[attr-defined]
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                c = float(b.conf[0]) if b.conf is not None else 0.0
                out.append([round(x1), round(y1), round(x2), round(y2), round(c, 3)])
        return out
