"""Backend B — our own fine-tuned 2-class (red/blue) detector (PRIMARY GOAL, stubbed).

Implements the same ``FighterDetector`` interface as the chain. The weights are not on
this machine yet (see ``scripts/recover_melik_detector.md``); until they are, this raises
``MissingWeightsError``. Flipping to it is a one-line config change, and its candidates
still go through the same ``select.py`` resolver (a 2-class detector can emit multiple
reds/blues or miss one — slotting is not assumed free).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .interface import Candidate, FighterDetector, FrameDetection
from .select import select_fighters


class MissingWeightsError(RuntimeError):
    """Backend B weights have not been trained/recovered yet."""


class FinetunedDetector(FighterDetector):
    def __init__(
        self,
        weights: str,
        *,
        imgsz: int = 512,
        min_cls_conf: float = 0.25,
        device: int | str = 0,
        half: bool = True,
        expected_names: dict[int, str] | None = None,
    ) -> None:
        if not weights or not Path(weights).is_file():
            raise MissingWeightsError(
                f"Backend B weights not found at {weights!r}. "
                "Recover/train them first — see scripts/recover_melik_detector.md."
            )
        from ultralytics import YOLO  # lazy

        self.model = YOLO(weights)
        # Polarity guard: red must be class 0, blue class 1 (see test_class_polarity).
        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        expected = expected_names or {0: "red", 1: "blue"}
        if names != expected:
            raise MissingWeightsError(
                f"Detector class polarity mismatch: model.names={names} expected={expected}. "
                "Refusing to run to avoid a silent red/blue swap."
            )
        self.imgsz = imgsz
        self.min_cls_conf = min_cls_conf
        self.device = device
        self.half = half
        self._wh: tuple[int, int] = (0, 0)

    @property
    def native_resolution(self) -> tuple[int, int]:
        return self._wh

    def detect(self, frame: np.ndarray, frame_idx: int) -> FrameDetection:
        h, w = frame.shape[:2]
        self._wh = (w, h)
        res = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.min_cls_conf,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]
        names = {int(k): str(v).lower() for k, v in res.names.items()}
        candidates: list[Candidate] = []
        if res.boxes is not None:
            for b in res.boxes:  # type: ignore[attr-defined]  # ultralytics Boxes is iterable
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                color = names[int(b.cls[0])]
                candidates.append(
                    Candidate(bbox=(x1, y1, x2, y2), det_conf=conf, cls_conf={color: conf})
                )
        red, blue = select_fighters(candidates, min_cls_conf=self.min_cls_conf)
        return FrameDetection(frame=frame_idx, red=red, blue=blue, n_candidates=len(candidates))
