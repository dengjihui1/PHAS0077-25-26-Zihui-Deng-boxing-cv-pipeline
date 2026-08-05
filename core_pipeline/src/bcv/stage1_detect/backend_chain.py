"""Backend A — the detect→classify chain (ships first; all weights on disk).

Stock COCO person detector finds people, each person crop is classified red/blue by
Melik's fine-tuned classifier, and ``select.py`` resolves the two fighter slots. Pose
is dropped (vestigial for punch detection). ultralytics/torch are imported lazily so
the pure modules (``select``, ``interface``) stay importable without the heavy deps.
"""

from __future__ import annotations

import numpy as np

from ..common.geometry import crop_person
from .classify import TrackClassMemory
from .interface import Box, Candidate, FighterDetector, FrameDetection
from .select import select_fighters


class ChainDetector(FighterDetector):
    def __init__(
        self,
        detector_weights: str,
        classifier_weights: str,
        *,
        det_imgsz: int = 512,
        cls_imgsz: int = 224,
        max_det: int = 5,
        iou: float = 0.95,
        crop_pad: float = 0.05,
        crop_min_x: int = 25,
        crop_min_y: int = 50,
        min_cls_conf: float = 0.5,
        box_hold_frames: int = 15,
        tracker: str = "botsort.yaml",
        device: int | str = 0,
        half: bool = True,
    ) -> None:
        from ultralytics import YOLO  # lazy: keeps select/interface torch-free

        self.det = YOLO(detector_weights)
        self.cls = YOLO(classifier_weights)
        self.det_imgsz = det_imgsz
        self.cls_imgsz = cls_imgsz
        self.max_det = max_det
        self.iou = iou
        self.crop_pad = crop_pad
        self.crop_min_x = crop_min_x
        self.crop_min_y = crop_min_y
        self.min_cls_conf = min_cls_conf
        self.tracker = tracker
        self.device = device
        self.half = half
        self._wh: tuple[int, int] = (0, 0)
        self._mem = TrackClassMemory(min_cls_conf=min_cls_conf, hold_frames=box_hold_frames)

    @property
    def native_resolution(self) -> tuple[int, int]:
        return self._wh

    def reset(self) -> None:
        # Drop tracker state + held labels so track IDs restart cleanly on a new video.
        if hasattr(self.det, "predictor") and self.det.predictor is not None:
            self.det.predictor.trackers = None
        self._mem.reset()

    def _classify_crops(self, crops: list[np.ndarray]) -> list[dict[str, float]]:
        """Classify every candidate crop in ONE batched forward pass.

        The per-frame classifier was the hot path: 3+ separate ``self.cls(crop)`` calls
        per frame, each a tiny GPU launch that left the device ~23% utilised (launch-
        overhead-bound). Batching all of a frame's crops into a single call cuts the
        launch count to one. Ultralytics returns results in input order, so we map each
        result back to its crop's index unchanged — preserving the exact candidate
        ordering that ``select_fighters``/``TrackClassMemory`` depend on.
        """
        if not crops:
            return []
        results = self.cls(crops, imgsz=self.cls_imgsz, verbose=False, device=self.device)
        out: list[dict[str, float]] = []
        for res in results:  # results[i] corresponds to crops[i] (input order preserved)
            names = res.names
            probs = res.probs.data.tolist()
            out.append({str(names[i]).lower(): float(p) for i, p in enumerate(probs)})
        return out

    def detect(self, frame: np.ndarray, frame_idx: int) -> FrameDetection:
        h, w = frame.shape[:2]
        self._wh = (w, h)
        res = self.det.track(
            frame,
            persist=True,
            classes=[0],  # COCO person only
            conf=0.0,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.det_imgsz,
            tracker=self.tracker,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        # Pass 1: collect every valid candidate crop (in detector/track order) so the
        # classifier can run once over the whole batch instead of once per crop.
        pending: list[tuple[tuple[int, int, int, int], float, int | None]] = []
        crops: list[np.ndarray] = []
        if res.boxes is not None:
            for b in res.boxes:  # type: ignore[attr-defined]  # ultralytics Boxes is iterable
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                det_conf = float(b.conf[0]) if b.conf is not None else 0.0
                track_id = int(b.id[0]) if b.id is not None else None
                crop = crop_person(
                    frame,
                    (x1, y1, x2, y2),
                    pad_frac=self.crop_pad,
                    min_w=self.crop_min_x,
                    min_h=self.crop_min_y,
                )
                if crop is None:
                    continue
                pending.append(((x1, y1, x2, y2), det_conf, track_id))
                crops.append(crop)

        # One batched classifier call (skipped entirely when there are no crops).
        cls_confs = self._classify_crops(crops)

        # Pass 2: resolve track memory in the ORIGINAL per-candidate order — resolve()
        # mutates per-track state, so its call order must match the single-call path.
        candidates: list[Candidate] = []
        for (bbox, det_conf, track_id), cls_conf in zip(pending, cls_confs, strict=True):
            # Hold a track's recent confident red/blue label across short dips.
            cls_conf = self._mem.resolve(track_id, cls_conf, frame_idx)
            candidates.append(
                Candidate(
                    bbox=bbox,
                    det_conf=det_conf,
                    cls_conf=cls_conf,
                    track_id=track_id,
                )
            )

        red, blue = select_fighters(candidates, min_cls_conf=self.min_cls_conf)
        return FrameDetection(frame=frame_idx, red=red, blue=blue, n_candidates=len(candidates))


# expose the resolved-Box type for typing convenience
__all__ = ["Box", "ChainDetector"]
