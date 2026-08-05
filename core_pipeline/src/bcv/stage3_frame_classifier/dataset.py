"""Centered-window dataset for the per-frame punch classifier.

A sample is a window of ``N = 2k+1`` crop frames centered on a target frame; its label
is the target (middle) frame's punch/no-punch, derived as *span membership* of the bout's
annotation runs (optionally dilated by ``k_pad``). Sliding stride-1 over the clip yields
one labelled sample per frame. Frames are returned as ``(C, T, H, W)`` float tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..common.annotations import frame_binary_target
from ..common.contracts import StrikeRun
from ..common.io import read_meta, validate_meta


class CroppedWindowDataset(Dataset):
    """Windows of crop frames with center-frame punch labels.

    Construct directly from arrays (tests) or via :meth:`from_artifact` (real Stage-2 output).
    """

    def __init__(
        self,
        frames: np.ndarray,  # (M, H, W, C) uint8/float
        labels: np.ndarray,  # (M,) int/bool
        valid: np.ndarray,  # (M,) bool — crop_valid from manifest
        frame_idx: np.ndarray,  # (M,) absolute frame indices
        *,
        k: int = 2,
        exclude_carried_forward: bool = False,
    ) -> None:
        self.frames = frames
        self.labels = np.asarray(labels, dtype=np.float32)
        self.valid = np.asarray(valid, dtype=bool)
        self.frame_idx = np.asarray(frame_idx, dtype=np.int64)
        self.k = k
        self.m = len(frames)
        usable = np.arange(self.m)
        if exclude_carried_forward:
            usable = usable[self.valid]
        self._index = usable

    @classmethod
    def from_artifact(
        cls,
        stage2_dir: str | Path,
        runs: list[StrikeRun],
        *,
        k: int = 2,
        k_pad: int = 1,
        exclude_carried_forward: bool = False,
        source_video: str | None = None,
        img_size: int | None = None,
    ) -> CroppedWindowDataset:
        import cv2

        stage2_dir = Path(stage2_dir)
        manifest = pd.read_parquet(stage2_dir / "crop_manifest.parquet")
        meta = read_meta(stage2_dir)
        if source_video is not None:
            validate_meta(meta, source_video=source_video, num_frames=len(manifest))

        cap = cv2.VideoCapture(str(stage2_dir / "crop.mp4"))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if img_size and (fr.shape[0] != img_size or fr.shape[1] != img_size):
                fr = cv2.resize(fr, (img_size, img_size), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        frames_arr = np.asarray(frames[: len(manifest)])

        frame_idx = manifest["frame"].to_numpy()
        target = frame_binary_target(runs, int(frame_idx.max()) + 1, dilate=k_pad)
        labels = target[frame_idx]
        valid = manifest["crop_valid"].to_numpy()
        return cls(
            frames_arr,
            labels,
            valid,
            frame_idx,
            k=k,
            exclude_carried_forward=exclude_carried_forward,
        )

    def __len__(self) -> int:
        return len(self._index)

    def subrange(self, lo_frac: float, hi_frac: float) -> CroppedWindowDataset:
        """A temporal slice over center frames in ``[lo_frac, hi_frac)`` sharing the arrays.

        Used for a within-fight split (train on the first half, eval on the second) without
        copying the multi-GB frame array.
        """
        import copy

        lo, hi = int(lo_frac * self.m), int(hi_frac * self.m)
        new = copy.copy(self)
        new._index = self._index[(self._index >= lo) & (self._index < hi)]
        return new

    def restrict_to_rounds(self, rounds: list[tuple[int, int]]) -> int:
        """Drop center frames outside any round span (exclude between-round rest).

        ``rounds`` are absolute-frame ``[start, end]`` (inclusive). Filters ``_index`` in
        place; returns how many usable frames were dropped. No-op if ``rounds`` is empty.
        """
        if not rounds:
            return 0
        fi = self.frame_idx[self._index]
        keep = np.zeros(len(fi), dtype=bool)
        for lo, hi in rounds:
            keep |= (fi >= lo) & (fi <= hi)
        dropped = int((~keep).sum())
        self._index = self._index[keep]
        return dropped

    def positive_fraction(self) -> float:
        lab = self.labels[self._index]
        return float(lab.mean()) if len(lab) else 0.0

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        center = int(self._index[i])
        # window with edge replication
        idxs = np.clip(np.arange(center - self.k, center + self.k + 1), 0, self.m - 1)
        clip_np = self.frames[idxs].astype(np.float32) / 255.0  # (T, H, W, C)
        clip = torch.from_numpy(clip_np).permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)
        return {
            "clip": clip,
            "label": torch.tensor(self.labels[center], dtype=torch.float32),
            "valid": torch.tensor(bool(self.valid[center])),
            "frame": torch.tensor(int(self.frame_idx[center]), dtype=torch.int64),
        }
