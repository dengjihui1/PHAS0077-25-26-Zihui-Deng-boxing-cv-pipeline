"""Stride-1 inference: score every frame's centered window -> frame_probs.parquet."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..common.contracts import FRAME_PROBS_SCHEMA
from .dataset import CroppedWindowDataset


def _window(frames: np.ndarray, center: int, k: int) -> np.ndarray:
    m = len(frames)
    idxs = np.clip(np.arange(center - k, center + k + 1), 0, m - 1)
    clip = frames[idxs].astype(np.float32) / 255.0  # (T,H,W,C)
    return clip.transpose(3, 0, 1, 2)  # (C,T,H,W)


@torch.no_grad()
def predict_logits(
    module: torch.nn.Module,
    dataset: CroppedWindowDataset,
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Return a logit for every frame (window centered on it), in frame order."""
    module.eval().to(device)
    frames, k, m = dataset.frames, dataset.k, dataset.m
    out = np.empty(m, dtype=np.float32)
    for start in range(0, m, batch_size):
        centers = range(start, min(start + batch_size, m))
        batch = torch.from_numpy(np.stack([_window(frames, c, k) for c in centers])).to(device)
        out[start : start + len(batch)] = module(batch).float().cpu().numpy().reshape(-1)
    return out


def build_frame_probs(
    logits: np.ndarray,
    dataset: CroppedWindowDataset,
    *,
    temperature: float = 1.0,
    roll_w: int = 11,
) -> pd.DataFrame:
    """Assemble the typed frame_probs table from per-frame logits."""
    p_raw = 1.0 / (1.0 + np.exp(-logits))
    p_punch = 1.0 / (1.0 + np.exp(-logits / temperature))
    m, k = dataset.m, dataset.k
    coverage = np.array(
        [len(set(np.clip(range(c - k, c + k + 1), 0, m - 1))) for c in range(m)], dtype=np.int16
    )
    p_smooth = pd.Series(p_punch).rolling(roll_w, center=True, min_periods=1).mean().to_numpy()
    df = pd.DataFrame(
        {
            "frame": dataset.frame_idx.astype(np.int32),
            "p_punch": p_punch.astype(np.float32),
            "p_raw": p_raw.astype(np.float32),
            "coverage": coverage,
            "crop_valid": dataset.valid.astype(bool),
            "p_smooth": p_smooth.astype(np.float32),
            "label": dataset.labels.astype(np.int8),
        },
        columns=list(FRAME_PROBS_SCHEMA),
    )
    return df.astype(FRAME_PROBS_SCHEMA)
