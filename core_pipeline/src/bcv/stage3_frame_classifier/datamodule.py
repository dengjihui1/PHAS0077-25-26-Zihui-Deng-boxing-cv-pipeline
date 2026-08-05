"""LightningDataModule wrapping the windowed crop datasets, with balanced sampling.

The per-frame target is ~15:1 negative, so the train loader uses a WeightedRandomSampler
that up-weights positives rather than relying on ``pos_weight`` alone.
"""

from __future__ import annotations

import lightning as L
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .dataset import CroppedWindowDataset


def _sample_labels(ds: CroppedWindowDataset) -> np.ndarray:
    return ds.labels[ds._index]


class PunchDataModule(L.LightningDataModule):
    def __init__(
        self,
        train: list[CroppedWindowDataset],
        val: list[CroppedWindowDataset],
        *,
        batch_size: int = 16,
        num_workers: int = 0,
        balanced: bool = True,
    ) -> None:
        super().__init__()
        self.train_sets = train
        self.val_sets = val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balanced = balanced

    def _sampler(self, concat: ConcatDataset) -> WeightedRandomSampler | None:
        if not self.balanced:
            return None
        labels = np.concatenate([_sample_labels(ds) for ds in self.train_sets])
        pos = labels.sum()
        neg = len(labels) - pos
        if pos == 0 or neg == 0:
            return None
        w_pos, w_neg = 0.5 / pos, 0.5 / neg
        weights = np.where(labels > 0.5, w_pos, w_neg).astype(float).tolist()
        return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

    def train_dataloader(self) -> DataLoader:
        concat: ConcatDataset = ConcatDataset(self.train_sets)
        sampler = self._sampler(concat)
        return DataLoader(
            concat,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            ConcatDataset(self.val_sets),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
