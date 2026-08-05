"""WindowPunchModule — a small temporal CNN that scores the center frame of a window.

Default backbone is a compact 3D CNN over the ``(C, T, H, W)`` clip; ``channel_stack_2d``
collapses time into channels for a 2D CNN, and ``k=0`` reduces the whole thing to a
single-frame baseline. Trained with ``BCEWithLogits(pos_weight)`` (the data is ~15:1
negative); headline metrics are PR-AUC / AUROC, never raw accuracy.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


class _ChannelStack2D(nn.Module):
    """Collapse (B,C,T,H,W) -> (B,C*T,H,W) and run a 2D CNN."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        return self.net(x.reshape(b, c * t, h, w))


def _small3d() -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(3, 16, 3, padding=1),
        nn.BatchNorm3d(16),
        nn.ReLU(),
        nn.MaxPool3d((1, 2, 2)),
        nn.Conv3d(16, 32, 3, padding=1),
        nn.BatchNorm3d(32),
        nn.ReLU(),
        nn.MaxPool3d((2, 2, 2)),
        nn.Conv3d(32, 64, 3, padding=1),
        nn.BatchNorm3d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
    )


def build_backbone(name: str, window_len: int) -> tuple[nn.Module, int]:
    if name == "small3d":
        return _small3d(), 64
    if name == "channel_stack_2d":
        return _ChannelStack2D(3 * window_len), 64
    raise ValueError(f"unknown backbone {name!r} (use 'small3d' or 'channel_stack_2d')")


class WindowPunchModule(L.LightningModule):
    pos_weight: torch.Tensor  # registered buffer; annotated for the type checker

    def __init__(
        self,
        *,
        backbone: str = "small3d",
        k: int = 2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
        focal_gamma: float = 0.0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        window_len = 2 * k + 1
        self.net, feat = build_backbone(backbone, window_len)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat, 1)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.val_ap = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.net(clip))).squeeze(-1)

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.hparams["focal_gamma"] > 0:
            p = torch.sigmoid(logits)
            pt = torch.where(target > 0.5, p, 1 - p)
            mod = (1 - pt).clamp_min(1e-6) ** self.hparams["focal_gamma"]
            bce = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=self.pos_weight, reduction="none"
            )
            return (mod * bce).mean()
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)

    def training_step(self, batch: dict, _idx: int) -> torch.Tensor:
        logits = self(batch["clip"])
        loss = self._loss(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True, batch_size=len(logits))
        return loss

    def validation_step(self, batch: dict, _idx: int) -> None:
        logits = self(batch["clip"])
        loss = self._loss(logits, batch["label"])
        probs = torch.sigmoid(logits)
        tgt = batch["label"].int()
        self.val_ap.update(probs, tgt)
        self.val_auroc.update(probs, tgt)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(logits))

    def on_validation_epoch_end(self) -> None:
        # guard: metrics need both classes present
        try:
            self.log("val_ap", self.val_ap.compute(), prog_bar=True)
            self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        except (ValueError, RuntimeError):
            pass
        self.val_ap.reset()
        self.val_auroc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]
        )
