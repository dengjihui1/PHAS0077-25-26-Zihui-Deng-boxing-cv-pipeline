from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from bcv.common.contracts import FRAME_PROBS_SCHEMA
from bcv.stage3_frame_classifier.datamodule import PunchDataModule
from bcv.stage3_frame_classifier.dataset import CroppedWindowDataset
from bcv.stage3_frame_classifier.infer import build_frame_probs, predict_logits
from bcv.stage3_frame_classifier.model import WindowPunchModule
from bcv.stage3_frame_classifier.run import Stage3Config, resolve_pos_weight


def _ds(m: int = 20, k: int = 2) -> CroppedWindowDataset:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (m, 8, 8, 3), dtype=np.uint8)
    labels = np.zeros(m, dtype=np.int8)
    labels[5:9] = 1
    valid = np.ones(m, dtype=bool)
    frame_idx = np.arange(100, 100 + m)
    return CroppedWindowDataset(frames, labels, valid, frame_idx, k=k)


def test_dataset_item_shape_and_label():
    ds = _ds(k=2)
    s = ds[6]
    assert s["clip"].shape == (3, 5, 8, 8)  # (C, T=2k+1, H, W)
    assert s["label"].item() == 1.0
    assert s["frame"].item() == 106


def test_window_edge_replication():
    ds = _ds(k=2)
    clip = ds[0]["clip"]  # center 0 -> [-2,-1,0,1,2] clipped to [0,0,0,1,2]
    assert torch.equal(clip[:, 0], clip[:, 1])  # frame 0 replicated at the left edge


def test_model_forward_loss_and_backward():
    ds = _ds()
    module = WindowPunchModule(backbone="small3d", k=2, pos_weight=3.0)
    clip = torch.stack([ds[i]["clip"] for i in range(4)])
    label = torch.tensor([0.0, 1.0, 0.0, 1.0])
    logits = module(clip)
    assert logits.shape == (4,)
    loss = module._loss(logits, label)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()  # grads flow end-to-end


def test_channel_stack_backbone():
    module = WindowPunchModule(backbone="channel_stack_2d", k=1)
    clip = torch.rand(2, 3, 3, 8, 8)  # T = 2k+1 = 3
    assert module(clip).shape == (2,)


def test_resolve_pos_weight_balanced_avoids_double_weighting():
    # _ds() is 4 positives / 16 negatives -> neg/pos = 4.0
    train = [_ds()]
    # balanced sampler ON: "auto" must NOT also up-weight (would double-count) -> 1.0
    assert resolve_pos_weight(Stage3Config(pos_weight="auto", balanced=True), train) == 1.0
    # balanced OFF: "auto" falls back to neg/pos
    assert resolve_pos_weight(Stage3Config(pos_weight="auto", balanced=False), train) == 4.0
    # an explicit float is always honoured verbatim, regardless of the sampler
    assert resolve_pos_weight(Stage3Config(pos_weight=2.5, balanced=True), train) == 2.5


def test_datamodule_sampler_toggle():
    ds = _ds()
    on = PunchDataModule([ds], [ds], balanced=True).train_dataloader()
    assert isinstance(on.sampler, WeightedRandomSampler)  # balanced -> weighted sampling
    off = PunchDataModule([ds], [ds], balanced=False).train_dataloader()
    assert not isinstance(off.sampler, WeightedRandomSampler)  # plain shuffle otherwise


def test_calibration_off_by_default():
    # predict-time temperature scaling is leaky on the target's own labels -> default off
    assert Stage3Config().calibration == "none"


def test_infer_shapes_and_schema():
    ds = _ds(m=15, k=2)
    module = WindowPunchModule(k=2)
    logits = predict_logits(module, ds, device="cpu", batch_size=4)
    assert logits.shape == (15,)
    df = build_frame_probs(logits, ds, temperature=1.5, roll_w=5)
    assert list(df.columns) == list(FRAME_PROBS_SCHEMA)
    assert df["frame"].tolist() == list(range(100, 115))
    assert (df["p_punch"] >= 0).all() and (df["p_punch"] <= 1).all()
    assert df["coverage"].iloc[7] == 5  # interior frame sees full window
