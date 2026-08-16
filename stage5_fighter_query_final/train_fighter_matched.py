"""Train Stage-5 models with one-to-one, per-fighter event targets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bcv.common.annotations import load_runs
from bcv.common.config import load_pipeline_config
from bcv.eval.window import match_events

TRAIN_BOUTS = {116, 117, 120, 121}
VAL_BOUTS = {122}
TEST_BOUTS = {115}
FIGHTERS = ("blue", "red")
CLASSES = ("null", "body_landed", "head_landed", "strike_blocked", "strike_missed")
LABELS = tuple(f"{fighter}_{label}" for fighter in FIGHTERS for label in CLASSES[1:])


class EventDataset(Dataset):
    def __init__(self, data: dict, classes: np.ndarray, indices: np.ndarray, train: bool, view_dropout: float) -> None:
        self.data = data
        self.classes = classes
        self.indices = indices
        self.train = train
        self.view_dropout = view_dropout

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        source = int(self.indices[item])
        features = torch.from_numpy(self.data["features"][source].astype(np.float32))
        mask = torch.from_numpy(self.data["view_mask"][source].copy())
        if self.train and mask.sum() > 1 and self.view_dropout > 0:
            drop = (torch.rand(len(mask)) < self.view_dropout) & mask
            if int((mask & ~drop).sum()) == 0:
                drop[torch.where(mask)[0][torch.randint(int(mask.sum()), (1,))]] = False
            mask &= ~drop
        return features, mask, torch.from_numpy(self.classes[source]), source


class MeanModel(nn.Module):
    """View-averaged baseline with a shared two-slot head.

    The 5-D path reads the two slots from the panel features' fighter axis; the
    4-D path adds a learned per-fighter offset instead.
    """

    def __init__(self, hidden: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim), nn.GELU())
        self.fighter = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.context = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.shared_head = nn.Linear(dim, len(CLASSES))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim == 5:
            per_view = self.proj(features.mean(dim=3))
            weights = mask.float().unsqueeze(-1).unsqueeze(-1)
            fighters = (per_view * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            per_view = self.proj(features.mean(dim=2))
            weights = mask.float().unsqueeze(-1)
            event = (per_view * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            fighters = event.unsqueeze(1) + self.fighter.unsqueeze(0)
        blue_raw, red_raw = fighters[:, 0], fighters[:, 1]
        blue = self.context(torch.cat([blue_raw, red_raw, blue_raw - red_raw], dim=1))
        red = self.context(torch.cat([red_raw, blue_raw, red_raw - blue_raw], dim=1))
        return torch.stack([self.shared_head(blue), self.shared_head(red)], dim=1)


class FighterQueryModel(nn.Module):
    """Temporal query per view, masked cross-view attention, one query per fighter.

    The two slots share one context transform and one 5-way head; each slot sees
    its own feature, the opponent's feature, and their difference.
    """

    def __init__(self, hidden: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim), nn.GELU())
        self.temporal_pos = nn.Parameter(torch.zeros(1, 1, 8, dim))
        self.temporal_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.fighter_queries = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        self.temporal_attention = nn.MultiheadAttention(dim, 4, dropout=dropout, batch_first=True)
        self.view_attention = nn.MultiheadAttention(dim, 4, dropout=dropout, batch_first=True)
        self.context = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.shared_head = nn.Linear(dim, len(CLASSES))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, views = features.shape[:2]
        if features.ndim == 5:
            fighters, temporal = features.shape[2:4]
            if temporal != self.temporal_pos.shape[2]:
                raise ValueError(
                    f"temporal_pos has {self.temporal_pos.shape[2]} bins but features have {temporal}"
                )
            tokens = self.proj(features) + self.temporal_pos[:, :, None, :temporal]
            flat = tokens.reshape(batch * views * fighters, temporal, -1)
            temporal_query = self.temporal_query.expand(batch * views * fighters, -1, -1)
            per_view, _ = self.temporal_attention(temporal_query, flat, flat, need_weights=False)
            per_view = (
                per_view.reshape(batch, views, fighters, -1)
                .permute(0, 2, 1, 3)
                .reshape(batch * fighters, views, -1)
            )
            fighter_query = self.fighter_queries.expand(batch, -1, -1).reshape(batch * fighters, 1, -1)
            expanded_mask = mask[:, None, :].expand(-1, fighters, -1).reshape(batch * fighters, views)
            fighter_features, _ = self.view_attention(
                fighter_query,
                per_view,
                per_view,
                key_padding_mask=~expanded_mask,
                need_weights=False,
            )
            fighter_features = fighter_features.reshape(batch, fighters, -1)
        else:
            temporal = features.shape[2]
            if temporal != self.temporal_pos.shape[2]:
                raise ValueError(
                    f"temporal_pos has {self.temporal_pos.shape[2]} bins but features have {temporal}"
                )
            tokens = self.proj(features) + self.temporal_pos[:, :, :temporal]
            flat = tokens.reshape(batch * views, temporal, -1)
            temporal_query = self.temporal_query.expand(batch * views, -1, -1)
            per_view, _ = self.temporal_attention(temporal_query, flat, flat, need_weights=False)
            per_view = per_view.reshape(batch, views, -1)
            fighter_queries = self.fighter_queries.expand(batch, -1, -1)
            fighter_features, _ = self.view_attention(
                fighter_queries,
                per_view,
                per_view,
                key_padding_mask=~mask,
                need_weights=False,
            )
        blue_raw, red_raw = fighter_features[:, 0], fighter_features[:, 1]
        blue = self.context(torch.cat([blue_raw, red_raw, blue_raw - red_raw], dim=1))
        red = self.context(torch.cat([red_raw, blue_raw, red_raw - blue_raw], dim=1))
        return torch.stack([self.shared_head(blue), self.shared_head(red)], dim=1)


def build_matched_targets(data: dict, pipeline_config: Path, radius: int) -> tuple[np.ndarray, dict]:
    pipeline = load_pipeline_config(pipeline_config)
    classes = np.zeros((len(data["bouts"]), 2), dtype=np.int64)
    audit = {}
    for bout in sorted(set(data["bouts"].astype(int).tolist())):
        source = np.flatnonzero(data["bouts"] == bout)
        spans = [(int(data["peaks"][index]) - radius, int(data["peaks"][index]) + radius) for index in source]
        runs = load_runs(pipeline.bout_dir(bout))
        bout_audit = {"proposals": len(source), "fighters": {}}
        for fighter_index, fighter in enumerate(FIGHTERS):
            gt = [run for run in runs if run.label.startswith(f"{fighter}_")]
            matches, missed, false = match_events(spans, [(run.start_frame, run.end_frame) for run in gt])
            for gt_index, proposal_index in matches:
                suffix = gt[gt_index].label.removeprefix(f"{fighter}_")
                classes[source[proposal_index], fighter_index] = CLASSES.index(suffix)
            bout_audit["fighters"][fighter] = {
                "gt": len(gt), "matched": len(matches), "missed": len(missed), "unmatched_proposals": len(false)
            }
        audit[str(bout)] = bout_audit
    return classes, audit


def multilabel(classes: np.ndarray) -> np.ndarray:
    result = np.zeros((len(classes), len(LABELS)), dtype=bool)
    for fighter in range(2):
        active = classes[:, fighter] > 0
        rows = np.flatnonzero(active)
        result[rows, fighter * 4 + classes[rows, fighter] - 1] = True
    return result


def f1_metrics(true_classes: np.ndarray, pred_classes: np.ndarray) -> dict:
    true = multilabel(true_classes)
    pred = multilabel(pred_classes)
    tp = (true & pred).sum(axis=0)
    fp = (~true & pred).sum(axis=0)
    fn = (true & ~pred).sum(axis=0)
    per_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(LABELS), dtype=float), where=(2 * tp + fp + fn) > 0)
    micro = 2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum())
    true_null = ~true.any(axis=1)
    pred_null = ~pred.any(axis=1)
    null_tp = int((true_null & pred_null).sum())
    null_fp = int((~true_null & pred_null).sum())
    null_fn = int((true_null & ~pred_null).sum())
    null_f1 = 2 * null_tp / max(1, 2 * null_tp + null_fp + null_fn)
    return {
        "n": len(true),
        "micro_f1": float(micro),
        "macro_f1": float(per_f1.mean()),
        "per_class_f1": dict(zip(LABELS, per_f1.tolist(), strict=True)),
        "per_class_positive": dict(zip(LABELS, true.sum(axis=0).astype(int).tolist(), strict=True)),
        "exact_match": float((true_classes == pred_classes).all(axis=1).mean()),
        "fighter_accuracy": float((true_classes == pred_classes).mean()),
        "null_f1": float(null_f1),
        "mean_true_labels": float(true.sum(axis=1).mean()),
        "mean_pred_labels": float(pred.sum(axis=1).mean()),
    }


def typed_event_metrics(
    pred_classes: np.ndarray,
    source: np.ndarray,
    data: dict,
    pipeline_config: Path,
    radius: int,
) -> dict:
    pipeline = load_pipeline_config(pipeline_config)
    pred_labels = multilabel(pred_classes)
    total_gt = total_pred = total_match = 0
    per_class = {}
    for label_index, label in enumerate(LABELS):
        gt_count = pred_count = match_count = 0
        for bout in TEST_BOUTS:
            gt = [(run.start_frame, run.end_frame) for run in load_runs(pipeline.bout_dir(bout)) if run.label == label]
            pred = [
                (int(data["peaks"][item]) - radius, int(data["peaks"][item]) + radius)
                for row, item in enumerate(source)
                if int(data["bouts"][item]) == bout and pred_labels[row, label_index]
            ]
            matches, _, _ = match_events(pred, gt)
            gt_count += len(gt)
            pred_count += len(pred)
            match_count += len(matches)
        f1 = 2 * match_count / max(1, pred_count + gt_count)
        per_class[label] = {"gt": gt_count, "pred": pred_count, "matched": match_count, "f1": f1}
        total_gt += gt_count
        total_pred += pred_count
        total_match += match_count
    return {
        "gt": total_gt,
        "pred": total_pred,
        "matched": total_match,
        "precision": total_match / max(1, total_pred),
        "recall": total_match / max(1, total_gt),
        "f1": 2 * total_match / max(1, total_pred + total_gt),
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
        "per_class": per_class,
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true_rows, pred_rows, source_rows = [], [], []
    for features, mask, targets, source in loader:
        logits = model(features.to(device), mask.to(device))
        true_rows.append(targets.numpy())
        pred_rows.append(logits.argmax(dim=2).cpu().numpy())
        source_rows.append(source.numpy())
    return np.concatenate(true_rows), np.concatenate(pred_rows), np.concatenate(source_rows)


def train_model(name: str, data: dict, classes: np.ndarray, args, device: str) -> dict:
    usable = data["usable"].astype(bool)
    bouts = data["bouts"]
    split_indices = {
        "train": np.flatnonzero(usable & np.isin(bouts, list(TRAIN_BOUTS))),
        "val": np.flatnonzero(usable & np.isin(bouts, list(VAL_BOUTS))),
        "test": np.flatnonzero(usable & np.isin(bouts, list(TEST_BOUTS))),
    }
    loaders = {
        split: DataLoader(
            EventDataset(data, classes, indices, split == "train", args.view_dropout if split == "train" else 0.0),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, indices in split_indices.items()
    }
    hidden = int(data["features"].shape[-1])
    model = (MeanModel if name == "mean_categorical" else FighterQueryModel)(hidden, args.dim, args.dropout).to(device)
    train_classes = classes[split_indices["train"]]
    class_weights = []
    for fighter in range(2):
        counts = np.bincount(train_classes[:, fighter], minlength=len(CLASSES)).astype(float)
        class_weights.append(
            np.power(len(train_classes) / np.maximum(counts, 1.0), args.class_weight_power)
        )
    weights = torch.tensor(np.stack(class_weights), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for features, mask, targets, _ in loaders["train"]:
            features, mask, targets = features.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, mask)
            loss = sum(
                nn.functional.cross_entropy(
                    logits[:, fighter],
                    targets[:, fighter],
                    weight=weights[fighter],
                    label_smoothing=0.03,
                )
                for fighter in range(2)
            ) / 2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            seen += len(targets)
        val_true, val_pred, _ = predict(model, loaders["val"], device)
        val = f1_metrics(val_true, val_pred)
        selection = val["macro_f1"] + 0.25 * val["micro_f1"] + 0.10 * val["null_f1"]
        history.append({"epoch": epoch, "loss": total_loss / seen, "selection": selection, "val": val})
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{name} epoch {epoch}: val macro={val['macro_f1']:.3f} "
                f"micro={val['micro_f1']:.3f} null={val['null_f1']:.3f}",
                flush=True,
            )
        if selection > best_score:
            best_score, best_epoch = selection, epoch
            torch.save(model.state_dict(), output_dir / "best.pt")
    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))
    val_true, val_pred, _ = predict(model, loaders["val"], device)
    test_true, test_pred, test_source = predict(model, loaders["test"], device)
    result = {
        "best_epoch": best_epoch,
        "best_selection": best_score,
        "counts": {split: len(indices) for split, indices in split_indices.items()},
        "class_counts_train": {
            fighter: dict(
                zip(
                    CLASSES,
                    np.bincount(train_classes[:, index], minlength=len(CLASSES))
                    .astype(int)
                    .tolist(),
                    strict=True,
                )
            )
            for index, fighter in enumerate(FIGHTERS)
        },
        "val": f1_metrics(val_true, val_pred),
        "test": f1_metrics(test_true, test_pred),
        "typed_event_test": typed_event_metrics(test_pred, test_source, data, args.pipeline_config, args.match_radius),
        "history": history,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-radius", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--view-dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = np.load(args.features, allow_pickle=False)
    data = {key: source[key] for key in source.files}
    classes, audit = build_matched_targets(data, args.pipeline_config, args.match_radius)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "target_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    results = {
        name: train_model(name, data, classes, args, device)
        for name in ("mean_categorical", "fighter_query_categorical")
    }
    comparison = {
        name: {
            "best_epoch": row["best_epoch"],
            "val_macro_f1": row["val"]["macro_f1"],
            "test_micro_f1": row["test"]["micro_f1"],
            "test_macro_f1": row["test"]["macro_f1"],
            "test_exact_match": row["test"]["exact_match"],
            "test_null_f1": row["test"]["null_f1"],
            "typed_event_f1": row["typed_event_test"]["f1"],
            "typed_event_macro_f1": row["typed_event_test"]["macro_f1"],
        }
        for name, row in results.items()
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
