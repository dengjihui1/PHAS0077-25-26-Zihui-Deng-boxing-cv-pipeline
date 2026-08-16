"""Evaluate activity-threshold decoding for the retained Stage 5 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# train_fighter_matched.py lives beside this script; import it directly.
SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

from train_fighter_matched import (  # noqa: E402
    TEST_BOUTS,
    VAL_BOUTS,
    FighterQueryModel,
    build_matched_targets,
    f1_metrics,
    typed_event_metrics,
)


def load_npz(path: Path) -> dict:
    handle = np.load(path, allow_pickle=False)
    return {key: handle[key] for key in handle.files}


@torch.no_grad()
def collect_all_logits(model, data, batch_size, device):
    model.eval()
    result = np.zeros((len(data["features"]), 2, 5), dtype=np.float32)
    for start in range(0, len(result), batch_size):
        end = min(len(result), start + batch_size)
        features = torch.from_numpy(data["features"][start:end].astype(np.float32)).to(device)
        mask = torch.from_numpy(data["view_mask"][start:end].copy()).to(device)
        result[start:end] = model(features, mask).cpu().numpy()
    return result


@torch.no_grad()
def softmax(value, axis=-1):
    shifted = value - value.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True).clip(min=1e-8)


def metrics_from_prediction(prediction, source, classes, data, pipeline_config, radius):
    return {
        "proposal": f1_metrics(classes[source], prediction),
        "typed_event": typed_event_metrics(prediction, source, data, pipeline_config, radius),
    }


def score(row):
    proposal = row["proposal"]
    return proposal["macro_f1"] + 0.25 * proposal["micro_f1"] + 0.10 * proposal["null_f1"]


def decode_activity(logits: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    pred = np.zeros(logits.shape[:2], dtype=np.int64)
    nonnull_logits = logits[:, :, 1:]
    best_nonnull = nonnull_logits.argmax(axis=-1) + 1
    if mode == "prob":
        probs = softmax(logits, axis=-1)
        activity_score = 1.0 - probs[:, :, 0]
        active = activity_score >= threshold
    elif mode == "prob_margin":
        probs = softmax(logits, axis=-1)
        activity_score = probs[:, :, 1:].max(axis=-1) - probs[:, :, 0]
        active = activity_score >= threshold
    elif mode == "logit_margin":
        activity_score = nonnull_logits.max(axis=-1) - logits[:, :, 0]
        active = activity_score >= threshold
    else:
        raise ValueError(mode)
    pred[active] = best_nonnull[active]
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-features", type=Path, required=True)
    parser.add_argument("--base-ckpt", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--match-radius", type=int, default=6)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_npz(args.panel_features)
    classes, target_audit = build_matched_targets(data, args.pipeline_config, args.match_radius)
    usable = data["view_mask"].any(axis=1)
    bouts = data["bouts"].astype(int)
    val_source = np.flatnonzero(usable & np.isin(bouts, list(VAL_BOUTS)))
    test_source = np.flatnonzero(usable & np.isin(bouts, list(TEST_BOUTS)))

    model = FighterQueryModel(int(data["features"].shape[-1]), 256, 0.25).to(device)
    model.load_state_dict(torch.load(args.base_ckpt, map_location=device))
    all_logits = collect_all_logits(model, data, args.batch_size, device)

    candidates = {}
    best = None
    grids = {
        "argmax": [0.0],
        "prob": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70],
        "prob_margin": [-0.60, -0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20],
        "logit_margin": [-2.0, -1.5, -1.0, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.0],
    }
    for mode, thresholds in grids.items():
        for threshold in thresholds:
            if mode == "argmax":
                prediction = all_logits[val_source].argmax(axis=-1)
            else:
                prediction = decode_activity(all_logits[val_source], mode, threshold)
            row = metrics_from_prediction(prediction, val_source, classes, data, args.pipeline_config, args.match_radius)
            key = f"{mode}:{threshold:.2f}"
            candidates[key] = {"mode": mode, "threshold": threshold, "score": score(row), "metrics": row}
            if best is None or candidates[key]["score"] > best["score"]:
                best = candidates[key]

    base_prediction = all_logits[test_source].argmax(axis=-1)
    if best["mode"] == "argmax":
        selected_prediction = base_prediction
    else:
        selected_prediction = decode_activity(all_logits[test_source], best["mode"], best["threshold"])
    base_test = metrics_from_prediction(base_prediction, test_source, classes, data, args.pipeline_config, args.match_radius)
    fused_test = metrics_from_prediction(selected_prediction, test_source, classes, data, args.pipeline_config, args.match_radius)
    result = {
        "method": "activity_threshold_decoding",
        "device": device,
        "selected_mode": best["mode"],
        "selected_threshold": best["threshold"],
        "selected_val": best["metrics"],
        "base_test": base_test,
        "fused_test": fused_test,
        "beats_base_typed_f1": fused_test["typed_event"]["f1"] > base_test["typed_event"]["f1"],
        "beats_base_typed_macro_f1": fused_test["typed_event"]["macro_f1"] > base_test["typed_event"]["macro_f1"],
        "candidate_scores": {key: value["score"] for key, value in candidates.items()},
        "target_audit": target_audit,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("selected_mode", "selected_threshold", "base_test", "fused_test")}, indent=2))


if __name__ == "__main__":
    main()
