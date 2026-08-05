"""Build synchronized multi-view VideoMAE features around Stage-4 consensus peaks."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, VideoMAEModel

from bcv.common.annotations import load_runs
from bcv.common.config import load_pipeline_config

LABELS = (
    "blue_body_landed",
    "blue_head_landed",
    "blue_strike_blocked",
    "blue_strike_missed",
    "red_body_landed",
    "red_head_landed",
    "red_strike_blocked",
    "red_strike_missed",
)
FIGHTERS = ("red", "blue")


def load_pretrained_encoder(checkpoint: str, device: str) -> tuple[VideoMAEModel, VideoMAEImageProcessor]:
    """Load VideoMAE while repairing the legacy q/v-bias checkpoint mapping."""
    wrapper, loading = VideoMAEForVideoClassification.from_pretrained(
        checkpoint,
        output_loading_info=True,
    )
    missing = set(loading["missing_keys"])
    unexpected = set(loading["unexpected_keys"])
    mismatched = set(loading["mismatched_keys"])
    expected_missing = {
        f"videomae.encoder.layer.{layer}.attention.attention.{projection}.bias"
        for layer in range(wrapper.config.num_hidden_layers)
        for projection in ("query", "key", "value")
    }
    expected_unexpected = {
        f"videomae.encoder.layer.{layer}.attention.attention.{projection}_bias"
        for layer in range(wrapper.config.num_hidden_layers)
        for projection in ("q", "v")
    }
    encoder_missing = {key for key in missing if key.startswith("videomae.")}
    encoder_unexpected = {key for key in unexpected if key.startswith("videomae.")}
    encoder_mismatched = {key for key in mismatched if key.startswith("videomae.")}
    if (
        encoder_missing != expected_missing
        or encoder_unexpected != expected_unexpected
        or encoder_mismatched
    ):
        raise RuntimeError(
            "unexpected VideoMAE encoder load report: "
            f"missing={sorted(encoder_missing - expected_missing)}, "
            f"unexpected={sorted(encoder_unexpected - expected_unexpected)}, "
            f"mismatched={sorted(encoder_mismatched)}"
        )

    checkpoint_path = hf_hub_download(checkpoint, "model.safetensors")
    with safe_open(checkpoint_path, framework="pt", device="cpu") as weights:
        patch_key = "videomae.embeddings.patch_embeddings.projection.weight"
        loaded_patch = wrapper.videomae.state_dict()[patch_key.removeprefix("videomae.")]
        expected_patch = weights.get_tensor(patch_key)
        if not torch.equal(loaded_patch, expected_patch):
            raise RuntimeError("pretrained patch embedding does not match the checkpoint")
        for layer in range(wrapper.config.num_hidden_layers):
            attention = wrapper.videomae.encoder.layer[layer].attention.attention
            q_bias = weights.get_tensor(
                f"videomae.encoder.layer.{layer}.attention.attention.q_bias"
            )
            v_bias = weights.get_tensor(
                f"videomae.encoder.layer.{layer}.attention.attention.v_bias"
            )
            attention.query.bias.data.copy_(q_bias)
            attention.key.bias.data.zero_()
            attention.value.bias.data.copy_(v_bias)
            if not torch.equal(attention.query.bias, q_bias):
                raise RuntimeError(f"failed to restore query bias in encoder layer {layer}")
            if not torch.equal(attention.value.bias, v_bias):
                raise RuntimeError(f"failed to restore value bias in encoder layer {layer}")

    encoder = wrapper.videomae.to(device).eval()
    print(
        f"verified pretrained VideoMAE encoder: {checkpoint} "
        f"({sum(parameter.numel() for parameter in encoder.parameters()):,} parameters)",
        flush=True,
    )
    return encoder, VideoMAEImageProcessor.from_pretrained(checkpoint)


@dataclass(frozen=True)
class BoxInterpolator:
    frames: np.ndarray
    boxes: np.ndarray
    max_gap: int

    def at(self, frame: int) -> tuple[float, float, float, float] | None:
        if len(self.frames) == 0:
            return None
        pos = int(np.searchsorted(self.frames, int(frame)))
        left = max(0, pos - 1)
        right = min(len(self.frames) - 1, pos)
        lf, rf = int(self.frames[left]), int(self.frames[right])
        lg, rg = abs(frame - lf), abs(rf - frame)
        left_ok, right_ok = lg <= self.max_gap, rg <= self.max_gap
        if not left_ok and not right_ok:
            return None
        if left_ok and right_ok and lf != rf:
            alpha = (frame - lf) / (rf - lf)
            box = (1.0 - alpha) * self.boxes[left] + alpha * self.boxes[right]
        elif left_ok and (not right_ok or lg <= rg):
            box = self.boxes[left]
        else:
            box = self.boxes[right]
        return tuple(float(value) for value in box)


def load_bbox_interpolators(path: Path, max_gap: int) -> dict[str, BoxInterpolator]:
    frames = {fighter: [] for fighter in FIGHTERS}
    boxes = {fighter: [] for fighter in FIGHTERS}
    with path.open(encoding="utf-8") as handle:
        for frame, line in enumerate(handle):
            candidates = json.loads(line) if line.strip() else []
            best: dict[str, tuple[float, list[float]]] = {}
            for candidate in candidates:
                confs = candidate.get("cls_confs", {})
                for fighter in FIGHTERS:
                    score = float(confs.get(fighter, 0.0))
                    if score >= 0.5 and (fighter not in best or score > best[fighter][0]):
                        best[fighter] = (score, candidate["bbox"])
            for fighter, (_score, box) in best.items():
                frames[fighter].append(frame)
                boxes[fighter].append(box)
    return {
        fighter: BoxInterpolator(
            np.asarray(frames[fighter], dtype=np.int64),
            np.asarray(boxes[fighter], dtype=np.float32).reshape(-1, 4),
            int(max_gap),
        )
        for fighter in FIGHTERS
    }


def square_crop_box(
    box: tuple[float, float, float, float], width: int, height: int, pad_frac: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    half = 0.5 * max(bw, bh) * (1.0 + 2.0 * pad_frac)
    side = min(max(32.0, 2.0 * half), float(width), float(height))
    half = side / 2.0
    x1, x2, y1, y2 = cx - half, cx + half, cy - half, cy + half
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 > height:
        y1 -= y2 - height
        y2 = height
    return round(x1), round(y1), round(x2), round(y2)


def proposal_indices(peak: int, unique_frames: int, video_length: int) -> np.ndarray:
    offsets = np.arange(unique_frames, dtype=np.int64) - unique_frames // 2
    return np.clip(int(peak) + offsets, 0, max(0, video_length - 1))


def make_panel(
    frames_rgb: np.ndarray,
    indices: np.ndarray,
    lookups: dict[str, BoxInterpolator],
    *,
    width: int,
    height: int,
    size: int,
    pad_frac: float,
    stripe: int,
) -> np.ndarray | None:
    panels = []
    half_width = size // 2
    for frame, frame_index in zip(frames_rgb, indices, strict=True):
        crops = {}
        for fighter in FIGHTERS:
            box = lookups[fighter].at(int(frame_index))
            if box is None:
                return None
            x1, y1, x2, y2 = square_crop_box(box, width, height, pad_frac)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            crops[fighter] = crop
        red = cv2.resize(crops["red"], (half_width, size), interpolation=cv2.INTER_AREA)
        blue = cv2.resize(crops["blue"], (size - half_width, size), interpolation=cv2.INTER_AREA)
        panel = np.concatenate([red, blue], axis=1)
        if stripe > 0:
            panel[:, : min(stripe, half_width), :] = (255, 0, 0)
            panel[:, max(half_width, size - stripe) :, :] = (0, 0, 255)
        panels.append(panel)
    return np.stack(panels)


def make_fighter_clips(
    frames_rgb: np.ndarray,
    indices: np.ndarray,
    lookups: dict[str, BoxInterpolator],
    *,
    width: int,
    height: int,
    size: int,
    pad_frac: float,
) -> np.ndarray | None:
    clips = {fighter: [] for fighter in FIGHTERS}
    for frame, frame_index in zip(frames_rgb, indices, strict=True):
        for fighter in FIGHTERS:
            box = lookups[fighter].at(int(frame_index))
            if box is None:
                return None
            x1, y1, x2, y2 = square_crop_box(box, width, height, pad_frac)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            clips[fighter].append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack([np.stack(clips[fighter]) for fighter in ("blue", "red")])


def labels_for_span(gt_runs, start: int, end: int) -> list[str]:
    return sorted({run.label for run in gt_runs if not (run.end_frame < start or run.start_frame > end)})


@torch.no_grad()
def encode_batch(
    model: VideoMAEModel,
    panels: list[np.ndarray],
    processor: VideoMAEImageProcessor,
    device: str,
    encoded_frames: int,
    separate_fighters: bool,
    panel_fighter_pooling: bool,
) -> np.ndarray:
    raw = np.stack(panels)
    unique_frames = raw.shape[2] if separate_fighters else raw.shape[1]
    sample = np.round(np.linspace(0, unique_frames - 1, encoded_frames)).astype(np.int64)
    if separate_fighters:
        batch, fighters = raw.shape[:2]
        raw = raw[:, :, sample].reshape(batch * fighters, encoded_frames, *raw.shape[3:])
    else:
        raw = raw[:, sample]
    pixels = torch.from_numpy(raw).permute(0, 1, 4, 2, 3).float().div_(255.0)
    mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
    std = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)
    pixels = ((pixels - mean) / std).to(device, non_blocking=True)
    with torch.autocast(device_type=device, enabled=device == "cuda"):
        tokens = model(pixel_values=pixels).last_hidden_state
    tubelet = int(model.config.tubelet_size)
    temporal = encoded_frames // tubelet
    if tokens.shape[1] % temporal != 0:
        raise RuntimeError(f"cannot reshape {tokens.shape} into {temporal} temporal bins")
    spatial = tokens.shape[1] // temporal
    spatial_tokens = tokens.reshape(tokens.shape[0], temporal, spatial, tokens.shape[2])
    if panel_fighter_pooling:
        grid = round(spatial**0.5)
        if grid * grid != spatial:
            raise RuntimeError(f"cannot split {spatial} spatial tokens into fighter regions")
        grid_tokens = spatial_tokens.reshape(tokens.shape[0], temporal, grid, grid, tokens.shape[2])
        red = grid_tokens[:, :, :, : grid // 2].mean(dim=(2, 3))
        blue = grid_tokens[:, :, :, grid // 2 :].mean(dim=(2, 3))
        features = torch.stack([blue, red], dim=1)
    else:
        features = spatial_tokens.mean(dim=2)
    if separate_fighters:
        features = features.reshape(batch, fighters, temporal, tokens.shape[2])
    return features.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-ckpt", default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--bouts", nargs="+", type=int, default=[116, 117, 120, 121, 122, 115])
    parser.add_argument("--max-proposals", type=int)
    parser.add_argument("--unique-frames", type=int, default=8)
    parser.add_argument("--encoded-frames", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--pad-frac", type=float, default=0.25)
    parser.add_argument("--identity-stripe", type=int, default=4)
    parser.add_argument("--max-box-gap", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--separate-fighters", action="store_true")
    parser.add_argument("--use-stage2-crops", action="store_true")
    parser.add_argument("--panel-fighter-pooling", action="store_true")
    args = parser.parse_args()
    input_modes = (args.separate_fighters, args.use_stage2_crops, args.panel_fighter_pooling)
    if sum(input_modes) > 1:
        parser.error("fighter, Stage-2, and panel-pooling modes are mutually exclusive")

    pipeline = load_pipeline_config(args.pipeline_config)
    bouts = tuple(args.bouts)
    proposals = []
    for bout in bouts:
        payload = json.loads((args.windows_dir / f"bout_{bout}_consensus_windows.json").read_text())
        gt_runs = load_runs(pipeline.bout_dir(bout))
        for row in payload["windows"]:
            peak = int(row["peak_frame"])
            start = peak - args.unique_frames // 2
            end = start + args.unique_frames - 1
            labels = labels_for_span(gt_runs, start, end)
            target = [int(label in labels) for label in LABELS]
            proposals.append(
                {
                    "proposal_id": f"b{bout}_p{int(row['window_id']):04d}",
                    "bout": bout,
                    "window_id": int(row["window_id"]),
                    "peak_frame": peak,
                    "start_frame": start,
                    "end_frame": end,
                    "labels": labels,
                    "target": target,
                }
            )
    if args.max_proposals is not None:
        proposals = proposals[: args.max_proposals]

    n = len(proposals)
    features: np.ndarray | None = None
    view_mask = np.zeros((n, 4), dtype=bool)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model, processor = load_pretrained_encoder(args.model_ckpt, device)
    hidden = int(model.config.hidden_size)
    temporal = args.encoded_frames // int(model.config.tubelet_size)
    has_fighter_axis = args.separate_fighters or args.panel_fighter_pooling
    feature_shape = (n, 4, 2, temporal, hidden) if has_fighter_axis else (n, 4, temporal, hidden)
    features = np.zeros(feature_shape, dtype=np.float16)

    proposal_index = {(row["bout"], row["window_id"]): index for index, row in enumerate(proposals)}
    for bout in bouts:
        bout_rows = [row for row in proposals if row["bout"] == bout]
        for split in range(4):
            video_path = (
                Path("output/stage2_crop")
                / f"Bout {bout}_Split 1-4"
                / f"split_{split}"
                / "crop.mp4"
                if args.use_stage2_crops
                else pipeline.split_video(bout, split)
            )
            bbox_path = pipeline.bout_dir(bout) / f"split_{split}_fighter_bboxes.json"
            if not video_path.exists() or (not args.use_stage2_crops and not bbox_path.exists()):
                print(f"{bout}/{split}: missing video or bbox", flush=True)
                continue
            reader = VideoReader(str(video_path), ctx=cpu(0))
            first = reader[0].asnumpy()
            height, width = int(first.shape[0]), int(first.shape[1])
            lookups = None if args.use_stage2_crops else load_bbox_interpolators(bbox_path, args.max_box_gap)
            pending_panels: list[np.ndarray] = []
            pending_indices: list[int] = []
            valid = rejected = 0

            def flush(view: int = split) -> None:
                nonlocal pending_panels, pending_indices
                if not pending_panels:
                    return
                encoded = encode_batch(
                    model,
                    pending_panels,
                    processor,
                    device,
                    args.encoded_frames,
                    args.separate_fighters,
                    args.panel_fighter_pooling,
                )
                for event_index, event_features in zip(pending_indices, encoded, strict=True):
                    features[event_index, view] = event_features.astype(np.float16)
                    view_mask[event_index, view] = True
                pending_panels, pending_indices = [], []

            for row in bout_rows:
                event_index = proposal_index[(bout, row["window_id"])]
                indices = proposal_indices(row["peak_frame"], args.unique_frames, len(reader))
                batch = reader.get_batch(indices).asnumpy()
                if args.use_stage2_crops:
                    panel = batch
                elif args.separate_fighters:
                    assert lookups is not None
                    panel = make_fighter_clips(
                        batch,
                        indices,
                        lookups,
                        width=width,
                        height=height,
                        size=args.crop_size,
                        pad_frac=args.pad_frac,
                    )
                else:
                    assert lookups is not None
                    panel = make_panel(
                        batch,
                        indices,
                        lookups,
                        width=width,
                        height=height,
                        size=args.crop_size,
                        pad_frac=args.pad_frac,
                        stripe=args.identity_stripe,
                    )
                if panel is None:
                    rejected += 1
                    continue
                valid += 1
                pending_panels.append(panel)
                pending_indices.append(event_index)
                if len(pending_panels) >= args.batch_size:
                    flush()
            flush()
            print(f"{bout}/{split}: {valid} valid, {rejected} rejected", flush=True)

    targets = np.asarray([row["target"] for row in proposals], dtype=np.uint8)
    bouts_array = np.asarray([row["bout"] for row in proposals], dtype=np.int16)
    peaks = np.asarray([row["peak_frame"] for row in proposals], dtype=np.int32)
    starts = np.asarray([row["start_frame"] for row in proposals], dtype=np.int32)
    ends = np.asarray([row["end_frame"] for row in proposals], dtype=np.int32)
    usable = view_mask.any(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "features.npz",
        features=features,
        view_mask=view_mask,
        targets=targets,
        bouts=bouts_array,
        peaks=peaks,
        starts=starts,
        ends=ends,
        usable=usable,
        labels=np.asarray(LABELS),
    )
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "proposal_id", "bout", "window_id", "peak_frame", "start_frame", "end_frame",
                "labels", "n_labels", "is_empty", "n_views", "usable",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(proposals):
            writer.writerow(
                {
                    "proposal_id": row["proposal_id"],
                    "bout": row["bout"],
                    "window_id": row["window_id"],
                    "peak_frame": row["peak_frame"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "labels": "|".join(row["labels"]),
                    "n_labels": len(row["labels"]),
                    "is_empty": len(row["labels"]) == 0,
                    "n_views": int(view_mask[index].sum()),
                    "usable": bool(usable[index]),
                }
            )
    summary = {
        "model_ckpt": args.model_ckpt,
        "unique_frames": args.unique_frames,
        "encoded_frames": args.encoded_frames,
        "separate_fighters": args.separate_fighters,
        "fighter_order": ["blue", "red"] if has_fighter_axis else None,
        "input_mode": (
            "stage2_union_crop"
            if args.use_stage2_crops
            else "separate_fighter_crops"
            if args.separate_fighters
            else "joint_panel_fighter_pooling"
            if args.panel_fighter_pooling
            else "joint_panel_global_pooling"
        ),
        "feature_shape": list(features.shape),
        "proposals": n,
        "usable": int(usable.sum()),
        "zero_view": int((~usable).sum()),
        "view_count": {str(k): int((view_mask.sum(axis=1) == k).sum()) for k in range(5)},
        "empty": int((targets.sum(axis=1) == 0).sum()),
        "positive_counts": dict(zip(LABELS, targets.sum(axis=0).astype(int).tolist(), strict=True)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
