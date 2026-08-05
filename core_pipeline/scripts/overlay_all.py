"""Prob-vs-truth timeline overlays for every runnable labelled split.

For each split that has a Stage-2 crop (115/116/117 -> 11 splits; 120/121/122 lack
detection boxes so the model can't run), score it with the frozen 0.887 checkpoint,
window it with the tuned Stage-4 config, and render a multi-row "ECG" timeline:
p_smooth (orange) with GT punch spans (green) and predicted windows (blue) so merges
(one blue span swallowing several green bands) are visible at a glance. Prints the
overlap error budget per split. NOTE: the model trained on 116+117, so only bout 115
is held-out — 116/117 overlays are on training data and read optimistically.
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from bcv.common.annotations import load_runs
from bcv.common.config import load_config, load_pipeline_config
from bcv.common.io import write_parquet
from bcv.eval.window import window_metrics
from bcv.stage3_frame_classifier.infer import build_frame_probs, predict_logits
from bcv.stage3_frame_classifier.model import WindowPunchModule
from bcv.stage3_frame_classifier.run import Stage3Config, build_dataset
from bcv.stage4_windowing.hysteresis import make_windows
from bcv.stage4_windowing.run import Stage4Config

CKPT = (".cometml-runs/boxing-stage3-frame/"
        "0f0def58ba61412ab2e4937c3b1ef645/checkpoints/epoch=7-step=11136.ckpt")
TRAIN_BOUTS = {116, 117}  # everything else is held-out for this checkpoint


def _timeline_plot(out_png, df, windows, gt, s4, title, frames_per_row=3000):
    frames = df["frame"].to_numpy()
    psm = df["p_smooth"].to_numpy()
    fmin, fmax = int(frames.min()), int(frames.max())
    n_rows = max(1, int(np.ceil((fmax - fmin + 1) / frames_per_row)))
    pred = [(w.start_frame, w.end_frame) for w in windows]

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 1.5 * n_rows + 0.6), squeeze=False)
    for r, ax in enumerate(axes[:, 0]):
        lo = fmin + r * frames_per_row
        hi = lo + frames_per_row
        sel = (frames >= lo) & (frames < hi)
        ax.plot(frames[sel], psm[sel], color="tab:orange", lw=0.7)
        ax.axhline(s4.t_high, color="gray", ls="--", lw=0.5)
        for gs, ge in gt:
            if ge >= lo and gs < hi:
                ax.axvspan(gs, ge, color="tab:green", alpha=0.35, lw=0)
        for ps, pe in pred:
            if pe >= lo and ps < hi:
                ax.axvspan(ps, pe, ymin=0.0, ymax=1.0, color="tab:blue", alpha=0.18, lw=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    handles = [Patch(color="tab:orange", label="p_smooth"),
               Patch(color="tab:green", alpha=0.35, label="GT punch"),
               Patch(color="tab:blue", alpha=0.18, label="pred window")]
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=7, ncol=3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage3_frame_classifier.yaml")
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--stage4-config", default="configs/stage4_windowing.yaml")
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--img-size", type=int, default=112)
    args = p.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    cfg = load_config(args.config, Stage3Config)
    s4 = load_config(args.stage4_config, Stage4Config)
    module = WindowPunchModule.load_from_checkpoint(args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    crop_root = pipeline.output_root / "stage2_crop"
    splits = []
    for fight in sorted(p for p in crop_root.iterdir() if p.is_dir()):
        bout = int(fight.name.split("_")[0].replace("Bout ", "").strip())
        for sd in sorted(fight.glob("split_*")):
            if (sd / "crop.mp4").exists():
                splits.append((bout, int(sd.name.replace("split_", ""))))
    print(f"runnable splits ({len(splits)}): {splits}", flush=True)

    print(f"\n{'bout/split':14s} {'held?':6s} {'det_recall':>10s} {'miss':>5s} "
          f"{'fake':>5s} {'merged':>7s} {'#GT':>5s} {'#win':>5s}")
    for bout, split in splits:
        ds = build_dataset(pipeline, cfg, bout, split)
        if ds is None:
            continue
        if args.img_size and ds.frames.shape[1] != args.img_size:
            ds.frames = np.stack([cv2.resize(f, (args.img_size, args.img_size)) for f in ds.frames])
        logits = predict_logits(module, ds, device=device, batch_size=256)
        df = build_frame_probs(logits, ds, temperature=1.0, roll_w=cfg.roll_w)

        windows = make_windows(
            df["frame"].to_numpy(), df["p_smooth"].to_numpy(), df["p_punch"].to_numpy(),
            t_high=s4.t_high, t_low=s4.t_low, min_duration=s4.min_duration,
            merge_gap=s4.merge_gap, split_valley=s4.split_valley,
            split_min_gap=s4.split_min_gap,
            split_peak_min_prob=s4.split_peak_min_prob,
            split_peak_min_distance=s4.split_peak_min_distance,
            split_peak_min_drop=s4.split_peak_min_drop,
        )
        fmin, fmax = int(df["frame"].min()), int(df["frame"].max())
        gt = [(r.start_frame, r.end_frame) for r in load_runs(pipeline.bout_dir(bout))
              if not (r.end_frame < fmin or r.start_frame > fmax)]
        wm = window_metrics([(w.start_frame, w.end_frame) for w in windows], gt)

        held = "no" if bout in TRAIN_BOUTS else "YES"
        out_dir = pipeline.artifact_dir(bout, split, "overlays")
        out_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(out_dir / "frame_probs.parquet", df)
        title = (f"Bout {bout} split {split}  [{'HELD-OUT' if held=='YES' else 'TRAIN'}]  "
                 f"det_recall {wm['detection_recall']:.2f}  miss {wm['n_missed']}  "
                 f"fake {wm['n_fake']}  merged {wm['n_merged']}/{wm['n_detected']}")
        _timeline_plot(out_dir / "timeline.png", df, windows, gt, s4, title)
        print(f"{f'{bout}/{split}':14s} {held:6s} {wm['detection_recall']:>10.2f} "
              f"{wm['n_missed']:>5d} {wm['n_fake']:>5d} "
              f"{wm['n_merged']:>3d}/{wm['n_detected']:<3d} {wm['n_gt_events']:>5d} "
              f"{wm['n_pred_windows']:>5d}  -> {out_dir / 'timeline.png'}", flush=True)


if __name__ == "__main__":
    main()
