"""Stage 3 orchestration: fit the windowed punch model, and predict per-frame probs."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import cv2
import lightning as L
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pydantic import BaseModel, ConfigDict

from ..common import viz
from ..common.annotations import discover_annotation_file, group_into_runs, load_annotations
from ..common.cometlog import make_logger
from ..common.config import PipelineConfig
from ..common.contracts import ArtifactMeta, StrikeRun
from ..common.io import write_meta, write_parquet
from ..common.video import VideoWriter
from .datamodule import PunchDataModule
from .dataset import CroppedWindowDataset
from .infer import build_frame_probs, predict_logits
from .model import WindowPunchModule

STAGE = "stage3_frame_classifier"
STAGE2 = "stage2_crop"


class Stage3Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = 2
    k_pad: int = 1
    backbone: str = "small3d"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    # With ``balanced`` sampling on, "auto" resolves to 1.0 (the sampler already balances
    # the stream — adding neg/pos pos_weight double-counts positives and over-confidences
    # the curve). With balanced off, "auto" => neg/pos. A float is always used verbatim.
    pos_weight: float | str = "auto"
    balanced: bool = True  # WeightedRandomSampler draws ~50/50 pos/neg in the train loader
    focal_gamma: float = 0.0
    dropout: float = 0.3
    batch_size: int = 16
    num_workers: int = 0
    max_epochs: int = 10
    img_size: int | None = None  # optional square downsample before the net (e.g. 112 for the 2D backbone)
    roll_w: int = 11
    exclude_carried_forward: bool = False
    exclude_between_rounds: bool = True  # train/eval only on in-round frames (rounds.json)
    # Calibration off by default: the raw sigmoid IS the probability. Temperature scaling
    # is only sound when fit on a *separate* held-out set — never on the prediction target.
    calibration: str = "none"
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str = "32-true"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return None


def runs_for_bout(pipeline: PipelineConfig, bout: int) -> list[StrikeRun]:
    anno = discover_annotation_file(pipeline.bout_dir(bout))
    if anno is None:
        return []
    return group_into_runs(load_annotations(anno).annotations)


def build_dataset(
    pipeline: PipelineConfig, cfg: Stage3Config, bout: int, split: int
) -> CroppedWindowDataset | None:
    crop_dir = pipeline.artifact_dir(bout, split, STAGE2)
    if not (crop_dir / "crop.mp4").exists():
        return None
    ds = CroppedWindowDataset.from_artifact(
        crop_dir,
        runs_for_bout(pipeline, bout),
        k=cfg.k,
        k_pad=cfg.k_pad,
        exclude_carried_forward=cfg.exclude_carried_forward,
        source_video=str(pipeline.split_video(bout, split)),
        img_size=cfg.img_size,
    )
    if cfg.exclude_between_rounds:
        from ..common.rounds import load_rounds
        rounds = load_rounds(pipeline.bout_dir(bout))
        if rounds:
            before = len(ds)
            dropped = ds.restrict_to_rounds(rounds)
            print(f"[stage3] bout {bout}/{split}: {len(rounds)} rounds, dropped "
                  f"{dropped}/{before} between-round frames", flush=True)
        else:
            print(f"[stage3] bout {bout}/{split}: no rounds.json -> using ALL frames "
                  f"(mark rounds in bcv-label or seed via scripts/seed_rounds.py)", flush=True)
    return ds


def collect_datasets(
    pipeline: PipelineConfig, cfg: Stage3Config, bouts: list[int], splits: list[int] | None = None
) -> list[CroppedWindowDataset]:
    out = []
    split_ids = splits if splits is not None else list(range(pipeline.num_views))
    for bout in bouts:
        for split in split_ids:
            ds = build_dataset(pipeline, cfg, bout, split)
            if ds is not None and len(ds) > 0:
                out.append(ds)
    return out


def _auto_pos_weight(train: list[CroppedWindowDataset]) -> float:
    labels = np.concatenate([ds.labels[ds._index] for ds in train])
    pos = float(labels.sum())
    neg = float(len(labels) - pos)
    return float(neg / pos) if pos > 0 else 1.0


def resolve_pos_weight(cfg: Stage3Config, train: list[CroppedWindowDataset]) -> float:
    """Pick the BCE ``pos_weight``, avoiding the sampler+pos_weight double-up-weighting.

    An explicit float is honoured verbatim. ``"auto"`` returns **1.0 when the balanced
    sampler is on** (it already presents a ~50/50 stream, so neg/pos pos_weight would
    double-count positives and over-confidence the curve) and ``neg/pos`` otherwise.
    """
    if cfg.pos_weight != "auto":
        return float(cfg.pos_weight)
    return 1.0 if cfg.balanced else _auto_pos_weight(train)


def downsample_datasets(
    datasets: list[CroppedWindowDataset], img_size: int | None
) -> list[CroppedWindowDataset]:
    """Resize each dataset's in-memory frames to ``img_size`` square (no-op if unset/equal)."""
    if not img_size:
        return datasets
    for ds in datasets:
        if ds.frames.shape[1] != img_size:
            ds.frames = np.stack([cv2.resize(f, (img_size, img_size)) for f in ds.frames])
    return datasets


def build_module(cfg: Stage3Config, pos_weight: float) -> WindowPunchModule:
    return WindowPunchModule(
        backbone=cfg.backbone,
        k=cfg.k,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        pos_weight=pos_weight,
        focal_gamma=cfg.focal_gamma,
        dropout=cfg.dropout,
    )


def run_fit(
    pipeline: PipelineConfig,
    cfg: Stage3Config,
    *,
    train_bouts: list[int] | None = None,
    val_bouts: list[int] | None = None,
    train_splits: list[int] | None = None,
    resume_from: str | Path | None = None,
) -> Path:
    _require_cuda_if_requested(cfg)
    train_b = train_bouts or pipeline.splits.train
    val_b = val_bouts if val_bouts is not None else pipeline.splits.val
    train = downsample_datasets(collect_datasets(pipeline, cfg, train_b, train_splits), cfg.img_size)
    if not train:
        raise RuntimeError(
            "no Stage-2 crop artifacts found for the train bouts — run Stage 1+2 first"
        )

    # Only validate/select a checkpoint on a GENUINELY held-out fight. When val == train
    # (or unset), aliasing train as val and monitoring val_loss would select on the training
    # data — a leak — so we instead train a fixed number of epochs and keep `last` (mirrors
    # scripts/cross_bout_experiment.py, which evaluates held-out *separately*).
    held_out = bool(val_b) and set(val_b) != set(train_b)
    val = downsample_datasets(collect_datasets(pipeline, cfg, val_b, train_splits), cfg.img_size) if held_out else []
    if held_out and not val:
        print(f"[stage3] no crops for val bouts {val_b} -> training without validation", flush=True)
        held_out = False

    pos_weight = resolve_pos_weight(cfg, train)
    selection = (
        f"best val_ap, early-stop patience={cfg.early_stopping_patience}"
        if held_out and cfg.early_stopping_patience is not None
        else ("best val_ap, no early stop" if held_out else "last checkpoint, no validation")
    )
    print(
        f"[stage3] train {train_b} ({sum(len(d) for d in train)} windows) | "
        f"val {'held-out ' + str(val_b) if held_out else 'NONE (fixed-epoch, save last)'} | "
        f"balanced={cfg.balanced} pos_weight={pos_weight:.3f} | selection={selection}",
        flush=True,
    )
    print(f"[stage3] device: {_device_summary(cfg)}", flush=True)
    module = build_module(cfg, pos_weight)
    # Lightning requires a non-empty val set object even when we skip validation; the
    # `limit_val_batches=0.0` below makes that object inert without reloading crop frames.
    dm = PunchDataModule(
        train, val if held_out else train,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, balanced=cfg.balanced,
    )

    ckpt_dir = pipeline.output_root / "checkpoints" / STAGE
    if held_out:
        ckpt = ModelCheckpoint(
            dirpath=ckpt_dir, filename="best", monitor="val_ap", mode="max", save_last=True
        )
    else:
        ckpt = ModelCheckpoint(dirpath=ckpt_dir, filename="best", save_last=True)
    callbacks: list[L.Callback] = [ckpt]
    if held_out and cfg.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_ap",
                mode="max",
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta,
                verbose=True,
            )
        )
    logger = make_logger(
        "boxing-stage3-frame",
        name=f"fit-train{'+'.join(map(str, train_b))}"
        + (f"-val{'+'.join(map(str, val_b))}" if held_out else ""),
    )
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,  # type: ignore[arg-type]
        logger=logger or False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
        limit_val_batches=1.0 if held_out else 0.0,
        num_sanity_val_steps=2 if held_out else 0,
    )
    trainer.fit(module, dm, ckpt_path=str(resume_from) if resume_from else None)
    if logger is not None:
        logger.finalize("success")
    return Path(ckpt.best_model_path or ckpt.last_model_path)


# ---------------------------------------------------------------------------
# Held-out evaluation experiments (formerly scripts/{cross_bout,temporal}_experiment.py
# and scripts/make_eval_probs.py — collapsed here so the CLI is the single source of truth
# and can reproduce the 0.887 cross-bout result).
# ---------------------------------------------------------------------------


class _EpochProgress(L.Callback):
    """One stdout line per epoch so background runs show progress (no val/progress bar)."""

    def on_train_epoch_end(self, tr: L.Trainer, _pl: L.LightningModule) -> None:
        loss = tr.callback_metrics.get("train_loss")
        msg = f"epoch {tr.current_epoch + 1}/{tr.max_epochs}"
        print(msg + (f"  train_loss={float(loss):.4f}" if loss is not None else ""), flush=True)


def train_fixed_epochs(
    pipeline: PipelineConfig,
    cfg: Stage3Config,
    train: list[CroppedWindowDataset],
    *,
    logger_name: str | None = None,
    save_ckpt: bool = True,
    resume_from: str | Path | None = None,
) -> tuple[WindowPunchModule, object | None]:
    """Train ``max_epochs`` with the balanced sampler and NO validation; the held-out eval
    is done separately by the caller (so there is no train-as-val selection leak). Returns
    the in-memory module and its Comet logger (or None) for the caller to log eval metrics."""
    _require_cuda_if_requested(cfg)
    pos_weight = resolve_pos_weight(cfg, train)
    print(
        f"[stage3] train {sum(len(d) for d in train)} windows across {len(train)} splits "
        f"| balanced={cfg.balanced} pos_weight={pos_weight:.3f} | fixed {cfg.max_epochs} epochs, no val",
        flush=True,
    )
    print(f"[stage3] device: {_device_summary(cfg)}", flush=True)
    module = build_module(cfg, pos_weight)
    dm = PunchDataModule(
        train, train, batch_size=cfg.batch_size, num_workers=cfg.num_workers, balanced=cfg.balanced
    )
    logger = make_logger("boxing-stage3-frame", name=logger_name) if logger_name else None
    callbacks: list = [_EpochProgress()]
    if save_ckpt:
        callbacks.append(ModelCheckpoint(dirpath=pipeline.output_root / "checkpoints" / STAGE,
                                         save_last=True))
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs, accelerator=cfg.accelerator, devices=cfg.devices,
        precision=cfg.precision,  # type: ignore[arg-type]
        logger=logger or False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=callbacks, limit_val_batches=0.0, num_sanity_val_steps=0,
    )
    trainer.fit(module, dm, ckpt_path=str(resume_from) if resume_from else None)
    return module, logger


def _predict_df(cfg: Stage3Config, module: WindowPunchModule, ds: CroppedWindowDataset) -> pd.DataFrame:
    device = "cuda" if cfg.accelerator in ("auto", "gpu") and _cuda() else "cpu"
    logits = predict_logits(module, ds, device=device, batch_size=256)
    return build_frame_probs(logits, ds, temperature=1.0, roll_w=cfg.roll_w)


def evaluate_probs(
    pipeline: PipelineConfig,
    df: pd.DataFrame,
    *,
    eval_bout: int,
    eval_split: int,
    out_subdir: str,
    extra_meta: dict | None = None,
    logger: object | None = None,
) -> dict:
    """Frame ROC/PR + Stage-4 window metrics for ``df`` against ``eval_bout``'s GT (restricted
    to df's frame range). Writes frame_probs.parquet, both plots, metrics.json; logs to Comet."""
    from ..common.annotations import load_runs
    from ..common.config import load_config
    from ..eval.frame import frame_metrics, plot_frame_eval
    from ..eval.window import plot_window_eval, window_metrics
    from ..stage4_windowing.hysteresis import make_windows
    from ..stage4_windowing.run import Stage4Config

    out_dir = pipeline.artifact_dir(eval_bout, eval_split, out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(out_dir / "frame_probs.parquet", df)
    fm = frame_metrics(df)
    plot_frame_eval(df, str(out_dir / "frame_roc_pr.png"))

    s4 = load_config("configs/stage4_windowing.yaml", Stage4Config)
    windows = make_windows(
        df["frame"].to_numpy(), df["p_smooth"].to_numpy(), df["p_punch"].to_numpy(),
        t_high=s4.t_high, t_low=s4.t_low, min_duration=s4.min_duration, merge_gap=s4.merge_gap,
        split_valley=s4.split_valley, split_min_gap=s4.split_min_gap,
        split_peak_min_prob=s4.split_peak_min_prob,
        split_peak_min_distance=s4.split_peak_min_distance,
        split_peak_min_drop=s4.split_peak_min_drop,
    )
    pred = [(w.start_frame, w.end_frame) for w in windows]
    fmin, fmax = int(df["frame"].min()), int(df["frame"].max())
    gt = [(r.start_frame, r.end_frame) for r in load_runs(pipeline.bout_dir(eval_bout))
          if not (r.end_frame < fmin or r.start_frame > fmax)]
    wm = window_metrics(pred, gt)
    plot_window_eval(wm, str(out_dir / "window_eval.png"))

    metrics = {"frame": fm, "window": wm, **(extra_meta or {})}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if logger is not None:
        logger.log_metrics({  # type: ignore[attr-defined]
            "test_auroc": fm["auroc"], "test_ap": fm["ap"],
            "test_pos_frac": fm.get("pos_frac", 0.0),
            "test_window_recall": wm["recall"], "test_window_precision": wm["precision"],
        })
        for png in ("frame_roc_pr", "window_eval"):
            logger.experiment.log_image(str(out_dir / f"{png}.png"), name=png)  # type: ignore[attr-defined]
    print(f"FRAME : AUROC {fm['auroc']:.3f}  AP {fm['ap']:.3f}  "
          f"(pos_frac {fm.get('pos_frac', 0):.3f}, n={fm['n']})", flush=True)
    print(f"WINDOW: recall {wm['recall']:.2f}  precision {wm['precision']:.2f}  "
          f"missed {wm['n_missed_events']}  false {wm['n_false_alarms']}", flush=True)
    print("plots ->", out_dir, flush=True)
    return metrics


def run_crossbout_eval(
    pipeline: PipelineConfig,
    cfg: Stage3Config,
    *,
    train_bouts: list[int],
    train_splits: list[int] | None = None,
    resume_from: str | Path | None = None,
    eval_bout: int,
    eval_split: int,
) -> dict:
    """Train on ``train_bouts``, evaluate the never-seen ``eval_bout`` — the real
    generalization test. Reproduces scripts/cross_bout_experiment.py via the CLI."""
    train = downsample_datasets(collect_datasets(pipeline, cfg, train_bouts, train_splits), cfg.img_size)
    if not train:
        raise RuntimeError(f"no cropped train splits for bouts {train_bouts}")
    eval_ds = build_dataset(pipeline, cfg, eval_bout, eval_split)
    if eval_ds is None:
        raise RuntimeError(f"no crop for eval bout {eval_bout} split {eval_split}")
    downsample_datasets([eval_ds], cfg.img_size)
    split_tag = "all" if train_splits is None else "+".join(map(str, train_splits))
    name = f"crossbout-train{'+'.join(map(str, train_bouts))}-splits{split_tag}-test{eval_bout}"
    module, logger = train_fixed_epochs(
        pipeline, cfg, train, logger_name=name, resume_from=resume_from
    )
    df = _predict_df(cfg, module, eval_ds)
    metrics = evaluate_probs(
        pipeline, df, eval_bout=eval_bout, eval_split=eval_split, out_subdir="eval_crossbout",
        extra_meta={"train_bouts": train_bouts, "train_splits": train_splits, "eval": [eval_bout, eval_split]},
        logger=logger,
    )
    if logger is not None:
        logger.finalize("success")  # type: ignore[attr-defined]
    return metrics


def run_temporal_eval(
    pipeline: PipelineConfig,
    cfg: Stage3Config,
    *,
    bout: int,
    split: int,
    train_frac: float = 0.5,
    gap_frac: float = 0.01,
) -> dict:
    """Within-fight sanity check: train on the first ``train_frac`` of one split, evaluate the
    held-out tail. Reproduces scripts/temporal_experiment.py via the CLI."""
    ds = build_dataset(pipeline, cfg, bout, split)
    if ds is None:
        raise RuntimeError(f"no crop artifact for bout {bout} split {split}")
    downsample_datasets([ds], cfg.img_size)
    m = ds.m
    train = ds.subrange(0.0, train_frac - gap_frac)
    module, _ = train_fixed_epochs(pipeline, cfg, [train], save_ckpt=False)
    df = _predict_df(cfg, module, ds)
    df_test = df.iloc[int(train_frac * m):].reset_index(drop=True)
    return evaluate_probs(
        pipeline, df_test, eval_bout=bout, eval_split=split, out_subdir="eval_temporal",
        extra_meta={"train_frac": train_frac, "bout": bout, "split": split},
    )


def eval_from_checkpoint(
    pipeline: PipelineConfig,
    cfg: Stage3Config,
    ckpt: str | Path,
    *,
    eval_bout: int,
    eval_split: int,
) -> dict:
    """Predict-only held-out eval from a saved checkpoint (no training → no OOM). Reproduces
    scripts/make_eval_probs.py via the CLI."""
    eval_ds = build_dataset(pipeline, cfg, eval_bout, eval_split)
    if eval_ds is None:
        raise RuntimeError(f"no crop for eval bout {eval_bout} split {eval_split}")
    downsample_datasets([eval_ds], cfg.img_size)
    module = WindowPunchModule.load_from_checkpoint(ckpt)
    df = _predict_df(cfg, module, eval_ds)
    return evaluate_probs(
        pipeline, df, eval_bout=eval_bout, eval_split=eval_split, out_subdir="eval_crossbout",
        extra_meta={"checkpoint": str(ckpt), "eval": [eval_bout, eval_split]},
    )


def _write_debug(
    crop_dir: Path,
    out_dir: Path,
    df: pd.DataFrame,
    fps: float,
    threshold: float,
    *,
    labels: np.ndarray | None = None,
    window_seconds: float = 10.0,
) -> None:
    cap = cv2.VideoCapture(str(crop_dir / "crop.mp4"))
    probs = df["p_smooth"].to_numpy()
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return
    h, w = frame.shape[:2]
    strip_h = 90
    # Rolling window so local on/off is legible instead of the whole fight squeezed to width.
    window_frames = round(window_seconds * fps) if window_seconds else None
    writer = VideoWriter(out_dir / "prob_trace.mp4", fps, w, h + strip_h)
    idx = 0
    while ok:
        vis = viz.prob_trace(frame, probs, idx, height=strip_h, threshold=threshold,
                             labels=labels, window_frames=window_frames)
        writer.write(vis)
        ok, frame = cap.read()
        idx += 1
    cap.release()
    writer.close()


def run_predict(
    pipeline: PipelineConfig,
    cfg: Stage3Config,
    ckpt: str | Path,
    *,
    bout: int,
    split: int,
    debug_video: bool = True,
    threshold: float = 0.5,
) -> Path:
    ds = build_dataset(pipeline, cfg, bout, split)
    if ds is None:
        raise RuntimeError(f"no Stage-2 crop for bout {bout} split {split}")
    module = WindowPunchModule.load_from_checkpoint(ckpt)
    device = "cuda" if cfg.accelerator in ("auto", "gpu") and _cuda() else "cpu"
    logits = predict_logits(module, ds, device=device, batch_size=max(8, cfg.batch_size))

    # Temperature scaling is intentionally NOT fit here: the only labels available at
    # predict time are the prediction target's own, so fitting on them is in-sample and
    # leaky (it flatters metrics and squashes the signal Stage 4 needs). Calibrate on a
    # separate held-out split if ever needed; default keeps the raw sigmoid (T=1).
    if cfg.calibration == "temperature":
        print(
            "[stage3] calibration='temperature' ignored in predict: refusing in-sample fit "
            "on the target's own labels (use a held-out calib split). Using T=1.",
            flush=True,
        )
    df = build_frame_probs(logits, ds, temperature=1.0, roll_w=cfg.roll_w)

    crop_dir = pipeline.artifact_dir(bout, split, STAGE2)
    out_dir = pipeline.artifact_dir(bout, split, STAGE)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(out_dir / "frame_probs.parquet", df)
    if debug_video:
        from ..common.io import read_meta

        fps = read_meta(crop_dir).fps
        # Only overlay ground truth when the bout is actually labelled — otherwise
        # the all-zero label column would render a misleading "no punches" truth band.
        gt = df["label"].to_numpy() if runs_for_bout(pipeline, bout) else None
        _write_debug(crop_dir, out_dir, df, fps, threshold, labels=gt)

    meta = ArtifactMeta(
        stage=STAGE,
        source_video=str(pipeline.split_video(bout, split)),
        fps=read_meta_fps(crop_dir),
        width=ds.frames.shape[2],
        height=ds.frames.shape[1],
        num_frames=len(df),
        git_sha=_git_sha(),
        created_utc=datetime.now(UTC).isoformat(),
        producer={
            "k": cfg.k,
            "window_len": 2 * cfg.k + 1,
            "backbone": cfg.backbone,
            "checkpoint": str(ckpt),
            "temperature": 1.0,
            "roll_w": cfg.roll_w,
            "pos_weight": module.hparams["pos_weight"],
        },
    )
    write_meta(out_dir, meta)
    return out_dir


def read_meta_fps(crop_dir: Path) -> float:
    from ..common.io import read_meta

    return read_meta(crop_dir).fps


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def _require_cuda_if_requested(cfg: Stage3Config) -> None:
    if cfg.accelerator != "gpu":
        return
    if not _cuda():
        raise RuntimeError(
            "Stage 3 was started with accelerator='gpu', but PyTorch cannot see CUDA. "
            "Check nvidia-smi, the active Python environment, and the installed torch build."
        )


def _device_summary(cfg: Stage3Config) -> str:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return (
            f"cuda available | requested accelerator={cfg.accelerator}, "
            f"devices={cfg.devices}, gpu0={name}"
        )
    return f"cuda unavailable | requested accelerator={cfg.accelerator}, devices={cfg.devices}"
