"""Stage 4: frame_probs.parquet -> strike windows.json (+ timeline debug plot)."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, ConfigDict

from ..common.annotations import load_runs
from ..common.config import PipelineConfig
from ..common.contracts import ArtifactMeta
from ..common.io import write_json, write_meta
from .hysteresis import Window, make_windows, tag_windows_with_gt

STAGE = "stage4_windowing"
STAGE3 = "stage3_frame_classifier"


class Stage4Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_high: float = 0.5
    t_low: float = 0.35
    min_duration: int = 3
    merge_gap: int = 2
    split_valley: float | None = None
    split_min_gap: int = 2
    split_peak_min_prob: float | None = None
    split_peak_min_distance: int = 8
    split_peak_min_drop: float = 0.1


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return None


def _debug_plot(out_png: Path, df: pd.DataFrame, windows: list[Window], gt_events) -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df["frame"], df["p_smooth"], color="orange", lw=0.8, label="p_smooth")
    for w in windows:
        ax.axvspan(w.start_frame, w.end_frame, color="tab:blue", alpha=0.25)
    for gs, ge, _ in gt_events:
        ax.axvspan(gs, ge, ymin=0.0, ymax=0.08, color="green")
    ax.set_ylim(0, 1)
    ax.set_xlabel("frame")
    ax.set_ylabel("P(punch)")
    ax.set_title("Stage 4 - predicted windows (blue) vs GT events (green band)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def run_stage4(
    pipeline: PipelineConfig, cfg: Stage4Config, *, bout: int, split: int, debug: bool = True
) -> Path:
    in_dir = pipeline.artifact_dir(bout, split, STAGE3)
    df = pd.read_parquet(in_dir / "frame_probs.parquet")
    frames = df["frame"].to_numpy()

    windows = make_windows(
        frames,
        df["p_smooth"].to_numpy(),
        df["p_punch"].to_numpy(),
        t_high=cfg.t_high,
        t_low=cfg.t_low,
        min_duration=cfg.min_duration,
        merge_gap=cfg.merge_gap,
        split_valley=cfg.split_valley,
        split_min_gap=cfg.split_min_gap,
        split_peak_min_prob=cfg.split_peak_min_prob,
        split_peak_min_distance=cfg.split_peak_min_distance,
        split_peak_min_drop=cfg.split_peak_min_drop,
    )
    gt_events = [(r.start_frame, r.end_frame, r.label) for r in load_runs(pipeline.bout_dir(bout))]
    tag_windows_with_gt(windows, gt_events)

    out_dir = pipeline.artifact_dir(bout, split, STAGE)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "windows.json", {"windows": [w.model_dump() for w in windows]})
    if debug:
        _debug_plot(out_dir / "timeline.png", df, windows, gt_events)

    write_meta(
        out_dir,
        ArtifactMeta(
            stage=STAGE,
            source_video=str(pipeline.split_video(bout, split)),
            fps=0.0,
            width=0,
            height=0,
            num_frames=len(df),
            git_sha=_git_sha(),
            created_utc=datetime.now(UTC).isoformat(),
            producer={
                "t_high": cfg.t_high,
                "t_low": cfg.t_low,
                "min_duration": cfg.min_duration,
                "merge_gap": cfg.merge_gap,
                "split_valley": cfg.split_valley,
                "split_min_gap": cfg.split_min_gap,
                "split_peak_min_prob": cfg.split_peak_min_prob,
                "split_peak_min_distance": cfg.split_peak_min_distance,
                "split_peak_min_drop": cfg.split_peak_min_drop,
                "n_windows": len(windows),
                # analyzed frame range: eval restricts GT to this so recall is meaningful
                "frame_min": int(frames.min()) if len(frames) else 0,
                "frame_max": int(frames.max()) if len(frames) else 0,
            },
        ),
    )
    return out_dir
