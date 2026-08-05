"""`bcv-label-prefill` — precompute initial red/blue bbox keyframes for the labeller.

Writes the GUI keyframe project (``output/label/bbox_keyframes_bout{N}_split{S}.json``) so
``bcv-label --bout N --split S`` opens already pre-filled — no in-app waiting. By default it
REUSES an existing Stage-1 ``detections.parquet`` if present (instant), else runs the chain.
Bulk over bouts/splits.

  uv run bcv-label-prefill --bout 120 --split 0                  # reuse detections, else chain
  uv run bcv-label-prefill --bouts 120 121 122 --splits 0 1 2 3  # bulk
  uv run bcv-label-prefill --bout 130 --split 0 --source chain   # force a fresh chain run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bcv.common.config import load_pipeline_config

from .boxes import FIGHTERS, BBoxProject, project_path, save_project
from .frames import FrameSource


def keyframes_from_detections(df: pd.DataFrame, stride: int) -> dict[str, dict[str, list[int] | None]]:
    """Sample a detections.parquet (DETECTION_SCHEMA) every ``stride`` frames into keyframes.

    Records present boxes and explicit absent (None) keyframes, so the GUI interpolation
    matches what the detector actually saw rather than bridging across absences.
    """
    out: dict[str, dict[str, list[int] | None]] = {c: {} for c in FIGHTERS}
    frames = df["frame"].to_numpy()
    for pos in range(0, len(df), stride):
        f = int(frames[pos])
        row = df.iloc[pos]
        for c in FIGHTERS:
            out[c][str(f)] = (
                [int(row[f"{c}_x1"]), int(row[f"{c}_y1"]), int(row[f"{c}_x2"]), int(row[f"{c}_y2"])]
                if bool(row[f"{c}_present"]) else None
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute initial bbox keyframes for bcv-label")
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--stage1-config", default="configs/stage1_detect.yaml")
    p.add_argument("--bout", type=int)
    p.add_argument("--bouts", type=int, nargs="+")
    p.add_argument("--split", type=int)
    p.add_argument("--splits", type=int, nargs="+", default=None)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--source", choices=["auto", "detections", "chain"], default="auto",
                   help="auto: reuse detections.parquet if present, else run the chain")
    p.add_argument("--out-dir", default=None, help="default: <output_root>/label")
    p.add_argument("--overwrite", action="store_true",
                   help="re-prefill even if a keyframe project already exists")
    args = p.parse_args()

    bouts = args.bouts or ([args.bout] if args.bout is not None else None)
    if not bouts:
        raise SystemExit("give --bout or --bouts")
    splits = args.splits or ([args.split] if args.split is not None else [0, 1, 2, 3])

    pipeline = load_pipeline_config(args.pipeline_config)
    out_dir = Path(args.out_dir) if args.out_dir else pipeline.output_root / "label"

    for b in bouts:
        for s in splits:
            video = pipeline.split_video(b, s)
            if not video.exists():
                print(f"skip {b}/{s}: no split video")
                continue
            pf = project_path(out_dir, b, s)
            if pf.exists() and not args.overwrite:
                print(f"skip {b}/{s}: {pf.name} exists (use --overwrite)")
                continue

            det = pipeline.artifact_dir(b, s, "stage1_detect") / "detections.parquet"
            use_chain = args.source == "chain" or (args.source == "auto" and not det.exists())

            if use_chain:
                from bcv.common.config import load_config
                from bcv.stage1_detect.run import Stage1Config

                from .prefill import prefill_from_chain
                fs = FrameSource(video)
                proj = BBoxProject(bout=b, split=s, num_frames=fs.num_frames)
                fs.release()
                stage_cfg = load_config(args.stage1_config, Stage1Config)
                print(f"{b}/{s}: running chain over {proj.num_frames} frames…", flush=True)
                n = prefill_from_chain(proj, pipeline, stage_cfg, stride=args.stride)
                src = "chain"
            else:
                df = pd.read_parquet(det)
                proj = BBoxProject(bout=b, split=s, num_frames=len(df))
                proj.keyframes = keyframes_from_detections(df, args.stride)
                n = sum(len(v) for v in proj.keyframes.values())
                src = "detections"

            save_project(pf, proj)
            print(f"{b}/{s}: {n} keyframe-frames ({src}) -> {pf}")


if __name__ == "__main__":
    main()
