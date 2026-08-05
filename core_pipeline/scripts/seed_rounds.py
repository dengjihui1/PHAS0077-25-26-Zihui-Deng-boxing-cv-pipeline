"""Seed rounds.json for ALREADY-LABELLED fights by inferring rounds from punch annotations.

Rounds are dense punch clusters separated by ~60-75s rests; this writes the inferred
fight-level spans to ``<bout_dir>/rounds.json`` (source=inferred), which the Stage-3 dataset
then uses to exclude between-round frames. For NEW/unlabelled fights there are no punches to
infer from — mark rounds in the bcv-label GUI instead (source=manual). Edit the json by hand
to refine any fight whose inference looks off.

  uv run python scripts/seed_rounds.py --bouts 115 116 117 120 121 122
"""
from __future__ import annotations

import argparse

from bcv.common.annotations import load_runs
from bcv.common.config import load_pipeline_config
from bcv.common.io import read_meta
from bcv.common.rounds import between_round_fraction, infer_rounds, save_rounds


def _fps_for(pipeline, bout: int) -> float:
    for stage in ("stage2_crop", "stage1_detect"):
        for split in range(pipeline.num_views):
            d = pipeline.artifact_dir(bout, split, stage)
            if (d / "meta.json").exists():
                try:
                    return read_meta(d).fps
                except Exception:
                    pass
    return 29.97


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--bouts", type=int, nargs="+", required=True)
    p.add_argument("--gap-s", type=float, default=30.0)
    p.add_argument("--pad-s", type=float, default=8.0)
    args = p.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    for b in args.bouts:
        runs = load_runs(pipeline.bout_dir(b))
        if not runs:
            print(f"bout {b}: no annotations -> skip (mark rounds in bcv-label instead)")
            continue
        fps = _fps_for(pipeline, b)
        max_frame = max(r.end_frame for r in runs) + int(20 * fps)
        rounds = infer_rounds(runs, fps=fps, gap_s=args.gap_s, pad_s=args.pad_s, max_frame=max_frame)
        out = save_rounds(pipeline.bout_dir(b), rounds, fps=fps, source="inferred")
        excl = between_round_fraction(max_frame, rounds) * 100
        spans = ", ".join(f"{lo}-{hi}" for lo, hi in rounds)
        print(f"bout {b}: {len(rounds)} rounds [{spans}] (~{excl:.0f}% between-round) -> {out}")


if __name__ == "__main__":
    main()
