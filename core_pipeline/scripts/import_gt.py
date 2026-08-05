"""Import canonical fighter-box GT into a SEPARATE output root, for detector evaluation.

The GT lives at ``new_splits/Bout N_Split 1-4/split_S_fighter_bboxes.json`` (the labels the
detector was trained on). This resolves each frame's candidates through the same
``select.py`` red/blue resolver the live chain uses, writing a GT ``detections.parquet``
under ``--out-root`` so it never collides with the live-chain predictions in ``output/``.
Then: ``bcv-eval detection --bout N --split S --gt-parquet <out-root>/stage1_detect/.../detections.parquet``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bcv.common.config import load_pipeline_config
from bcv.stage1_detect.import_bboxes import import_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--bouts", type=int, nargs="+", default=[117])
    p.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--out-root", default="output_gt")
    args = p.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config).model_copy(
        update={"output_root": Path(args.out_root)}
    )
    for b in args.bouts:
        for s in args.splits:
            jsonl = pipeline.bout_dir(b) / f"split_{s}_fighter_bboxes.json"
            if not jsonl.exists():
                print(f"skip {b}/{s}: no GT json at {jsonl}")
                continue
            out = import_split(
                pipeline, bout=b, split=s, bbox_json=str(jsonl),
                min_cls_conf=0.5, debug_video=False,
            )
            print(f"GT {b}/{s} -> {out}")


if __name__ == "__main__":
    main()
