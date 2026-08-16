"""Audit fixed-length Stage-5 labels around robust consensus peaks.

Diagnostic only: it inspects label purity and peak-tolerance recall on every bout,
including the held-out test bout 115. The output is never used for parameter or model
selection; Bout 115 is included purely so the final test numbers can be sanity-checked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bcv.common.annotations import load_runs
from bcv.common.config import load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    result = {}
    for bout in (116, 117, 120, 121, 122, 115):
        payload = json.loads((args.windows_dir / f"bout_{bout}_consensus_windows.json").read_text())
        peaks = [int(row["peak_frame"]) for row in payload["windows"]]
        gt = [(r.start_frame, r.end_frame, r.label) for r in load_runs(pipeline.bout_dir(bout))]
        lengths = {}
        for length in (8, 16, 32):
            counts = {"empty": 0, "single_label": 0, "multi_label": 0, "per_fighter_unambiguous": 0}
            half = length // 2
            for peak in peaks:
                start, end = peak - half, peak - half + length - 1
                labels = {label for gs, ge, label in gt if not (ge < start or gs > end)}
                if not labels:
                    counts["empty"] += 1
                elif len(labels) == 1:
                    counts["single_label"] += 1
                else:
                    counts["multi_label"] += 1
                per_fighter: dict[str, set[str]] = {}
                for label in labels:
                    fighter, strike_type = label.split("_", 1)
                    per_fighter.setdefault(fighter, set()).add(strike_type)
                if labels and all(len(types) <= 1 for types in per_fighter.values()):
                    counts["per_fighter_unambiguous"] += 1
            n = len(peaks)
            lengths[str(length)] = {
                **counts,
                **{f"{key}_fraction": value / n if n else 0.0 for key, value in counts.items()},
            }
        tolerance_recall = {}
        for tolerance in (2, 4, 8, 12, 16):
            hits = sum(
                any(start - tolerance <= peak <= end + tolerance for peak in peaks)
                for start, end, _label in gt
            )
            tolerance_recall[str(tolerance)] = hits / len(gt) if gt else 0.0
        result[str(bout)] = {
            "n_peaks": len(peaks),
            "n_gt_events": len(gt),
            "clip_lengths": lengths,
            "gt_peak_recall_by_tolerance": tolerance_recall,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
