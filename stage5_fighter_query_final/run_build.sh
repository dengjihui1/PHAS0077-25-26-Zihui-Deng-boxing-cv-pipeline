#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"

OUT="Zihui/stage5_multiview_structured_20260728/data"
mkdir -p "$OUT"

uv run python Zihui/stage5_multiview_structured_20260728/build_features.py \
  --pipeline-config configs/pipeline.yaml \
  --windows-dir Zihui/stage4_multiview_consensus_20260727/results_cv/robust_windows \
  --output-dir "$OUT" \
  --batch-size 8
