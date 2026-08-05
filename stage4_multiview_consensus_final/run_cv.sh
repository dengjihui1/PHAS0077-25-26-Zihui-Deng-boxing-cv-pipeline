#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"

OUT="Zihui/stage4_multiview_consensus_20260727/results_cv"
mkdir -p "$OUT"

uv run python Zihui/stage4_multiview_consensus_20260727/sweep_consensus.py \
  --pipeline-config configs/pipeline.yaml \
  --probs-root Zihui/stage5_latest_20260720/output/stage3_frame_classifier \
  --baseline-windows-root Zihui/stage4_stage5_clean_windows_20260724/output/stage4_windowing \
  --output-dir "$OUT"
