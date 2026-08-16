#!/usr/bin/env bash
# Stage 4 sweep + leave-one-bout-out development check, writing to results_cv/.
# Same inputs and env overrides as run.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/core_pipeline"
export PATH="${BCV_CONDA_BIN:-/home/ubuntu/conda/bin}:$PATH"

PROBS_ROOT="${BCV_STAGE3_PROBS_ROOT:-/home/ubuntu/boxing-cv-pipeline/Zihui/stage5_latest_20260720/output/stage3_frame_classifier}"
BASELINE_ROOT="${BCV_BASELINE_WINDOWS_ROOT:-/home/ubuntu/boxing-cv-pipeline/Zihui/stage4_stage5_clean_windows_20260724/output/stage4_windowing}"
OUT="${BCV_STAGE4_CV_OUT:-$ROOT/stage4_multiview_consensus_final/results_cv}"
mkdir -p "$OUT"

uv run python "$ROOT/stage4_multiview_consensus_final/sweep_consensus.py" \
  --pipeline-config configs/pipeline.yaml \
  --probs-root "$PROBS_ROOT" \
  --baseline-windows-root "$BASELINE_ROOT" \
  --output-dir "$OUT"
