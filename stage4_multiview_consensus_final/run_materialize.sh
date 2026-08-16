#!/usr/bin/env bash
# Stage 4 (final step): materialize the modal robust configuration and its windows.
#
# The sweep (run_cv.sh) explores the grid and picks per-fold parameters; this script
# writes the modal leave-one-bout-out configuration's metrics and the synchronised
# windows that Stage 5 consumes. Run it after run_cv.sh and before run_build.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/core_pipeline"
export PATH="${BCV_CONDA_BIN:-/home/ubuntu/conda/bin}:$PATH"

PROBS_ROOT="${BCV_STAGE3_PROBS_ROOT:-/home/ubuntu/boxing-cv-pipeline/Zihui/stage5_latest_20260720/output/stage3_frame_classifier}"
OUT="${BCV_STAGE4_OUT:-$ROOT/stage4_multiview_consensus_final/results_cv}"
mkdir -p "$OUT"

uv run python "$ROOT/stage4_multiview_consensus_final/materialize_robust.py" \
  --pipeline-config configs/pipeline.yaml \
  --probs-root "$PROBS_ROOT" \
  --output-dir "$OUT"
