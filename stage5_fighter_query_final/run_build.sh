#!/usr/bin/env bash
# Stage 5 feature build: 8-frame VideoMAE panels around Stage-4 consensus peaks.
#
# Requires the robust windows written by run_materialize.sh and the raw bout videos.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/core_pipeline"
export PATH="${BCV_CONDA_BIN:-/home/ubuntu/conda/bin}:$PATH"

WINDOWS_DIR="${BCV_ROBUST_WINDOWS_ROOT:-$ROOT/stage4_multiview_consensus_final/results_cv/robust_windows}"
OUT="${BCV_STAGE5_DATA:-$ROOT/stage5_fighter_query_final/data}"
mkdir -p "$OUT"

uv run python "$ROOT/stage5_fighter_query_final/build_features.py" \
  --pipeline-config configs/pipeline.yaml \
  --windows-dir "$WINDOWS_DIR" \
  --output-dir "$OUT" \
  --batch-size 8
