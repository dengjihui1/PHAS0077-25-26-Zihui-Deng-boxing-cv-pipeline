#!/usr/bin/env bash
# Stage 5 feature build: 8-frame VideoMAE panels around Stage-4 consensus peaks.
#
# Needs the robust consensus windows (materialize_robust.py output) and the raw
# bout videos. Point WINDOWS_DIR at your robust_windows folder.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/core_pipeline"
export PATH="/home/ubuntu/conda/bin:$PATH"

WINDOWS_DIR="$ROOT/stage4_multiview_consensus_final/results_cv/robust_windows"
OUT="$ROOT/stage5_fighter_query_final/data"
mkdir -p "$OUT"

uv run python "$ROOT/stage5_fighter_query_final/build_features.py" \
  --pipeline-config configs/pipeline.yaml \
  --windows-dir "$WINDOWS_DIR" \
  --output-dir "$OUT" \
  --batch-size 8
