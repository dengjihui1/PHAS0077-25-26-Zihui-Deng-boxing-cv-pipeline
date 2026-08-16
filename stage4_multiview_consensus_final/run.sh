#!/usr/bin/env bash
# Stage 4 sweep: rank-normalised multi-view consensus (see README.md in this folder).
#
# Requires the bcv package (`uv sync` inside core_pipeline) and two artefact trees.
# Override the BCV_* variables for a different layout; defaults target the original
# server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/core_pipeline"
export PATH="${BCV_CONDA_BIN:-/home/ubuntu/conda/bin}:$PATH"

PROBS_ROOT="${BCV_STAGE3_PROBS_ROOT:-/home/ubuntu/boxing-cv-pipeline/Zihui/stage5_latest_20260720/output/stage3_frame_classifier}"
BASELINE_ROOT="${BCV_BASELINE_WINDOWS_ROOT:-/home/ubuntu/boxing-cv-pipeline/Zihui/stage4_stage5_clean_windows_20260724/output/stage4_windowing}"
OUT="${BCV_STAGE4_OUT:-$ROOT/stage4_multiview_consensus_final/results}"
mkdir -p "$OUT"

uv run python "$ROOT/stage4_multiview_consensus_final/sweep_consensus.py" \
  --pipeline-config configs/pipeline.yaml \
  --probs-root "$PROBS_ROOT" \
  --baseline-windows-root "$BASELINE_ROOT" \
  --output-dir "$OUT"
