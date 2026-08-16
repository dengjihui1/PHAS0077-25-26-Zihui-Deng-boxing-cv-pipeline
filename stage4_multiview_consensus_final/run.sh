#!/usr/bin/env bash
# Stage 4 sweep: rank-normalised multi-view consensus (see README.md in this folder).
#
# Requires the bcv package (run `uv sync` inside core_pipeline) and two artefact
# trees from the original server. Point PROBS_ROOT / BASELINE_ROOT at your own
# copies if you reproduce elsewhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/core_pipeline"
export PATH="/home/ubuntu/conda/bin:$PATH"

PROBS_ROOT="/home/ubuntu/boxing-cv-pipeline/Zihui/stage5_latest_20260720/output/stage3_frame_classifier"
BASELINE_ROOT="/home/ubuntu/boxing-cv-pipeline/Zihui/stage4_stage5_clean_windows_20260724/output/stage4_windowing"
OUT="$ROOT/stage4_multiview_consensus_final/results"
mkdir -p "$OUT"

uv run python "$ROOT/stage4_multiview_consensus_final/sweep_consensus.py" \
  --pipeline-config configs/pipeline.yaml \
  --probs-root "$PROBS_ROOT" \
  --baseline-windows-root "$BASELINE_ROOT" \
  --output-dir "$OUT"
