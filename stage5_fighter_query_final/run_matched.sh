#!/usr/bin/env bash
# Stage 5 training: per-fighter categorical fighter-query over the cached features.
# Runs both the mean-fusion baseline and the fighter-query model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/core_pipeline"
export PATH="${BCV_CONDA_BIN:-/home/ubuntu/conda/bin}:$PATH"

uv run python "$ROOT/stage5_fighter_query_final/train_fighter_matched.py" \
  --features "${BCV_STAGE5_FEATURES:-$ROOT/stage5_fighter_query_final/data/features.npz}" \
  --pipeline-config configs/pipeline.yaml \
  --output-dir "${BCV_STAGE5_MODELS:-$ROOT/stage5_fighter_query_final/models_matched}" \
  --epochs 100 \
  --batch-size 64
