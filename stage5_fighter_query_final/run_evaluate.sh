#!/usr/bin/env bash
# Stage 5 final evaluation: activity-threshold decoding of the retained checkpoint.
# Reproduces results/final_retained_result.json (typed event F1 0.448 / macro 0.257).
#
# The trained checkpoint is not bundled (size); train it with run_matched.sh first
# or restore it from the server, then point BASE_CKPT at the fighter_query best.pt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/core_pipeline"
export PATH="/home/ubuntu/conda/bin:$PATH"

BASE_CKPT="$ROOT/stage5_fighter_query_final/models_matched/fighter_query_categorical/best.pt"

uv run python "$ROOT/stage5_fighter_query_final/evaluate_activity_threshold.py" \
  --panel-features "$ROOT/stage5_fighter_query_final/data/features.npz" \
  --base-ckpt "$BASE_CKPT" \
  --pipeline-config configs/pipeline.yaml \
  --output "$ROOT/stage5_fighter_query_final/results/final_retained_result.json" \
  --batch-size 64
