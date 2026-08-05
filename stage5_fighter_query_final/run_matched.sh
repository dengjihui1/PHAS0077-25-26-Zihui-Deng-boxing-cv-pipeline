#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"

uv run python Zihui/stage5_multiview_structured_20260728/train_fighter_matched.py \
  --features Zihui/stage5_multiview_structured_20260728/data/features.npz \
  --pipeline-config configs/pipeline.yaml \
  --output-dir Zihui/stage5_multiview_structured_20260728/models_matched \
  --epochs 100 \
  --batch-size 64
