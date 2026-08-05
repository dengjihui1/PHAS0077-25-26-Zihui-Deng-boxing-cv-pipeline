#!/usr/bin/env bash
# Self-serve crops for the 3 labelled-but-no-box fights (120/121/122) using the Stage 1
# CHAIN detector (stock YOLO + red/blue classifier), then Stage 2 crop. After this they
# become runnable/held-out for scripts/overlay_all.py. Weaker boxes than a fine-tuned
# detector (Backend B, not built) — a bad crop can masquerade as a model miss.
set -euo pipefail
cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"
PC="--pipeline-config configs/pipeline.yaml"

for b in 120 121 122; do
  for s in 0 1 2 3; do
    echo "=== DETECT bout $b split $s ($(date +%H:%M:%S)) ==="
    uv run bcv-detect --config configs/stage1_detect.yaml $PC --bout "$b" --split "$s"
  done
done
echo "### STAGE 1 (120/121/122, 12 splits) DONE"

for b in 120 121 122; do
  for s in 0 1 2 3; do
    echo "=== CROP bout $b split $s ($(date +%H:%M:%S)) ==="
    uv run bcv-crop --config configs/stage2_crop.yaml $PC --bout "$b" --split "$s" --no-debug-video
  done
done
echo "### STAGE 2 (120/121/122, 12 splits) DONE"
