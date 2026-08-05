#!/usr/bin/env bash
# Cross-bout run: train Stage 3 on bouts 116+117 (2 views each), eval on held-out bout 115.
set -euo pipefail
cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"
BB=/home/ubuntu/data/melik_bboxes
PC="--pipeline-config configs/pipeline.yaml"

echo "### import bout 117 boxes (splits 0,1)"
for s in 0 1; do
  uv run python -m bcv.stage1_detect.import_bboxes $PC --bout 117 --split "$s" --bbox-json "$BB/bout117_split${s}.json"
done

echo "### crop bout 117 (splits 0,1) — union crop only"
for s in 0 1; do
  uv run bcv-crop --config configs/stage2_crop.yaml $PC --bout 117 --split "$s" --no-debug-video
done

echo "### cross-bout train (116+117) -> eval held-out 115 split 1"
uv run python scripts/cross_bout_experiment.py \
  --train-bouts 116 117 --eval-bout 115 --eval-split 1 --max-epochs 8
echo "### DONE"
