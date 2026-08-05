#!/usr/bin/env bash
# Overnight: Stage 2 crops (parallel) then Stage 3 punch-classifier eval (sequential) for
# 120/121/122. Run AFTER Stage 1 detection has written all 12 detections.parquet.
#
# Crops are CPU-bound (decode+crop+write) -> parallelize. Punch eval loads the model + a
# whole crop.mp4 (~3GB decoded) into RAM, so it runs ONE split at a time to avoid OOM on
# the 30GB box. Each step logs per-split to /tmp; a summary of frame AUROC/AP prints at the
# end. NOTE: these crops come from the chain detector, which misses the RED fighter ~40% of
# the time (see memory) — so the union crop is often blue-only/full-frame fallback, which
# will depress punch metrics. That is a real, honest result about crop-quality -> punch-quality.
set -uo pipefail
cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PAIRS="120:0 120:1 120:2 120:3 121:0 121:1 121:2 121:3 122:0 122:1 122:2 122:3"

echo "===== STAGE 2 CROPS (parallel) $(date +%H:%M:%S) ====="
crop_one() {
  local b="${1%%:*}" s="${1##*:}"
  if uv run bcv-crop --config configs/stage2_crop.yaml --pipeline-config configs/pipeline.yaml \
       --bout "$b" --split "$s" --no-debug-video > "/tmp/crop_${b}_${s}.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] crop OK   $b/$s"
  else
    echo "[$(date +%H:%M:%S)] crop FAIL $b/$s (see /tmp/crop_${b}_${s}.log)"
  fi
}
export -f crop_one
printf '%s\n' $PAIRS | xargs -P 6 -I{} bash -c 'crop_one "$@"' _ {}

echo "===== STAGE 3 PUNCH-CLASSIFIER EVAL (sequential, OOM-safe) $(date +%H:%M:%S) ====="
for p in $PAIRS; do
  b="${p%%:*}"; s="${p##*:}"
  log="/tmp/puncheval_${b}_${s}.log"
  if uv run python scripts/make_eval_probs.py --eval-bout "$b" --eval-split "$s" > "$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] $b/$s  $(grep -oE 'AUROC [0-9.]+  AP [0-9.]+.*' "$log" | head -1)"
  else
    echo "[$(date +%H:%M:%S)] punch-eval FAIL $b/$s (see $log)"
  fi
done

echo "===== SUMMARY (frame AUROC / AP per split) $(date +%H:%M:%S) ====="
for p in $PAIRS; do
  b="${p%%:*}"; s="${p##*:}"
  echo -n "  $b/$s: "; grep -oE 'AUROC [0-9.]+  AP [0-9.]+.*' "/tmp/puncheval_${b}_${s}.log" 2>/dev/null | head -1 || echo "(no result)"
done
echo "### FINISH 120/121/122 DONE $(date +%H:%M:%S)"
