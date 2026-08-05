#!/usr/bin/env bash
# Run the Stage-1 chain detector over many (bout:split) pairs in PARALLEL processes.
# The chain is launch-overhead-bound (~24 fps/proc, GPU ~25% util), so several procs share
# the idle GPU near-linearly (measured ~1.9x at 2 procs). CPU threads are capped per proc
# to avoid oversubscription on the 8-core box.
#   PROCS=4 OUT=output_pred bash scripts/detect_parallel.sh 117:0 117:1 117:2 117:3
set -uo pipefail
cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OUT="${OUT:-}"

run_one() {
  local b="${1%%:*}" s="${1##*:}"
  local log="/tmp/detect_${b}_${s}.log"
  echo "[$(date +%H:%M:%S)] START $b/$s  (live progress: tail -f $log)"
  # Stream full output to a per-split log so live fps/ETA is visible (no tail buffering).
  uv run bcv-detect --config configs/stage1_detect.yaml --pipeline-config configs/pipeline.yaml \
    --bout "$b" --split "$s" ${OUT:+--output-root "$OUT"} > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $b/$s  ($(grep -oE '[0-9.]+ fps' "$log" | tail -1))"
}
export -f run_one

printf '%s\n' "$@" | xargs -P "${PROCS:-4}" -I{} bash -c 'run_one "$@"' _ {}
echo "### detect_parallel DONE ($# splits)"
