#!/usr/bin/env bash
# Render Stage-3 punch-classifier overlay videos (prob_trace.mp4 = P(punch) trace + GT band
# on the crop) for many bout:split pairs in parallel. Reads saved frame_probs.parquet, no
# model re-run. Per-split logs in /tmp/trace_<b>_<s>.log.
#   PROCS=6 bash scripts/render_traces.sh 120:0 120:1 ...
set -uo pipefail
cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"
export OMP_NUM_THREADS=1
render_one() {
  local b="${1%%:*}" s="${1##*:}"
  if uv run python scripts/render_prob_trace.py --bout "$b" --split "$s" > "/tmp/trace_${b}_${s}.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK   $b/$s -> $(grep -oE 'output/.*prob_trace.mp4' /tmp/trace_${b}_${s}.log | head -1)"
  else
    echo "[$(date +%H:%M:%S)] FAIL $b/$s (see /tmp/trace_${b}_${s}.log)"
  fi
}
export -f render_one
printf '%s\n' "$@" | xargs -P "${PROCS:-6}" -I{} bash -c 'render_one "$@"' _ {}
echo "### RENDER TRACES DONE ($# splits)"
