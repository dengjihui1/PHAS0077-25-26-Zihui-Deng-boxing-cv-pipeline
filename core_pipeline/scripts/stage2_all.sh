#!/usr/bin/env bash
# Stage 2 over ALL splits: crop.mp4 (the cropped fighter video) + crop-box debug, in the
# new layout output/stage2_crop/<fight>/split_N/. Run after stage1_all.sh.
set -euo pipefail
cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"
PC="--pipeline-config configs/pipeline.yaml"

crop() {
  echo "--- crop: bout $1 split $2 ---"
  # --no-debug-video: just the small crop.mp4 (the cropped fighter video), skip the
  # heavy crop-on-source overlay (the stage1 boxes.mp4 already shows the crop box).
  uv run bcv-crop --config configs/stage2_crop.yaml $PC --bout "$1" --split "$2" --no-debug-video
}

for s in 1 2 3;   do crop 115 "$s"; done
for s in 0 1 2 3; do crop 116 "$s"; done
for s in 0 1 2 3; do crop 117 "$s"; done
echo "### STAGE 2 (all 11 splits) DONE"
