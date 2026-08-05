#!/usr/bin/env bash
# Stage 1 over ALL splits we have boxes for: import -> detections.parquet + a debug video
# (red/blue fighter boxes + the green union "crop box" drawn on the source) so we can
# review every fight's boxes before cropping. New layout: output/stage1_detect/<fight>/...
set -euo pipefail
cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"
BB=/home/ubuntu/data/melik_bboxes
PC="--pipeline-config configs/pipeline.yaml"

imp() {
  echo "--- import + box-debug: bout $1 split $2 ---"
  uv run python -m bcv.stage1_detect.import_bboxes $PC --bout "$1" --split "$2" \
    --bbox-json "$BB/bout$1_split$2.json" --debug-video
}

for s in 1 2 3;     do imp 115 "$s"; done
for s in 0 1 2 3;   do imp 116 "$s"; done
for s in 0 1 2 3;   do imp 117 "$s"; done
echo "### STAGE 1 (all 11 splits) DONE"
