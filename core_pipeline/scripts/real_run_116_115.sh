#!/usr/bin/env bash
# First real cross-bout run: train Stage 3 on bout 116, blind-test on bout 115.
# Memory-safe on a 30GB box: train on 116 splits 0,1 (~6.6GB), test on 115 split 1.
set -euo pipefail
cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"
BB=/home/ubuntu/data/melik_bboxes
PC="--pipeline-config configs/pipeline.yaml"

echo "### 1. import bboxes -> detections.parquet"
uv run python -m bcv.stage1_detect.import_bboxes $PC --bout 116 --split 1 --bbox-json $BB/bout116_split1.json
uv run python -m bcv.stage1_detect.import_bboxes $PC --bout 115 --split 1 --bbox-json $BB/bout115_split1.json
# (116/0 already imported in the interactive step)

echo "### 2. crop (no debug video, for speed)"
for spec in "116 0" "116 1" "115 1"; do
  set -- $spec
  uv run bcv-crop --config configs/stage2_crop.yaml $PC --bout "$1" --split "$2" --no-debug-video
done

echo "### 3. train Stage 3 on bout 116 (splits 0,1), val reuses train"
rm -f output/checkpoints/stage3_frame_classifier/*.ckpt
FIT=$(uv run bcv-frame-clf --config configs/stage3_frame_classifier.yaml fit \
      --train-bouts 116 --val-bouts 116 --max-epochs 15)
echo "$FIT" | tail -3
CKPT=$(echo "$FIT" | grep -oP '(?<=best checkpoint: ).*$')
echo "CKPT=$CKPT"

echo "### 4. predict on held-out bout 115 split 1 (debug video on)"
uv run bcv-frame-clf --config configs/stage3_frame_classifier.yaml predict --ckpt "$CKPT" --bout 115 --split 1

echo "### 5. window + eval on bout 115 split 1"
uv run bcv-window --config configs/stage4_windowing.yaml $PC --bout 115 --split 1 >/dev/null
echo "--- FRAME EVAL (blind bout 115) ---"
uv run bcv-eval $PC --bout 115 --split 1 frame 2>&1 | grep -E '"auroc"|"ap"|"n_pos"|"pos_frac"'
echo "--- WINDOW EVAL (blind bout 115) ---"
uv run bcv-eval $PC --bout 115 --split 1 window 2>&1 | grep -E '"recall"|"precision"|"exact"|"n_gt|"n_pred|"n_missed|"n_false|frames_hist|eval_frame'
echo "### DONE"
