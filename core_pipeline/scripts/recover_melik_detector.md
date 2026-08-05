# Recovering Melik's fine-tuned fighter detector (Stage 1, Backend B)

Backend B — our own 2-class (red/blue) YOLO detector — is the **primary** Stage-1 goal,
but its assets are **not on this machine** (verified: `find` returns nothing for either).
They only ever lived on Melik's box / Roboflow. This track is **non-blocking**: the whole
pipeline runs on Backend A (the detect→classify chain) until these arrive.

## Ask Melik for

1. **`runs/detect/train15/weights/best.pt`** — the fine-tuned 2-class red/blue **detector**
   weights (immediate better backend + warm-start for re-training).
2. **The `labelled_data/` detection dataset + its `data.yaml`** (red/blue fighter boxes) —
   or the **Roboflow workspace/project** name it lives in (referenced as
   `boxer-crop-classification` for the *classifier*; the *detector* dataset is `labelled_data`).
3. *(Bonus)* the newer "big-data" classifier weights `boxer_yolo26_classifier_new_bigger42`
   (only `args.yaml` is on disk here; `best.pt` is missing).

## When assets arrive

1. Drop weights under `models/` (git-ignored).
2. `python scripts/make_detector_dataset.py` → normalize labels to YOLO det format,
   **two classes** `names: {0: red, 1: blue}`, **bout-level** train/val split.
3. `bcv-detector-finetune --config configs/stage1_detector_finetune.yaml`
   (warm-start from `train15/best.pt` if recovered, else cold-start from `yolo26x.pt`).
4. **Polarity gate** (`tests/test_class_polarity.py`): assert `model.names == {0:'red',1:'blue'}`
   and that index 0 fires on the red fighter, 1 on blue — guards a silent red/blue swap.
5. Flip `backend: finetuned` in `configs/stage1_detect.yaml`; Backend B feeds the **same**
   `select.py` slot resolver and emits the identical `detections.parquet` schema.
6. A/B check: red/blue box IoU agreement between Backend A and B on one bout before committing.
