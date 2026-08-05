# Stage 4 Multi-View Consensus

This experiment replaces independent per-camera Stage 4 decisions with one synchronized
bout-level proposal stream.

## Why

All four camera splits share the same annotation frame indices, but the inherited pipeline
runs Stage 3/4 independently per split. The same physical punch can therefore produce
duplicated, missing, or temporally inconsistent windows. Those inconsistent windows are a
major source of noisy multi-label Stage 5 clips.

## Method

1. Align available split probabilities by frame.
2. Compare raw and within-view rank normalization.
3. Compare max, top-2 mean, mean, and median view fusion.
4. Detect short local-maximum proposals with temporal NMS.
5. Select parameters on Bout 122 only; evaluate Bout 115 once.

The script writes `sweep.csv`, `selected_result.json`, and one consensus `windows.json`
file per bout below `results/`. Existing outputs are read-only.

`run_cv.sh` writes a second copy below `results_cv/` and adds leave-one-bout-out
development-set checks without changing the original result folder.

The five development folds selected the same modal configuration (`rank`, mean fusion,
threshold 0.80, minimum peak distance 6, radius 6). `materialize_robust.py` writes that
configuration's final metrics and synchronized windows for downstream Stage 5 use.

## Run

```bash
nohup bash Zihui/stage4_multiview_consensus_20260727/run.sh \
  > Zihui/stage4_multiview_consensus_20260727/train.log 2>&1 &
```
