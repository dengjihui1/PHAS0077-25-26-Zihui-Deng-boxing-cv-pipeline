# Boxing CV Final Code Submission Folder

This folder contains the useful local code for the final retained boxing computer-vision pipeline. It keeps the full inherited `core_pipeline` for reproducibility, but excludes failed exploratory branches, large checkpoints, cached features, videos, and temporary server logs.

## Folder Structure

```text
final_code_submission_clean_20260805/
  .gitignore
  README.md
  core_pipeline/
    src/
    configs/
    scripts/
    tests/
    pyproject.toml
    uv.lock
  stage4_multiview_consensus_final/
    sweep_consensus.py
    materialize_robust.py
    analyze_stage5_readiness.py
    run.sh
    run_cv.sh
    results_cv/
  stage5_fighter_query_final/
    build_features.py
    train_fighter_matched.py
    run_build.sh
    run_matched.sh
    results/
```

## What Is Included

### Core pipeline

`core_pipeline` contains the reusable five-stage project package and its tests. It includes the original Stage 1/2/3 implementation, Stage 4 base utilities, scripts, configs, and dependency files.

This is inherited project code copied unchanged from the local original repository. It is retained so the final Stage 4 and Stage 5 scripts can be understood and run with their original dependencies.

Main relevant Stage 3 files:

- `core_pipeline/src/bcv/stage3_frame_classifier/dataset.py`
- `core_pipeline/src/bcv/stage3_frame_classifier/model.py`
- `core_pipeline/src/bcv/stage3_frame_classifier/infer.py`
- `core_pipeline/src/bcv/stage3_frame_classifier/run.py`
- `core_pipeline/scripts/cross_bout_experiment.py`

### Final Stage 4 method

`stage4_multiview_consensus_final` contains the retained Stage 4 improvement:

```text
rank-normalized multi-view consensus
```

It aligns synchronized split-level Stage 3 probabilities, rank-normalizes each view, mean-fuses available views, detects consensus peaks, applies temporal NMS, and outputs one proposal stream per bout.

The comparison below uses the same Bout 115 ground truth and strict one-to-one event-matching definition.

| Method | Precision | Recall | Strict event F1 |
|---|---:|---:|---:|
| Previous independent-view reference | 0.925 | 0.209 | 0.341 |
| Rank-normalized multi-view consensus | 0.808 | 0.754 | 0.780 |

### Final Stage 5 method

`stage5_fighter_query_final` contains the retained Stage 5 route:

```text
verified Kinetics VideoMAE features + matched per-fighter categorical fighter-query
```

It starts from Stage 4 consensus proposals, uses short synchronized multi-view panels, and predicts a separate state for each fighter:

```text
null, body landed, head landed, blocked, missed
```

The final retained Bout 115 evaluation was:

| Metric | Result |
|---|---:|---:|
| Typed event precision | 0.443 |
| Typed event recall | 0.453 |
| Typed event F1 | 0.448 |
| Typed macro-F1 | 0.257 |

For context, the archived original route used single-view 32-frame clips and eight-way clip classification. Its Bout 115 result was accuracy 0.204 and macro-F1 0.102. Because proposal construction, targets, and metrics changed, this is historical pipeline progress rather than a direct metric comparison. See `stage5_fighter_query_final/RESULTS.md` for the retained evaluation record and the included reproducibility artifact.

## What Is Excluded

The following are deliberately excluded from this clean folder:

- failed Stage 5 exploration branches;
- model checkpoints and `.pt` files;
- feature caches such as `.npz` or memmaps;
- raw videos and generated clips;
- server `nohup`/training logs except lightweight retained result JSONs;
- `__pycache__` and test cache folders.

## Running The Code

The original project uses `uv` and should be run from the server project root:

```bash
cd /home/ubuntu/boxing-cv-pipeline
export PATH="/home/ubuntu/conda/bin:$PATH"
```

Typical Stage 3 reproduction command:

```bash
uv run python scripts/cross_bout_experiment.py --train-bouts 116 117 --eval-bout 115
```

Final Stage 4 cross-bout consensus sweep:

```bash
bash Zihui/stage4_multiview_consensus_final/run_cv.sh
```

Final Stage 5 retained route:

```bash
bash Zihui/stage5_fighter_query_final/run_build.sh
bash Zihui/stage5_fighter_query_final/run_matched.sh
```

The copied folder is for clean submission/reading. On the server, paths may need to be placed under `/home/ubuntu/boxing-cv-pipeline/Zihui/` as in the original experiment folders.
