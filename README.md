# Boxing CV — Final Code Submission

Retained code for the five-stage boxing computer-vision pipeline described in the
dissertation. It keeps the inherited `core_pipeline` package for reproducibility and
adds the two final experiment folders plus their result artifacts.

## What is here

```text
core_pipeline/                    inherited five-stage package (bcv) + tests
stage4_multiview_consensus_final/ Stage 4: rank-normalised multi-view consensus
stage5_fighter_query_final/       Stage 5: fighter-query VideoMAE classifier
report_analysis/                  script + CSVs behind the report figures/tables
```

## Final results (held-out Bout 115)

| Stage | Method | Result |
|---|---|---|
| 4 | rank-normalised multi-view consensus | strict event F1 0.780 (precision 0.808, recall 0.754) |
| 5 | fighter-query VideoMAE, argmax decoding | typed event F1 0.448, typed macro-F1 0.257 |

Independent-view Stage 4 baseline: precision 0.925, recall 0.209, F1 0.341.

Artifacts: `stage4_multiview_consensus_final/results_cv/` and
`stage5_fighter_query_final/results/final_retained_result.json`.

## Dependencies and install

Python ≥ 3.11, managed with [uv](https://docs.astral.sh/uv/):

```bash
cd core_pipeline
uv sync --extra detect --extra train --extra label
```

## Reproduce the tests (works offline, no data needed)

The self-contained check is the package test suite, which uses small synthetic videos
and annotations generated on the fly:

```bash
cd core_pipeline
uv run pytest -q          # 80 tests, ~5 s
```

## Reproduce Stage 4 / Stage 5

The bout videos, punch labels, fighter boxes, Stage-3 probabilities and trained
checkpoints are controlled project data and are not bundled. The experiment scripts
are included so the runs can be repeated once those artefacts are restored:

```bash
bash stage4_multiview_consensus_final/run_cv.sh    # Stage 4 sweep + leave-one-bout-out
bash stage5_fighter_query_final/run_build.sh       # Stage 5 feature cache
bash stage5_fighter_query_final/run_matched.sh     # Stage 5 training
bash stage5_fighter_query_final/run_evaluate.sh    # Stage 5 final evaluation
```

Each `run_*.sh` documents which input tree it expects and points at the original
server paths; edit the `*_ROOT` variables for a different layout.

## Report analysis

`report_analysis/` contains `build_report_package.py` (which produced the report's
figures/tables) and the resulting CSVs: view-count ablation, temporal-tolerance curve,
Stage 5 outcome-family F1 and class support. These back the numbers quoted in the
dissertation Results section.
