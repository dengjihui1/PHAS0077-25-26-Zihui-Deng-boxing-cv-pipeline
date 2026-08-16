# Boxing CV — Multi-View Strike Localisation

A five-stage computer-vision pipeline for boxing video analysis: it detects the two
fighters in synchronised camera views, crops them, scores every frame for punch
evidence, fuses the views into a single stream of strike proposals, and predicts each
fighter's outcome. The core contribution of this work is the **multi-view temporal
consensus** (Stage 4) and the **fighter-query outcome model** (Stage 5).

## Pipeline at a glance

| Stage | Function | Implementation | Origin |
|---|---|---|---|
| 1 | Detect the two fighters in each view | YOLO detector proposes boxes, a dedicated red/blue classifier assigns fighter colour, short-horizon tracking stabilises the fixed red/blue identity slots | inherited |
| 2 | Crop the pair | padded square built from the union of the two boxes, EMA-smoothed over time and carried through brief detection gaps | inherited |
| 3 | Per-frame punch evidence | small windowed CNN scores the central frame of each cropped context → per-frame punch probability | inherited, retrained with more bouts |
| 4 | Strike-event localisation | **rank-normalised multi-view consensus**: within-view rank, mean fusion, peak detection + temporal NMS | **new** (replaces per-view hysteresis windowing) |
| 5 | Per-fighter outcome | **fighter-query VideoMAE** over 8-frame synchronised multi-view panels (null / body / head / blocked / missed per fighter) | **new** (not present in the original project) |

Stages 1–3 are inherited from the original project (the supervisor's 2026 hand-off
implementation); Stages 4–5 are the contributions of this dissertation.

## Stage 3 — punch finder

Stage 3 is a small windowed CNN that scores the central frame of every cropped context,
emitting a per-frame punch probability that the later stages treat as evidence rather than
a final decision.

![Frame-level information exposed by the punch finder](docs/punchFinder_FrameInformation.png)

The frame-level view shows what the punch finder exposes at each time step: the fighter
crops it reads, the frame index, and the punch probability it emits for that frame. This
per-frame trace is exactly the signal the multi-view consensus consumes in Stage 4.

![Punch finder running over a bout](docs/punchFinder_InAction.png)

Over a full bout, the punch finder highlights the frames where a strike occurs. This gives
a direct visual check that the frame-level evidence tracks real punches before any
downstream localisation is applied.

## Stage 4 — multi-view temporal consensus

The original pipeline windowed each camera view independently, which was precise but
missed many events. Stage 4 rank-normalises each view's probability trace, fuses the
available views by their mean, and keeps local peaks as a single synchronised proposal
stream.

![Precision–recall movement on held-out Bout 115](docs/fig_stage4_precision_recall_movement.png)

The precision–recall movement shows what the consensus buys: recall rises from 0.209 to
0.754 while precision stays above 0.80, moving strict event F1 from 0.341 to 0.780.

![View-count ablation on held-out Bout 115](docs/fig_stage4_view_count_ablation.png)

The view-count ablation confirms the mechanism — strict event F1 grows with the number of
synchronised views (0.661 with one view to 0.780 with three), so the cameras contribute
complementary rather than redundant evidence.

## Stage 5 — fighter-query outcome model

Each Stage 4 peak anchors an eight-frame synchronised panel. A verified Kinetics-pretrained
VideoMAE encodes every view, and red/blue fighter queries read the shared evidence through
separate identity slots to predict one five-state outcome per fighter (null, body landed,
head landed, blocked, missed).

![Stage 5 fighter-query multi-view architecture](docs/fig_stage5_fighter_query_architecture.png)

![Class support versus per-class F1](docs/fig_stage5_support_vs_f1.png)

The support-versus-F1 plot explains the remaining difficulty: body-landed and blocked
outcomes have very few training examples and correspondingly weak recognition, while
frequent outcomes such as missed are far more reliable.

![Pipeline bottleneck summary](docs/fig_pipeline_bottleneck_waterfall.png)

The waterfall summarises the pipeline-level story — temporal localisation improves
substantially, while fine-grained outcome recognition remains bounded by label scarcity
and short RGB clips.

## Results (held-out Bout 115)

| Stage | Method | Result |
|---|---|---|
| 4 | rank-normalised multi-view consensus | strict event F1 **0.341 → 0.780** (recall 0.209 → 0.754) |
| 5 | fighter-query VideoMAE, argmax decoding | typed event F1 **0.448**, typed macro-F1 **0.257** |

The archived original route (per-view windowing + 32-frame eight-way classification)
scored 0.204 accuracy / 0.102 macro-F1 on the same bout; the two routes use different
proposal, target and metric definitions, so the comparison indicates pipeline progress
rather than a strictly matched metric change.

## Repository layout

```text
core_pipeline/                     inherited five-stage package (bcv) + tests
stage4_multiview_consensus_final/  Stage 4 multi-view consensus (this work)
stage5_fighter_query_final/        Stage 5 fighter-query classifier (this work)
docs/                              figures used in this README
```

## Install and reproduce

Python ≥ 3.11, managed with [uv](https://docs.astral.sh/uv/):

```bash
cd core_pipeline
uv sync --extra detect --extra train --extra label
uv run pytest -q          # 87 tests, offline, ~5 s
```

The Stage 4 / Stage 5 experiment scripts are included so the runs can be repeated once
the controlled bout data are restored:

```bash
bash stage4_multiview_consensus_final/run_cv.sh    # Stage 4 sweep + leave-one-bout-out
bash stage5_fighter_query_final/run_build.sh       # Stage 5 feature cache
bash stage5_fighter_query_final/run_matched.sh     # Stage 5 training
bash stage5_fighter_query_final/run_evaluate.sh    # Stage 5 final evaluation
```

Bout videos, punch labels, fighter boxes, Stage-3 probabilities and trained checkpoints
are controlled project data and are not bundled; each `run_*.sh` documents its expected
inputs.

## Results and provenance

- `stage4_multiview_consensus_final/results_cv/` — parameter sweep, leave-one-bout-out
  folds, and the held-out Bout 115 result.
- `stage5_fighter_query_final/results/final_retained_result.json` — final Stage 5 numbers.
- `stage5_fighter_query_final/results/disentangled_result.json` — clean-GT and activity
  diagnostics.

## Acknowledgements

I would like to express my sincere gratitude to the PhD students Brian Chiang, Luke
Johnson and Melik Oughton for their foundational research and code, on which this project
builds, and to SWA for generously providing the data and information used throughout this
work. I am also deeply grateful to my supervisors, Gabriel Facini and Nikita Pond, for
their invaluable guidance, encouragement and support.
