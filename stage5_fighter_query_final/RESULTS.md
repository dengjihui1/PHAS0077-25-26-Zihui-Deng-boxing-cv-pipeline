# Stage 5 Retained Results

Train on Bouts 116, 117, 120, 121; validate on Bout 122; evaluate once on held-out
Bout 115. No Bout 115 tuning.

## Final retained evaluation

The retained Kinetics VideoMAE fighter-query model uses plain per-fighter argmax
decoding (the activity-threshold sweep selected `argmax`, threshold 0.0). Its Bout 115
typed-event result is in `results/final_retained_result.json`:

| Metric | Result |
|---|---:|
| Typed event precision | 0.442589 |
| Typed event recall | 0.452991 |
| Typed event F1 | 0.447730 |
| Typed macro-F1 | 0.256955 |

These round to 0.443 / 0.453 / 0.448 / 0.257 in the report. Reproduce with
`run_evaluate.sh` (needs the trained checkpoint, which is not bundled).

## Outcome-family F1 on the final retained run

Pooling blue and red per-class counts from `final_retained_result.json`:

| Outcome | GT | Pred | Matched | F1 |
|---|---:|---:|---:|---:|
| body landed | 12 | 2 | 0 | 0.000 |
| head landed | 124 | 133 | 47 | 0.366 |
| blocked | 65 | 28 | 9 | 0.194 |
| missed | 267 | 316 | 156 | 0.535 |

Bout 115 has one blue body-landed and nine red blocked events (per-class table inside
`final_retained_result.json`).

## Earlier reproducibility run

`results/matched_fighter_query_result.json` records an earlier run of the same model
family: typed event F1 0.444211, macro-F1 0.253956. Kept for provenance, not the final
result.

## Clean-GT and activity diagnostics

`results/disentangled_result.json` backs the report's bottleneck claims:
- clean GT-centred type recognition: accuracy 0.545, macro-F1 0.274;
- per-fighter activity macro-F1: 0.684.
