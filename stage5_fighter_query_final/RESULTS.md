# Stage 5 Retained Results

Evaluation protocol: train on Bouts 116, 117, 120, and 121; validate on Bout 122; evaluate once on held-out Bout 115.

## Retained final evaluation

The retained verified Kinetics VideoMAE fighter-query model used per-fighter categorical decoding. Its final Bout 115 typed-event result was:

| Metric | Result |
|---|---:|
| Precision | 0.442589 |
| Recall | 0.452991 |
| Typed event F1 | 0.447730 |
| Typed macro-F1 | 0.256955 |

These values are rounded to 0.443, 0.453, 0.448, and 0.257 in the repository README files.

## Included result artifact

`results/matched_fighter_query_result.json` records an earlier reproducibility run of the same model family. Its typed-event F1 is 0.444211 and typed macro-F1 is 0.253956. It is retained as a lightweight code-package artifact, but it is not the final retained evaluation reported above.

The final evaluation was reproduced by the validation-selected original argmax decoding setting, recorded in the project activity-threshold evaluation. No Bout 115 tuning was used.
