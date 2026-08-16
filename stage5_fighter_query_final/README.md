# Stage 5 Fighter-Query Outcome Recognition

This directory contains the retained Stage 5 route. It takes the robust Stage 4
consensus peaks, extracts eight-frame synchronized multi-view VideoMAE features, and
uses a fighter-query categorical head to predict one state for each fighter:

```text
null, body landed, head landed, blocked, missed
```

The explicit null class keeps false Stage 4 proposals in the end-to-end pipeline rather
than forcing every proposal to be a strike. Training uses Bouts 116, 117, 120, and 121;
Bout 122 is used for validation and Bout 115 is held out for final evaluation.

The retained final Bout 115 result is typed event precision 0.443, recall 0.453, F1
0.448, and typed macro-F1 0.257. `RESULTS.md` explains the result files included in this
clean code package.

```bash
nohup bash run_build.sh   > build.log    2>&1 &
nohup bash run_matched.sh > train.log    2>&1 &
nohup bash run_evaluate.sh > evaluate.log 2>&1 &
```
