# Stage 5 Multi-View Structured Classifier

This experiment consumes the robust synchronized Stage 4 peaks and compares three
models on exactly the same cached VideoMAE features:

1. `mean_direct`: temporal/view mean fusion and a direct eight-label head.
2. `attention_direct`: peak-aware temporal attention and masked cross-view attention.
3. `attention_structured`: the attention model plus fighter activity, outcome, and
   head/body auxiliary heads with probability consistency.

Eight unique frames are extracted around each peak. Empty proposals are retained as
negative examples. The protocol is train 116/117/120/121, validation 122, and held-out
test 115.

```bash
nohup bash Zihui/stage5_multiview_structured_20260728/run_build.sh \
  > Zihui/stage5_multiview_structured_20260728/build.log 2>&1 &

nohup bash Zihui/stage5_multiview_structured_20260728/run_train.sh \
  > Zihui/stage5_multiview_structured_20260728/train.log 2>&1 &
```
