# Changes

- Added direct feature extraction from synchronized raw views and canonical bbox JSONL.
- Added 8-frame consensus-peak labels including explicit empty proposals.
- Added cached temporal VideoMAE features with variable-view masks.
- Added mean-fusion and attention-fusion baselines on the same features.
- Added structured per-fighter activity/outcome/target supervision.
- Added proposal-level and end-to-end typed-event metrics.

No existing clips, manifests, checkpoints, probabilities, or windows are modified.

## 2026-07-28 Follow-up

- Fixed a critical Transformers 5.12 compatibility bug: `VideoMAEModel.from_pretrained`
  did not load the prefixed encoder weights. The builder now loads the Kinetics wrapper,
  extracts `.videomae`, migrates legacy `q_bias`/`v_bias`, zeros key bias, and verifies
  checkpoint equality before feature extraction.
- Added synchronized 8-frame feature datasets for global joint panels, separate full-size
  fighter crops, Stage-2 union crops, and spatially pooled fighter tokens.
- Added one-to-one per-fighter GT matching and categorical null/type models.
- Added class-weight sensitivity runs and 8-frame versus 32-frame context checks.
- Added clean GT-anchor feature construction and a disentangled activity/type experiment.
- Added a partial VideoMAE fine-tuning implementation for clean GT anchors. The Stage-2
  input run was rejected; the canonical-panel run overloaded server I/O before a result
  could be confirmed.

## 2026-07-28 Final Follow-up

- Sequentially cached 8,195 clean-GT panels and completed canonical-panel partial
  fine-tuning. It reached clean-GT Bout 115 macro-F1 0.111 and was rejected.
- A proposal-panel cache was then built sequentially in the isolated
  `stage5_hierarchical_finetune_20260728` branch: 2,438 events and 8,337 views.
- Last-block proposal fine-tuning was stopped on validation after three monotonically
  worsening epochs; Bout 115 was not used to make the stopping decision.
- SSV2, Kinetics+SSV2 concatenation, hierarchical labels, and validation-selected late
  fusion were tested. None exceeded the retained 0.444 typed F1 and 0.254 macro-F1.
