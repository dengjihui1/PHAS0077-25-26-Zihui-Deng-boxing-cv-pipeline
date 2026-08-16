# Report analysis

`build_report_package.py` generated the figures and tables used in the dissertation
Results section (Stage 4 precision/recall movement, view-count ablation, temporal
tolerance, Stage 5 outcome-family F1 and class support). It reads Stage-3 probabilities,
Stage-4 windows and Stage-5 predictions from the server project tree.

> The script still points at the original server folder names
> (`stage4_multiview_consensus_20260727`, `stage5_multiview_structured_20260728`, …), so it
> is kept for provenance of the report's figure/table logic rather than as a directly
> runnable script inside this submission.

The CSVs here are the numeric tables behind those figures:

- `stage4_view_count_summary.csv` — Bout 115 F1 by number of available views.
- `stage4_temporal_tolerance_curve.csv` — F1 versus matching radius.
- `stage4_fusion_component_ablation.csv`, `stage4_leave_one_view_out.csv`,
  `stage4_proposal_density_by_bout.csv` — further Stage 4 diagnostics.
- `stage5_outcome_family_summary.csv` — blue+red pooled outcome F1 (final retained run).
- `stage5_class_support_vs_f1.csv` — per-class support and F1 (final retained run).

The Stage 5 CSVs match `../stage5_fighter_query_final/results/final_retained_result.json`.
