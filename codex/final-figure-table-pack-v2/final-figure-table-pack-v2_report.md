# final-figure-table-pack-v2

Generated: 2026-06-05T11:22:30

## Protocol Guard

- This package only regenerates paper-ready data-derived figures and tables from accepted CSV/JSON/MD evidence.
- It does not train models, tune hyperparameters, search features, choose horizons, or select by Batch 10.
- Batch 1-8 are source cells, Batch 9 is labeled target-development/adaptation and selection, and Batch 10 is final blind-test reporting only.
- Batch 10 plots report the locked strict baseline selected from Batch 9; they are not used to choose models.
- Target-label budget rows are sensitivity evidence only; low-error subsets are not promoted as selected models.
- Raw-provenance v3 is diagnostic feature-traceability evidence, not raw-only superiority.

## Figures

- `figure_protocol_flowchart_v2.png` / SVG / PDF
  - Source: authoritative mainline, strict protocol reports, and project split facts.
  - Safe interpretation: fixed supervised small labeled target-domain adaptation protocol.
  - Boundary: not unsupervised adaptation, not cross-dataset validation.

- `figure_batch9_candidate_horizon_model_selection_v2.png` / SVG / PDF
  - Source: `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict-eol-protocol-v1_candidate_validation_summary.csv`.
  - Safe interpretation: Batch9 validation selects source_plus_target_adapt_loocv + ExtraTrees + 1000 cycles.
  - Boundary: color scale is clipped for readability and contains no Batch10 ranking.

- `figure_batch10_predicted_vs_true_eol_v2.png` / SVG / PDF
  - Source: `C:\Users\ROG\Desktop\paper\codex\strict-eol-robustness-v1\strict-eol-robustness-v1_per_cell.csv` and `C:\Users\ROG\Desktop\paper\codex\strict-eol-robustness-v1\strict-eol-robustness-v1_summary.json`.
  - Safe interpretation: final Batch10 predicted-vs-true EOL for the Batch9-selected strict baseline.
  - Boundary: correlation and bias are over eight blind-test cells only.

- `figure_batch10_per_cell_abs_error_v2.png` / SVG / PDF
  - Source: `C:\Users\ROG\Desktop\paper\codex\strict-eol-robustness-v1\strict-eol-robustness-v1_per_cell.csv` and `C:\Users\ROG\Desktop\paper\codex\strict-eol-robustness-v1\strict-eol-robustness-v1_summary.json`.
  - Safe interpretation: per-cell absolute EOL/RUL error at H = 1000.
  - Boundary: small n means each cell strongly affects summary metrics.

- `figure_target_label_budget_sensitivity_v2.png` / SVG / PDF
  - Source: `C:\Users\ROG\Desktop\paper\codex\target-label-budget-sensitivity-v1\budget_aggregate_results.csv` and `C:\Users\ROG\Desktop\paper\codex\target-label-budget-sensitivity-v1\budget_subset_results.csv`.
  - Safe interpretation: fixed ExtraTrees 1000-cycle sensitivity to predefined labeled Batch9 budgets.
  - Boundary: no subset is selected by Batch10; k=8 reproduces the strict main result.

- `figure_raw_provenance_v3_summary.png` / SVG / PDF
  - Source: `C:\Users\ROG\Desktop\paper\codex\strict-raw-provenance-v3\selected_raw_features_v3.csv`, `C:\Users\ROG\Desktop\paper\codex\strict-raw-provenance-v3\raw_feature_audit_v3.csv`, and `C:\Users\ROG\Desktop\paper\codex\strict-raw-provenance-v3\raw_model_results_v3.csv`.
  - Safe interpretation: selected raw v3 model uses 20 capacity-family raw features computed from observed cycles only.
  - Boundary: provenance/availability diagnostic, not a replacement for the strict main baseline.

- `figure_evidence_hierarchy_v2.png` / SVG / PDF
  - Source: strict protocol/robustness, target-label budget, raw-provenance v3, neural v1, residual ablation, and conformal uncertainty summaries.
  - Safe interpretation: separates formal main evidence, comparator, diagnostics, prototype, negative result, and limited uncertainty.
  - Boundary: no SOTA, neural-superiority, residual-improvement, raw-only-superiority, or strong-uncertainty claim.

## Tables

- `table_main_results_v2.csv` / `table_main_results_v2.md`
  - Sources: strict protocol and robustness, raw-provenance v3, neural v1, residual ablation, conformal uncertainty.
  - Safe interpretation: compact evidence hierarchy for manuscript main table or supplementary summary.

- `table_target_label_budget.csv` / `table_target_label_budget.md`
  - Sources: budget aggregate CSV and subset CSV.
  - Safe interpretation: mean/median/IQR/range summarize all predefined subsets at each labeled Batch9 budget.

- `table_raw_provenance_v3.csv` / `table_raw_provenance_v3.md`
  - Sources: raw v3 selected features, feature audit, model results, and summary JSON.
  - Safe interpretation: raw feature family, forbidden-input audit, and Batch9-selected diagnostic summary.

## Key Numerical Snapshot

- Strict main result: Batch10 RMSE/MAE = 56.19/49.42 cycles, selected by Batch9 only.
- Source-only comparator: Batch10 RMSE/MAE = 64.27/55.21 cycles.
- Target-label budget mean RMSE: k=0/1/2/4/8 gives 64.27/61.17/60.55/59.64/56.19 cycles.
- Raw-provenance v3: corr_capacity_top20 + SVR-RBF gives Batch10 RMSE/MAE = 74.12/63.16 cycles.
- Neural v1 prototype: Batch10 RMSE/MAE = 68.68/60.30 cycles.
- Residual calibration negative result: Batch10 RMSE/MAE = 100.26/92.90 cycles.
- Conformal uncertainty diagnostic: finite 80% half-width = 129.42 cycles with 8/8 Batch10 coverage; 90% and 95% standard intervals are unbounded with n_cal=8.

## Main Table Snapshot

evidence_level                          result_line                                            source_artifact                    protocol                                    model_or_strategy horizon_cycles batch9_rmse_cycles batch9_mae_cycles batch10_rmse_cycles batch10_mae_cycles n_test_cells                           uncertainty_or_diagnostic_metric                                                paper_role                                                                                                  safe_interpretation                                                        do_not_overclaim
   Formal main                 Strict main baseline           strict-eol-protocol-v1; strict-eol-robustness-v1 Source + target-adapt LOOCV          ExtraTrees, strict processed fused features           1000              75.26             61.96               56.19              49.42            8                                                                           Current main quantitative proof-of-concept                                         Batch9-selected strict baseline; Batch10 is final blind-test reporting only.  Not final proposed architecture; supervised labeled Batch9 adaptation.
    Comparator        Strict source-only comparator strict-eol-protocol-v1; target-label-budget-sensitivity-v1                 Source only                     ExtraTrees, no Batch9 adaptation           1000              82.59             68.68               64.27              55.21            8                                                                             Comparator for labeled target-adapt gain                                                           Shows the fixed model without labeled Batch9 target cells.                                    Not a globally selected main result.
    Diagnostic         Raw-provenance v3 diagnostic                                   strict-raw-provenance-v3 Source + target-adapt LOOCV                        corr_capacity_top20 + SVR-RBF           1000              29.87             21.66               74.12              63.16            8                   20 selected capacity-family raw features Raw-feature provenance and online-availability diagnostic                Direct raw our_data/*.pkl features support traceability, not superiority over the strict main result. Does not beat strict ExtraTrees Batch10 RMSE; not raw-only superiority.
     Prototype                  Neural v1 prototype                                        neural-eol-model-v1 Source + target-adapt LOOCV                                          Neural QNet           1000              22.50             18.61               68.68              60.30            8                                                                                  Physics-consistent neural prototype                                      Useful prototype, but weaker than the strict ExtraTrees main result on Batch10.                                            No neural superiority claim.
      Negative Residual calibration negative result                                   eol-ablation-residual-v1       Residual nested LOOCV             shrink_to_target_mean_w075 on ExtraTrees           1000              60.60             47.29              100.26              92.90            8                                                                                       Negative/risk-control evidence                   Batch9 improvement does not transfer; Batch10 worsens, so the residual correction is not retained.                                          No residual-improvement claim.
   Uncertainty               Uncertainty diagnostic                        strict-eol-conformal-uncertainty-v1 Source + target-adapt LOOCV Fixed strict ExtraTrees + split conformal diagnostic           1000              75.26             61.96               56.19              49.42            8 80% finite interval half-width 129.42 cycles; coverage 8/8                            Limited uncertainty diagnostic Finite 80% interval only; 90% and 95% standard split-conformal intervals are unbounded with eight calibration cells.                                 No strong calibrated-uncertainty claim.

## Remaining Limits

- Batch10 contains only eight cells.
- The main strict result uses historical processed capacity/SOH-like features and should be worded as measured/estimated historical capacity-SOH information.
- Raw v3 uses direct raw pkl features and helps provenance defense, but its Batch10 error is higher than the strict main baseline.
- Uncertainty evidence is limited to diagnostic coverage/interval reporting, not strong calibrated uncertainty.
