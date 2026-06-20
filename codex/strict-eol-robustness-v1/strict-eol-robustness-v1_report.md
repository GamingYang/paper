# Strict EOL Robustness V1

Generated: 2026-06-04T12:53:36

## Protocol Guard

- Fixed split: Batch 1-8 source, Batch 9 target adaptation/model selection, Batch 10 blind test.
- This script does not train, tune, or reselect a model. It only reports final blind-test statistics for the existing `strict-eol-protocol-v1` predictions.
- The selected candidate is recovered from `strict-eol-protocol-v1_candidate_validation_summary.csv`, which contains Batch 9 validation metrics only.
- Batch 10 predictions are filtered to `selection_scope == global_best_by_batch9`; they are not sorted or used to choose a model, horizon, feature set, or parameter.
- Selected Batch 9 path: `source_plus_target_adapt_loocv` + `extra_trees` + 1000 observed cycles, selected by `batch9_leave_one_cell_out` with Batch 9 RMSE 75.26 cycles.

Top Batch 9 validation rows:

| protocol                       | model         |   horizon | selection_set   | selection_method          |   selection_eol_rmse |   selection_eol_mae |
|:-------------------------------|:--------------|----------:|:----------------|:--------------------------|---------------------:|--------------------:|
| source_plus_target_adapt_loocv | extra_trees   |      1000 | target_adapt    | batch9_leave_one_cell_out |                75.26 |               61.96 |
| source_plus_target_adapt_loocv | svr_rbf       |       800 | target_adapt    | batch9_leave_one_cell_out |                79.57 |               53.07 |
| source_plus_target_adapt_loocv | random_forest |      1000 | target_adapt    | batch9_leave_one_cell_out |                81.75 |               61.09 |
| source_only                    | extra_trees   |      1000 | target_adapt    | direct_batch9_eval        |                82.59 |               68.68 |
| source_plus_target_adapt_loocv | random_forest |       500 | target_adapt    | batch9_leave_one_cell_out |                83.2  |               72.26 |
| source_plus_target_adapt_loocv | elastic_net   |      1000 | target_adapt    | batch9_leave_one_cell_out |                89.32 |               71.83 |
| source_plus_target_adapt_loocv | extra_trees   |       800 | target_adapt    | batch9_leave_one_cell_out |                91.08 |               78.14 |
| source_plus_target_adapt_loocv | gbr           |       500 | target_adapt    | batch9_leave_one_cell_out |                92.82 |               84.2  |

## Main Metrics

Metrics use the eight Batch 10 blind-test cells. At a fixed observed horizon, EOL error and derived RUL error are numerically identical.

| Metric | Value |
|---|---:|
| n cells | 8 |
| RMSE | 56.19 cycles |
| MAE | 49.42 cycles |
| Median absolute error | 47.36 cycles |
| Max absolute error | 93.21 cycles |
| Bias, mean signed error | 38.23 cycles |
| Pearson true-vs-pred | 0.990 |
| Spearman true-vs-pred | 0.976 |

## Bootstrap CI

Cell-level bootstrap resampling was performed on the eight Batch 10 cells with replacement. This is final-result uncertainty reporting, not model selection.

| Metric | Resamples | 2.5% | Median | 97.5% |
|---|---:|---:|---:|---:|
| RMSE | 20000 | 36.84 | 55.68 | 72.61 |
| MAE | 20000 | 31.35 | 49.33 | 68.10 |

## Per-cell Error

| cell_id   |   true_eol_cycle |   pred_eol_cycle |   eol_error_cycles |   abs_eol_error_cycles |   true_rul_at_horizon_cycles |   pred_rul_at_horizon_cycles |
|:----------|-----------------:|-----------------:|-------------------:|-----------------------:|-----------------------------:|-----------------------------:|
| 10-1      |             1702 |          1785.01 |              83.01 |                  83.01 |                          702 |                       785.01 |
| 10-2      |             1697 |          1726.96 |              29.96 |                  29.96 |                          697 |                       726.96 |
| 10-3      |             1848 |          1897.98 |              49.98 |                  49.98 |                          848 |                       897.98 |
| 10-4      |             1811 |          1904.21 |              93.21 |                  93.21 |                          811 |                       904.21 |
| 10-5      |             2030 |          2086.82 |              56.82 |                  56.82 |                         1030 |                      1086.82 |
| 10-6      |             2285 |          2316.89 |              31.89 |                  31.89 |                         1285 |                      1316.89 |
| 10-7      |             1783 |          1788.73 |               5.73 |                   5.73 |                          783 |                       788.73 |
| 10-8      |             1400 |          1355.26 |             -44.74 |                  44.74 |                          400 |                       355.26 |

Largest absolute error: 10-4 with 93.21 cycles. The blind-test result is not dominated by a single catastrophic cell, but the small n means each cell has visible influence.

Figures:

- `strict-eol-robustness-v1_predicted_vs_true_eol.png`
- `strict-eol-robustness-v1_per_cell_abs_error.png`
- `strict-eol-robustness-v1_bootstrap_rmse_distribution.png`

## Sensitivity

Leave-one-cell-out recomputation removes one Batch 10 cell at a time after the final prediction table is fixed.

| left_out_cell_id   |   left_out_abs_error_cycles |   rmse_cycles |   mae_cycles |   delta_rmse_vs_full_cycles |
|:-------------------|----------------------------:|--------------:|-------------:|----------------------------:|
| 10-1               |                       83.01 |         51.22 |        44.62 |                       -4.97 |
| 10-2               |                       29.96 |         58.99 |        52.2  |                        2.8  |
| 10-3               |                       49.98 |         57.02 |        49.34 |                        0.83 |
| 10-4               |                       93.21 |         48.65 |        43.16 |                       -7.54 |
| 10-5               |                       56.82 |         56.09 |        48.36 |                       -0.09 |
| 10-6               |                       31.89 |         58.84 |        51.92 |                        2.66 |
| 10-7               |                        5.73 |         60.03 |        55.66 |                        3.84 |
| 10-8               |                       44.74 |         57.64 |        50.09 |                        1.45 |

LOO RMSE range: 48.65 to 60.03 cycles. LOO MAE range: 43.16 to 55.66 cycles.

## Paper-ready Wording

Under the fixed cross-batch protocol, Batch 1-8 were used as the source domain, Batch 9 was used for target-domain development and Batch-9-only model/horizon selection, and Batch 10 was reserved as the blind target test. The Batch-9-selected ExtraTrees baseline using the first 1000 observed cycles achieved an EOL/RUL RMSE of 56.19 cycles and MAE of 49.42 cycles on the eight Batch 10 cells. Cell-level bootstrap resampling gave 95% confidence intervals of 36.84-72.61 cycles for RMSE and 31.35-68.10 cycles for MAE. Leave-one-cell-out analysis yielded RMSE values from 48.65 to 60.03 cycles, indicating that the reported blind-test performance is not explained by a single removed cell.

## Limitations

- Batch 10 contains only eight cells, so the bootstrap interval is wide and should be framed as cell-level test-set uncertainty.
- The bootstrap and LOO analyses do not measure retraining variability; they reuse the already selected final Batch 10 predictions.
- This remains a strict baseline/proof-of-concept, not the final proposed architecture.
- The protocol uses labeled Batch 9 target EOL values for supervised small-target adaptation and should not be described as unsupervised domain adaptation.
- The upstream strict baseline uses processed historical capacity/SOH features, including `y_nominal`; these must be framed as historical measured or estimated capacity/SOH information until a stricter raw-capacity or `dq` variant is promoted.
