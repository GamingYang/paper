# SOH Stage/Tail Bootstrap Diagnostic v1

## Scope and Protocol

This package is a no-retraining diagnostic over saved Batch 10 target_test SOH predictions. It recomputes metrics from `C:\Users\ROG\Desktop\paper\codex\visualization-update-20260420\teal_ci_visualization_summary_20260420.xlsx` and reads existing JSON metrics only as run provenance checks.

Protocol compliance: source-domain training remains Batch 1-8, target-domain development/adaptation remains Batch 9, and Batch 10 is used here only for blind-test reporting. No model selection, hyperparameter tuning, checkpoint selection, retraining, RUL, or EOL analysis is performed in this diagnostic.

## Diagnostic Definitions

Cycle stage uses per-cell min-max normalized cycle position: `(cycle - min_cycle) / (max_cycle - min_cycle)`. Stage bins are `early=[0,1/3]`, `mid=(1/3,2/3]`, and `late=(2/3,1]`.

SOH bins are fixed before analysis as: `y_true <= 0.85`, `0.85 < y_true <= 0.90`, `0.90 < y_true <= 0.95`, `0.95 < y_true <= 1.00`, and `y_true > 1.00`.

Tail cycles are defined as the last 20% per cell (`cycle_position_norm > 0.80`). Tail monotonicity uses a per-cell trailing rolling mean of predictions with window=21 cycles and `min_periods=1`; adjacent smoothed increases greater than 1e-06 count as monotonic violations. Signed bias is `mean(prediction - y_true)`, so positive values indicate overprediction.

The cell-level bootstrap uses 10000 resamples over the 8 Batch 10 cells. These are empirical resampling intervals for macro metrics, not calibrated predictive uncertainty.

## Key Numbers

| model               |     rmse |   macro_rmse_mean_of_cell_rmse |      mae |   signed_bias_pred_minus_true |   sample_count |   cell_count |
|:--------------------|---------:|-------------------------------:|---------:|------------------------------:|---------------:|-------------:|
| final-retained-v1   | 0.009499 |                       0.008301 | 0.005704 |                     -0.001316 |          14556 |            8 |
| no-physical-base-v1 | 0.020716 |                       0.019506 | 0.013759 |                      0.005719 |          14556 |            8 |
| no-mmd-v1           | 0.009677 |                       0.008721 | 0.005784 |                      0.000608 |          14556 |            8 |
| no-attention-v1     | 0.011677 |                       0.010535 | 0.008145 |                     -0.00148  |          14556 |            8 |

The final retained model has pooled Batch 10 RMSE 0.009499, macro RMSE 0.008301, and MAE 0.005704. Removing the low-order nonlinear degradation prior increases pooled RMSE by 0.011217. Removing MMD changes pooled RMSE by 0.000178, while removing attention changes it by 0.002177.

## Stage-Wise RMSE

| model               |    early |      mid |     late |
|:--------------------|---------:|---------:|---------:|
| final-retained-v1   | 0.00312  | 0.009111 | 0.013341 |
| no-physical-base-v1 | 0.008601 | 0.013933 | 0.031926 |
| no-mmd-v1           | 0.003379 | 0.00646  | 0.015093 |
| no-attention-v1     | 0.00676  | 0.009897 | 0.016291 |

## Tail Diagnostics

| model               |     rmse |      mae |   signed_bias_pred_minus_true |   max_abs_error |   monotonic_violation_count |   positive_predicted_slope_fraction |
|:--------------------|---------:|---------:|------------------------------:|----------------:|----------------------------:|------------------------------------:|
| final-retained-v1   | 0.013864 | 0.011054 |                      0.000142 |        0.073595 |                         679 |                            0.233735 |
| no-physical-base-v1 | 0.03662  | 0.028629 |                      0.02351  |        0.107718 |                         981 |                            0.337694 |
| no-mmd-v1           | 0.017501 | 0.013588 |                      0.007161 |        0.094996 |                         685 |                            0.2358   |
| no-attention-v1     | 0.018135 | 0.01513  |                      0.010529 |        0.047578 |                         663 |                            0.228227 |

In the last 20% of each cell, the final retained model has RMSE 0.013864, MAE 0.011054, and signed bias 0.000142. The no-physical-base ablation increases tail RMSE by 0.022757. This supports describing the cubic base only as a useful low-order nonlinear degradation prior that helps stabilize the tail; it should not be framed as a strong electrochemical mechanism.

## Cell Bootstrap

| model               |   observed_macro_rmse |   macro_rmse_p2_5 |   macro_rmse_p97_5 |   observed_macro_mae |   macro_mae_p2_5 |   macro_mae_p97_5 |
|:--------------------|----------------------:|------------------:|-------------------:|---------------------:|-----------------:|------------------:|
| final-retained-v1   |              0.008301 |          0.005816 |           0.011278 |             0.005519 |         0.003731 |          0.007955 |
| no-physical-base-v1 |              0.019506 |          0.013844 |           0.025825 |             0.013972 |         0.009847 |          0.018156 |
| no-mmd-v1           |              0.008721 |          0.006216 |           0.011543 |             0.005625 |         0.003976 |          0.007842 |
| no-attention-v1     |              0.010535 |          0.007964 |           0.013699 |             0.007829 |         0.005442 |          0.011112 |

For the final retained model, the empirical cell-bootstrap interval is 0.005816-0.011278 for macro RMSE and 0.003731-0.007955 for macro MAE.

## Manuscript Implications

The diagnostic strengthens the SOH-only ablation story in a restrained way: the final retained model is best overall and the physical-base removal remains the clearest degradation, including in the tail. The MMD effect is small and mixed relative to the final model, so it should be described as a lightweight cross-domain regularizing/enhancement component rather than a dominant source of improvement. Attention shows a clearer but secondary contribution.

Residual bands or intervals derived from these outputs should be called empirical. The bootstrap summarizes variability across the eight Batch 10 cells and does not establish calibrated predictive uncertainty.

## Figures

- `C:\Users\ROG\Desktop\paper\codex\soh-stage-tail-bootstrap-v1\figures\stage_wise_rmse_comparison.png`
- `C:\Users\ROG\Desktop\paper\codex\soh-stage-tail-bootstrap-v1\figures\tail_bias_late_stage_diagnostics.png`

## Output Tables

- `overall_metrics_recomputed`: pooled and macro Batch 10 metrics recomputed from saved predictions.
- `cycle_stage_metrics`: early/mid/late metrics by per-cell normalized cycle position.
- `soh_bin_metrics`: fixed y_true SOH-bin metrics.
- `tail_metrics`: last-20%-cycle metrics plus smoothed monotonicity diagnostics.
- `cell_bootstrap_ci`: empirical cell-level bootstrap intervals over the 8 Batch 10 cells.
- `final_vs_ablations_deltas`: ablation deltas relative to the final retained model.
