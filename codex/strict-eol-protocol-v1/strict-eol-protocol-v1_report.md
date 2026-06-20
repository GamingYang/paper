# Strict EOL Protocol V1

- Finished at: 2026-06-03T16:49:37
- Elapsed seconds: 218.4
- Data dir: `C:\Users\ROG\Desktop\paper\PYthon训练代码\processed_clean`

## Protocol

- Source domain: Batch 1-8.
- Target development/adaptation set: Batch 9.
- Blind test set: Batch 10.
- Candidate model and horizon selection uses Batch 9 only.
- `source_plus_target_adapt_loocv` uses leave-one-Batch-9-cell-out validation, so the held-out target cell is not used for supervised adaptation.
- Batch 10 is evaluated only after Batch-9-based selection; candidate rows are not sorted by Batch 10.

## Global Selection

| Scope | Protocol | Model | Horizon | Batch 9 RMSE | Batch 10 RMSE | Batch 10 MAE |
|---|---|---|---:|---:|---:|---:|
| global_best_by_batch9 | source_plus_target_adapt_loocv | extra_trees | 1000 | 75.26 | 56.19 | 49.42 |

## Selected Baselines

Each row below selected its horizon by Batch 9 validation before Batch 10 evaluation.

| Scope | Protocol | Model | Horizon | Batch 9 RMSE | Batch 10 RMSE | Batch 10 MAE |
|---|---|---|---:|---:|---:|---:|
| best_horizon_for_source_only_elastic_net | source_only | elastic_net | 300 | 99.86 | 250.68 | 196.57 |
| best_horizon_for_source_only_extra_trees | source_only | extra_trees | 1000 | 82.59 | 64.27 | 55.21 |
| best_horizon_for_source_only_gbr | source_only | gbr | 800 | 99.60 | 141.04 | 85.89 |
| best_horizon_for_source_only_kernel_ridge_rbf | source_only | kernel_ridge_rbf | 50 | 1790.95 | 1503.65 | 1465.96 |
| best_horizon_for_source_only_random_forest | source_only | random_forest | 1000 | 92.85 | 65.99 | 58.09 |
| best_horizon_for_source_only_ridge | source_only | ridge | 1000 | 127.35 | 171.55 | 141.35 |
| best_horizon_for_source_only_source_mean_eol | source_only | source_mean_eol | 1000 | 220.64 | 251.78 | 203.97 |
| best_horizon_for_source_only_source_median_eol | source_only | source_median_eol | 100 | 243.03 | 244.92 | 188.25 |
| best_horizon_for_source_only_svr_rbf | source_only | svr_rbf | 500 | 131.12 | 102.34 | 67.07 |
| best_horizon_for_source_plus_target_adapt_loocv_elastic_net | source_plus_target_adapt_loocv | elastic_net | 1000 | 89.32 | 210.56 | 157.69 |
| best_horizon_for_source_plus_target_adapt_loocv_extra_trees | source_plus_target_adapt_loocv | extra_trees | 1000 | 75.26 | 56.19 | 49.42 |
| best_horizon_for_source_plus_target_adapt_loocv_gbr | source_plus_target_adapt_loocv | gbr | 500 | 92.82 | 82.78 | 70.34 |
| best_horizon_for_source_plus_target_adapt_loocv_kernel_ridge_rbf | source_plus_target_adapt_loocv | kernel_ridge_rbf | 50 | 1644.35 | 1443.19 | 1402.45 |
| best_horizon_for_source_plus_target_adapt_loocv_random_forest | source_plus_target_adapt_loocv | random_forest | 1000 | 81.75 | 58.49 | 54.52 |
| best_horizon_for_source_plus_target_adapt_loocv_ridge | source_plus_target_adapt_loocv | ridge | 1000 | 108.81 | 149.35 | 125.38 |
| best_horizon_for_source_plus_target_adapt_loocv_source_mean_eol | source_plus_target_adapt_loocv | source_mean_eol | 100 | 212.40 | 257.13 | 212.63 |
| best_horizon_for_source_plus_target_adapt_loocv_source_median_eol | source_plus_target_adapt_loocv | source_median_eol | 200 | 217.96 | 256.54 | 211.75 |
| best_horizon_for_source_plus_target_adapt_loocv_svr_rbf | source_plus_target_adapt_loocv | svr_rbf | 800 | 79.57 | 96.96 | 79.08 |
| global_best_by_batch9 | source_plus_target_adapt_loocv | extra_trees | 1000 | 75.26 | 56.19 | 49.42 |

## Batch 9 Candidate Ranking

This table is validation-only and is safe for model/horizon selection.

| Protocol | Model | Horizon | Batch 9 RMSE | Batch 9 MAE | Selection method |
|---|---|---:|---:|---:|---|
| source_plus_target_adapt_loocv | extra_trees | 1000 | 75.26 | 61.96 | batch9_leave_one_cell_out |
| source_plus_target_adapt_loocv | svr_rbf | 800 | 79.57 | 53.07 | batch9_leave_one_cell_out |
| source_plus_target_adapt_loocv | random_forest | 1000 | 81.75 | 61.09 | batch9_leave_one_cell_out |
| source_only | extra_trees | 1000 | 82.59 | 68.68 | direct_batch9_eval |
| source_plus_target_adapt_loocv | random_forest | 500 | 83.20 | 72.26 | batch9_leave_one_cell_out |
| source_plus_target_adapt_loocv | elastic_net | 1000 | 89.32 | 71.83 | batch9_leave_one_cell_out |
| source_plus_target_adapt_loocv | extra_trees | 800 | 91.08 | 78.14 | batch9_leave_one_cell_out |
| source_plus_target_adapt_loocv | gbr | 500 | 92.82 | 84.20 | batch9_leave_one_cell_out |
| source_only | random_forest | 1000 | 92.85 | 76.43 | direct_batch9_eval |
| source_plus_target_adapt_loocv | random_forest | 800 | 93.26 | 81.75 | batch9_leave_one_cell_out |
| source_plus_target_adapt_loocv | elastic_net | 800 | 95.03 | 79.95 | batch9_leave_one_cell_out |
| source_plus_target_adapt_loocv | elastic_net | 300 | 99.10 | 60.95 | batch9_leave_one_cell_out |
| source_only | gbr | 800 | 99.60 | 83.79 | direct_batch9_eval |
| source_plus_target_adapt_loocv | gbr | 800 | 99.73 | 84.49 | batch9_leave_one_cell_out |
| source_only | elastic_net | 300 | 99.86 | 69.73 | direct_batch9_eval |
| source_only | extra_trees | 800 | 100.07 | 81.80 | direct_batch9_eval |
| source_plus_target_adapt_loocv | gbr | 1000 | 102.07 | 76.51 | batch9_leave_one_cell_out |
| source_only | gbr | 1000 | 102.19 | 78.85 | direct_batch9_eval |
| source_plus_target_adapt_loocv | ridge | 1000 | 108.81 | 77.08 | batch9_leave_one_cell_out |
| source_plus_target_adapt_loocv | elastic_net | 500 | 110.09 | 85.38 | batch9_leave_one_cell_out |

## Interpretation

- The reported EOL RMSE in cycles is numerically the same as RUL RMSE at a fixed observation cycle when RUL is derived as predicted EOL minus current cycle.
- This run is stricter than the earlier EOL fingerprint pilot because Batch 10 is not used for selecting the model or horizon.
- `y_nominal` features must be described as historical measured/estimated capacity-SOH information. A later raw-capacity or `dq` feature audit is still needed.
- The feature audit covers every evaluated horizon and split.
- The ExtraTrees/GBR family remains a strong baseline. It is not yet the final publishable architecture.

## Output Files

- Candidate validation table: `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict-eol-protocol-v1_candidate_validation_summary.csv`
- Selected Batch 10 table: `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict-eol-protocol-v1_selected_test_summary.csv`
- Selected predictions: `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict-eol-protocol-v1_selected_test_predictions.csv`
- Feature audit: `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict-eol-protocol-v1_feature_audit.csv`
- JSON summary: `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict-eol-protocol-v1_summary.json`
