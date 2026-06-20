# Strict Raw-Feature Provenance V3

- Finished at: 2026-06-04T22:03:54
- Elapsed seconds: 661.6
- Raw data directory: `C:\Users\ROG\Desktop\paper\our_data`
- Fixed observation horizon: 1000 cycles
- Feature source: direct `our_data/*.pkl` parse. The script does not load processed `clean_*_x.npy`, `clean_*_meta.csv`, or `y_nominal` as model inputs.

## Protocol

- Batch 1-8 are source cells.
- Batch 9 is the labeled target development/adaptation set and the only set used for model/feature-strategy selection.
- Batch 10 is blind target test and is evaluated only for the global Batch9-selected configuration.
- The 1000-cycle horizon is fixed as a provenance diagnostic aligned with the current strict 1000-cycle main result; it is not selected using Batch10.
- EOL/RUL labels, full cell length, observed ratio, identifiers, horizon, observed-count fields, and future cycles are excluded from model inputs.

## Selected Batch9 Configuration

| Strategy | Features | Model | Protocol | Batch9 RMSE/MAE | Batch10 RMSE/MAE | Beats 56.19? |
|---|---:|---|---|---:|---:|---|
| corr_capacity_top20 | 20 | svr_rbf | source_plus_target_adapt_loocv | 29.87/21.66 | 74.12/63.16 | no |

## Interpretation

- The selected raw-only diagnostic is weaker than the 56.19-cycle strict processed-feature reference; it is provenance evidence, not a replacement result.
- RUL is derived after EOL prediction as `predicted_EOL - 1000`; therefore EOL and RUL errors are numerically identical at this fixed horizon.
- Batch9 labels make this supervised small labeled target-domain adaptation, not unsupervised domain adaptation.

## Raw Feature Provenance

| Family | Candidate features | Selected features | Raw source |
|---|---:|---:|---|
| capacity | 225 | 20 | raw Capacity (mAh) summaries/trends from observed cycle records |
| dq | 45 | 0 | raw cell['dq'] sequence summaries/trends if present |
| time_status | 225 | 0 | raw Time (s) duration and Status-phase summaries from observed records |
| voltage_current | 315 | 0 | raw Voltage (V), Current (mA), and capacity-voltage summaries from observed records |

## Batch9 Candidate Ranking

| Rank | Strategy | Model | Protocol | Batch9 RMSE | Batch9 MAE | Features |
|---:|---|---|---|---:|---:|---:|
| 1 | corr_capacity_top20 | svr_rbf | source_plus_target_adapt_loocv | 29.87 | 21.66 | 20 |
| 2 | corr_fused_top20 | svr_rbf | source_plus_target_adapt_loocv | 30.94 | 25.56 | 20 |
| 3 | corr_fused_top50 | svr_rbf | source_plus_target_adapt_loocv | 40.08 | 36.34 | 50 |
| 4 | corr_fused_top20 | svr_rbf | source_only | 40.38 | 32.62 | 20 |
| 5 | corr_fused_top100 | svr_rbf | source_plus_target_adapt_loocv | 40.89 | 38.62 | 100 |
| 6 | corr_dq_top20 | svr_rbf | source_plus_target_adapt_loocv | 41.46 | 37.42 | 20 |
| 7 | corr_time_status_top20 | svr_rbf | source_plus_target_adapt_loocv | 45.68 | 37.17 | 20 |
| 8 | corr_capacity_top20 | svr_rbf | source_only | 46.06 | 43.48 | 20 |
| 9 | corr_capacity_top20 | random_forest | source_only | 54.18 | 45.25 | 20 |
| 10 | corr_fused_top20 | random_forest | source_only | 55.70 | 48.86 | 20 |
| 11 | corr_time_status_top20 | svr_rbf | source_only | 56.72 | 46.81 | 20 |
| 12 | corr_capacity_top20 | extra_trees | source_only | 57.25 | 44.61 | 20 |
| 13 | family_fused | random_forest | source_plus_target_adapt_loocv | 57.55 | 52.44 | 810 |
| 14 | corr_fused_top20 | extra_trees | source_plus_target_adapt_loocv | 58.31 | 49.24 | 20 |
| 15 | corr_fused_top50 | random_forest | source_plus_target_adapt_loocv | 59.38 | 47.40 | 50 |
| 16 | corr_fused_top20 | random_forest | source_plus_target_adapt_loocv | 61.41 | 49.95 | 20 |
| 17 | corr_fused_top20 | elastic_net | source_plus_target_adapt_loocv | 63.27 | 52.37 | 20 |
| 18 | corr_fused_top20 | extra_trees | source_only | 63.70 | 53.55 | 20 |
| 19 | corr_fused_top50 | svr_rbf | source_only | 64.02 | 58.82 | 50 |
| 20 | corr_fused_top100 | svr_rbf | source_only | 64.11 | 56.72 | 100 |
| 21 | family_time_status_only | random_forest | source_plus_target_adapt_loocv | 64.59 | 55.37 | 225 |
| 22 | corr_time_status_top20 | ridge | source_plus_target_adapt_loocv | 65.41 | 57.12 | 20 |
| 23 | corr_capacity_top20 | random_forest | source_plus_target_adapt_loocv | 65.60 | 55.37 | 20 |
| 24 | corr_dq_top20 | svr_rbf | source_only | 66.46 | 59.62 | 20 |
| 25 | corr_time_status_top20 | ridge | source_only | 66.73 | 59.03 | 20 |
| 26 | family_time_status_only | random_forest | source_only | 67.59 | 54.88 | 225 |
| 27 | corr_fused_top100 | elastic_net | source_plus_target_adapt_loocv | 67.89 | 51.94 | 100 |
| 28 | corr_capacity_top20 | elastic_net | source_plus_target_adapt_loocv | 68.72 | 60.84 | 20 |
| 29 | corr_capacity_top20 | extra_trees | source_plus_target_adapt_loocv | 68.79 | 53.93 | 20 |
| 30 | corr_fused_top20 | ridge | source_plus_target_adapt_loocv | 71.69 | 61.02 | 20 |

## Selected Raw Features

| Rank | Feature | Family | Source note |
|---:|---|---|---|
| 1 | `discharge_capacity_span_mAh_log_var` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 2 | `cap_span_mAh_log_var` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 3 | `charge_capacity_span_mAh_log_var` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 4 | `cap_max_mAh_log_var` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 5 | `cap_span_mAh_slope_last_20` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 6 | `discharge_capacity_span_mAh_slope_last_20` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 7 | `cap_max_mAh_slope_last_20` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 8 | `charge_capacity_span_mAh_slope_last_20` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 9 | `cap_span_mAh_slope_last_50` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 10 | `discharge_capacity_span_mAh_slope_last_50` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 11 | `charge_capacity_span_mAh_slope_last_50` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 12 | `cap_max_mAh_slope_last_50` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 13 | `cap_max_mAh_drop_last_50` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 14 | `charge_capacity_span_mAh_drop_last_50` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 15 | `cap_span_mAh_slope_last_100` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 16 | `discharge_capacity_span_mAh_slope_last_100` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 17 | `charge_capacity_span_mAh_slope_last_100` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 18 | `cap_max_mAh_slope_last_100` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 19 | `charge_capacity_span_mAh_drop_last_100` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |
| 20 | `cap_max_mAh_drop_last_100` | capacity | raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only |

## Output Files

- Script: `C:\Users\ROG\Desktop\paper\codex\strict-raw-provenance-v3\run_strict_raw_provenance_v3.py`
- Report: `C:\Users\ROG\Desktop\paper\codex\strict-raw-provenance-v3\strict-raw-provenance-v3_report.md`
- Summary JSON: `C:\Users\ROG\Desktop\paper\codex\strict-raw-provenance-v3\strict-raw-provenance-v3_summary.json`
- Feature audit CSV: `C:\Users\ROG\Desktop\paper\codex\strict-raw-provenance-v3\raw_feature_audit_v3.csv`
- Model results CSV: `C:\Users\ROG\Desktop\paper\codex\strict-raw-provenance-v3\raw_model_results_v3.csv`
- Selected raw features CSV: `C:\Users\ROG\Desktop\paper\codex\strict-raw-provenance-v3\selected_raw_features_v3.csv`
