# leakage-negative-control-v1

- Finished at: 2026-06-06T19:00:57.303275
- Elapsed seconds: 132.56
- Purpose: supplementary feature-family / horizon / leakage negative-control audit.
- Protocol: Batch 1-8 source, Batch 9 validation/adaptation, Batch 10 report-only after Batch9 selection.
- This is not a new model-selection run for the main paper; it is reviewer-defense material.

## Valid Feature-Family Selection by Batch 9

| Family | Selected horizon | Features | Batch9 RMSE | Batch9 MAE | Batch10 RMSE report-only | Batch10 MAE report-only |
|---|---:|---:|---:|---:|---:|---:|
| strict_fused_all | 1000 | 108 | 75.26 | 61.96 | 56.19 | 49.42 |
| soh_capacity_only | 800 | 59 | 86.76 | 70.24 | 138.56 | 88.31 |
| x_curve_only | 1000 | 49 | 87.46 | 84.34 | 199.74 | 167.79 |

## Leakage Blacklist Check

- Forbidden inputs: `batch_id, battery_index, cell_id, eol_cycle, full_cell_length, horizon, last_observed_cycle, n_cycles_total, n_observed, observed_ratio, rul_cycle, split`.
- Intersection between forbidden inputs and strict model columns at 1000 cycles: `[]`.
- Direct endpoint, RUL, full life length, observed-ratio, horizon, observed-count, cell id and batch id fields are not allowed in the valid feature families.

## Invalid Leakage Negative Control

The invalid negative control injects fields derived from true EOL, including `invalid_observed_ratio_true_eol = horizon / eol_cycle` and `invalid_remaining_cycles_at_horizon = eol_cycle - horizon`. These fields are deliberately invalid and must never be used as model inputs. Their purpose is to demonstrate why the blacklist is necessary.

| Horizon | Invalid Batch9 RMSE | Invalid Batch10 RMSE report-only |
|---:|---:|---:|
| 50 | 0.0002 | 0.0004 |
| 100 | 0.0199 | 0.0097 |
| 200 | 0.3714 | 0.6056 |
| 300 | 0.3556 | 0.4163 |
| 500 | 0.4326 | 0.2336 |
| 800 | 0.5717 | 0.9564 |
| 1000 | 2.0148 | 0.3158 |

## Manuscript-Safe Interpretation

- Safe: report valid feature-family and horizon sensitivity as Batch9 validation-only evidence.
- Safe: state that endpoint-derived fields collapse the task and are therefore explicitly blacklisted.
- Unsafe: use invalid leakage negative-control numbers as performance evidence.
- Unsafe: select any horizon or feature family by Batch10 report-only metrics.

## Output Files

- `leakage-negative-control-v1_horizon_family_leakage_stress_results.csv`
- `leakage-negative-control-v1_selected_by_batch9_valid_families.csv`
- `leakage-negative-control-v1_feature_family_used_features.csv`
- plot files: `['leakage-negative-control-v1_horizon_feature_family_batch9_rmse.png', 'leakage-negative-control-v1_invalid_leakage_negative_control.png']`
- `leakage-negative-control-v1_summary.json`