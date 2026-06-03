# RUL baseline suite v1

- Finished at: 2026-05-28T21:15:19
- Elapsed seconds: 114.0
- Data dir: `C:\Users\ROG\Desktop\paper\PYthon训练代码\processed_clean`

## Result Summary

| Model | Protocol | Dev RMSE | Test RMSE | Test MAE | Test Macro RMSE |
|---|---|---:|---:|---:|---:|
| extra_trees_direct_rul_source_plus_full_target_adapt | source_plus_full_labeled_target_adapt_training | 12.72 | 223.34 | 175.84 | 201.51 |
| extra_trees_direct_rul_source | source_only | 146.32 | 229.71 | 184.85 | 211.21 |
| median_eol_source | source_only | 238.60 | 236.56 | 181.08 | 184.19 |
| mean_eol_source | source_only | 217.56 | 241.22 | 195.11 | 200.61 |
| median_eol_source_plus_target_adapt | source_plus_labeled_target_adapt_eol_prior | 207.82 | 244.88 | 202.00 | 208.71 |
| mean_eol_source_plus_target_adapt | source_plus_labeled_target_adapt_eol_prior | 206.75 | 245.34 | 202.78 | 209.62 |
| extra_trees_direct_rul_source_linear_calibrated | source_plus_full_labeled_target_adapt_calibration | 130.52 | 247.73 | 214.65 | 236.29 |
| cubic_cycle_to_rul_source | source_only | 203.69 | 255.94 | 212.73 | 238.66 |

## Interpretation

- The naive source-domain median EOL baseline is already strong because RUL is structurally close to EOL minus current cycle.
- ExtraTrees with engineered partial-curve statistics improves over the quick neural RUL run, but it remains far from a publishable RUL result.
- Full target-adapt supervision is reported as a small-labeled-target upper protocol, not as an unsupervised domain adaptation result.
