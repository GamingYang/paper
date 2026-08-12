# soh-static-mmd-v1 Static MMD SOH Control

## Protocol

- SOH-only experiment on `processed_clean_fixed_split`.
- Batch 1-8 source data are used for supervised training.
- Batch 9 (`target_adapt`) is used for MMD adaptation and checkpoint/model selection.
- Batch 10 (`target_test`) is used only once for final reporting after selection.
- Batch 10 is not used for training, tuning, checkpoint selection, hyperparameter choice, seed selection, or figure selection.
- Network structure, seed, epochs, MSE warmup/MMD Guide/Pure MSE Sprint schedule, and `TARGET_MMD_WEIGHT=5e-05` follow `final-retained-v1`.
- Static MMD change: `MMD_loss = compute_mmd(source_features, target_features)`; the exponential SOH-dependent `dynamic_weight` is removed.

## Outputs

- Training script: `C:\Users\ROG\Desktop\paper\codex\soh-static-mmd-v1\train_soh_static_mmd_scheme_a.py`
- Summary JSON: `C:\Users\ROG\Desktop\paper\codex\soh-static-mmd-v1\soh-static-mmd-v1_best_metrics.json`
- Results XLSX: `C:\Users\ROG\Desktop\paper\codex\soh-static-mmd-v1\soh-static-mmd-v1_results.xlsx`
- Summary XLSX: `C:\Users\ROG\Desktop\paper\codex\soh-static-mmd-v1\soh-static-mmd-v1_summary.xlsx`
- Summary CSV: `C:\Users\ROG\Desktop\paper\codex\soh-static-mmd-v1\soh-static-mmd-v1_summary.csv`
- History CSV: `C:\Users\ROG\Desktop\paper\codex\soh-static-mmd-v1\soh-static-mmd-v1_history.csv`
- Report: `C:\Users\ROG\Desktop\paper\codex\soh-static-mmd-v1\soh-static-mmd-v1_report.md`
- Checkpoint: `C:\Users\ROG\Desktop\paper\codex\soh-static-mmd-v1\soh-static-mmd-v1_best_by_target_adapt.pt`

## Static MMD Metrics

| split | pooled RMSE | macro RMSE | MAE | MAPE (%) |
|---|---:|---:|---:|---:|
| target_adapt | 0.007172364 | 0.007002264 | 0.005044632 | 0.532033861 |
| Batch10 target_test | 0.009634693 | 0.008418250 | 0.005827995 | 0.627271295 |

## Archived Comparison

| model | Batch10 pooled RMSE | Batch10 macro RMSE | Batch10 MAE | Batch10 MAPE (%) |
|---|---:|---:|---:|---:|
| static MMD | 0.009634693 | 0.008418250 | 0.005827995 | 0.627271295 |
| final-retained-v1 | 0.009499246 | 0.008300783 | 0.005704158 | 0.610362053 |
| no-mmd-v1 | 0.009677417 | 0.008721296 | 0.005783882 | 0.623825967 |

## Short Interpretation

- Static MMD Batch10 pooled RMSE is 0.009634693, versus archived final-retained-v1 0.009499246 and archived no-mmd-v1 0.009677417.
- target_adapt pooled RMSE is 0.007172364; this is the only split used for checkpoint/model selection.
