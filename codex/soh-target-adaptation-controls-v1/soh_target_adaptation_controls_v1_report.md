# SOH Target-Adaptation Controls v1

This package adds SOH-only source-only / no-target-adaptation control experiments, Batch 9 target-development cell-budget sensitivity, and a model complexity/efficiency table.

## Protocol Guardrails

- Source domain: Batch 1-8.
- Target-development/adaptation domain: Batch 9.
- Blind-test domain: Batch 10.
- Batch 10 is not used for training, MMD alignment, checkpoint selection, target subset selection, or hyperparameter tuning.
- The cubic base is treated as a low-order nonlinear degradation prior, not a strong electrochemical mechanism model.
- MMD is treated as a lightweight alignment regularizer; this experiment does not turn it into the main contribution.

## Key Outputs

- Summary workbook: `C:\Users\ROG\Desktop\paper\codex\soh-target-adaptation-controls-v1\soh_target_adaptation_controls_v1_summary.xlsx`
- Overall CSV: `C:\Users\ROG\Desktop\paper\codex\soh-target-adaptation-controls-v1\soh_target_adaptation_controls_v1_overall.csv`
- Target-budget CSV: `C:\Users\ROG\Desktop\paper\codex\soh-target-adaptation-controls-v1\soh_target_budget_summary.csv`
- Complexity CSV: `C:\Users\ROG\Desktop\paper\codex\soh-target-adaptation-controls-v1\soh_complexity_efficiency_table.csv`
- Figures: `C:\Users\ROG\Desktop\paper\codex\soh-target-adaptation-controls-v1\figures`

## Budget Summary

| k Batch9 cells | n runs | Batch10 RMSE mean | Batch10 MAE mean | Batch9 full RMSE mean |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0.010805 | 0.006587 | 0.008471 |
| 1 | 1 | 0.010467 | 0.006936 | 0.009597 |
| 2 | 1 | 0.011190 | 0.006552 | 0.008292 |
| 4 | 1 | 0.010373 | 0.006535 | 0.008587 |
| 8 | 1 | 0.009448 | 0.006433 | 0.007946 |

## Run-Level Results

| run | protocol | target cells | selection | Batch10 RMSE | Batch10 macro RMSE | Batch10 MAE | best epoch |
|---|---|---|---|---:|---:|---:|---:|
| k0_source_only | source_only_no_target_adaptation | - | source_internal_validation | 0.010805 | 0.009265 | 0.006587 | 148 |
| k1_target_subset_1 | source_plus_fixed_batch9_subset_mmd_and_selection | 64 | batch9_target_subset | 0.010467 | 0.009626 | 0.006936 | 85 |
| k2_target_subset_2 | source_plus_fixed_batch9_subset_mmd_and_selection | 64,65 | batch9_target_subset | 0.011190 | 0.009623 | 0.006552 | 89 |
| k4_target_subset_4 | source_plus_fixed_batch9_subset_mmd_and_selection | 63,64,65,68 | batch9_target_subset | 0.010373 | 0.009371 | 0.006535 | 87 |
| k8_target_subset_all | source_plus_fixed_batch9_subset_mmd_and_selection | 61,62,63,64,65,66,67,68 | batch9_target_subset | 0.009448 | 0.008778 | 0.006433 | 68 |

## Complexity And Efficiency

| run | trainable params | checkpoint MB | train min | inference ms/sample | samples/s |
|---|---:|---:|---:|---:|---:|
| k0_source_only | 31466 | 0.391 | 7.07 | 0.04591 | 21782.3 |
| k1_target_subset_1 | 31466 | 0.392 | 10.52 | 0.04673 | 21400.3 |
| k2_target_subset_2 | 31466 | 0.392 | 10.72 | 0.04615 | 21666.3 |
| k4_target_subset_4 | 31466 | 0.392 | 11.14 | 0.04644 | 21531.3 |
| k8_target_subset_all | 31466 | 0.392 | 11.86 | 0.04652 | 21495.7 |

## Archived Final-Retained Reference

The archived final-retained-v1 row is listed for manuscript cross-checking only; it was not rerun or reselected here.

- Batch 9 RMSE: 0.007234
- Batch 10 pooled RMSE: 0.009499
- Batch 10 macro RMSE: 0.008301
- Batch 10 MAE: 0.005704

## Plot Skill Use

- Skill directory used: `C:\Users\ROG\Desktop\paper\paper-plot-skills-main\plot-from-data`
- Styles used: `line_confidence_band` for target-budget curve and `bar_grouped_hatch` for grouped bars.
- Generated numerical figures are data-derived Matplotlib outputs; they are not AI-generated replacement images.

## Manuscript Interpretation

- Use k=0 as the clean source-only / no-target-adaptation control because it selects checkpoints only on source internal validation.
- Use the k>0 rows as fixed nested target-development budget sensitivity, not as Batch 10-driven model selection.
- Because the default `single` plan uses one predefined Batch 9 subset per budget, do not claim monotonic or distribution-level label-budget improvement unless the `expanded` plan is run.
- If the k=8 rerun differs from archived final-retained-v1, report it as a same-protocol rerun and keep final-retained-v1 as the locked archived main result unless the whole protocol is formally restarted.
- Do not claim that MMD is the dominant source of improvement from this table alone.
