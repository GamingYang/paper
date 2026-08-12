# SOH Target-Adaptation Controls Acceptance - 2026-06-22

## Scope

SOH-only experiment package for:

- source-only / no-target-adaptation control,
- Batch 9 target-development cell-budget sensitivity,
- model complexity and efficiency table,
- data-derived figures generated with the local `paper-plot-skills-main/plot-from-data` style guidance.

No RUL/EOL result is introduced here.

## Package

Directory:

`C:\Users\ROG\Desktop\paper\codex\soh-target-adaptation-controls-v1`

Key files:

- `run_soh_target_adaptation_controls_v1.py`
- `soh_target_adaptation_controls_v1_summary.xlsx`
- `soh_target_adaptation_controls_v1_overall.csv`
- `soh_target_budget_summary.csv`
- `soh_complexity_efficiency_table.csv`
- `soh_target_adaptation_controls_v1_report.md`
- `soh_target_adaptation_controls_v1_summary.json`
- `figures/soh_batch9_budget_sensitivity_line_confidence_band_style.png`
- `figures/soh_source_only_vs_target_budget_bar_grouped_hatch_style.png`
- `figures/soh_complexity_efficiency_bar_grouped_hatch_style.png`

## Completion Check

PASS: source-only / no-target-adaptation control was run.

- Run: `k0_source_only`
- Selection split: source internal validation only.
- Source validation cells: 7, 17, 18, 24, 27, 28, 51, 56.
- Batch 9 and Batch 10 are not used for training, MMD alignment, checkpoint selection, target-subset selection, or hyperparameter tuning in this run.

PASS: Batch 9 small target-domain budget sensitivity was run.

- Fixed Batch 9 target-cell order: 64, 65, 63, 68, 67, 62, 66, 61.
- Nested budgets: k=1 uses 64; k=2 uses 64,65; k=4 uses 63,64,65,68; k=8 uses all Batch 9 cells.
- Batch 10 is used only for blind-test reporting after each run is fixed.

PASS: model complexity and efficiency table was generated.

- Trainable parameters: 31,466 for each neural configuration.
- Checkpoint size: about 0.391-0.392 MB.
- Batch 10 inference latency: about 0.046 ms/sample on the current CUDA environment.

PASS: paper-plot-skills-main plotting guidance was used.

- Skill directory: `C:\Users\ROG\Desktop\paper\paper-plot-skills-main\plot-from-data`
- Styles used: `line_confidence_band` and `bar_grouped_hatch`.
- Three PNG/PDF figure pairs were generated and visually checked as nonblank.

## Main Results

| Run | k Batch9 cells | Selection split | Batch10 RMSE | Batch10 macro RMSE | Batch10 MAE | Best epoch |
|---|---:|---|---:|---:|---:|---:|
| k0_source_only | 0 | source_internal_validation | 0.010805 | 0.009265 | 0.006587 | 148 |
| k1_target_subset_1 | 1 | batch9_target_subset | 0.010467 | 0.009626 | 0.006936 | 85 |
| k2_target_subset_2 | 2 | batch9_target_subset | 0.011190 | 0.009623 | 0.006552 | 89 |
| k4_target_subset_4 | 4 | batch9_target_subset | 0.010373 | 0.009371 | 0.006535 | 87 |
| k8_target_subset_all | 8 | batch9_target_subset | 0.009448 | 0.008778 | 0.006433 | 68 |

Archived final-retained-v1 reference:

- Batch 9 RMSE: 0.007234.
- Batch 10 pooled RMSE: 0.009499.
- Batch 10 macro RMSE: 0.008301.
- Batch 10 MAE: 0.005704.

The k=8 rerun is close to the archived final-retained-v1 Batch 10 pooled RMSE, but it is not a replacement for the locked main result unless the formal protocol is restarted and reselected from Batch 9 only.

## Interpretation Boundaries

- The k=0 row is the clean source-only comparator.
- The k>0 rows support a cautious target-development budget diagnostic.
- The default `single` plan uses one predefined nested Batch 9 subset per budget. It does not prove monotonic label-budget improvement across all possible subsets.
- The k=2 point is worse than k=0/k1/k4/k8 on pooled Batch 10 RMSE, so do not write a smooth monotonic curve.
- The k=8 row is the best in this diagnostic on Batch 10 pooled RMSE, but Batch 10 must not be used to select future models, target subsets, or hyperparameters.
- MMD should still be described as a lightweight alignment regularizer, not the dominant source of improvement.
- The cubic base should remain described as a low-order nonlinear degradation prior, not as a strong electrochemical mechanism model.

## Manuscript Use

Recommended use:

- Put `k0_source_only` in the main comparison or protocol-control table.
- Use the k=0/1/2/4/8 budget figure as supplementary or a short robustness/control result.
- Put the complexity and efficiency table in either the main experiments section or supplementary material.
- State clearly that the budget curve is based on a fixed nested Batch 9 order.

Not recommended:

- Do not claim full target-label-budget monotonicity from the current `single` plan.
- Do not replace final-retained-v1 with the k=8 rerun without formally restarting the model-freezing protocol.
- Do not use these results to strengthen RUL/EOL claims.

