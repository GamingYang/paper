# SOH Physical Degree Ablation v1

## Scope And Protocol

This run is a formal SOH-only ablation/diagnostic for the physical/degradation-prior polynomial degree in the full residual temporal model. It preserves `processed_clean_fixed_split`: Batch 1-8 are the source domain, Batch 9 is target-domain development/adaptation, and Batch 10 is blind-test reporting only.

For each degree, the prior coefficients were fitted on source training data only. The residual CRNN/LSTM/Attention model, seed, optimizer schedule, nominal SOH labels, lightweight MMD settings, and metrics follow the retained Scheme A script. Checkpoints were selected by Batch 9 target-adapt RMSE. Batch 10 metrics were computed after each degree checkpoint was fixed and were not used for tuning, degree selection, or final-model identity changes.

## Overall Results

| Degree | Best epoch by Batch 9 | Batch 9 RMSE | Batch 10 pooled RMSE | Batch 10 macro RMSE | Batch 10 MAE | Batch 10 MAPE (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 104 | 0.008571 | 0.009122 | 0.008489 | 0.006095 | 0.6456 |
| 2 | 116 | 0.008363 | 0.009811 | 0.008180 | 0.005961 | 0.6421 |
| 3 | 125 | 0.008384 | 0.008862 | 0.007894 | 0.005670 | 0.6031 |
| 4 | 116 | 0.007806 | 0.010469 | 0.009204 | 0.006541 | 0.7050 |
| 5 | 140 | 0.007441 | 0.009723 | 0.009185 | 0.006394 | 0.6829 |

Batch 9 ranking selects degree 5 with RMSE 0.007441. Batch 10 reporting is separate: the lowest blind-test pooled RMSE in this diagnostic is degree 3 with RMSE 0.008862. This Batch 10 ordering is reported for transparency only and must not be used to choose the degree.

## Interpretation

Degree 3 is not the Batch 9 winner in this run. The manuscript should therefore avoid claiming degree-3 optimality; at most, the cubic prior can be described as a compact complexity/oscillation compromise if supported by the broader retained-model evidence.

The ablation should be read as evidence about the polynomial prior's complexity, not as a claim that a specific polynomial degree encodes a strong electrochemical mechanism. In particular, the cubic base should be phrased as a compact low-order nonlinear degradation prior. MMD is inherited from the retained model as a lightweight cross-domain alignment component; these results do not by themselves isolate or overclaim the effect of MMD.

This diagnostic does not alter the established `final-retained-v1` identity. Any manuscript decision to replace or revise that model should be made by the parent workflow after comparing this diagnostic with the existing retained-model evidence.

## Artifacts

- Summary workbook: `soh_physical_degree_ablation_v1_summary.xlsx`
- Main figure: `figures/degree_vs_batch9_batch10_rmse.png`
- Per-cell heatmap: `figures/batch10_per_battery_rmse_heatmap_by_degree.png`
