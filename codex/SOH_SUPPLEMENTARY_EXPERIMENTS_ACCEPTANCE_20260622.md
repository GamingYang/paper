# SOH Supplementary Experiments Acceptance 20260622

Scope: independent SOH prediction paper only. RUL/EOL was not advanced in this round.

## 1. Completion Status

This round completed and validated three SOH supplemental experiment packages:

| Package | Status | Main output directory |
|---|---|---|
| Stage/tail/bootstrap diagnostics | PASS | `codex/soh-stage-tail-bootstrap-v1/` |
| MMD representation diagnostic | PASS | `codex/soh-mmd-representation-diagnostic-v1/` |
| Full-model physical-prior degree ablation | PASS with caveats | `codex/soh-physical-degree-ablation-v1/` |

An independent read-only validation subagent audited the three packages and reported overall PASS. The only minor documentation issue was that the stage/tail workbook did not restate the full split mapping; this has been fixed in `soh_stage_tail_bootstrap_v1.py` and the refreshed workbook now includes:

- source domain: Batch 1-8.
- target development/adaptation: Batch 9.
- blind test: Batch 10, reporting only.

## 2. Protocol Compliance

All packages preserve the SOH fixed protocol:

- Data protocol: `processed_clean_fixed_split`.
- Source domain: Batch 1-8.
- Target-domain development/adaptation: Batch 9.
- Blind-test reporting: Batch 10.
- Batch 10 was not used for tuning, checkpoint selection, degree selection, or final model replacement.
- No RUL/EOL analysis was performed.
- MMD is framed as lightweight representation/domain regularization, not as the dominant performance source.
- The cubic base is framed as a low-order nonlinear degradation prior, not as a strong electrochemical mechanism.
- Bootstrap intervals are described as empirical resampling intervals, not calibrated predictive uncertainty.

## 3. Stage/Tail/Bootstrap Diagnostics

Output files:

- `codex/soh-stage-tail-bootstrap-v1/soh_stage_tail_bootstrap_v1.py`
- `codex/soh-stage-tail-bootstrap-v1/soh_stage_tail_bootstrap_v1_summary.xlsx`
- `codex/soh-stage-tail-bootstrap-v1/soh_stage_tail_bootstrap_v1_report.md`
- `codex/soh-stage-tail-bootstrap-v1/figures/stage_wise_rmse_comparison.png`
- `codex/soh-stage-tail-bootstrap-v1/figures/tail_bias_late_stage_diagnostics.png`

Headline Batch 10 metrics:

| Model | Pooled RMSE | Macro RMSE | MAE | Signed bias |
|---|---:|---:|---:|---:|
| final-retained-v1 | 0.009499 | 0.008301 | 0.005704 | -0.001316 |
| no-physical-base-v1 | 0.020716 | 0.019506 | 0.013759 | 0.005719 |
| no-mmd-v1 | 0.009677 | 0.008721 | 0.005784 | 0.000608 |
| no-attention-v1 | 0.011677 | 0.010535 | 0.008145 | -0.001480 |

Stage-wise RMSE:

| Model | Early | Mid | Late |
|---|---:|---:|---:|
| final-retained-v1 | 0.003120 | 0.009111 | 0.013341 |
| no-physical-base-v1 | 0.008601 | 0.013933 | 0.031926 |
| no-mmd-v1 | 0.003379 | 0.006460 | 0.015093 |
| no-attention-v1 | 0.006760 | 0.009897 | 0.016291 |

Tail last-20% diagnostics:

| Model | Tail RMSE | Tail MAE | Tail signed bias | Positive slope fraction |
|---|---:|---:|---:|---:|
| final-retained-v1 | 0.013864 | 0.011054 | 0.000142 | 0.233735 |
| no-physical-base-v1 | 0.036620 | 0.028629 | 0.023510 | 0.337694 |
| no-mmd-v1 | 0.017501 | 0.013588 | 0.007161 | 0.235800 |
| no-attention-v1 | 0.018135 | 0.015130 | 0.010529 | 0.228227 |

Cell-level bootstrap over the 8 Batch 10 cells:

| Model | Observed macro RMSE | 95% empirical interval | Observed macro MAE | 95% empirical interval |
|---|---:|---:|---:|---:|
| final-retained-v1 | 0.008301 | 0.005816 to 0.011278 | 0.005519 | 0.003731 to 0.007955 |
| no-physical-base-v1 | 0.019506 | 0.013844 to 0.025825 | 0.013972 | 0.009847 to 0.018156 |
| no-mmd-v1 | 0.008721 | 0.006216 to 0.011543 | 0.005625 | 0.003976 to 0.007842 |
| no-attention-v1 | 0.010535 | 0.007964 to 0.013699 | 0.007829 | 0.005442 to 0.011112 |

Manuscript use:

- This package strengthens the physical/degradation-prior story.
- The strongest new statement is that removing the low-order degradation prior sharply worsens late-stage and tail behavior.
- Tail RMSE increases from 0.013864 to 0.036620 without the prior.
- Tail signed bias shifts from near zero, +0.000142, to clear overprediction, +0.023510, without the prior.
- MMD remains small/mixed in RMSE terms and should not be overclaimed.

## 4. MMD Representation Diagnostic

Output files:

- `codex/soh-mmd-representation-diagnostic-v1/soh_mmd_representation_diagnostic_v1.py`
- `codex/soh-mmd-representation-diagnostic-v1/soh_mmd_representation_diagnostic_v1_summary.xlsx`
- `codex/soh-mmd-representation-diagnostic-v1/soh_mmd_representation_diagnostic_v1_report.md`
- `codex/soh-mmd-representation-diagnostic-v1/figures/mmd_representation_domain_bars.png`
- `codex/soh-mmd-representation-diagnostic-v1/figures/final_vs_no_mmd_pca_2d.png`

Key final-retained vs no-MMD findings:

| Diagnostic | final-retained-v1 | no-mmd-v1 | Direction |
|---|---:|---:|---|
| RBF multikernel MMD^2, z-scored fused features | 0.153337 | 0.464046 | final lower by about 67.0% |
| Raw feature MMD^2 | 0.108920 | 0.441094 | final lower |
| Centroid gap, z-scored | 1.684173 | 3.851309 | final lower |
| Domain classifier accuracy | 0.940667 | 0.951500 | final slightly less separable |
| Domain classifier AUC | 0.979493 | 0.988835 | final slightly less separable |

Manuscript use:

- This gives representation-level support for MMD as a domain regularizer.
- It should be written as auxiliary support: MMD reduces source-Batch9 feature discrepancy.
- It should not be written as a major direct accuracy driver, because Batch 10 RMSE differences remain small and multi-seed evidence is nuanced.
- Batch 10 was not loaded for this diagnostic.

## 5. Physical-Prior Degree Ablation

Output files:

- `codex/soh-physical-degree-ablation-v1/run_soh_physical_degree_ablation_v1.py`
- `codex/soh-physical-degree-ablation-v1/soh_physical_degree_ablation_v1_summary.xlsx`
- `codex/soh-physical-degree-ablation-v1/soh_physical_degree_ablation_v1_report.md`
- `codex/soh-physical-degree-ablation-v1/figures/degree_vs_batch9_batch10_rmse.png`
- `codex/soh-physical-degree-ablation-v1/figures/batch10_per_battery_rmse_heatmap_by_degree.png`
- Per-degree result folders: `degree_1/` through `degree_5/`.

All requested degrees 1-5 completed. `progress.json` reports completed degrees `[1,2,3,4,5]`, no remaining degrees, and `error: null`.

Overall results:

| Degree | Best epoch by Batch 9 | Batch 9 RMSE | Batch 10 pooled RMSE | Batch 10 macro RMSE | Batch 10 MAE | Batch 10 MAPE (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 104 | 0.008571 | 0.009122 | 0.008489 | 0.006095 | 0.6456 |
| 2 | 116 | 0.008363 | 0.009811 | 0.008180 | 0.005961 | 0.6421 |
| 3 | 125 | 0.008384 | 0.008862 | 0.007894 | 0.005670 | 0.6031 |
| 4 | 116 | 0.007806 | 0.010469 | 0.009204 | 0.006541 | 0.7050 |
| 5 | 140 | 0.007441 | 0.009723 | 0.009185 | 0.006394 | 0.6829 |

Key interpretation:

- Batch 9 model-selection criterion favors degree 5: RMSE 0.007441.
- Batch 10 reporting favors degree 3: pooled RMSE 0.008862.
- Because Batch 10 must not be used for degree selection, this result does not justify selecting degree 3 because it has the lowest Batch 10 RMSE in this diagnostic.
- Degree 3 is not the Batch 9 winner in this run. The manuscript should avoid claiming degree-3 optimality.
- The safest wording is that the cubic prior is the current compact low-order nonlinear prior used in the retained model and can be interpreted as a complexity/oscillation compromise, not as a proven optimal polynomial order.

Important caveats:

- Reproducibility difference: the degree-3 ablation uses the same cubic coefficients as original `final-retained-v1`, but does not numerically reproduce the original run.
  - Original final-retained-v1: best epoch 140, Batch 9 RMSE 0.007234276, Batch 10 RMSE 0.009499246.
  - Degree-3 ablation run: best epoch 125, Batch 9 RMSE 0.008383574, Batch 10 RMSE 0.008862434.
- This likely reflects training nondeterminism or implementation/run-script differences and should be stated if using the degree ablation as formal evidence.
- `full_run_stderr.log` contains two `RankWarning: Polyfit may be poorly conditioned` warnings from `np.polyfit(...)`, likely tied to high-degree polynomial fitting. This supports restrained wording around higher-degree priors.

## 6. Acceptance Decision

The requested supplemental SOH experiments are complete and usable, with the following manuscript decisions:

1. Use the stage/tail/bootstrap package in the main ablation/results section to support the physical/degradation-prior contribution.
2. Use the MMD diagnostic as supplementary representation evidence only.
3. Use the degree ablation as a cautionary experiment:
   - It does not prove cubic optimality.
   - It shows high-degree selection can look better on Batch 9 but not necessarily on Batch 10.
   - It strengthens the argument for restrained "low-order nonlinear prior" wording rather than "optimal cubic physical law".
4. Do not replace the established `final-retained-v1` main model based on these new Batch 10 observations.
5. Do not change the paper to an RUL/EOL paper.

## 7. Remaining Non-Experimental Paper Work

No further SOH experiment is required before manuscript integration for a conservative SCI/Q2 route, unless the target venue demands external-dataset validation.

Remaining work is writing and packaging:

- Build the final SOH manuscript table pack.
- Update MMD language across the manuscript.
- Update physical-prior language across the manuscript.
- Insert stage/tail/bootstrap and MMD diagnostic figures/tables.
- Decide whether degree ablation goes in main text or supplement.
- Add verified citations for Introduction and Related Work.
