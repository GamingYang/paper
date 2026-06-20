# SUBAGENT ROUND8 AUDIT 20260606

## Round Objective

User requested two additions under the locked EOL/RUL mainline:

1. Add same-protocol feature-family / horizon / leakage-audit supplementary materials to strengthen reviewer defense.
2. Try higher-grade physics-consistent EOL/RUL neural models under the same Batch-9-only selection protocol.

Non-negotiable protocol:

- Batch 1-8: source domain.
- Batch 9: labeled target development/adaptation/model-selection set.
- Batch 10: blind test only after selection is fixed.
- Batch 10 must not be used for training, horizon selection, feature selection, model selection, checkpoint selection, seed selection, or ranking.

## Accepted Outputs

### 1. Feature-Horizon-Leakage Defense V2

Path:

`C:\Users\ROG\Desktop\paper\codex\feature-horizon-leakage-defense-v2`

Status: accepted as manuscript/supplementary defense material.

Key results:

| Feature family | Best horizon by Batch 9 | Features | Batch 9 RMSE | Batch 9 MAE | Status |
|---|---:|---:|---:|---:|---|
| fused_all | 1000 | 108 | 75.26 | 61.96 | main strict processed family |
| soh_capacity | 800 | 59 | 86.76 | 70.24 | useful but weaker |
| x_curve | 1000 | 49 | 87.46 | 84.34 | useful but weaker |
| raw/provenance | n/a | n/a | n/a | n/a | separate raw diagnostic only |

Reviewer-defense value:

- Confirms the selected strict feature family and 1000-cycle horizon using Batch 9 only.
- Confirms early 50/100-cycle rows are not below 100 cycles, so the safe wording remains early-to-mid-life, not very-early.
- Confirms forbidden fields are not used as model inputs.
- Provides two usable supplement figures:
  - `horizon_vs_batch9_rmse.png`
  - `feature_family_batch9_rmse.png`

Manuscript use:

- Add this as a supplementary feature-family/horizon audit table.
- In Methods, state that the feature-family and horizon audit uses Batch 9 validation only and does not load Batch 10.

### 2. Leakage Negative Control V1

Path:

`C:\Users\ROG\Desktop\paper\codex\leakage-negative-control-v1`

Status: accepted as an optional reviewer-defense stress test, not a model result.

Valid rows after QA:

| Family | Selected horizon | Batch 9 RMSE | Batch 10 RMSE report-only |
|---|---:|---:|---:|
| strict_fused_all | 1000 | 75.26 | 56.19 |
| soh_capacity_only | 800 | 86.76 | 138.56 |
| x_curve_only | 1000 | 87.46 | 199.74 |

Invalid leakage stress test:

- Injected deliberately invalid fields:
  - `invalid_observed_ratio_true_eol = horizon / eol_cycle`
  - `invalid_remaining_cycles_at_horizon = eol_cycle - horizon`
- These invalid fields collapse Batch 9 RMSE to near zero across horizons.
- This proves why endpoint-derived fields must be blacklisted.

QA note:

- An initial local script version accidentally allowed `invalid_*` fields into the valid fused family. That contamination produced unrealistically low valid scores. The bug was caught, fixed, and rerun. The final report now matches strict v1: fused_all + 1000 cycles gives Batch 9 RMSE 75.26 and Batch 10 RMSE 56.19.

Manuscript use:

- Use only as a short supplement note: endpoint-derived variables collapse the task and are therefore explicitly excluded.
- Do not cite invalid leakage scores as performance.

### 3. Physics-Consistent Neural EOL V3

Path:

`C:\Users\ROG\Desktop\paper\codex\physics-consistent-neural-eol-v3`

Status: accepted as a strict negative neural prototype.

Selected by Batch 9:

| Protocol | Model | Horizon | Seed | Batch 9 RMSE | Batch 9 MAE | Batch 10 RMSE | Batch 10 MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| source_plus_target_adapt_loocv | physics_consistent_mlp_v3 | 1000 | 11 | 122.64 | 96.05 | 83.52 | 62.53 |

Comparison:

- Strict ExtraTrees reference: Batch 10 RMSE/MAE = 56.19/49.42 cycles.
- Neural v3 does not beat the strict ExtraTrees baseline.

Value:

- Protocol appears clean: held-out Batch 9 cell is not used for training, early stopping, or checkpoint selection in its fold.
- It enforces `predicted_EOL >= horizon` and non-negative derived RUL.
- It shows that a small MLP plus hard non-negative RUL constraint is not enough to improve the paper's main result.

Manuscript use:

- If included, place in supplement or Discussion as a negative neural control.
- Do not write that neural models improve the baseline.

### 4. Physics Neural EOL Alt V4

Path:

`C:\Users\ROG\Desktop\paper\codex\physics-neural-eol-alt-v4`

Status: accepted as an aggressive negative model-innovation probe.

Selected by Batch 9:

| Protocol | Model | Degree | Horizon | Batch 9 RMSE | Batch 9 MAE | Batch 10 RMSE | Batch 10 MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| source_plus_target_adapt_loocv | physics_neural_polyresid_d3 | 3 | 1000 | 114.03 | 99.82 | 166.08 | 150.74 |

Polynomial-base ablation:

| Family | Protocol | Degree | Best horizon | Batch 9 RMSE | Batch 9 MAE |
|---|---|---:|---:|---:|---:|
| physics_neural_polyresid | source_plus_target_adapt_loocv | 3 | 1000 | 114.03 | 99.82 |
| physics_neural_polyresid | source_plus_target_adapt_loocv | 4 | 1000 | 182.94 | 151.24 |
| physics_neural_polyresid | source_plus_target_adapt_loocv | 2 | 1000 | 742.88 | 708.78 |
| poly_base_only | base_only_no_training | 3 | 1000 | 141.37 | 105.36 |
| poly_base_only | base_only_no_training | 4 | 1000 | 353.40 | 313.50 |
| poly_base_only | base_only_no_training | 2 | 1000 | 1187.93 | 1147.65 |

Interpretation:

- In the compact target-adapt LOOCV neural setting, degree 3 is the best low-order nonlinear base.
- Degree 3 should be described as a practical low-order degradation prior, not a physical law.
- The selected v4 model fails badly on Batch 10 and must not replace the strict ExtraTrees baseline.
- Ordered interval head has no quantile-order violations, but eight Batch 10 cells cannot support strong calibration claims.

## Verification Performed

Commands/checks completed:

- `py_compile` passed for all four scripts:
  - `feature-horizon-leakage-defense-v2`
  - `physics-consistent-neural-eol-v3`
  - `physics-neural-eol-alt-v4`
  - `leakage-negative-control-v1`
- JSON files parsed successfully.
- Key CSV row counts:
  - `batch9_validation_by_feature_family_horizon.csv`: 21 rows.
  - `candidate_validation.csv`: 24 rows.
  - `physics-neural-eol-alt-v4_batch9_candidate_table.csv`: 9 rows.
  - `leakage-negative-control-v1_horizon_family_leakage_stress_results.csv`: 28 rows.

## Current Scientific Judgment

The round strengthened the paper's defense but did not produce a stronger main model.

Main result remains:

```text
strict-eol-protocol-v1
source_plus_target_adapt_loocv + ExtraTrees + 1000 cycles
Batch 10 RMSE/MAE = 56.19 / 49.42 cycles
```

What improved:

- The feature-family/horizon selection story is now better defended.
- Leakage blacklist is better defended with both a clean audit and an invalid negative-control stress test.
- The cubic-base question now has a limited Batch9-only ablation answer.
- Neural model failures are protocol-clean and useful as negative controls.

What did not improve:

- No neural model beat strict ExtraTrees.
- No result supports a stronger Applied Energy-style model-innovation claim.
- The evidence still supports a conservative Q2/SCI framing better than a high-impact SOTA framing.

## Writing Guidance

Safe claims to add:

- A supplementary Batch9-only audit confirms that fused processed historical capacity/SOH and curve fingerprints outperform either family alone.
- A leakage negative control shows that endpoint-derived observed-ratio variables would collapse the task, which justifies the strict blacklist.
- A polynomial-base ablation supports cubic as a practical low-order degradation prior in the compact target-adapt setting.
- Strict neural prototypes with non-negative RUL and ordered quantile outputs did not outperform the tabular ExtraTrees baseline.

Claims to avoid:

- Do not claim neural superiority.
- Do not claim the cubic base is a physical law.
- Do not claim very-early 50/100-cycle performance.
- Do not claim raw-only superiority.
- Do not use invalid leakage negative-control scores as performance.
- Do not imply Batch 10 was used for feature/horizon/model selection.

## Recommended Next Step

For the Q2 target, stop broad neural sweeps for now and integrate the accepted defense materials into:

1. Supplementary feature-family/horizon audit table.
2. Supplementary leakage blacklist and negative-control table.
3. Discussion paragraph on failed neural prototypes.
4. Limitations paragraph clarifying cubic-base evidence.

Further model work should be attempted only if it has a sharply different hypothesis, such as raw-signal reconstruction of processed curve fingerprints, external-batch validation, or a model that can exploit more target-domain structure without increasing leakage risk.
