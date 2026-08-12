# SOH Q2 Paper Outline And Task Checklist

Created: 2026-06-22  
Project root: `C:\Users\ROG\Desktop\paper`  
Status: active SOH-only mainline unless explicitly replaced.

## 1. Current Decision

The SOH work should be treated as an independent manuscript line, not as a
minor section inside the current EOL/RUL manuscript.

Recommended positioning:

```text
Physics-guided hybrid transfer learning for cross-batch lithium-ion battery
SOH prediction.
```

The current evidence is plausible for a conservative Q2 / ordinary SCI
submission if the claims remain controlled and the remaining protocol and
manuscript checks are closed. It is not currently safe to claim Applied
Energy-level novelty.

## 2. Do Not Mix With The RUL Mainline

This SOH line is separate from the EOL/RUL line.

- SOH main task: predict nominal SOH.
- RUL/EOL main task: predict cell-level EOL first, derive RUL as EOL minus
  current cycle.
- Current SOH final model is a neural physics-guided hybrid transfer model.
- Current RUL/EOL formal result uses an ExtraTrees strict EOL-first baseline.

Do not describe the current work as one already-unified SOH-RUL neural model
unless a new bridge experiment proves the chain:

```text
raw cycling data -> predicted SOH trajectory -> degradation fingerprints ->
predicted EOL -> derived RUL
```

## 3. Current SOH Evidence

Authoritative SOH result files:

- `codex/final-retained-v1/final-retained-v1_best_metrics.json`
- `codex/supplementary-experiments/seed42_summary.csv`
- `codex/external-baseline-suite-v1/external_baseline_suite_v1_summary.json`
- `codex/kernel-baseline-suite-v2/kernel_baseline_suite_v2_summary.json`
- `codex/paper-draft-v3-20260425/manuscript_v3_generation_summary_20260425.json`
- `codex/paper-draft-v3-20260425/method_experiment_draft_v3.docx`

Fixed split:

| Split | Role | Cells / samples |
|---|---|---:|
| Batch 1-8 | source training | 61 cells / 115256 samples |
| Batch 9 | target adaptation and model selection | 8 cells / 16310 samples |
| Batch 10 | final blind test | 8 cells / 14556 samples |

Main model:

```text
physical cubic degradation base + residual temporal network +
attention-assisted temporal aggregation + life-cycle adaptive MMD
```

Main SOH metrics:

| Model | Batch10 SOH RMSE | Batch10 MAE | Role |
|---|---:|---:|---|
| final-retained | 0.009499 | 0.005704 | main retained model |
| w/o MMD | 0.009677 | 0.005784 | ablation |
| w/o Attention | 0.011677 | 0.008145 | ablation |
| CNN-LSTM baseline | 0.016075 | 0.009299 | external baseline |
| w/o physical base | 0.020716 | 0.013759 | ablation |
| PCA-GPR | 0.023108 | 0.014784 | kernel baseline |
| PCA-RBF-SVR | 0.023823 | 0.014490 | kernel baseline |
| pure cubic physical base | 0.032649 | 0.023746 | physical-only baseline |
| LSTM baseline | 0.032742 | 0.023522 | external baseline |
| GRU baseline | 0.033109 | 0.023577 | external baseline |
| PCA-LinearSVR | 0.036993 | 0.025585 | kernel baseline |

Interpretation:

- The physical cubic base is the strongest ablation-supported component.
- Attention provides a meaningful auxiliary improvement.
- MMD is positive but small; do not overclaim it as the core novelty.
- The retained model beats common recurrent, CNN-LSTM, kernel/statistical, and
  pure physical baselines under the prepared split.

## 4. Proposed Manuscript Outline

### Title

Candidate:

```text
Physics-Guided Hybrid Transfer Learning for Cross-Batch State-of-Health
Prediction of Lithium-Ion Batteries
```

### Abstract

Must include:

- cross-batch SOH prediction problem;
- fixed source / target-development / blind-test protocol;
- physical cubic degradation base;
- residual temporal network with attention;
- life-cycle adaptive MMD as modest target-domain alignment;
- final Batch10 RMSE/MAE;
- controlled limitation: within-dataset cross-batch validation only.

### 1. Introduction

Core pain point:

```text
Battery SOH models trained on historical batches may degrade on new batches.
Reliable cross-batch SOH estimation is needed for BMS monitoring, maintenance,
warranty, and second-life screening before full life information is available.
```

Contributions:

1. A physics-guided hybrid transfer model combining a low-order degradation
   prior and residual temporal learning.
2. An attention-assisted residual sequence model for individualized deviation
   around the physical base.
3. A life-cycle adaptive MMD regularizer for cross-batch feature alignment.
4. A fixed source / target-development / blind-test evaluation with baselines
   and module ablations.

### 2. Related Work

Subsections:

- Data-driven battery SOH estimation.
- Physics-informed or physics-guided SOH prediction.
- Transfer learning / domain adaptation for battery health estimation.
- Hybrid neural sequence models for degradation modeling.

### 3. Dataset And Protocol

Must define:

- dataset: 77 commercial LFP/graphite cells;
- nominal SOH label;
- `processed_clean_fixed_split`;
- Batch 1-8 source, Batch 9 target development/adaptation, Batch 10 blind
  test;
- target-domain checkpoint/model selection uses Batch 9, not Batch 10;
- metrics: RMSE, MAE, MAPE, macro RMSE, per-cell RMSE.

### 4. Method

Required method blocks:

1. Cubic physical degradation base.
2. Residual temporal network.
3. Attention-assisted temporal aggregation.
4. Life-cycle adaptive MMD.
5. Training and selection protocol.

Important wording:

- The cubic base is a low-order nonlinear degradation prior, not a complete
  electrochemical mechanism.
- Attention is an auxiliary representation module, not the main novelty.
- MMD is a regularizer with modest but positive contribution.
- Smoothness penalty was tested and removed; do not include it as final method.

### 5. Experiments

Tables:

1. Fixed split and data statistics.
2. Main comparison with baselines.
3. Ablation study.
4. Per-cell Batch10 results.
5. Optional multi-seed summary if rerun.

Figures:

1. Overall method framework.
2. Batch10 SOH prediction curves for final model.
3. Per-battery RMSE comparison.
4. Ablation prediction curves or grouped error bars.
5. Optional residual / uncertainty / error-distribution figure.

### 6. Results And Discussion

Key points:

- The retained model gives the best Batch10 SOH RMSE among prepared models.
- Removing the physical base causes the largest degradation.
- Removing attention increases error and weakens temporal aggregation.
- Removing MMD only slightly worsens the result, so its role should be stated
  cautiously.
- Small target-development set and within-dataset split are limitations.

### 7. Conclusion

Keep conservative:

- The model improves cross-batch SOH prediction under the prepared protocol.
- The physical base and residual temporal learner are complementary.
- Future work: external dataset, broader operating conditions, stronger
  uncertainty, and optional SOH-to-RUL bridge.

## 5. Immediate Task Checklist

### P0: Must Finish Before A Serious SOH Draft

- [ ] Freeze the SOH-only mainline and stop mixing RUL/EOL claims into the SOH
      manuscript.
- [ ] Verify that all SOH metrics are reproduced from saved prediction tables
      or result workbooks.
- [ ] Audit the SOH protocol: Batch 10 must not be used for checkpoint,
      hyperparameter, model, seed, or figure selection.
- [ ] Audit preprocessing: scaler / normalization / imputation / PCA must not
      be fit using Batch 10.
- [ ] Produce a compact SOH evidence table from final, ablation, external
      baseline, and kernel baseline files.
- [ ] Regenerate or polish SOH figures with consistent English labels and
      Times New Roman / journal-style formatting.
- [ ] Write a clean SOH manuscript v1 from the SOH outline, not by editing the
      EOL/RUL manuscript.

### P1: Strongly Recommended For Q2 Defense

- [ ] Run or verify multi-seed SOH results for final, no-physical-base,
      no-attention, and no-MMD.
- [ ] Add per-cell Batch10 error table and discussion.
- [ ] Add a short reviewer-risk section for small Batch10 n=8.
- [ ] Add a normalization/leakage audit note in supplementary materials.
- [ ] Verify literature citations for physics-guided SOH, transfer learning,
      and battery sequence models.

### P2: Optional Upgrades

- [ ] Add a missing/noisy input robustness test if runtime permits.
- [ ] Add uncertainty or prediction-band diagnostic, clearly framed as
      diagnostic if not formally calibrated.
- [ ] Add an SOH-to-RUL bridge experiment only if the final strategy becomes a
      unified SOH-RUL paper.
- [ ] Test an external public dataset only if the data and preprocessing burden
      are acceptable.

## 6. Forbidden Claims

Do not claim:

- The SOH paper already proves RUL prediction.
- The current SOH model is an electrochemical mechanism model.
- MMD is the dominant source of improvement.
- The result is cross-dataset or cross-chemistry validated.
- The result is temperature-aware.
- Applied Energy-level novelty is already guaranteed.
- Batch 10 was used for any selection step.

## 7. Next Recommended Move

Open a clean SOH-focused window or use the SOH skill in the current window, then
ask for:

```text
Use the battery-soh-q2-paper skill. Build a SOH-only evidence audit and draft
plan from the current project files. Do not use RUL/EOL as the main task.
```
