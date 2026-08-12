# Advanced Attention Candidate Validation Report

Date: 2026-07-22

## 1. Purpose

This experiment tested two attention candidates for the lithium-ion battery SOH
manuscript:

1. `multi-expert-gated-attention-v1`
2. `stage-aware-branch-attention-v1`

Both were trained under the existing fixed protocol:

```text
Batch 1-8  -> source training
Batch 9    -> target-domain MMD / checkpoint selection
Batch 10   -> final blind reporting only
```

No Batch10 sample was used for training, tuning, or checkpoint selection.

## 2. Formula Document

Detailed formulas and checks are saved at:

```text
C:\Users\ROG\Desktop\paper\codex\advanced-attention-candidates-v1\ATTENTION_FORMULA_DERIVATION_AND_CHECK_20260722.md
```

The two formulas were checked before training for:

- temporal attention normalization;
- expert-gate normalization;
- unchanged downstream feature dimension;
- unchanged physical-residual prediction head;
- unchanged Batch1-8 / Batch9 / Batch10 protocol;
- no Batch10-dependent design term.

## 3. Training Setup

Training script:

```text
C:\Users\ROG\Desktop\paper\codex\advanced-attention-candidates-v1\run_advanced_attention_candidates_v1.py
```

Configuration:

| Item | Setting |
|---|---|
| seeds | 42, 7, 123 |
| epochs | 150 |
| label | nominal SOH |
| source | Batch1-8 |
| target adaptation / model selection | Batch9 |
| blind test | Batch10 |
| optimizer / MMD schedule | same as adaptive-attention-v1 |

Training completed successfully. `training_stderr.log` is empty.

## 4. Three-Seed Result

| Method | Mean RMSE | Std RMSE | Mean Macro RMSE | Mean MAE |
|---|---:|---:|---:|---:|
| Original attention | 0.010381 | 0.000867 | 0.009171 | 0.006253 |
| Adaptive attention v1 | **0.009906** | **0.000557** | **0.008764** | **0.006189** |
| Multi-expert gate | 0.010993 | 0.001433 | 0.009635 | 0.006618 |
| Stage-aware branch | 0.014712 | 0.005948 | 0.011676 | 0.007592 |

Conclusion from overall metrics:

```text
Neither candidate is strong enough to replace adaptive-attention-v1 as the
current final attention module.
```

## 5. Per-Seed Result

| Method | Seed | Batch10 RMSE | Macro RMSE | MAE | Early RMSE | Middle RMSE | Late RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multi-expert gate | 42 | 0.012515 | 0.011328 | 0.007891 | 0.005610 | 0.016713 | 0.029898 |
| Multi-expert gate | 7 | 0.010793 | 0.008804 | 0.006034 | 0.003510 | 0.015738 | 0.012535 |
| Multi-expert gate | 123 | 0.009671 | 0.008773 | 0.005929 | 0.003621 | 0.013137 | 0.023722 |
| Stage-aware branch | 42 | 0.021025 | 0.016379 | 0.010131 | 0.005642 | 0.031192 | 0.017937 |
| Stage-aware branch | 7 | 0.009212 | 0.008153 | 0.005769 | 0.003738 | 0.012389 | 0.022793 |
| Stage-aware branch | 123 | 0.013900 | 0.010496 | 0.006875 | 0.002937 | 0.020805 | 0.011174 |

## 6. Mechanism Interpretation

### 6.1 Multi-Expert Gate

The multi-expert gate did learn stage-dependent expert allocation:

| Stage | Expert 1 | Expert 2 | Expert 3 |
|---|---:|---:|---:|
| Early | 0.448985 | 0.270845 | 0.280170 |
| Middle | 0.262715 | 0.416667 | 0.320618 |
| Late | 0.297419 | 0.346238 | 0.356343 |

This supports the formula-level claim that the gate responds to degradation
state. However, the performance does not justify making it the final model.

Interpretation:

- The gate has plausible stage allocation.
- Late-stage behavior improves in seed7, but not consistently.
- Mean overall RMSE is worse than original attention and adaptive-attention-v1.
- The method may be useful as a future refinement, but not as current main
  evidence.

### 6.2 Stage-Aware Branch

The stage branch shows strong stage-dependent modulation:

| Stage | Mean attention entropy | Mean max weight | Mean temperature | Mean abs(gamma) | Mean abs(beta) |
|---|---:|---:|---:|---:|---:|
| Early | 2.284916 | 0.213715 | 0.845335 | 0.507974 | 0.285358 |
| Middle | 1.312420 | 0.525269 | 0.669573 | 0.354289 | 0.191004 |
| Late | 1.042729 | 0.606140 | 0.578019 | 0.548726 | 0.335330 |

This means the branch becomes sharper and more strongly modulated in the late
stage. It is mechanistically interpretable, but unstable.

Interpretation:

- It improves late-stage RMSE on average.
- It strongly worsens middle-stage RMSE and three-seed stability.
- Seed42 collapses in the middle stage, causing poor overall RMSE.
- The branch has a useful idea, but needs regularization or a safer fusion
  design before being considered a main method.

## 7. Visual Outputs

Figures are saved under:

```text
C:\Users\ROG\Desktop\paper\codex\advanced-attention-candidates-v1\figures
```

Main figures:

| Figure | Purpose |
|---|---|
| `fig01_three_seed_error_summary.png` | overall three-seed RMSE / macro RMSE / MAE |
| `fig02_seedwise_rmse_comparison.png` | per-seed RMSE comparison |
| `fig03_seed42_stage_rmse_comparison.png` | seed42 early/middle/late RMSE |
| `fig04_three_seed_stage_rmse.png` | three-seed stage-wise RMSE |
| `fig05_overall_late_tradeoff.png` | overall-late accuracy tradeoff |
| `fig06_attention_profiles_seed42.png` | stage-wise attention profiles |
| `fig07_multi_expert_gate_stage_seed42.png` | multi-expert gate allocation |
| `fig08_stage_branch_modulation_seed42.png` | stage-branch modulation strength |
| `fig09_seed42_training_curves.png` | Batch9 checkpoint-selection curves |

The requested path `codex/paper-plot-skill-figures-v1` was not present in the
current project. The available local plotting skill used for figure style was:

```text
C:\Users\ROG\Desktop\paper\paper-plot-skills-main\plot-from-data
```

## 8. Paper-Level Decision

Recommended decision:

```text
Keep adaptive-attention-v1 as the current paper candidate.
Do not replace it with either multi-expert-gated-attention-v1 or
stage-aware-branch-attention-v1.
```

How to use these experiments in the paper:

1. Do not present them as the final proposed method.
2. Use them as internal diagnostic evidence that stronger stage-aware structures
   can shift attention toward late degradation but may damage overall
   cross-batch generalization.
3. If mentioned, frame them cautiously in a discussion or supplementary
   diagnostic paragraph.
4. For the main text, v1 remains the more defensible attention mechanism because
   it has the best mean RMSE, lowest variance, and a clean formula-level
   self-adaptive story.

## 9. Next Technical Direction

The evidence suggests that the next useful improvement should not simply add a
larger branch or more experts. A safer next version should constrain the new
stage mechanism:

- use residual mixing with adaptive-attention-v1 rather than full replacement;
- penalize excessive attention sharpness or gate collapse;
- add a middle-stage preservation term if late improvement is targeted;
- make the stage branch influence only temperature/gate, not the full hidden
  representation;
- validate with the same three-seed protocol before manuscript adoption.

In short:

```text
The formulas are scientifically meaningful, but the current trained candidates
are not strong enough to become the final attention model.
```
