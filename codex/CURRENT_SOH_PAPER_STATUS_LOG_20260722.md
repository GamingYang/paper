# Current SOH Paper Status Log

Date: 2026-07-22

This log records the current state of the independent lithium-ion battery SOH
prediction manuscript after the adaptive-attention-v1 experiment and the
late-aware-v2 diagnostic. It does not modify or delete any previous Markdown
guidance files.

## 1. Active Manuscript Line

The active paper should remain SOH-only:

```text
Physics-guided hybrid transfer learning for cross-batch lithium-ion battery
SOH prediction.
```

Do not merge the current SOH manuscript with the RUL/EOL line unless a separate
SOH-to-RUL bridge experiment is completed.

The fixed protocol remains:

```text
Batch 1-8  -> source training
Batch 9    -> target-domain adaptation / checkpoint selection
Batch 10   -> final blind-test reporting only
```

Batch 10 must not be used for training, hyperparameter tuning, checkpoint
selection, seed selection, model selection, or figure selection.

## 2. Current Model Status

The previous authoritative retained model was:

```text
physical cubic degradation base + residual CNN-LSTM + ordinary temporal
attention + lifecycle-adaptive MMD
```

Main code path:

```text
C:\Users\ROG\Desktop\paper\codex\final-retained-v1\train_final_retained_scheme_a.py
```

Its seed42 Batch10 result:

| Model | Seed | Batch10 RMSE | Macro RMSE | MAE |
|---|---:|---:|---:|---:|
| original temporal attention | 42 | 0.009499 | 0.008301 | 0.005704 |

## 3. Adaptive-Attention-v1 Status

The v1 attention candidate keeps the same physical-residual and MMD protocol,
but replaces ordinary temporal attention with degradation-state-guided curve
attention.

Code path:

```text
C:\Users\ROG\Desktop\paper\codex\adaptive-attention-v1\run_adaptive_attention_v1.py
```

Core idea:

```text
s_i = [p_i, y_base,i, 1 - y_base,i]
c_i,t = [V_i,t, Delta V_i,t, Delta^2 V_i,t]

q_i = tanh(W_s s_i)
e_i,t = q_i^T tanh(W_h h_i,t + W_c c_i,t + q_i)
tau_i = tau_min + softplus(W_tau s_i)
alpha_i,t = softmax(e_i,t / tau_i)
z_i = sum_t alpha_i,t h_i,t
```

Three-seed Batch10 results:

| Seed | Best epoch | Batch10 RMSE | Macro RMSE | MAE | Early RMSE | Middle RMSE | Late RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 85 | 0.009278 | 0.008216 | 0.005733 | 0.003545 | 0.012223 | 0.026287 |
| 7 | 135 | 0.010338 | 0.008995 | 0.006085 | 0.003635 | 0.014774 | 0.016522 |
| 123 | 93 | 0.010103 | 0.009080 | 0.006749 | 0.005068 | 0.013395 | 0.022435 |

Three-seed mean:

| Model | Mean Batch10 RMSE | Std | Mean Macro RMSE | Mean MAE |
|---|---:|---:|---:|---:|
| original attention | 0.010381 | 0.000867 | 0.009171 | 0.006253 |
| adaptive-attention-v1 | 0.009906 | 0.000557 | 0.008764 | 0.006189 |

Interpretation:

- v1 improves three-seed RMSE and macro RMSE relative to the original attention.
- v1 provides clearer attention-stage adaptivity in the visualization.
- In seed42, v1 slightly worsens the late-stage RMSE relative to original
  attention, so the late-stage issue should be discussed cautiously.
- v1 should remain the current candidate attention version, but not be declared
  final until mechanism ablations are completed.

## 4. Late-Aware-v2 Diagnostic

The late-aware-v2 run kept the v1 attention module unchanged and only added a
late-stage gated source-supervision weight:

```text
r_i = sigmoid(20 * (p_i - 0.60))
L_source = mean((1 + 1.5 r_i) * (y_i - yhat_i)^2)
```

Code path:

```text
C:\Users\ROG\Desktop\paper\codex\adaptive-attention-late-aware-v2\run_adaptive_attention_late_aware_v2.py
```

Result summary, seed42, 150 epochs:

| Model | Seed | Best epoch | Batch9 RMSE | Batch10 RMSE | Macro RMSE | MAE | Early RMSE | Middle RMSE | Late RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original temporal attention | 42 | 140 | n/a | 0.009499 | 0.008301 | 0.005704 | 0.003661 | 0.012651 | 0.025532 |
| adaptive-attention-v1 | 42 | 85 | 0.008864 | 0.009278 | 0.008216 | 0.005733 | 0.003546 | 0.012224 | 0.026285 |
| late-aware-v2 | 42 | 148 | 0.008887 | 0.011113 | 0.009766 | 0.006901 | 0.003814 | 0.014905 | 0.030478 |

Interpretation:

- late-aware-v2 is not a viable replacement for v1 under this single-seed test.
- The late-gated loss worsens overall RMSE, middle-stage RMSE, and late-stage
  RMSE.
- The attention visualization shows that v2 produces more extreme attention
  concentration and shifts early/middle attention toward late tokens, which may
  explain the reduced generalization.
- Do not run three seeds for this v2 configuration unless a new formula is
  proposed. Treat this run as a negative diagnostic.

Important protocol note:

The late-aware-v2 idea was triggered by a Batch10 late-stage observation.
Therefore, it should be framed as a post-hoc diagnostic unless the full model
selection protocol is restarted or an independent holdout is introduced.

## 5. Current Recommendation

Do not adopt late-aware-v2 as the final attention model.

Keep adaptive-attention-v1 as the current attention candidate because it has:

- better three-seed mean RMSE than the original attention;
- better three-seed mean macro RMSE;
- visible degradation-stage-dependent attention behavior;
- a more defensible formula-level story than simple concatenation.

However, do not overclaim v1. The correct manuscript stance is:

```text
The proposed degradation-state-guided curve attention improves overall
cross-batch SOH prediction under the current three-seed protocol, while its
late-stage behavior remains a limitation that requires further diagnostic or
regularization work.
```

## 6. Markdown Guidance Audit

Cleanup checklist path:

```text
C:\Users\ROG\Desktop\paper\codex\MD_GUIDANCE_UPDATE_AUDIT_20260722.txt
```

Key conclusion:

- `battery-soh-q2-paper` is not obsolete, but it needs an update.
- The SOH-only protocol and claim boundaries remain correct.
- The current model identity, attention description, artifact paths, and
  metrics are outdated in that skill/reference.
- June 2026 evidence/acceptance Markdown files should be treated as historical
  evidence packages unless refreshed.
- Old Chinese manuscript drafts should be regenerated or repaired before reuse
  because they contain old ordinary-attention equations and may have encoding
  corruption.

## 7. Immediate Next Steps

1. Freeze v1 as the current candidate, not v2.
2. If time permits, run mechanism ablations for v1:
   - w/o degradation state;
   - w/o curve-shape tokens;
   - w/o slope and curvature;
   - fixed temperature;
   - shuffled degradation state.
3. Update the `battery-soh-q2-paper` skill only after the final attention
   identity is frozen.
4. Rewrite Methodology 3.4 around degradation-state-guided curve attention.
5. Rewrite Section 4.5 so ablations test scientific hypotheses rather than only
   deleting modules.
6. Keep all Batch10 visualizations as reporting-only evidence.
