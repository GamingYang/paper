# Current Window Context And Attention Ablation Guardrail

Date: 2026-07-22

Purpose:

This file is a long-context safety note for future work in this Codex window.
If context is compressed, lost, or partially missing, read this file first
before continuing SOH attention experiments or manuscript writing.

## 1. Current Task Must Stay Fixed

The current SOH manuscript task is not to invent endless new attention modules.

The current main attention candidate is:

```text
adaptive-attention-v1
```

The current main model story is:

```text
physical cubic degradation base + residual temporal network +
degradation-state-guided curve attention v1 + life-cycle adaptive MMD
```

The current writing goal is:

```text
Show that the proposed attention is self-adaptive because degradation state,
local curve morphology, and adaptive focus intensity each contribute to
lithium-ion battery SOH curve modeling.
```

Do not restart from ordinary attention. Do not replace v1 with multi-expert or
stage-branch attention unless a new fixed-protocol experiment proves a stable
three-seed improvement.

## 2. Fixed Experimental Boundary

Use the fixed SOH protocol:

```text
Batch 1-8  -> source training
Batch 9    -> target-domain development/adaptation and checkpoint selection
Batch 10   -> final blind test only
```

Batch10 must not be used for:

- training;
- tuning;
- checkpoint selection;
- hyperparameter choice;
- seed selection;
- model selection;
- deciding which figure to optimize.

If a new idea is motivated by Batch10 error inspection, label it as a diagnostic
unless the full protocol is restarted or an independent holdout is added.

## 3. Current Evidence State

Do not forget these results:

| Model | Three-seed mean RMSE | Std | Mean macro RMSE | Mean MAE | Decision |
|---|---:|---:|---:|---:|---|
| Original attention | 0.010381 | 0.000867 | 0.009171 | 0.006253 | baseline |
| Adaptive attention v1 | 0.009906 | 0.000557 | 0.008764 | 0.006189 | current main candidate |
| Multi-expert gate | 0.010993 | 0.001433 | 0.009635 | 0.006618 | not final |
| Stage-aware branch | 0.014712 | 0.005948 | 0.011676 | 0.007592 | not final |

Interpretation:

- `adaptive-attention-v1` is currently the best overall choice.
- Multi-expert gate and stage-aware branch are mechanistically meaningful but
  not stable enough to replace v1.
- The next work should focus on mechanism ablations for v1, not another v3/v4
  attention search.

Key paths:

```text
C:\Users\ROG\Desktop\paper\codex\adaptive-attention-v1
C:\Users\ROG\Desktop\paper\codex\adaptive-attention-late-aware-v2
C:\Users\ROG\Desktop\paper\codex\advanced-attention-candidates-v1
C:\Users\ROG\.codex\skills\battery-soh-q2-paper-v2
```

## 4. The Ablation Philosophy

The ablation study must not be:

```text
remove one module -> report worse result -> claim module is useful
```

That is too shallow.

The ablation study must test scientific hypotheses:

```text
Which physical or curve-related variable explains why the attention is adaptive?
```

The real claim is not merely:

```text
attention is useful.
```

The real claim is:

```text
degradation state, local curve morphology, and adaptive focus intensity each
provide independent information for modeling lithium-ion battery SOH curves.
```

Therefore, every ablation must answer one mechanism question.

## 5. Five Thinking Rules For Ablation Design

### 5.1 Reverse Review Method

Do not first pile up literature.

First ask:

```text
What do normal attention-based SOH predictors implicitly assume?
```

Likely shared assumptions:

1. The same temporal attention rule can be used for all life stages.
2. A sequence hidden state alone is enough to decide temporal importance.
3. Cycle position or SOH state does not need to explicitly condition attention.
4. Local curve slope and curvature are treated as ordinary input features, not
   as variables that should reshape attention.
5. Attention sharpness is fixed by learned logits, not explicitly adapted to
   degradation stage.

Our ablation experiments should attack these assumptions one by one.

### 5.2 Defect Amplification

Before running an ablation, ask like a strict reviewer:

```text
Could this design be just parameter count, feature concatenation, or random
regularization rather than real adaptive attention?
```

Common reviewer doubts:

1. The improvement may come from more parameters.
2. The degradation state may only duplicate cycle information.
3. Curve tokens may be noise or ordinary feature augmentation.
4. Adaptive temperature may be cosmetic.
5. Batch9 selection may not transfer to Batch10.

Each ablation should be able to answer at least one of these doubts.

### 5.3 Transfer And Structural Analogy

Do not randomly attach modules.

Use a mathematical analogy:

```text
Battery SOH degradation stage is to attention query/temperature
as task condition is to conditional representation learning.
```

This justifies why `s_i` should condition query and temperature.

Another analogy:

```text
Local voltage slope/curvature is to SOH curve modeling
as edge/gradient information is to visual structure recognition.
```

This justifies why local curve morphology `c_i,t` should enter attention
scoring, not merely final prediction.

### 5.4 Feasibility Quick Test

Do not immediately run a giant experimental suite.

Start with a small verification set:

1. implement the ablation;
2. run seed42 or one quick 150-epoch check if consistent with existing protocol;
3. inspect Batch9 selection and Batch10 performance;
4. only promote it to three seeds if it does not collapse.

However, final manuscript claims require stronger evidence than one seed.

### 5.5 Storyline Prewriting

Before each ablation, write:

```text
hypothesis -> evidence -> contribution
```

The experiment should serve this logic. Do not add an ablation just because it
is easy to run.

## 6. The Three Independent Contributions To Prove

### 6.1 Contribution A: Degradation State

Mechanism variable:

```text
s_i = [p_i, y_base,i, 1 - y_base,i]
```

Claim:

```text
The attention distribution should depend on the current degradation state,
because early, middle, and late SOH curves contain different predictive cues.
```

Suggested ablations:

1. `w/o degradation state`
   Remove `s_i` from query and temperature.

2. `cycle-only state`
   Use only `p_i`, remove `y_base,i` and `1 - y_base,i`.

3. `shuffled degradation state`
   Randomly shuffle `s_i` across samples while keeping input sequences intact.

Expected interpretation:

- If performance drops after removing or shuffling `s_i`, the model is not only
  gaining from extra parameters.
- If cycle-only is weaker than full state, the physical base contributes more
  than raw cycle position.

### 6.2 Contribution B: Local Curve Morphology

Mechanism variable:

```text
c_i,t = [V_i,t, Delta V_i,t, Delta^2 V_i,t]
```

Claim:

```text
Local voltage level, slope, and curvature help attention locate degradation
patterns related to SOH curve evolution.
```

Suggested ablations:

1. `w/o curve tokens`
   Remove `c_i,t` from attention scoring.

2. `voltage-only curve token`
   Use `V_i,t` but remove slope and curvature.

3. `w/o slope-curvature`
   Keep voltage token, remove `Delta V_i,t` and `Delta^2 V_i,t`.

Expected interpretation:

- If full curve tokens outperform voltage-only, then local change rate and
  curvature are not decorative.
- If removing curve tokens weakens stage-wise attention profiles, then the
  module is genuinely curve-aware.

### 6.3 Contribution C: Adaptive Focus Intensity

Mechanism variable:

```text
tau_i = tau_min + softplus(W_tau s_i)
```

Claim:

```text
The model should not use the same attention sharpness for all samples.
Early-stage curves may need smoother attention, while middle/late-stage curves
may require sharper focus on local degradation cues.
```

Suggested ablations:

1. `fixed temperature`
   Replace `tau_i` with a constant.

2. `no temperature scaling`
   Use softmax directly on attention logits.

3. `temperature from cycle only`
   Let temperature depend only on `p_i`.

Expected interpretation:

- If adaptive temperature improves mean RMSE or macro RMSE, the model benefits
  from sample-specific focus intensity.
- If temperature varies by degradation stage in visualization, it supports
  interpretability.

## 7. Recommended Minimal Ablation Set

If time is limited, run this minimal set first:

| Ablation | Tests which claim? | Priority |
|---|---|---|
| Original attention | baseline: no stage-conditioned curve attention | P0 |
| Adaptive attention v1 | full proposed attention | P0 |
| w/o degradation state | contribution of `s_i` | P0 |
| w/o curve tokens | contribution of `c_i,t` | P0 |
| fixed temperature | contribution of adaptive `tau_i` | P0 |
| shuffled degradation state | anti-spurious-control test | P1 |
| voltage-only curve token | slope/curvature contribution | P1 |
| cycle-only state | physical-base state contribution | P1 |

Do not run all variants blindly before checking runtime and whether the first
P0 results make sense.

## 8. Required Reporting Format For Each Ablation

Each ablation should be reported with four layers:

1. Performance:
   RMSE, macro RMSE, MAE, early/middle/late RMSE.

2. Stability:
   three-seed mean and standard deviation when promoted beyond quick test.

3. Mechanism:
   attention entropy, max attention weight, center of mass, temperature profile.

4. Interpretation:
   one sentence linking the result back to the hypothesis.

Example:

```text
Removing degradation state weakens Batch10 macro RMSE and reduces the
stage-wise separation of attention profiles, indicating that the proposed
attention does not merely aggregate temporal features but conditions temporal
focus on the battery degradation state.
```

## 9. Claims That Are Allowed

Allowed if supported by ablation results:

- The proposed attention is degradation-state-conditioned.
- Local curve morphology contributes to temporal aggregation.
- Adaptive temperature provides sample-dependent focus intensity.
- The attention mechanism is more interpretable than ordinary temporal
  attention because its intermediate quantities can be visualized by stage.

## 10. Claims To Avoid

Do not claim:

- The method fully solves late-stage prediction.
- The attention is universally superior under every seed and every stage.
- The cubic physical base is a full electrochemical model.
- MMD is the dominant innovation.
- Multi-expert or stage-branch attention is the final method.
- Batch10 was used to choose the model.

## 11. Current Decision

For the manuscript:

```text
Freeze adaptive-attention-v1 as the current main attention module.
Use the next ablation experiments to prove the independent roles of:
1. degradation state;
2. local curve morphology;
3. adaptive focus intensity.
```

For future experiments:

```text
Do not chase new attention versions unless the v1 mechanism ablations fail or
the teacher explicitly asks for a new architecture.
```

For writing:

```text
First write the hypothesis-evidence-contribution chain, then let ablations
serve that chain.
```
