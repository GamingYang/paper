# Advanced Attention Candidate Formula Derivation And Check

Date: 2026-07-22

This note fixes two candidate attention mechanisms for a small feasibility
study under the existing SOH protocol:

```text
Batch 1-8  -> source supervised training
Batch 9    -> target-domain MMD / checkpoint selection
Batch 10   -> final blind reporting only
```

The two candidates are not declared as final methods before three-seed results
are inspected.

## 1. Shared Notation

For sample `i`:

| Symbol | Meaning | Shape |
|---|---|---|
| `x_i` | raw voltage window | `L` |
| `p_i` | normalized cycle position | `1` |
| `y_base,i` | cubic physical SOH base | `1` |
| `H_i = {h_i,t}` | CNN-LSTM temporal features | `T x d_h` |
| `C_i = {c_i,t}` | curve-shape tokens | `T x d_c` |
| `s_i` | degradation state vector | `d_s` |
| `z_i` | aggregated temporal feature | `d_h` |

The retained physical base is:

```text
y_base,i = a p_i^3 + b p_i^2 + c p_i + d
```

The degradation state used by both candidates is:

```text
s_i = [p_i, y_base,i, 1 - y_base,i]
```

The local curve token is:

```text
c_i,t = [v_i,t, Delta v_i,t, Delta^2 v_i,t]
```

where `v_i,t` is the interpolated voltage token aligned with the CNN-LSTM
temporal length `T`.

The final SOH predictor remains physical-residual:

```text
y_hat_i = y_base,i + f_theta([z_i, phi(p_i)])
```

where `phi(p_i)` is the same physical-position embedding used in v1.

## 2. Candidate A: Multi-Expert Gated Curve Attention

### 2.1 Motivation

The original attention and v1 attention both use one attention surface for all
degradation stages. Candidate A assumes that early, middle, and late degradation
may require different temporal aggregation patterns. It therefore uses several
attention experts and lets a soft gate decide how much each expert contributes.

### 2.2 Expert Attention

Use `K = 3` experts:

```text
k in {early expert, middle expert, late expert}
```

Each expert has its own stage-conditioned query:

```text
q_i^(k) = tanh(W_s^(k) s_i)
```

Each expert scores each temporal token with hidden features, curve-shape tokens,
and the expert query:

```text
u_i,t^(k) = tanh(W_h^(k) h_i,t + W_c^(k) c_i,t + q_i^(k))
e_i,t^(k) = <q_i^(k), u_i,t^(k)>
```

Each expert has an adaptive temperature:

```text
tau_i^(k) = tau_min + softplus(W_tau^(k) s_i)
```

The expert-specific temporal attention is:

```text
alpha_i,t^(k) = exp(e_i,t^(k) / tau_i^(k))
                / sum_j exp(e_i,j^(k) / tau_i^(k))
```

The expert context is:

```text
z_i^(k) = sum_t alpha_i,t^(k) h_i,t
```

### 2.3 Gate Fusion

The gate decides which expert is more suitable for the current degradation
state:

```text
g_i = softmax(W_g tanh(W_r s_i))
```

where:

```text
g_i = [g_i^(1), g_i^(2), g_i^(3)]
sum_k g_i^(k) = 1
```

The final temporal representation is:

```text
z_i = sum_k g_i^(k) z_i^(k)
```

### 2.4 What Is Adaptive Here

Adaptivity appears at three levels:

1. Each expert computes different temporal attention.
2. Expert temperature changes with degradation state.
3. The gate softly changes the expert mixture for each sample.

Compared with v1:

```text
v1: one attention mechanism changes its query/temperature with stage.
Candidate A: multiple attention experts are softly selected by stage.
```

### 2.5 Check

- `alpha_i,t^(k)` sums to 1 over temporal tokens for each expert.
- `g_i^(k)` sums to 1 over experts for each sample.
- `z_i^(k)` has shape `d_h`.
- `z_i` has shape `d_h`, so the downstream residual head remains unchanged.
- The method adds structure to attention, not only extra loss weight.

## 3. Candidate B: Degradation-Stage-Aware Branch Attention

### 3.1 Motivation

Candidate B assumes the model should first build an explicit degradation-stage
embedding, then use it to modulate temporal features before attention. This is
different from simply concatenating the stage state into attention scores.

### 3.2 Stage Branch Input

Use both global state and curve summary.

First define a curve summary:

```text
m_i = mean_t([v_i,t, |Delta v_i,t|, |Delta^2 v_i,t|])
```

Then construct the branch input:

```text
b_i = [s_i, m_i]
```

### 3.3 Stage Embedding

The branch generates a latent degradation-stage embedding:

```text
r_i = tanh(W_2 ReLU(W_1 b_i))
```

Here `r_i` is not a hard early/middle/late label. It is a continuous state
representation learned from cycle position, physical-base SOH, remaining health,
and local curve shape.

### 3.4 Feature Modulation

The stage embedding controls the temporal features through a scaled FiLM-style
modulation:

```text
gamma_i = tanh(W_gamma r_i)
beta_i  = tanh(W_beta r_i)
h'_i,t = h_i,t * (1 + rho_gamma gamma_i) + rho_beta beta_i
```

For stable training, the feasibility run uses small constants:

```text
rho_gamma = 0.5
rho_beta = 0.1
```

This means the model first asks:

```text
What degradation state am I seeing?
```

and then changes how the CNN-LSTM features are expressed before attention.

### 3.5 Stage-Conditioned Attention

The attention query and temperature are generated from the branch embedding:

```text
q_i = tanh(W_q r_i)
tau_i = tau_min + softplus(W_tau r_i)
```

The score is:

```text
u_i,t = tanh(W_h h'_i,t + W_c c_i,t + q_i)
e_i,t = <q_i, u_i,t>
```

The attention distribution is:

```text
alpha_i,t = exp(e_i,t / tau_i) / sum_j exp(e_i,j / tau_i)
```

The final representation is:

```text
z_i = sum_t alpha_i,t h'_i,t
```

### 3.6 What Is Adaptive Here

Adaptivity appears through:

1. an explicit degradation-stage branch `r_i`;
2. stage-conditioned feature modulation `h'_i,t`;
3. stage-conditioned query `q_i`;
4. stage-conditioned temperature `tau_i`.

Compared with v1:

```text
v1: degradation state directly conditions attention.
Candidate B: degradation state and curve summary first form a branch embedding,
then the embedding modulates temporal features and attention.
```

### 3.7 Check

- `r_i` is sample-level, not token-level, matching the idea of degradation state.
- `gamma_i` and `beta_i` have shape `d_h`, so they can modulate each `h_i,t`.
- `rho_gamma` and `rho_beta` stabilize the branch and do not use test data.
- `alpha_i,t` sums to 1 over temporal tokens.
- `z_i` has shape `d_h`, so the downstream residual head remains unchanged.
- The branch provides interpretable quantities: stage embedding, modulation
  strength, attention temperature, and attention profile.

## 4. Training Objective

Both candidates keep the retained training objective:

```text
L = MSE_source + lambda_mmd L_MMD(source, target_adapt)
```

MMD is active only during the middle training phase, following the existing
v1 protocol. No Batch10 samples are used for optimization or checkpoint
selection.

## 5. Feasibility Criteria

A candidate is promising only if it satisfies most of the following:

1. Three-seed mean Batch10 RMSE is no worse than adaptive-attention-v1.
2. Macro RMSE is stable or improved.
3. Stage-wise RMSE does not show a large late-stage collapse.
4. Attention/gate/branch visualizations show stage-dependent behavior.
5. The method can be explained without claiming Batch10-driven tuning.

## 6. Formula-Level Final Check Before Coding

- Both candidates keep the same data protocol, physical base, residual head,
  MMD schedule, optimizer, seeds, and epochs as v1.
- Both candidates output a `d_h` temporal representation.
- Neither candidate uses Batch10 information in the formula or training
  objective.
- Candidate A tests whether stage-wise expert specialization helps.
- Candidate B tests whether explicit degradation-state encoding and feature
  modulation help.
- The two candidates are structurally different enough to compare.
