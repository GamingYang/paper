# SOH MMD Storytelling Literature Audit

Date: 2026-06-23

Scope: SOH-only manuscript positioning. RUL/EOL is not used as evidence for
the current manuscript.

## 1. Bottom Line

The previous conservative wording, "MMD is only a small auxiliary module", is
too narrow if we judge contribution only by Batch10 RMSE. A stronger and still
defensible position is:

> The paper introduces a degradation-aware exponential MMD regularizer that
> increases source-target feature alignment pressure as the estimated
> degradation state deepens. Its value should be argued as a physically
> motivated transfer-learning idea and representation-shaping mechanism, not
> merely as a large RMSE reducer.

This does not mean that MMD can be claimed as a dominant accuracy driver. In
our current experiments, the final model and w/o-MMD model are very close in
Batch10 RMSE, and the multi-seed mean of w/o-MMD is slightly lower. However,
the representation diagnostic is favorable to MMD:

| Diagnostic | Final | w/o MMD | Interpretation |
|---|---:|---:|---|
| RBF multikernel MMD^2, z-scored fused features | 0.153337 | 0.464046 | final is about 67% lower |
| Domain classifier AUC | 0.979493 | 0.988835 | source/target slightly less separable |
| Domain classifier accuracy | 0.940667 | 0.951500 | source/target slightly less separable |

The best story is therefore not "MMD greatly improves RMSE", but:

> Under a strong CRNN residual predictor where RMSE is already near saturation,
> degradation-aware MMD provides a principled way to shape latent
> cross-batch representations, especially for deeper degradation regions.

## 2. What Is Actually New In Our MMD

The local implementation is not a plain static MMD. In
`codex/final-retained-v1/train_final_retained_scheme_a.py`, the regularizer is
implemented as `LifetimeAdaptiveMMD`:

```python
current_soh = torch.clamp(torch.mean(target_base_soh), max=1.0)
dynamic_weight = torch.exp(self.lambda_scale * (1.0 - current_soh))
return dynamic_weight * self.compute_mmd(source_features, target_features)
```

This is important. The MMD strength is not just controlled by a fixed global
coefficient. It is internally reshaped by an exponential function of the target
base SOH. As SOH decreases, `1 - current_soh` increases, so the feature
alignment penalty becomes stronger.

Recommended name:

```text
degradation-aware exponential MMD
```

Conservative formal name:

```text
lifecycle-adaptive MMD regularization
```

Aggressive but still defensible manuscript phrase:

```text
an exponential degradation-aware domain alignment term that strengthens
cross-batch feature matching in deeper degradation states
```

Avoid:

- "we invent MMD";
- "MMD is the dominant source of accuracy improvement";
- "the exponential weight is derived from a full electrochemical mechanism";
- "the MMD module proves universal cross-chemistry generalization".

## 3. Literature: MMD Is Not New, But Our Specific Battery Framing May Be

### 3.1 Battery SOH papers already use MMD

1. Chen et al., "A Self-attention Knowledge Domain Adaptation Network for
   Commercial Lithium-ion Batteries State-of-health Estimation under Shallow
   Cycles" (arXiv 2023).

   Link: https://arxiv.org/abs/2304.05084

   This paper uses self-attention distillation plus multi-kernel MMD for
   unsupervised transfer from full-cycle source data to shallow-cycle target
   data. It frames MMD as a bridge for distribution discrepancy caused by SOC
   range and operating-condition shift.

   Relation to us:

   - It overlaps with our use of MMD for battery SOH domain adaptation.
   - It does not appear, from the accessible text, to use an SOH-dependent
     exponential weighting of the MMD strength.
   - Its story is "shallow-cycle unlabeled target domain"; our story is
     "cross-batch SOH prediction with degradation-state-dependent alignment".

2. da Silva et al., "Conformalized Transfer Learning for Li-ion Battery State
   of Health Forecasting under Manufacturing and Usage Variability" (arXiv
   2026).

   Link: https://arxiv.org/abs/2603.24475

   This paper combines LSTM, MMD-based domain adaptation, and conformal
   prediction. It explicitly uses MMD to align latent feature distributions
   between simulated/source and target domains.

   Relation to us:

   - It confirms that MMD-based battery SOH transfer learning is an active and
     publishable research direction.
   - It uses MMD with a global regularization coefficient selected by
     validation; it does not appear to be an exponential SOH/lifecycle-weighted
     MMD.
   - Its extra novelty is uncertainty calibration; our extra novelty is
     low-order degradation prior plus residual temporal learning plus
     lifecycle-adaptive alignment.

3. General weighted MMD literature exists.

   Yan et al., "Mind the Class Weight Bias: Weighted Maximum Mean Discrepancy
   for Unsupervised Domain Adaptation" (arXiv 2017).

   Link: https://arxiv.org/abs/1705.00609

   This paper shows that ordinary MMD can be problematic when class priors
   differ, and proposes weighted MMD to account for class-prior shift.

   Relation to us:

   - This prevents us from claiming "weighted MMD" as a general first.
   - But its weights are class-prior/pseudo-label weights for classification,
     not degradation-state exponential weights for battery SOH regression.

4. Dynamic MMD/domain alignment also exists outside batteries.

   Example: EEG domain adaptation papers use dynamic distribution alignment
   with MMD/CMMD.

   Link: https://arxiv.org/abs/2502.16485

   Relation to us:

   - Dynamic alignment is a known transfer-learning idea.
   - Our novelty must be battery-specific: the alignment weight is tied to a
     degradation/SOH proxy rather than generic training dynamics or class
     distribution.

### 3.2 Search status on exact duplication

Searches performed included combinations of:

- "life-cycle adaptive MMD";
- "lifecycle adaptive maximum mean discrepancy";
- "exponential weighted MMD lithium-ion battery SOH";
- "degradation-stage weighted MMD battery SOH";
- "adaptive MMD lithium-ion battery state of health".

Within the accessible web results, I did not find a battery SOH paper that
clearly matches our exact form:

```text
MMD_weight = exp(lambda * (1 - estimated_SOH))
```

This is not proof of global novelty. It is enough to justify a cautious claim:

> To the best of our knowledge, few existing battery SOH transfer models couple
> MMD alignment intensity directly to the estimated degradation state through an
> exponential lifecycle weighting function.

Use "few existing" or "has received limited attention", not "no one has ever".

## 4. Why The Exponential Weight Has Physical Story Value

There is a reasonable battery-degradation argument behind the exponential
weight. Lithium-ion capacity fade is often nonlinear and stage-dependent. Some
published work has even argued for simple exponential or stretched-exponential
capacity-fade laws.

Useful reference:

Cuervo-Reyes and Flueckiger, "One law to rule them all: Stretched exponential
master curve of capacity fade for Li-ion batteries" (arXiv 2019).

Link: https://arxiv.org/abs/1902.05787

Relevant idea:

- Despite complex degradation mechanisms, capacity-retention trajectories can
  collapse onto a simple stretched-exponential master curve in their study.
- The paper connects this simple law to diffusion-to-traps style physical
  reasoning.

How to use it:

> Motivated by the nonlinear and often acceleration-prone nature of battery
> capacity fade, the proposed MMD weight adopts an exponential degradation
> schedule. The schedule does not claim to reproduce electrochemical kinetics;
> instead, it encodes a simple inductive bias that later degradation states may
> require stronger cross-batch representation alignment.

This is a better version of "life curves look exponential". It is physically
inspired, but not overclaiming derivation from electrochemistry.

## 5. Why The Simple Cubic Base Can Be A Strength

The cubic base should not be defended as a full physical model. It can be
defended as a low-order nonlinear degradation prior. This fits a broader
literature pattern: simple, interpretable degradation models can still be
publishable when they solve a practical modeling problem.

Useful reference:

Salucci et al., "Multivariable Fractional Polynomials for lithium-ion batteries
degradation models under dynamic conditions" (arXiv 2021).

Link: https://arxiv.org/abs/2102.08111

Relevant idea:

- The paper explicitly promotes fractional polynomial regression because of
  simplicity, ease of computation, interpretability, generalizability, and BMS
  implementability.
- It does not sell the method as a full electrochemical model.

How to use it:

> Instead of replacing an electrochemical model, the cubic base serves as a
> compact trend prior. It deliberately leaves cell-specific deviations to the
> residual temporal network. This separation follows a simple-but-useful design
> philosophy: use a low-order trend to anchor the global degradation direction,
> and reserve neural capacity for deviations that the low-order prior cannot
> capture.

This is the "大道至简" story, translated into SCI language.

## 6. Examples Of Papers Where Contribution Is Not Only A Large Accuracy Jump

### Example A: Benchmark paper

Wang et al., "Open access dataset, code library and benchmarking deep learning
approaches for state-of-health estimation of lithium-ion batteries", Journal of
Energy Storage, 2024.

Link: https://scholar.xjtu.edu.cn/en/publications/open-access-dataset-code-library-and-benchmarking-deep-learning-a/

This paper's contribution is not a single new high-accuracy model. It publishes
a dataset/code library, standardizes preprocessing, benchmarks five DL models,
and discusses future directions. Its story is field infrastructure and fair
comparison.

Lesson for us:

> A paper can be valuable because it proposes a better evaluation frame or a
> clearer modeling principle, not only because it wins every metric by a large
> margin.

Our analogous angle:

> fixed cross-batch source/development/blind-test protocol plus module-level
> diagnostics.

### Example B: ACLA Neural-ODE hybrid paper

Li et al., "A novel Neural-ODE model for the state of health estimation of
lithium-ion battery using charging curve" (arXiv 2025).

Link: https://arxiv.org/pdf/2505.05803

The paper proposes an ANODE/CNN/LSTM/attention hybrid. In its Table 5, ACLA is
strong on HUST, but on TJU the NODE baseline has a lower SOH RMSE than ACLA
while ACLA is better on the EOL metric. The paper still frames ACLA as a useful
generalization/robustness architecture across datasets and metrics.

Lesson for us:

> A method can be positioned around robustness, architecture rationale, or
> cross-dataset behavior even when it is not uniformly best in every cell,
> metric, or dataset.

Our analogous angle:

> MMD does not need to dominate every RMSE comparison if it demonstrably shapes
> representation alignment and fits the cross-batch adaptation story.

### Example C: Domain-knowledge-guided linear model

Lanubile et al., "Domain knowledge-guided machine learning framework for state
of health estimation in Lithium-ion batteries" (arXiv 2024).

Link: https://arxiv.org/abs/2409.14575

This paper uses domain-knowledge health indicators and a linear regression
model. The story is not a deep model novelty race; it is physical insight,
real-world feature availability, partial profiles, and BMS practicality.

Lesson for us:

> Simple modules can be publishable when they encode the right inductive bias
> and are validated in a realistic operating/evaluation frame.

Our analogous angle:

> cubic base + degradation-aware MMD is a simple inductive-bias pair: one anchors
> the output trend, the other controls latent alignment pressure by degradation
> stage.

### Example D: Fractional-polynomial degradation model

Salucci et al. promote polynomial-style statistical degradation modeling for
interpretability and implementability rather than black-box accuracy alone.

Lesson for us:

> Simplicity can be framed as an engineering contribution if it reduces
> complexity, supports BMS deployment, and remains competitive.

Our analogous angle:

> The physical base alone is weak, but the prior-plus-residual decomposition is
> strong. The contribution is not the cubic curve by itself; it is the division
> of labor between compact degradation trend and neural residual.

## 7. Horizontal Comparison

| Work type | What they sell | Is accuracy always dramatically better? | Relation to our story |
|---|---|---|---|
| SKDAN shallow-cycle SOH | self-attention + MK-MMD for transfer from full-cycle to shallow-cycle data | claims good RMSE, but core novelty is transfer setting | MMD in battery SOH exists; our distinction is SOH-dependent exponential weighting and cross-batch protocol |
| Conformalized TL SOH | LSTM + MMD + conformal prediction | contribution includes reliability/uncertainty, not just point RMSE | MMD is accepted in SOH transfer; our missing piece is stronger uncertainty if aiming higher |
| Weighted MMD DA | class-prior weighted MMD | method contribution is theoretical/DA bias correction | weighted MMD exists generally; our weight is degradation-state based for SOH regression |
| ACLA Neural-ODE hybrid | ANODE + CNN/LSTM/attention generalization | not uniformly best on every dataset/metric | supports writing contribution around generalization/robustness rather than universal metric wins |
| SOHbenchmark | dataset/code/preprocessing/benchmark | not a new SOTA predictor | supports protocol and evidence-chain contribution |
| MFP regression | simple polynomial statistical degradation model | competitive rather than always best | supports "simple low-order prior + interpretability" story |
| Our SOH model | cubic trend prior + residual CRNN + attention + degradation-aware exponential MMD + fixed blind test | MMD has small RMSE effect but strong representation diagnostic | should be framed as physical-residual decomposition and degradation-aware transfer regularization |

## 8. Revised Innovation Points

### Innovation 1: Physical-residual decomposition

Old weak version:

> We use a cubic physical base and a neural network.

Better version:

> We propose a physical-residual SOH decomposition in which a low-order
> nonlinear degradation prior anchors the global capacity-fade trajectory, while
> a residual temporal network learns cell-specific deviations under cross-batch
> shift.

Evidence:

- Final Batch10 RMSE: 0.009499.
- w/o physical base Batch10 RMSE: 0.020716.
- Stage/tail diagnostics show the physical prior is particularly useful in
  later-stage/tail behavior.

### Innovation 2: Degradation-aware exponential MMD

Old weak version:

> We add MMD for domain adaptation.

Better version:

> We introduce a degradation-aware exponential MMD regularizer that increases
> domain-alignment pressure as the target-development degradation state deepens,
> reflecting the intuition that late-life SOH trajectories exhibit stronger
> nonlinear divergence and require more careful cross-batch feature alignment.

Evidence:

- Code-level formula: `exp(lambda_scale * (1 - current_soh)) * MMD`.
- Representation diagnostic: fused-feature MMD^2 decreases from 0.464046 to
  0.153337 relative to w/o-MMD.
- Accuracy effect is small; therefore present it as representation and
  regularization evidence.

### Innovation 3: Strict cross-batch evidence chain

Old weak version:

> We compare with baselines.

Better version:

> We evaluate the proposed model under a fixed cross-batch source /
> target-development / blind-test protocol, and combine main comparisons,
> ablations, source-only controls, target-budget sensitivity, representation
> diagnostics, stage/tail analysis, and efficiency measurements.

Evidence:

- Batch 1-8: source training.
- Batch 9: target-development/adaptation and checkpoint selection.
- Batch 10: blind-test reporting only.
- Baselines include CNN-LSTM, LSTM, GRU, physical-only, PCA-GPR, PCA-SVR.
- Additional controls include source-only and Batch9 budget sensitivity.

## 9. Recommended Manuscript Wording

### Introduction gap paragraph

Existing transfer-learning SOH studies commonly use MMD or adversarial losses
to reduce source-target discrepancy, but most treat the alignment objective as
a static distribution-matching term. Battery degradation, however, is not
uniform over the life cycle. As SOH decreases, cell-to-cell divergence and
nonlinear aging behavior may become more pronounced. This motivates a
degradation-aware alignment strategy in which the strength of feature
distribution matching is modulated by the estimated degradation state.

### Method paragraph

The proposed lifecycle-adaptive MMD is defined as a multi-kernel MMD term
multiplied by an exponential degradation weight. Let the target-development
base SOH estimate in a mini-batch be denoted by `s_t`. The alignment weight is
`exp(alpha * (1 - s_t))`, where `alpha` is a scale factor. This schedule assigns
lower alignment pressure near the beginning of life and stronger alignment
pressure as the estimated SOH decreases. The term is not intended as a complete
electrochemical degradation law; rather, it encodes a compact monotonic prior
that later-life features should be more carefully aligned across batches.

### Results paragraph

The direct RMSE improvement of MMD is modest, which is expected because the
residual CRNN already provides a strong predictor and the blind-test RMSE is
near a low-error regime. Nevertheless, representation diagnostics show that the
degradation-aware MMD reduces source-target fused-feature discrepancy under an
RBF multi-kernel MMD metric. This supports interpreting the module as a
regularizer that shapes cross-batch feature geometry rather than as an isolated
accuracy booster.

### Discussion paragraph

The results suggest that physically motivated simplicity can be useful in
cross-batch SOH modeling. The cubic base alone is insufficient, but it provides
a compact nonlinear trend around which a temporal residual model can learn
cell-level deviations. Similarly, the exponential MMD schedule is not a
mechanistic aging model, but it supplies a degradation-shaped inductive bias
for domain alignment. This design may encourage future work to replace static
domain losses with degradation-state-aware transfer objectives.

## 10. What We Should Not Do

- Do not say MMD is the strongest module; the physical base has much stronger
  ablation evidence.
- Do not hide the multi-seed nuance. State that MMD's point-error gain is small
  and mixed, while its representation alignment effect is clearer.
- Do not claim "first MMD for battery SOH"; this is false.
- Do not claim the exponential weight is a proven battery aging law.
- Do not claim cubic base is electrochemical. It is a low-order nonlinear
  degradation prior.
- Do not use Batch10 to choose a new MMD wording, coefficient, degree, or
  model variant.

## 11. Final Position

We can be more confident, but the confidence should move from "we beat every
baseline by MMD" to "we propose a clear and simple inductive-bias framework".

Recommended contribution sentence:

> This work proposes a physics-guided hybrid transfer framework for cross-batch
> lithium-ion battery SOH prediction, combining a low-order degradation prior,
> residual temporal correction, and a degradation-aware exponential MMD
> regularizer that adapts feature alignment strength to the estimated life-cycle
> state.

This version is stronger than the previous conservative story, while still
surviving a careful reviewer check.

