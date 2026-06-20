# Authoritative Mainline: Early-to-Mid-Life EOL/RUL Prognostics

Created: 2026-06-03  
Project root: `C:\Users\ROG\Desktop\paper`  
Status: authoritative project mainline until the user explicitly replaces it.

## 1. Locked Research Direction

The project mainline is:

```text
Cross-batch small-target early-to-mid-life EOL/RUL prediction for lithium-ion
batteries, using capacity-degradation fingerprints, physical consistency, and
domain adaptation.
```

Chinese working title:

```text
面向跨批次小样本目标域的早中期 EOL/RUL 预测：
融合容量退化指纹、物理一致性约束与域自适应的电池寿命预后框架
```

Do not revert to a generic "CNN-LSTM-Attention-MMD for SOH/RUL" story unless
the user explicitly asks to abandon this mainline.

## 2. Engineering Pain Point

The practical problem is:

> In a new battery batch or new operating domain, full EOL labels are expensive
> and slow to obtain. However, BMS operation, warranty assessment, second-life
> screening, and maintenance planning need reliable lifetime-risk estimation
> before complete degradation trajectories are available.

Therefore, the paper should emphasize **early-to-mid-life lifetime prognostics
under cross-batch and small-target-domain conditions**, not merely pointwise
SOH/RUL curve fitting.

## 3. Core Task Reframing

Current processed RUL labels satisfy:

```text
RUL(current cycle) = EOL cycle - current cycle
```

Therefore, the preferred task is:

```text
Predict cell-level EOL first, then derive pointwise RUL from:
predicted_RUL(cycle) = predicted_EOL - current_cycle
```

This is the default RUL/EOL framing for future experiments and paper writing.

## 4. Fixed Data Protocol

The fixed split is non-negotiable:

```text
Batch 1-8  -> source domain
Batch 9    -> target_adapt / target-domain development set
Batch 10   -> target_test / blind-test set
```

Rules:

- Batch 10 must never be used for training, target adaptation, checkpoint
  selection, horizon selection, model selection, feature selection, or
  hyperparameter tuning.
- If Batch 9 labels are used, call the protocol **small labeled target-domain
  adaptation**, not unsupervised domain adaptation.
- Batch 9 must not be used simultaneously as training data and validation data
  for the same `source+target_adapt` result. Use a cell-level split inside
  Batch 9 or leave-one-target-cell-out validation for model/horizon selection.
- Keep `source_only`, `source+target_adapt`, and `target_test blind-test`
  results clearly separated.
- Reports may include Batch 10 final metrics, but must not sort, select, or
  label a model as "best" according to Batch 10. Any "best" model/horizon must
  be selected using Batch 9 validation only.

## 5. Current Valid Evidence

Latest summary:

`C:\Users\ROG\Desktop\paper\codex\rul-eol-experiments-v1\RUL_EOL_experiment_summary_20260603.md`

Latest strict-protocol baseline:

`C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict-eol-protocol-v1_report.md`

Latest subagent result audit:

`C:\Users\ROG\Desktop\paper\codex\SUBAGENT_ROUND6_AUDIT_20260604.md`

Previous subagent result audits:

`C:\Users\ROG\Desktop\paper\codex\SUBAGENT_ROUND5_AUDIT_20260604.md`

`C:\Users\ROG\Desktop\paper\codex\SUBAGENT_ROUND4_AUDIT_20260604.md`

`C:\Users\ROG\Desktop\paper\codex\SUBAGENT_ROUND3_AUDIT_20260604.md`

`C:\Users\ROG\Desktop\paper\codex\SUBAGENT_ROUND2_AUDIT_20260603.md`

`C:\Users\ROG\Desktop\paper\codex\SUBAGENT_RESULTS_AUDIT_20260603.md`

Data audit:

`C:\Users\ROG\Desktop\paper\codex\data-audit-v1\battery_data_audit.md`

Exploratory cell-level EOL fingerprint result:

`C:\Users\ROG\Desktop\paper\codex\rul-eol-experiments-v1\cell-level-eol-fingerprint-v1_report.md`

Exploratory v1 results after removing direct `observed_ratio` leakage:

| Observed horizon | Best target-test EOL/RUL RMSE |
|---:|---:|
| 50 cycles | 158.73 cycles |
| 100 cycles | 133.49 cycles |
| 200 cycles | 147.33 cycles |
| 300 cycles | 140.49 cycles |
| 500 cycles | 77.08 cycles |
| 800 cycles | 97.61 cycles |
| 1000 cycles | 52.50 cycles |

Target-adapt-selected ExtraTrees with source+target_adapt at 500 observed
cycles achieves:

```text
target_test EOL/RUL RMSE ≈ 77.08 cycles
target_test EOL/RUL MAE  ≈ 59.39 cycles
```

This supports an **early-to-mid-life** claim. It does not support a very-early
50/100-cycle below-100-cycle claim.

Strict-protocol result accepted on 2026-06-03:

- Script:
  `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict_eol_protocol_v1.py`
- Report:
  `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict-eol-protocol-v1_report.md`
- Candidate selection used Batch 9 only.
- `source_plus_target_adapt_loocv` used leave-one-Batch-9-cell-out validation.
- Batch 10 was evaluated only after Batch-9-based model/horizon selection.
- Forbidden model inputs were excluded:
  `horizon`, `n_observed`, `last_observed_cycle`, `observed_ratio`,
  full cell length, EOL/RUL labels, and cell/batch identifiers.
- Global Batch-9-selected baseline:
  `source_plus_target_adapt_loocv` + `extra_trees` + 1000 observed cycles.
- Batch 9 validation RMSE: 75.26 cycles.
- Batch 10 blind-test RMSE: 56.19 cycles.
- Batch 10 blind-test MAE: 49.42 cycles.
- This is a strict baseline/proof-of-concept, not the final proposed
  publishable architecture.
- This protocol uses labeled Batch 9 EOL targets for supervised small-target
  adaptation. Do not call it unsupervised domain adaptation.
- `y_nominal` features remain conditionally safe only if framed as historical
  measured/estimated capacity-SOH information. A later raw-capacity or `dq`
  variant is still required to reduce reviewer attack surface.

Subagent experiments audited on 2026-06-03:

- Raw-pkl/dq strict baseline:
  `C:\Users\ROG\Desktop\paper\codex\raw-pkl-dq-eol-fingerprint-v2\raw-pkl-dq-eol-fingerprint-v2_report.md`
- Raw-pkl/dq v2 status: accepted as a diagnostic baseline only.
- Raw-pkl/dq v2 selected result:
  `source_plus_target_adapt_loocv` + `elastic_net` + 1000 cycles.
- Raw-pkl/dq v2 Batch 9 validation RMSE: 52.15 cycles.
- Raw-pkl/dq v2 Batch 10 blind-test RMSE: 125.54 cycles.
- Raw-pkl/dq v2 interpretation: the strict raw-feature pipeline is useful for
  online-availability defense, but current high-dimensional raw features
  generalize poorly and do not replace the strict ExtraTrees baseline.
- Neural EOL v1:
  `C:\Users\ROG\Desktop\paper\codex\neural-eol-model-v1\neural-eol-model-v1_report.md`
- Neural v1 status: accepted as a prototype only.
- Neural v1 selected result:
  `source_plus_target_adapt_loocv` + `neural_eol_qnet` + 1000 cycles.
- Neural v1 Batch 9 validation RMSE: 22.50 cycles.
- Neural v1 Batch 10 blind-test RMSE: 68.68 cycles.
- Neural v1 interpretation: it is a useful physics-consistent neural prototype
  but does not beat the strict ExtraTrees baseline on Batch 10.
- Raw feature selection v1:
  `C:\Users\ROG\Desktop\paper\codex\raw-feature-selection-v1\raw-feature-selection-v1_report.md`
- Raw feature selection v1 status: accepted as strict diagnostic evidence.
- Raw feature selection v1 selected result:
  `source_plus_target_adapt_loocv` + `svr_rbf` + `corr_fused_top20` +
  1000 cycles.
- Raw feature selection v1 Batch 9 validation RMSE: 30.94 cycles.
- Raw feature selection v1 Batch 10 blind-test RMSE: 76.97 cycles.
- Raw feature selection v1 Batch 10 blind-test MAE: 67.46 cycles.
- Raw feature selection v1 interpretation: feature selection greatly improves
  raw-pkl generalization versus raw-pkl v2, but it still does not beat the
  strict ExtraTrees baseline.
- Neural EOL v2:
  `C:\Users\ROG\Desktop\paper\codex\neural-eol-model-v2\neural-eol-model-v2_report.md`
- Neural v2 status: reject as formal validation evidence; archive as a
  negative neural prototype.
- Neural v2 selected result:
  `source_plus_target_adapt_loocv` + `summary_mlp_small_dropout035` +
  1000 cycles, seeds 11/42/123.
- Neural v2 Batch 9 validation RMSE: 8.15 cycles.
- Neural v2 Batch 10 blind-test RMSE: 146.07 cycles.
- Neural v2 Batch 10 blind-test MAE: 126.85 cycles.
- Neural v2 interpretation: the Batch 9 score is not a clean outer-fold
  estimate because the held-out Batch 9 cell is reused for
  early-stopping/checkpoint selection before being scored. It also has a poor
  Batch 10 result. Do not claim neural improvement from v2.
- Strict EOL robustness v1:
  `C:\Users\ROG\Desktop\paper\codex\strict-eol-robustness-v1\strict-eol-robustness-v1_report.md`
- Strict EOL robustness v1 status: accepted as the paper-ready robustness
  package for the current strict ExtraTrees result.
- Strict EOL robustness v1 Batch 10 metrics: RMSE 56.19 cycles, MAE 49.42
  cycles, MedAE 47.36 cycles, max absolute error 93.21 cycles, bias +38.23
  cycles, Pearson 0.990, Spearman 0.976.
- Strict EOL robustness v1 uncertainty/sensitivity: cell-level bootstrap 95%
  CI is 36.84-72.61 cycles for RMSE and 31.35-68.10 cycles for MAE;
  leave-one-cell-out RMSE range is 48.65-60.03 cycles.
- EOL ablation residual v1:
  `C:\Users\ROG\Desktop\paper\codex\eol-ablation-residual-v1\eol-ablation-residual-v1_report.md`
- EOL ablation residual v1 status: accepted as a negative/risk-control result.
- EOL ablation residual v1 feature-family result: `fused_all` at 1000 cycles is
  best by Batch 9 validation RMSE 75.26 cycles; `soh_capacity_family` best
  Batch 9 RMSE is 86.76 cycles; `x_curve_family` best Batch 9 RMSE is 87.46
  cycles; raw/dq-like features are not identifiable in the strict processed
  feature table.
- EOL ablation residual v1 residual result: Batch 9 selected
  `shrink_to_target_mean_w075` with RMSE 60.60 cycles, but Batch 10 RMSE
  worsens to 100.26 cycles and MAE to 92.90 cycles. Do not use residual
  calibration as the proposed method.
- Paper storyline v1:
  `C:\Users\ROG\Desktop\paper\codex\paper-storyline-v1\paper-storyline-v1_report.md`
- Paper storyline v1 status: accepted as a manuscript scaffold. Recommended
  positioning is conservative engineering-energy / energy informatics /
  battery data-driven SCI first, with Applied Energy-class venues treated only
  as a long-term target.
- The current main quantitative baseline remains strict ExtraTrees with Batch
  10 RMSE 56.19 cycles.

Subagent manuscript-material integration audited on 2026-06-04:

- Round 4 audit:
  `C:\Users\ROG\Desktop\paper\codex\SUBAGENT_ROUND4_AUDIT_20260604.md`
- Feature availability/leakage audit:
  `C:\Users\ROG\Desktop\paper\codex\feature-availability-leakage-audit-v1\feature-availability-leakage-audit-v1_report.md`
- Final figure/table pack:
  `C:\Users\ROG\Desktop\paper\codex\final-figure-table-pack-v1\final-figure-table-pack-v1_report.md`
- EOL/RUL manuscript scaffold:
  `C:\Users\ROG\Desktop\paper\codex\eol-rul-manuscript-draft-v1\eol-rul-manuscript-draft-v1.md`
- Q2 literature benchmark:
  `C:\Users\ROG\Desktop\paper\codex\q2-literature-benchmark-v1\q2-literature-benchmark-v1_report.md`
- Round 4 status: accepted as paper-integration material. It does not change
  the main quantitative result and must not be used to claim neural
  improvement, residual-calibration improvement, raw-only superiority, or
  unsupervised domain adaptation.
- The figure pack is draft-manuscript usable after journal-style polishing.
- The manuscript draft is a conservative scaffold, not a submission-ready paper.
- The literature benchmark is directional; DOI, metric, and quartile details
  must be verified before final citation use.

Subagent Round 5 audited on 2026-06-04:

- Round 5 audit:
  `C:\Users\ROG\Desktop\paper\codex\SUBAGENT_ROUND5_AUDIT_20260604.md`
- SCI readiness gap audit:
  `C:\Users\ROG\Desktop\paper\codex\sci-readiness-gap-audit-v1\sci-readiness-gap-audit-v1_report.md`
- Literature verification/citation bank:
  `C:\Users\ROG\Desktop\paper\codex\literature-verification-citation-bank-v2\literature-verification-citation-bank-v2_report.md`
- Manuscript draft v2:
  `C:\Users\ROG\Desktop\paper\codex\eol-rul-manuscript-draft-v2\eol-rul-manuscript-draft-v2.md`
- Strict conformal uncertainty diagnostic:
  `C:\Users\ROG\Desktop\paper\codex\strict-eol-conformal-uncertainty-v1\strict-eol-conformal-uncertainty-v1_report.md`
- Round 5 status: accepted as paper-readiness/risk-reduction evidence. It
  does not change the main point-prediction result.
- Conformal diagnostic status: finite 80% interval with half-width 129.42
  cycles and 8/8 Batch 10 coverage; standard 90% and 95% finite-width
  conformal intervals are unavailable with eight Batch 9 calibration cells.
  Use as a limited uncertainty diagnostic only.
- Current publication readiness: conservative SCI/SCIE is conditionally
  plausible; Q2 is borderline but possible with final writing and one more
  reviewer-risk-reduction experiment; Applied Energy / Journal of Energy
  Chemistry / high Energy-AI level is not ready.
- Model innovation decision: do not continue broad neural tuning. For Q2, the
  next best experiments are target-label budget sensitivity and stricter
  raw-feature provenance. For high-level venues, a genuinely validated
  physics-consistent EOL/RUL model or external/multi-condition validation is
  required.

Subagent Round 6 audited on 2026-06-04:

- Round 6 audit:
  `C:\Users\ROG\Desktop\paper\codex\SUBAGENT_ROUND6_AUDIT_20260604.md`
- Manuscript draft v3 with inserted citation IDs:
  `C:\Users\ROG\Desktop\paper\codex\eol-rul-manuscript-draft-v3\eol-rul-manuscript-draft-v3.md`
- Target-label budget sensitivity:
  `C:\Users\ROG\Desktop\paper\codex\target-label-budget-sensitivity-v1\target-label-budget-sensitivity-v1_report.md`
- Strict raw-feature provenance v3:
  `C:\Users\ROG\Desktop\paper\codex\strict-raw-provenance-v3\strict-raw-provenance-v3_report.md`
- Q2 submission package:
  `C:\Users\ROG\Desktop\paper\codex\q2-submission-package-v1\q2-submission-package-v1_report.md`
- Round 6 status: accepted as Q2-submission strengthening evidence.
- Target-label budget result: all 108 predefined subsets were evaluated under
  the fixed ExtraTrees 1000-cycle setting. Mean Batch10 RMSE changes from
  source-only 64.27 to k=1 61.17, k=2 60.55, k=4 59.64, and k=8 56.19 cycles.
  k=8 exactly reproduces the strict main result. These are sensitivity results,
  not Batch10-selected model choices.
- Strict raw provenance v3 result: direct raw `our_data/*.pkl` features only,
  no processed `clean_*_x.npy` or `y_nominal` model inputs. Batch9 selected
  `corr_capacity_top20 + svr_rbf + source_plus_target_adapt_loocv` at
  1000 cycles; Batch9 RMSE/MAE 29.87/21.66 cycles; Batch10 RMSE/MAE
  74.12/63.16 cycles. Use as provenance evidence, not raw-only superiority.
- Q2 readiness after Round 6: conservative SCI/SCIE about 80%; realistic
  Q2-direction about 70%; high-impact venues remain not ready.

Subagent Round 7 planned on 2026-06-05:

- Task file:
  `C:\Users\ROG\Desktop\paper\codex\SUBAGENT_ROUND7_TASKS_20260605.md`
- Round 7 is a submission-integration round, not a broad model-tuning round.
- Planned outputs:
  Visio-style schematic figure prompt bank, manuscript draft v4 integration,
  final data-derived figure/table pack v2, and Q2 reviewer-risk closeout.
- AI-generated images may be used only for conceptual schematics. Numerical
  result figures must be generated from accepted CSV/JSON/MD evidence.
- Round 7 does not change the main quantitative result unless its outputs are
  later audited and accepted.

Subagent Round 7 audited on 2026-06-05:

- Audit file:
  `C:\Users\ROG\Desktop\paper\codex\SUBAGENT_ROUND7_AUDIT_20260605.md`
- Round 7 status: accepted as a manuscript-integration and
  submission-closeout round.
- Accepted outputs:
  `paper-visio-figure-prompts-v1`,
  `eol-rul-manuscript-draft-v4`,
  `final-figure-table-pack-v2`, and
  `q2-reviewer-risk-closeout-v1`.
- Main quantitative result remains unchanged:
  ExtraTrees at 1000 cycles, Batch10 RMSE/MAE 56.19/49.42 cycles.
- Draft v4 integrates target-label budget sensitivity:
  k=0/1/2/4/8 mean Batch10 RMSE =
  64.27/61.17/60.55/59.64/56.19 cycles over 108 predefined subsets.
- Draft v4 integrates strict raw-provenance v3:
  direct raw `our_data/*.pkl`, `corr_capacity_top20 + SVR-RBF`, Batch10
  RMSE/MAE 74.12/63.16 cycles; provenance evidence only.
- Figure/table pack v2 is accepted as the current data-derived visual package
  with PNG/SVG/PDF outputs and no AI-generated numerical figures.
- Reviewer-risk closeout estimates readiness after P0 completion as:
  conservative SCI/SCIE 80-85%, realistic Q2-direction 70-75%,
  Journal of Energy Storage-style stretch 55-60%, Applied Energy/high
  Energy-AI not recommended.
- No broad model-tuning round is recommended before conservative Q2-direction
  submission. Remaining work is P0 closeout, final references, figure/table
  placement, feature/leakage supplement, and target-journal verification.

## 6. Invalid Or Dangerous Claims

Do not claim:

- "Very early RUL prediction below 100 cycles error" unless future 50/100-cycle
  experiments actually achieve it under strict protocol.
- "Unsupervised domain adaptation" when full labeled Batch 9 EOL/SOH labels are
  used.
- "Cross-dataset" or "cross-chemistry" validation; current evidence is
  cross-batch within the same dataset.
- "Temperature-aware" modeling; current data audit found no temperature field.
- "The cubic physical base is a physical law." It is a low-order nonlinear
  degradation prior, not an absolute mechanistic law.
- Any result based on `observed_ratio = observed_cycles / full_cell_length`;
  this was a leakage feature and is invalid.
- Any result selected by sorting on Batch 10 RMSE. Such results are exploratory
  only until rerun under Batch 9-only selection.

## 7. Physical Base Position

Base-only polynomial ablation:

`C:\Users\ROG\Desktop\paper\codex\rul-eol-experiments-v1\polynomial-base-ablation-v1_report.md`

Preferred wording:

> The cubic base is the lowest-order nonlinear degradation prior that captures
> curvature and accelerated aging better than degree 1/2 while avoiding the
> oscillation and complexity risk of higher-degree polynomials.

Required future experiment:

```text
Full-model physical-base degree ablation: degree 2 vs degree 3 vs degree 4.
```

## 8. Main Method To Build Next

The next intended main model should not be ExtraTrees as the final method.
ExtraTrees is a strong baseline/proof-of-concept.

Target final method:

```text
Physics-consistent domain-adaptive EOL/RUL network
```

Recommended architecture:

```text
SOH/capacity trajectory encoder
+ voltage/current curve fingerprint encoder
+ target-domain adaptation module
+ EOL distribution head
+ auxiliary degradation-rate / SOH-slope head
+ physics consistency losses
```

Core constraints:

```text
predicted_RUL(cycle) = predicted_EOL - current_cycle
predicted_EOL >= current_cycle
predicted_RUL >= 0
RUL should decrease as cycle increases
SOH degradation rate should be physically plausible
```

Uncertainty should be added when feasible:

```text
P10/P50/P90 EOL or RUL quantiles
80/90/95% interval coverage
calibration error
```

## 9. Required Baselines And Ablations

Required baselines:

- source mean EOL
- source median EOL
- cycle-only regression
- polynomial base-only
- Severson-style early-cycle delta features if extractable
- ExtraTrees / RandomForest / SVR / Ridge or KernelRidge
- old quick neural RUL baseline
- final proposed neural EOL/RUL model

Required ablations:

- w/o SOH slope/drop fingerprint
- w/o voltage/current fingerprint
- w/o target adaptation
- w/o physical consistency
- w/o uncertainty head, if uncertainty is implemented
- physical base degree 2/3/4

Required reporting:

- horizons: 50, 100, 200, 300, 500, 800, 1000 cycles
- target_adapt metrics
- target_test metrics
- per-battery errors
- model/horizon selected only from Batch 9
- Batch 10 evaluated once after selection
- feature whitelist/blacklist for online availability

Feature blacklist for model inputs:

```text
eol_cycle
rul_cycle
observed_ratio
full_cell_length
n_cycles_total
last_observed_cycle
n_observed
cell_id
battery_index
batch_id
```

Feature caution:

- `horizon/current_cycle` is allowed as an experimental condition, but when
  separate per-horizon models are trained it should not be included as a model
  input.
- SOH-derived features from `y_nominal` must be described as historical
  measured/estimated capacity/SOH features. A stricter `dq`/raw-capacity-only
  reconstruction should be added before final manuscript claims.
- Processed `x` fingerprint features must be audited to ensure they use only
  current and historical voltage/current/capacity/time curves.

## 10. Subagent Working Rule

This main conversation should own:

- mainline decisions
- protocol definitions
- final integration
- manuscript positioning
- acceptance/rejection of subagent outputs

Subagents may own bounded side tasks:

- leakage audit
- literature checklist
- one script or one ablation with disjoint files
- verification/report generation

Subagents must not redefine the mainline. If a subagent suggests a different
story, treat it as advice only; this file remains authoritative unless the user
explicitly replaces it.

## 11. Next Concrete Experiments

Experiment A: strict selection protocol

- Status: completed as `strict-eol-protocol-v1` on 2026-06-03.
- Accepted result: Batch-9-selected ExtraTrees at 1000 cycles, Batch 10 RMSE
  56.19 cycles and MAE 49.42 cycles.
- Remaining caveat: uses labeled Batch 9 EOL targets and `y_nominal`
  historical SOH features.

Experiment B: neural EOL model

- Start with fixed horizons 100, 300, 500, 1000.
- Compare against ExtraTrees.
- Add EOL consistency and non-negative RUL constraints.

Experiment C: leakage and online-availability audit

- Audit every feature for online availability.
- Separate measured-capacity-history setting from voltage/current-only setting.

Experiment D: full-model polynomial base ablation

- degree 2, degree 3, degree 4.
- Same selection protocol.

This file should be read before any future RUL/EOL experiment or manuscript
section.
