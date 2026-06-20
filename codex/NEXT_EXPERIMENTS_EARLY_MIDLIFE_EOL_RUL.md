# Next Experiments: Early-to-Mid-Life EOL/RUL Mainline

Created: 2026-06-03  
Authoritative mainline: `codex/MAINLINE_EARLY_MIDLIFE_EOL_RUL_20260603.md`

## Stage 1: Strict Protocol Lock

Goal: remove any hidden target-test model/horizon selection.

Status: completed on 2026-06-03 as:

- `codex/strict-eol-protocol-v1/strict_eol_protocol_v1.py`
- `codex/strict-eol-protocol-v1/strict-eol-protocol-v1_report.md`

Accepted strict baseline:

- Batch-9-selected global candidate:
  `source_plus_target_adapt_loocv` + `extra_trees` + 1000 observed cycles.
- Batch 9 validation RMSE: 75.26 cycles.
- Batch 10 blind-test RMSE: 56.19 cycles.
- Batch 10 blind-test MAE: 49.42 cycles.
- Feature audit now covers every evaluated horizon and split.
- Caveat: this is supervised small-target adaptation using labeled Batch 9 EOL
  targets, not unsupervised domain adaptation.
- Caveat: `y_nominal` features are acceptable only as historical
  measured/estimated SOH; add a stricter raw-capacity or `dq` variant later.

Tasks:

1. Build `strict_eol_protocol_v1.py`.
2. Inputs:
   - source: Batch 1-8
   - target_adapt: Batch 9
   - target_test: Batch 10
3. Candidate horizons:
   - 50, 100, 200, 300, 500, 800, 1000 cycles
4. Candidate models:
   - source median EOL
   - cycle-only
   - ExtraTrees
   - SVR
   - Ridge/KernelRidge
   - proposed neural model when available
5. Selection:
   - choose model/horizon using Batch 9 only
   - for `source+target_adapt`, split Batch 9 cells into target-train and
     target-validation, or use leave-one-target-cell-out validation
   - do not use the same Batch 9 target cells for both supervised adaptation
     and validation scoring
6. Final:
   - evaluate Batch 10 once
   - do not sort model candidates by Batch 10
7. Outputs:
   - summary CSV
   - per-cell prediction CSV
   - leakage audit table
   - Markdown report

## Stage 2: Online Feature Audit

Goal: prove each feature is available at prediction time.

Status: partially completed for strict-eol-protocol-v1. The current audit
excludes explicit leakage fields across all horizons/splits, but it does not
yet prove that processed `x` and `y_nominal` are raw-online signals.

Next required audit:

- inspect construction of processed `x`.
- build a raw-capacity or `dq`-based EOL fingerprint variant.
- The attempted `raw-capacity-eol-fingerprint-v1` worker output is not accepted
  as new evidence because it reused `clean_*_x.npy` and `y_nominal` rather than
  proving raw-pkl provenance or reconstructing features from raw data.
- `raw-pkl-dq-eol-fingerprint-v2` completed the raw-pkl reconstruction and is
  accepted as a diagnostic baseline. It did not improve the main result:
  Batch 10 RMSE 125.54 cycles after Batch-9-selected ElasticNet at 1000 cycles.
- `raw-feature-selection-v1` completed the next raw-feature reduction step
  using Batch 9 validation only. It selected `corr_fused_top20` + `svr_rbf` at
  1000 cycles, with Batch 9 RMSE 30.94 cycles and Batch 10 RMSE 76.97 cycles.
  This improves raw-pkl v2 but still does not beat strict ExtraTrees at 56.19
  cycles.
- Next raw-feature step: use `raw-feature-selection-v1` as diagnostic evidence
  for interpretable feature families, then run feature-family ablations and
  robustness checks. Do not present it as the current best model.
- document whether measured capacity/SOH is assumed available from diagnostic
  cycles or estimated by the BMS.

Feature categories:

- safe:
  - current cycle index
  - historical voltage/current/capacity up to current horizon
  - historical measured/estimated SOH up to current horizon
- unsafe:
  - full cell length
  - observed ratio using final life length
  - future SOH/capacity
  - target_test-derived horizon/model choices
  - `last_observed_cycle` when defined as `min(horizon, EOL)`
  - `n_observed` when it can reveal that a cell ended before the horizon
  - `cell_id`, `battery_index`, `batch_id` as model features
- needs clarification:
  - measured capacity per cycle, because real BMS may not have exact full
    discharge capacity every cycle.
  - `y_nominal`-derived SOH features; formal writing must describe them as
    historical measured/estimated capacity/SOH and should add a `dq`/raw
    capacity reconstruction experiment.

Output:

- feature name
- source file
- whether online available
- whether future information is used
- recommended paper wording

## Stage 3: Neural EOL/RUL Model

Goal: turn the ExtraTrees proof-of-concept into a publishable model.

Status: first two prototypes completed as `neural-eol-model-v1` and
`neural-eol-model-v2`.

Current neural result:

- Batch-9-selected model:
  `source_plus_target_adapt_loocv` + `neural_eol_qnet` + 1000 cycles.
- Batch 9 validation RMSE: 22.50 cycles.
- Batch 10 blind-test RMSE: 68.68 cycles.
- Batch 10 blind-test MAE: 60.30 cycles.
- It does not beat strict ExtraTrees at 56.19 cycles RMSE.
- Neural v2 selected model:
  `source_plus_target_adapt_loocv` + `summary_mlp_small_dropout035` +
  1000 cycles, seeds 11/42/123.
- Neural v2 Batch 9 validation RMSE: 8.15 cycles.
- Neural v2 Batch 10 blind-test RMSE: 146.07 cycles.
- Neural v2 Batch 10 blind-test MAE: 126.85 cycles.
- Neural v2 is rejected as formal validation evidence. Its held-out Batch 9
  cells are excluded from gradient training but reused for
  early-stopping/checkpoint selection, so the low Batch 9 error is optimistic.
  Its poor Batch 10 error also fails the main baseline.

Next neural step:

- treat v1/v2 as prototypes, not winning models.
- stop unconstrained MLP tuning until a stronger protocol-level idea is added.
- fix neural validation first: use nested validation or fixed epochs so the
  outer Batch 9 held-out cell is not used for checkpoint selection.
- remove derived horizon inputs such as `fixed_horizon_fraction`.
- prefer conservative residual correction on top of the strict ExtraTrees
  prediction, or a tightly constrained physics-consistent network using only
  feature families that survived the raw-feature audit.
- keep multi-seed results mandatory.
- evaluate uncertainty coverage for P10/P50/P90.
- use raw-pkl feature provenance or clearly justify processed `x` and
  `y_nominal` as online historical measurements.

Residual calibration update from 2026-06-04:

- `eol-ablation-residual-v1` tested conservative residual calibration on top of
  strict ExtraTrees.
- Batch 9 selected `shrink_to_target_mean_w075` with RMSE 60.60 cycles, but
  Batch 10 worsened to RMSE 100.26 cycles and MAE 92.90 cycles.
- Treat this as a negative/risk-control result. Do not promote target-mean
  shrinkage or residual calibration as the proposed method.
- Lesson: 8-cell Batch 9 residual calibration can overfit target-domain
  idiosyncrasies even when nested LOOCV is used.

Architecture:

```text
SOH/capacity trajectory encoder
+ voltage/current curve fingerprint encoder
+ target-domain adaptation
+ EOL point/quantile head
+ degradation-rate auxiliary head
+ physics consistency losses
```

First version:

- horizons: 100, 300, 500, 1000
- target: EOL cycle
- losses:
  - EOL MSE/Huber
  - non-negative RUL consistency
  - degradation-rate auxiliary loss if feasible
- compare against ExtraTrees under same protocol

## Stage 4: Physical Base Full-Model Ablation

Goal: justify cubic base beyond base-only evidence.

Compare:

- degree 2
- degree 3
- degree 4

Use same source/target_adapt/target_test protocol.

## Stage 5: Statistics And Robustness

Goal: reduce reviewer attack surface.

Status: partly completed on 2026-06-04 as
`codex/strict-eol-robustness-v1/`.

Accepted robustness facts:

- Batch 10 RMSE: 56.19 cycles.
- Batch 10 MAE: 49.42 cycles.
- Median absolute error: 47.36 cycles.
- Max absolute error: 93.21 cycles.
- Bias: +38.23 cycles.
- Pearson/Spearman true-vs-pred: 0.990 / 0.976.
- Cell-level bootstrap 95% CI: RMSE 36.84-72.61 cycles; MAE 31.35-68.10
  cycles.
- Leave-one-cell-out RMSE range: 48.65-60.03 cycles.

Required:

- multi-seed for neural models, only if neural models are revived under clean
  validation
- horizon-vs-error plot
- feature availability/leakage audit table
- final manuscript figure/table pack

Round 4 update on 2026-06-04:

- Audit file: `codex/SUBAGENT_ROUND4_AUDIT_20260604.md`.
- `codex/final-figure-table-pack-v1/` is accepted as draft manuscript
  figure/table material.
- `codex/feature-availability-leakage-audit-v1/` is accepted as Methods or
  Supplementary leakage/availability evidence.
- `codex/eol-rul-manuscript-draft-v1/` is accepted as a conservative paper
  scaffold, not a submission-ready manuscript.
- `codex/q2-literature-benchmark-v1/` is accepted as directional literature
  positioning, with DOI/metric/quartile verification still required.
- Next highest-value work is paper integration and citation verification, not
  another unconstrained model-tuning round.

Round 5 update on 2026-06-04:

- Audit file: `codex/SUBAGENT_ROUND5_AUDIT_20260604.md`.
- `codex/sci-readiness-gap-audit-v1/` is accepted. Verdict: conservative
  SCI/SCIE is conditionally plausible; Q2 is borderline but possible; high-level
  SCI is not ready.
- `codex/literature-verification-citation-bank-v2/` is accepted as citation
  material. It contains 20 verified rows and a related-work claim map.
- `codex/eol-rul-manuscript-draft-v2/` is accepted as a stronger scaffold, but
  not submission-ready until citations are inserted and placeholders are
  removed.
- `codex/strict-eol-conformal-uncertainty-v1/` is accepted as a limited
  uncertainty diagnostic. It reproduces the strict point predictions exactly,
  gives a finite 80% conformal interval with half-width 129.42 cycles and 8/8
  Batch 10 coverage, but 90% and 95% conformal intervals are unbounded because
  only eight Batch 9 calibration cells are available.
- Next highest-value experiments: target-label budget sensitivity and stricter
  raw-feature provenance under the same strict protocol.
- Do not continue broad neural tuning unless the new neural design has nested
  checkpoint selection and a clear physics-consistency hypothesis.

Round 6 update on 2026-06-04:

- Audit file: `codex/SUBAGENT_ROUND6_AUDIT_20260604.md`.
- `codex/eol-rul-manuscript-draft-v3/` is accepted as the current manuscript
  base with inserted citation IDs.
- `codex/target-label-budget-sensitivity-v1/` is accepted. It evaluates 108
  predefined subsets and shows Batch10 RMSE means of 64.27, 61.17, 60.55,
  59.64, and 56.19 cycles for k=0/1/2/4/8 labeled Batch9 cells, respectively.
- `codex/strict-raw-provenance-v3/` is accepted as raw-feature provenance.
  It uses direct raw `our_data/*.pkl` features only and obtains Batch10
  RMSE/MAE 74.12/63.16 cycles for the Batch9-selected raw capacity top-20
  SVR-RBF diagnostic.
- `codex/q2-submission-package-v1/` is accepted as Q2 target planning material.
- Next work is manuscript integration/finalization: fold target-label budget
  and strict raw-provenance v3 into the manuscript, update tables/figures, pick
  a target journal, then format citations and cover letter accordingly.

Round 7 task plan on 2026-06-05:

- Task file:
  `codex/SUBAGENT_ROUND7_TASKS_20260605.md`.
- Purpose: move from accepted evidence toward a Q2-direction submission package.
- Subagent A: Visio-style scientific figure prompt bank for schematic
  manuscript illustrations. This is for conceptual diagrams only, not
  generated numerical result plots.
- Subagent B: manuscript draft v4 integration, folding Round6 target-label
  budget and raw-provenance results into draft v3.
- Subagent C: final figure/table pack v2, regenerating data-derived figures
  and tables from accepted CSV/JSON/MD evidence.
- Subagent D: Q2 reviewer-risk closeout, including attack/response matrix,
  claim boundary check, and pre-submission todo list.
- Round7 does not change the main quantitative result unless a later audit
  explicitly accepts new evidence.

Round 7 audit on 2026-06-05:

- Audit file:
  `codex/SUBAGENT_ROUND7_AUDIT_20260605.md`.
- Accepted as manuscript-integration/submission-closeout work.
- Accepted directories:
  `codex/paper-visio-figure-prompts-v1`,
  `codex/eol-rul-manuscript-draft-v4`,
  `codex/final-figure-table-pack-v2`,
  `codex/q2-reviewer-risk-closeout-v1`.
- Current next work is not broad model tuning. It is:
  final claim-control pass, citation style conversion, figure/table placement,
  feature availability/leakage supplement, data/code/declaration statements,
  and target-journal institutional verification.
- Readiness estimate after P0 closeout:
  conservative SCI/SCIE 80-85%, realistic Q2-direction 70-75%,
  Journal of Energy Storage-style stretch 55-60%.

## Recommended Subagent Split

Main conversation:

- owns mainline, protocol, final integration, paper narrative.

Subagent A: leakage auditor

- reads scripts and feature tables.
- reports unsafe features and protocol risks.
- does not edit main experiment code unless asked.

Subagent B: baseline worker

- owns strict baseline suite files only.
- writes CSV/JSON/MD report.

Subagent C: neural model worker

- owns neural EOL model script only.
- no edits to baseline scripts.

Subagent D: literature explorer

- searches recent SOH/RUL/EOL early prediction, domain adaptation, uncertainty.
- returns citation candidates and exact claims only.

Never let subagents redefine the mainline.
