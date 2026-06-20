# Target-Label Budget Sensitivity V1

- Finished at: 2026-06-04T21:02:48
- Elapsed seconds: 84.0
- Data dir: `C:\Users\ROG\Desktop\paper\PYthon训练代码\processed_clean`
- Strict baseline script imported read-only: `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict_eol_protocol_v1.py`

## Fixed Protocol

- EOL-first target: cell-level EOL cycle; RUL at a fixed observation cycle is derived as predicted EOL minus current cycle.
- Source domain: Batch 1-8.
- Target labels: predefined subsets of labeled Batch 9 cells.
- Blind test: Batch 10 only.
- Model/horizon/features are fixed before this sensitivity run: ExtraTrees, 1000 observed cycles, strict feature blacklist.
- Batch 10 results are reported for all predefined subsets as sensitivity evidence only; no subset is selected by Batch 10.
- Because Batch 9 labels are used, the setting is supervised small labeled target-domain adaptation, not unsupervised domain adaptation.

## Subset Budget Design

| Budget k | Subsets evaluated | Strategy |
|---:|---:|---|
| 0 | 1 | source_only |
| 1 | 8 | all_combinations |
| 2 | 28 | all_combinations |
| 4 | 70 | all_combinations |
| 8 | 1 | all_combinations |

## Aggregate Batch 10 Sensitivity

| k labeled Batch 9 cells | Subsets | RMSE mean | RMSE median | RMSE range | MAE mean | MAE median | MAE range |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 64.27 | 64.27 | 64.27-64.27 | 55.21 | 55.21 | 55.21-55.21 |
| 1 | 8 | 61.17 | 60.86 | 57.74-64.20 | 54.30 | 54.11 | 48.63-58.04 |
| 2 | 28 | 60.55 | 60.42 | 54.45-66.24 | 53.91 | 53.91 | 46.15-60.72 |
| 4 | 70 | 59.64 | 59.46 | 51.99-67.91 | 53.20 | 53.55 | 43.94-62.70 |
| 8 | 1 | 56.19 | 56.19 | 56.19-56.19 | 49.42 | 49.42 | 49.42-49.42 |

## Anchor Rows

| Row | k | Target labels | Batch 10 RMSE | Batch 10 MAE |
|---|---:|---|---:|---:|
| Source-only anchor | 0 | none | 64.27 | 55.21 |
| Full Batch 9 adaptation anchor | 8 | all Batch 9 cells | 56.19 | 49.42 |

Strict baseline reproduction:

- k=8 matched 8 Batch 10 cells against the strict selected prediction CSV; max absolute prediction difference = 0.000000 cycles.

## Interpretation

- Full labeled Batch 9 adaptation improves the fixed ExtraTrees 1000-cycle anchor from 64.27 to 56.19 RMSE cycles.
- The k=8 minus k=0 RMSE difference is -8.09 cycles; lower is better.
- The k=1, k=2, and k=4 rows show substantial subset dependence, so few-label results should be discussed as sensitivity evidence rather than as selected models.
- Any low-error few-label subset is not a replacement for the main strict baseline because this task predefines and reports all subsets instead of selecting by Batch 10.
- This supports a cautious small-labeled-target adaptation narrative only when paired with the fixed protocol caveats.

## Caveats

- Batch 10 has only eight cells, so the aggregate ranges are sensitive to individual target-test cells.
- Batch 9 has only eight labeled cells; k=1 and k=2 are especially unstable by design.
- The experiment uses historical measured/estimated SOH-capacity features from `y_nominal`; wording must keep the online-availability caveat.
- This is not unsupervised domain adaptation and not Batch10-based model selection.
- The main point result remains the pre-existing strict ExtraTrees full-Batch-9 anchor.

## Output Files

- Script: `C:\Users\ROG\Desktop\paper\codex\target-label-budget-sensitivity-v1\run_target_label_budget_sensitivity_v1.py`
- Subset results: `C:\Users\ROG\Desktop\paper\codex\target-label-budget-sensitivity-v1\budget_subset_results.csv`
- Aggregate results: `C:\Users\ROG\Desktop\paper\codex\target-label-budget-sensitivity-v1\budget_aggregate_results.csv`
- Summary JSON: `C:\Users\ROG\Desktop\paper\codex\target-label-budget-sensitivity-v1\target-label-budget-sensitivity-v1_summary.json`
- Optional plot: `C:\Users\ROG\Desktop\paper\codex\target-label-budget-sensitivity-v1\target_label_budget_rmse_v1.png`
