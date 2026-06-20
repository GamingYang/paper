# Feature-Horizon Leakage Defense V2

- Finished at: 2026-06-06T18:54:45
- Elapsed seconds: 109.1
- Strict source script: `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict_eol_protocol_v1.py`
- Data directory: `C:\Users\ROG\Desktop\paper\PYthon训练代码\processed_clean`

## Protocol Guardrails

- Batch 1-8 are source cells.
- Batch 9 is the only development/adaptation/validation set used here.
- Batch 10 is not loaded by this script and is not used for feature-family, horizon, model, seed, epoch, checkpoint, or ranking decisions.
- All rows below use `source_plus_target_adapt_loocv`: each Batch 9 held-out cell is scored after training on Batch 1-8 plus the other Batch 9 cells.
- The model is fixed to the strict-v1 `extra_trees` baseline to isolate feature-family and horizon effects.

## Batch 9 Validation-Only Horizon Table

| Family | Horizon | Features | Batch 9 RMSE | Batch 9 MAE |
|---|---:|---:|---:|---:|
| fused_all | 50 | 108 | 219.15 | 151.50 |
| fused_all | 100 | 108 | 189.67 | 144.64 |
| fused_all | 200 | 108 | 174.49 | 120.23 |
| fused_all | 300 | 108 | 156.88 | 104.71 |
| fused_all | 500 | 108 | 118.42 | 101.14 |
| fused_all | 800 | 108 | 91.08 | 78.14 |
| fused_all | 1000 | 108 | 75.26 | 61.96 |
| soh_capacity | 50 | 59 | 219.87 | 160.36 |
| soh_capacity | 100 | 59 | 208.02 | 165.92 |
| soh_capacity | 200 | 59 | 163.83 | 135.13 |
| soh_capacity | 300 | 59 | 97.82 | 78.97 |
| soh_capacity | 500 | 59 | 104.48 | 89.33 |
| soh_capacity | 800 | 59 | 86.76 | 70.24 |
| soh_capacity | 1000 | 59 | 89.60 | 68.81 |
| x_curve | 50 | 49 | 213.32 | 148.12 |
| x_curve | 100 | 49 | 184.66 | 140.45 |
| x_curve | 200 | 49 | 172.96 | 128.11 |
| x_curve | 300 | 49 | 166.65 | 124.56 |
| x_curve | 500 | 49 | 159.76 | 119.67 |
| x_curve | 800 | 49 | 98.68 | 80.50 |
| x_curve | 1000 | 49 | 87.46 | 84.34 |

## Feature-Family Summary

| Family | Best horizon by Batch 9 | Features | Batch 9 RMSE | Batch 9 MAE | Status |
|---|---:|---:|---:|---:|---|
| fused_all | 1000 | 108 | 75.26 | 61.96 | available |
| soh_capacity | 800 | 59 | 86.76 | 70.24 | available |
| x_curve | 1000 | 49 | 87.46 | 84.34 | available |
| raw/provenance | n/a | n/a | n/a | n/a | separate diagnostic available; not in strict processed family table |

## Leakage Blacklist Audit

- Forbidden fields audited: `batch_id, battery_index, cell_id, eol_cycle, full_cell_length, full_length, full_life_length, horizon, identifier, last_observed_cycle, n_cycles_total, n_observed, observed_ratio, pkl_file, rul_cycle, split`.
- Forbidden fields used as model inputs: `0`.
- Direct leakage status: PASS if the count above is zero.
- `horizon`, `n_observed`, `last_observed_cycle`, `observed_ratio`, full-life length fields, EOL/RUL labels, and cell/batch identifiers are excluded from all evaluated families.
- `soh_*` features are conditionally manuscript-safe only as historical measured/estimated capacity-SOH signals.
- `x_*` features are conditionally manuscript-safe as processed curve fingerprints from observed rows; final writing should state the provenance caveat.

## Manuscript-Safe Wording

Safe to write:

- The supplementary Batch 9 validation-only audit compares fixed-horizon feature families under the same strict small labeled target-adaptation protocol.
- Model inputs exclude EOL/RUL labels, full-life length or observed-ratio variables, horizon/count metadata, and cell/batch identifiers.
- The fused processed-feature family gives the best Batch 9 validation result in this audit, while SOH/capacity and processed curve families remain useful but weaker standalone families.
- Raw/provenance evidence should be reported as a separate diagnostic based on direct `our_data/*.pkl` parsing, not as the best-performing main result.

Do not write:

- Do not claim Batch 10 was used to choose the horizon or feature family.
- Do not claim raw-only or voltage-only superiority from this audit.
- Do not call the Batch 9 labeled adaptation protocol unsupervised domain adaptation.
- Do not present processed `y_nominal`/SOH summaries as leakage-free labels; frame them as historical measured/estimated health observations.
- Do not claim very-early 50/100-cycle performance is below 100 cycles unless a strict Batch-9-selected result supports it.

## Core Result

Best validation-only row: `fused_all` at 1000 cycles, Batch 9 RMSE 75.26 cycles and MAE 61.96 cycles.

## Output Files

- Validation CSV: `C:\Users\ROG\Desktop\paper\codex\feature-horizon-leakage-defense-v2\batch9_validation_by_feature_family_horizon.csv`
- Best-family CSV: `C:\Users\ROG\Desktop\paper\codex\feature-horizon-leakage-defense-v2\feature_family_best_batch9_validation.csv`
- Blacklist audit CSV: `C:\Users\ROG\Desktop\paper\codex\feature-horizon-leakage-defense-v2\leakage_blacklist_audit.csv`
- Feature inventory CSV: `C:\Users\ROG\Desktop\paper\codex\feature-horizon-leakage-defense-v2\feature_family_column_inventory.csv`
- Raw diagnostic summary CSV: `C:\Users\ROG\Desktop\paper\codex\feature-horizon-leakage-defense-v2\raw_provenance_separate_diagnostic_summary.csv`
- Horizon figure: `C:\Users\ROG\Desktop\paper\codex\feature-horizon-leakage-defense-v2\horizon_vs_batch9_rmse.png`
- Feature-family figure: `C:\Users\ROG\Desktop\paper\codex\feature-horizon-leakage-defense-v2\feature_family_batch9_rmse.png`
- JSON summary: `C:\Users\ROG\Desktop\paper\codex\feature-horizon-leakage-defense-v2\feature-horizon-leakage-defense-v2_summary.json`
