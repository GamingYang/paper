# Dataset Overview Figure v1

Generated: 2026-07-10

## Purpose

This figure is a data-set analysis and signal-characterization overview for
the SOH manuscript. It visualizes the raw `our_data/*.pkl` cycling records and
the existing `processed_clean` split without using model predictions or
changing any experiment result.

The layout follows the uploaded reference figures: one wide capacity trajectory
panel plus multiple smaller coordinate plots showing distribution, density,
raw signal, processed voltage map, and time-frequency views.

## Input Sources

- Raw cycling records: `our_data/*.pkl`
- Processed split and labels: `PYthon训练代码/processed_clean/`
- Split summary: `codex/6月29日最新证据链/tables/fixed_split_summary.csv`
- Manuscript context: `codex/初稿撰写manuscript/SOH.docx`

## Outputs

Main figure:

- `figures/Figure_dataset_overview_multi_panel_v1.png`
- `figures/Figure_dataset_overview_multi_panel_v1.pdf`
- `figures/Figure_dataset_overview_multi_panel_v1.svg`

Cached data tables:

- `tables/raw_cell_capacity_summary.csv`
- `tables/raw_capacity_trajectories_sampled.csv`

Regeneration script:

- `scripts/make_dataset_overview_figure_v1.py`

## Panel Map

| Panel | Content | Data source |
|---|---|---|
| a | Discharge capacity trajectories for all cells | raw `dq` from `our_data/*.pkl` |
| b | Cycle-SOH sample density | `processed_clean/*_meta.csv` |
| c | Cycle-life distribution by batch | raw cell summaries |
| d | Initial capacity vs cycle life | raw cell summaries |
| e | Raw voltage/current profile for cell `10-1` | raw cycle DataFrame |
| f | Early/mid/late charge-voltage curves | raw cycle DataFrame |
| g | Processed 100-point charge-voltage map for cell `10-1` | `clean_target_test_x.npy` |
| h | SOH distribution by split | `processed_clean/*_meta.csv` |
| i | RUL-SOH label coupling | `processed_clean/*_meta.csv` |
| j | Voltage-current phase view | raw cycle DataFrame |
| k | Voltage time-frequency view | raw cycle DataFrame |

## Notes

- This is a dataset/feature visualization, not a SOH prediction result figure.
- Batch 1-8 are displayed as source, Batch 9 as target-development, and Batch 10
  as blind test, matching the manuscript protocol.
- The representative raw-signal panels use cell `10-1`; they are illustrative
  examples of the recorded voltage/current/capacity fields, not a selected model
  result.
