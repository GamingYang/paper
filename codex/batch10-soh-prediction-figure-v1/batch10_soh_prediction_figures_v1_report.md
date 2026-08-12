# Batch10 SOH Prediction Figures v1

Generated: 2026-07-29

## Purpose

Regenerate the Batch10 per-cell SOH prediction visualization using a richer
paper-style layout inspired by the supplied reference image. The figures use
the accepted Batch10 final-model prediction table and do not retrain or select
models.

## Input Evidence

- `codex\6月29日最新证据链\tables\batch10_final_predictions_by_cycle.csv`
- `codex\6月29日最新证据链\tables\batch10_per_cell_metrics_final.csv`

## Outputs

- Eight-cell overview: `codex\batch10-soh-prediction-figure-v1\figures\Figure_Batch10_per_cell_SOH_prediction_curves_enhanced_2x4.png`
- Best-cell detail: `codex\batch10-soh-prediction-figure-v1\figures\Figure_Batch10_best_RMSE_cell_SOH_prediction_detail.png`

Both figures are exported as PNG, PDF, and SVG.

## Best Cell

The smallest Batch10 per-cell RMSE is `10-5` / Battery
`73`:

- RMSE: 0.004201 SOH
- MAE: 0.002817 SOH
- MAPE: 0.300%
- Signed bias: -0.001189 SOH

## Visual Encoding

- Solid dark curve: actual SOH.
- Coloured prediction points: predicted SOH, coloured by absolute SOH error.
- Orange/blue fill: signed residual direction.
- Inset: residual-density distribution in `x10^-3` SOH units.
