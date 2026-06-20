# Strict EOL Conformal Uncertainty V1

- Finished at: 2026-06-04T20:41:47
- Elapsed seconds: 9.2
- Data dir: `C:\Users\ROG\Desktop\paper\PYthon训练代码\processed_clean`
- Strict baseline script imported read-only: `C:\Users\ROG\Desktop\paper\codex\strict-eol-protocol-v1\strict_eol_protocol_v1.py`

## Fixed Setting

- Protocol: `source_plus_target_adapt_loocv`.
- Model: `extra_trees` with the strict baseline ExtraTrees parameters.
- Observed horizon: 1000 cycles.
- Source: Batch 1-8; calibration/adaptation: Batch 9; blind test: Batch 10.
- Batch 9 leave-one-cell-out predictions are used only as calibration residuals.
- The final point model is trained once on source plus all Batch 9 allowed data and then evaluated on Batch 10.

## Point Prediction Check

| Set | Cells | RMSE cycles | MAE cycles |
|---|---:|---:|---:|
| Batch 9 LOO calibration | 8 | 75.26 | 61.96 |
| Batch 10 blind test | 8 | 56.19 | 49.42 |

Strict baseline consistency check:

- Matched Batch 10 cells: 8; max absolute prediction difference vs strict selected CSV: 0 cycles.

## Batch 9 Calibration Residuals

| Cell | EOL | Predicted EOL | Error | Absolute error |
|---|---:|---:|---:|---:|
| 9-1 | 2057 | 1977.14 | -79.86 | 79.86 |
| 9-2 | 2143 | 2165.82 | 22.82 | 22.82 |
| 9-3 | 1905 | 1895.74 | -9.26 | 9.26 |
| 9-4 | 1975 | 1993.29 | 18.29 | 18.29 |
| 9-5 | 2168 | 2297.42 | 129.42 | 129.42 |
| 9-6 | 1742 | 1809.62 | 67.62 | 67.62 |
| 9-7 | 2012 | 1963.11 | -48.89 | 48.89 |
| 9-8 | 2308 | 2427.49 | 119.49 | 119.49 |

## Split-Conformal Intervals

The split-conformal absolute-error radius uses the finite-sample order statistic `ceil((n_cal + 1) * nominal_coverage)` on the eight Batch 9 LOO residuals. With only eight calibration cells, 80% is finite at the maximum residual; 90% and 95% require rank 9 and are therefore unbounded under the standard guarantee.

| Nominal | Calibration rank | Radius cycles | Finite interval | Batch 10 coverage | Mean width cycles |
|---:|---:|---:|---|---:|---:|
| 80% | 8 | 129.42 | yes | 1.000 | 258.84 |
| 90% | 9 | unbounded | no | 1.000 | unbounded |
| 95% | 9 | unbounded | no | 1.000 | unbounded |

## Batch 10 Per-Cell 80% Conformal Intervals

| Cell | EOL | Prediction | Abs error | 80% lower | 80% upper | Covered |
|---|---:|---:|---:|---:|---:|---|
| 10-1 | 1702 | 1785.01 | 83.01 | 1655.59 | 1914.43 | true |
| 10-2 | 1697 | 1726.96 | 29.96 | 1597.54 | 1856.38 | true |
| 10-3 | 1848 | 1897.98 | 49.98 | 1768.56 | 2027.40 | true |
| 10-4 | 1811 | 1904.21 | 93.21 | 1774.79 | 2033.63 | true |
| 10-5 | 2030 | 2086.82 | 56.82 | 1957.40 | 2216.24 | true |
| 10-6 | 2285 | 2316.89 | 31.89 | 2187.47 | 2446.31 | true |
| 10-7 | 1783 | 1788.73 | 5.73 | 1659.31 | 1918.15 | true |
| 10-8 | 1400 | 1355.26 | 44.74 | 1225.84 | 1484.68 | true |

## ExtraTrees Tree-Spread Diagnostic

Tree-spread intervals are included only as an ensemble-spread diagnostic. They are not calibrated conformal intervals and should not be used as formal uncertainty evidence.

| Central tree fraction | Batch 10 coverage | Mean width cycles |
|---:|---:|---:|
| 80% | 0.750 | 186.02 |
| 90% | 1.000 | 250.52 |
| 95% | 1.000 | 319.98 |

## Interpretation And Caveats

- The 80% conformal interval is finite but conservative because the finite-sample rank is 8/9, using the largest Batch 9 residual as the radius.
- The 90% and 95% conformal intervals are mathematically unbounded with only eight calibration cells; any finite max-residual substitute would not carry the standard split-conformal guarantee.
- Batch 10 coverage is reported as a diagnostic only because Batch 10 has eight cells and was not used to tune the model or interval width.
- This run does not replace the selected strict ExtraTrees baseline and does not tune model, horizon, or features.
- The setting remains supervised small labeled target-domain adaptation because labeled Batch 9 EOL values are used.
- `y_nominal`-derived features still require cautious wording as historical measured/estimated capacity-SOH features.

## Output Files

- Script: `C:\Users\ROG\Desktop\paper\codex\strict-eol-conformal-uncertainty-v1\run_strict_eol_conformal_uncertainty_v1.py`
- Batch 9 calibration residuals: `C:\Users\ROG\Desktop\paper\codex\strict-eol-conformal-uncertainty-v1\batch9_calibration_residuals.csv`
- Batch 10 conformal intervals: `C:\Users\ROG\Desktop\paper\codex\strict-eol-conformal-uncertainty-v1\batch10_conformal_intervals.csv`
- Summary JSON: `C:\Users\ROG\Desktop\paper\codex\strict-eol-conformal-uncertainty-v1\strict-eol-conformal-uncertainty-v1_summary.json`
