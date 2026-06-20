# Cell-Level EOL Fingerprint V1

- Finished at: 2026-06-03T14:25:17
- Data dir: `C:\Users\ROG\Desktop\paper\PYthon训练代码\processed_clean`

## Best Results

| Model | Protocol | Horizon | Dev EOL RMSE | Test EOL RMSE | Test EOL MAE |
|---|---|---:|---:|---:|---:|
| gbr | source_only | 1000 | 109.82 | 52.50 | 46.44 |
| gbr | source_plus_target_adapt | 1000 | 26.16 | 59.61 | 52.35 |
| extra_trees | source_plus_target_adapt | 1000 | 12.26 | 61.16 | 54.66 |
| extra_trees | source_only | 1000 | 77.60 | 62.64 | 54.99 |
| random_forest | source_plus_target_adapt | 1000 | 29.69 | 63.16 | 59.54 |
| random_forest | source_only | 1000 | 91.08 | 76.97 | 62.65 |
| extra_trees | source_plus_target_adapt | 500 | 25.59 | 77.08 | 59.39 |
| gbr | source_plus_target_adapt | 500 | 44.12 | 87.30 | 71.45 |
| extra_trees | source_only | 500 | 105.14 | 90.07 | 66.77 |
| random_forest | source_plus_target_adapt | 500 | 35.36 | 90.27 | 70.76 |
| random_forest | source_only | 500 | 106.83 | 93.02 | 75.27 |
| svr_rbf | source_plus_target_adapt | 800 | 23.95 | 97.61 | 76.50 |
| svr_rbf | source_only | 500 | 118.42 | 101.43 | 67.27 |
| gbr | source_only | 500 | 123.99 | 101.77 | 82.64 |
| svr_rbf | source_plus_target_adapt | 500 | 100.34 | 104.27 | 90.07 |
| ridge | source_plus_target_adapt | 500 | 110.25 | 110.71 | 93.26 |
| elastic_net | source_plus_target_adapt | 500 | 89.12 | 116.90 | 87.25 |
| elastic_net | source_only | 500 | 141.13 | 129.99 | 108.57 |
| ridge | source_plus_target_adapt | 800 | 84.92 | 132.92 | 118.70 |
| svr_rbf | source_plus_target_adapt | 100 | 125.76 | 133.49 | 121.83 |

## Interpretation

- Cell-level EOL error is numerically equal to pointwise RUL error when RUL is computed as predicted EOL minus current cycle.
- This protocol uses early observed SOH/capacity history and processed voltage-curve fingerprints.
- The `source_plus_target_adapt` protocol is supervised small-target adaptation, not unsupervised domain adaptation.
