# Online EOL Baselines V1

- Finished at: 2026-06-03T14:21:02
- Data dir: `C:\Users\ROG\Desktop\paper\PYthon训练代码\processed_clean`
- EOL threshold: SOH=0.8

## Best Target-Test Results

| Method | Min history | Prior | RMSE | MAE | Macro RMSE | Early RMSE | Mid RMSE | Late RMSE |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| poly3_all_history | 200 | median | 351.02 | 195.95 | 304.65 | 512.82 | 107.69 | 52.20 |
| poly3_all_history | 100 | median | 416.59 | 225.89 | 353.77 | 612.26 | 107.69 | 52.20 |
| poly3_all_history | 50 | median | 474.46 | 253.62 | 420.78 | 699.64 | 107.69 | 52.20 |
| poly4_all_history | 200 | median | 478.08 | 297.40 | 418.85 | 685.66 | 206.60 | 62.83 |
| poly4_all_history | 100 | median | 529.35 | 329.28 | 458.33 | 764.82 | 206.60 | 62.83 |
| poly4_all_history | 50 | median | 529.68 | 329.55 | 458.62 | 765.32 | 206.60 | 62.83 |
| poly2_all_history | 200 | median | 642.37 | 446.49 | 629.28 | 728.00 | 669.95 | 96.75 |
| poly2_all_history | 100 | median | 677.91 | 479.05 | 660.55 | 796.42 | 669.95 | 96.75 |
| poly2_all_history | 50 | median | 701.75 | 498.75 | 681.17 | 841.08 | 669.95 | 96.75 |
| robust_recent_slope | 200 | median | 1399.85 | 1009.02 | 1366.35 | 1945.76 | 816.15 | 33.94 |
| linear_recent_100 | 200 | median | 1407.98 | 994.67 | 1373.31 | 1976.30 | 765.24 | 27.21 |
| linear_recent_300 | 200 | median | 1425.23 | 1062.97 | 1394.27 | 1926.23 | 969.49 | 58.94 |

## Note

- These baselines use only the SOH/capacity history available up to the current cycle.
- They are not comparable to voltage-only models unless the task definition allows measured capacity/SOH history.
- If this protocol is accepted, RUL should be framed as online EOL extrapolation rather than pointwise black-box regression.
