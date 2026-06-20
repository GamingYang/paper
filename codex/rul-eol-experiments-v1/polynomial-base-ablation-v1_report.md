# Polynomial Base Ablation V1

- Finished at: 2026-06-03T14:26:41
- Data dir: `C:\Users\ROG\Desktop\paper\PYthon训练代码\processed_clean`

## SOH Base-Only Target-Test

| Degree | RMSE | MAE | Macro RMSE |
|---:|---:|---:|---:|
| 1 | 0.034577 | 0.025329 | 0.033671 |
| 2 | 0.034397 | 0.025100 | 0.033514 |
| 3 | 0.032649 | 0.023746 | 0.031250 |
| 4 | 0.032923 | 0.023934 | 0.031576 |
| 5 | 0.032476 | 0.023620 | 0.030882 |

## RUL Base-Only Target-Test

| Degree | RMSE cycles | MAE cycles | Macro RMSE cycles |
|---:|---:|---:|---:|
| 1 | 274.49 | 226.05 | 261.86 |
| 2 | 256.84 | 213.07 | 240.12 |
| 3 | 255.94 | 212.73 | 238.66 |
| 4 | 255.65 | 212.94 | 238.66 |
| 5 | 255.28 | 212.84 | 238.01 |

## Interpretation

- This is a base-only ablation. It tests the global degradation prior, not the full residual neural model.
- If degree 4/5 improves only marginally while adding oscillation risk, degree 3 is easier to justify as the lowest flexible nonlinear prior.
- A journal version should still include a full-model base-degree ablation for degree 2/3/4.
