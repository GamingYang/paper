# MMD Exponential Weight Curve

This figure visualizes the lifecycle-adaptive MMD multiplier used by the SOH model.

- Formula: `exp(5.5 * (1 - SOH))`.
- Static MMD baseline multiplier: `1.0`.
- Outer target MMD coefficient in training: `5e-05`.
- SOH range shown: 1.00 to 0.80, matching the nominal SOH range of the fixed split.

Key points:

| SOH | Dynamic multiplier | Effective MMD coefficient |
|---:|---:|---:|
| 1.00 | 1.0000 | 0.00005000 |
| 0.95 | 1.3165 | 0.00006583 |
| 0.90 | 1.7333 | 0.00008666 |
| 0.85 | 2.2819 | 0.00011409 |
| 0.80 | 3.0042 | 0.00015021 |

Outputs:

- `C:\Users\ROG\Desktop\paper\codex\soh-mmd-exponential-weight-figure-v1\figures\soh_mmd_exponential_weight_curve.png`
- `C:\Users\ROG\Desktop\paper\codex\soh-mmd-exponential-weight-figure-v1\figures\soh_mmd_exponential_weight_curve.pdf`
- `C:\Users\ROG\Desktop\paper\codex\soh-mmd-exponential-weight-figure-v1\mmd_exponential_weight_curve.csv`