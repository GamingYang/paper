# SOH MMD Representation Diagnostic v1

## Protocol

This package diagnoses fused SOH representations using source Batch 1-8 and Batch 9 target-development/adaptation features only. Batch 10 is not loaded and is treated as an untouched blind-test set outside this diagnostic. The diagnostic is not used for checkpoint selection or hyperparameter tuning.

## Key final vs w/o MMD findings

| Metric | final-retained-v1 | no-mmd-v1 | Direction |
| --- | ---: | ---: | --- |
| RBF multikernel MMD^2, z-scored fused features | 0.153337 | 0.464046 | -67.0% vs w/o MMD |
| Domain classifier AUC | 0.979493 | 0.988835 | -0.9% vs w/o MMD |
| Domain classifier accuracy | 0.940667 | 0.951500 | -1.1% vs w/o MMD |

## Interpretation

The retained model shows representation-level support for the MMD regularizer on at least one domain-mixing diagnostic. This should be written as auxiliary representation evidence, not as a dominant accuracy-driver claim.

Lower feature MMD or lower domain classifier separability suggests stronger source/Batch9 feature mixing. Because the known Batch10 RMSE difference between final retained and w/o MMD is small, this diagnostic should be presented cautiously as representation-level support or as mixed evidence, not as proof that MMD dominates prediction accuracy.

## Configuration

- run_name: soh_mmd_representation_diagnostic_v1
- created_at: 2026-06-21T23:01:55
- project_root: C:\Users\ROG\Desktop\paper
- output_dir: C:\Users\ROG\Desktop\paper\codex\soh-mmd-representation-diagnostic-v1
- data_dir: C:\Users\ROG\Desktop\paper\PYthon训练代码\processed_clean
- label_mode: nominal
- data_protocol: processed_clean_fixed_split
- seed: 42
- device: cuda
- batch10_policy: not loaded; blind-test reporting only outside this diagnostic
- feature_distance_per_domain: 3000
- domain_classifier_per_domain: 10000
- soh_bin_per_domain: 1500
- mmd_rbf_multikernel_setup: biased MMD^2; Gaussian bandwidth base is mean off-diagonal squared distance on the deterministic balanced sample; kernel_mul=2.0; kernel_num=5; bandwidth_i=base/kernel_mul^(kernel_num//2)*kernel_mul^i
- domain_classifier_setup: balanced deterministic source/Batch9 sample; 70/30 stratified split; StandardScaler + LogisticRegression

## Figures

- `C:\Users\ROG\Desktop\paper\codex\soh-mmd-representation-diagnostic-v1\figures\mmd_representation_domain_bars.png`
- `C:\Users\ROG\Desktop\paper\codex\soh-mmd-representation-diagnostic-v1\figures\final_vs_no_mmd_pca_2d.png`
