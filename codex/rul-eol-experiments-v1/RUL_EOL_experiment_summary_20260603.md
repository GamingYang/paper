# RUL/EOL 实验阶段汇报与后续计划

日期：2026-06-03  
项目目录：`C:\Users\ROG\Desktop\paper`

## 1. 本轮目标

本轮主要回答五个问题：

1. 当前数据集中到底有哪些原始字段，是否包含温度信息。
2. RUL 预测误差能否从 200+ cycles 压到 100 cycles 以内。
3. RUL 是否应该继续做逐点回归，还是转成 EOL 预测更合理。
4. 三次物理退化基座是否有必要，二次/四次/五次是否更合适。
5. 后续论文故事应如何围绕工程痛点组织。

## 2. 数据审计结论

审计脚本：

- `C:\Users\ROG\Desktop\paper\codex\data-audit-v1\audit_battery_data.py`
- `C:\Users\ROG\Desktop\paper\codex\data-audit-v1\battery_data_audit.md`
- `C:\Users\ROG\Desktop\paper\codex\data-audit-v1\raw_field_inventory.csv`
- `C:\Users\ROG\Desktop\paper\codex\data-audit-v1\processed_cell_inventory.csv`

原始 `our_data/*.pkl` 结构：

- `rul`：每个 cycle 的剩余寿命。
- `dq`：每个 cycle 的容量值。
- `data`：每个 cycle 的 DataFrame。

每个 cycle 的 DataFrame 字段：

- `Status`
- `Cycle number`
- `Current (mA)`
- `Voltage (V)`
- `Capacity (mAh)`
- `Time (s)`

未发现温度字段。`processed_clean` 元数据中也没有温度列。

processed 数据划分：

| Split | Samples | Cells | Batches | Cycle Range | SOH Range | RUL Range |
|---|---:|---:|---|---|---|---|
| source | 115256 | 61 | 1-8 | 1-2689 | 0.7998-1.1197 | 0-2688 |
| target_adapt | 16310 | 8 | 9 | 1-2308 | 0.8000-1.1138 | 0-2307 |
| target_test | 14556 | 8 | 10 | 1-2285 | 0.7997-1.1222 | 0-2284 |

关键结论：`cycle + rul_cycle` 在每个电池内部恒定，说明当前 RUL 标签本质是：

```text
RUL(current cycle) = EOL cycle - current cycle
```

因此，RUL 任务更适合转成 **EOL cycle 预测**，再由 EOL 推出任意 cycle 的 RUL。

## 3. 已完成实验与结果

### 3.1 第一版深度 RUL quick baseline

脚本：

- `C:\Users\ROG\Desktop\paper\codex\rul-quick-v1\train_rul_quick_v1.py`

结果：

| Split | RMSE cycles | MAE cycles | Macro RMSE cycles |
|---|---:|---:|---:|
| target_adapt | 168.57 | 137.46 | 159.11 |
| target_test | 268.16 | 228.31 | 254.81 |

判断：效果不够好，甚至弱于后续简单树模型和 EOL 先验 baseline。这个模型不适合作为主结果。

### 3.2 RUL baseline suite

脚本：

- `C:\Users\ROG\Desktop\paper\codex\rul-baseline-suite-v1\run_rul_baselines_v1.py`
- `C:\Users\ROG\Desktop\paper\codex\rul-baseline-suite-v1\rul-baseline-suite-v1_report.md`

主要结果：

| Model | Protocol | Test RMSE cycles |
|---|---|---:|
| ExtraTrees direct RUL + source + full target_adapt | 小样本目标域监督上界 | 223.34 |
| ExtraTrees direct RUL + source only | 纯跨域泛化 | 229.71 |
| source 中位 EOL baseline | 纯寿命先验 | 236.56 |
| source 平均 EOL baseline | 纯寿命先验 | 241.22 |
| cubic cycle-to-RUL source | 物理多项式先验 | 255.94 |
| quick neural RUL | 上一版深度模型 | 268.16 |

经验：RUL 逐点回归很容易只学到 `cycle -> average EOL` 的寿命分布先验。后续必须把 `EOL baseline` 和 `cycle-only baseline` 列为必要对照。

### 3.3 在线 SOH 曲线外推 EOL

脚本：

- `C:\Users\ROG\Desktop\paper\codex\rul-eol-experiments-v1\run_online_eol_baselines_v1.py`
- `C:\Users\ROG\Desktop\paper\codex\rul-eol-experiments-v1\online-eol-baselines-v1_report.md`

最佳 target_test：

| Method | Min history | Test RMSE | Test MAE | Early RMSE | Mid RMSE | Late RMSE |
|---|---:|---:|---:|---:|---:|---:|
| poly3_all_history | 200 | 351.02 | 195.95 | 512.82 | 107.69 | 52.20 |

结论：直接用历史 SOH 曲线外推到 0.8 SOH 不可靠。临近 EOL 时表现很好，但早期误差极大。

经验：不能把“多项式外推到 80%”包装成创新点。它在 late stage 有用，但 early/mid stage 不够稳。

### 3.4 Cell-level EOL fingerprint 实验

脚本：

- `C:\Users\ROG\Desktop\paper\codex\rul-eol-experiments-v1\run_cell_level_eol_fingerprint_v1.py`
- `C:\Users\ROG\Desktop\paper\codex\rul-eol-experiments-v1\cell-level-eol-fingerprint-v1_report.md`
- `C:\Users\ROG\Desktop\paper\codex\rul-eol-experiments-v1\cell-level-eol-fingerprint-v1_h500_extra_trees_feature_importance.csv`

特征构成：

- 前 N 圈 SOH/容量轨迹统计。
- 最近 20/50/100/200 圈 SOH slope 和 drop。
- SOH 多项式拟合系数。
- processed `x` 曲线的均值、最后一圈、变化量等统计特征。
- cycle 100 与 cycle 10 的曲线差异特征。

重要提醒：第一次运行中出现了数据泄露字段 `observed_ratio = observed_cycles / full_cell_length`，该字段使用了完整寿命长度，已删除。删除前 5.8 cycles 的结果作废。

去泄露后结果：

| Horizon | Best Test EOL/RUL RMSE cycles |
|---:|---:|
| 50 | 158.73 |
| 100 | 133.49 |
| 200 | 147.33 |
| 300 | 140.49 |
| 500 | 77.08 |
| 800 | 97.61 |
| 1000 | 52.50 |

如果按更严格的 target_adapt 选模逻辑：

| Horizon | Model | Protocol | Target Adapt RMSE | Target Test RMSE |
|---:|---|---|---:|---:|
| 50 | ExtraTrees | source+target_adapt | 44.28 | 193.99 |
| 100 | ExtraTrees | source+target_adapt | 29.52 | 161.52 |
| 200 | ExtraTrees | source+target_adapt | 26.10 | 152.64 |
| 300 | ExtraTrees | source+target_adapt | 21.01 | 161.35 |
| 500 | ExtraTrees | source+target_adapt | 25.59 | 77.08 |
| 800 | ExtraTrees | source+target_adapt | 14.78 | 137.13 |
| 1000 | ExtraTrees | source+target_adapt | 12.26 | 61.16 |

结论：

- 100 cycles 以内可以做到，但目前需要至少约 500 圈历史。
- 50/100/200 圈极早期预测仍未达到 100 cycles。
- 当前最有价值的任务不是“全寿命逐点 RUL 回归”，而是“跨批次小样本目标域的早中期 EOL 预测”。

500 圈 ExtraTrees 特征重要性前几项：

| Feature | Meaning |
|---|---|
| `soh_slope_last_100` | 最近 100 圈 SOH 衰退速率 |
| `soh_slope_last_50` | 最近 50 圈 SOH 衰退速率 |
| `soh_slope_last_20` | 最近 20 圈 SOH 衰退速率 |
| `soh_min` | 观测窗口内最低 SOH |
| `soh_last` | 当前观测窗口最后 SOH |
| `soh_drop_last_100` | 最近 100 圈 SOH 下降量 |

说明：模型主要依赖容量退化速率，而不是复杂黑盒特征。这一点对论文解释有利。

## 4. 三次物理基座消融

脚本：

- `C:\Users\ROG\Desktop\paper\codex\rul-eol-experiments-v1\run_polynomial_base_ablation_v1.py`
- `C:\Users\ROG\Desktop\paper\codex\rul-eol-experiments-v1\polynomial-base-ablation-v1_report.md`

SOH base-only target_test：

| Degree | RMSE | MAE | Macro RMSE |
|---:|---:|---:|---:|
| 1 | 0.034577 | 0.025329 | 0.033671 |
| 2 | 0.034397 | 0.025100 | 0.033514 |
| 3 | 0.032649 | 0.023746 | 0.031250 |
| 4 | 0.032923 | 0.023934 | 0.031576 |
| 5 | 0.032476 | 0.023620 | 0.030882 |

RUL base-only target_test：

| Degree | RMSE cycles | MAE cycles | Macro RMSE cycles |
|---:|---:|---:|---:|
| 1 | 274.49 | 226.05 | 261.86 |
| 2 | 256.84 | 213.07 | 240.12 |
| 3 | 255.94 | 212.73 | 238.66 |
| 4 | 255.65 | 212.94 | 238.66 |
| 5 | 255.28 | 212.84 | 238.01 |

结论：

- 三次明显优于一次和二次。
- 四次相对三次没有稳定提升。
- 五次 base-only 略好，但提升极小，且外推振荡/过拟合风险更高。

推荐表述：

> 三次多项式不是“绝对物理真理”，而是最低阶、能表达非线性加速退化和曲率变化的全局退化先验。它相比二次有明显提升，相比四/五次复杂度更低、外推更稳定。

后续论文仍建议补完整模型的 degree=2/3/4 消融，而不只做 base-only。

## 5. 本轮经验总结

### 5.1 任务定义比模型结构更重要

逐点 RUL 回归效果不好，因为 RUL 标签本质上由 EOL 决定。直接逐点训练会产生大量强相关样本，但真正独立的 EOL 标签只有 77 个电池。模型很容易学到平均寿命先验，而不是电池个体退化机制。

更合理的路线：

```text
先预测 cell-level EOL，再计算任意 cycle 的 RUL。
```

### 5.2 数据泄露必须严防

本轮发现并修复了一个泄露字段：`observed_ratio = observed_cycles / full_cell_length`。由于 `full_cell_length` 就是 EOL，该字段不能用于早期预测。

后续所有特征必须满足：

```text
只使用当前 cycle 及以前的数据。
不使用完整寿命长度。
不使用 target_test 做模型选择。
```

### 5.3 100 cycles 以内是可达的，但不是极早期

当前 500 圈窗口已经达到 77 cycles RMSE；1000 圈约 52-61 cycles。但 100 圈仍在 130-160 cycles 附近。因此论文主张应谨慎：

- 可以说 early-to-mid life prediction。
- 不宜声称 very early prediction 已经做到高精度。

### 5.4 工程痛点可以成立

建议工程痛点：

> 新批次/新工况电池到手时，完整 EOL 标签获取成本极高；但 BMS、质保、梯次利用和安全维护需要在早中期就判断电池寿命风险。现有逐点 SOH/RUL 模型要么依赖完整寿命数据，要么跨批次泛化不稳。因此需要一种面向小样本目标域的早中期 EOL/RUL 预后框架。

这个痛点比“我用了某某网络结构”更像 Applied Energy 的故事。

## 6. 推荐论文主线

建议标题方向：

```text
Physics-consistent domain-adaptive early-to-mid-life EOL/RUL prediction
for lithium-ion batteries under small target-domain samples
```

中文表达：

```text
面向跨批次小样本目标域的早中期 EOL/RUL 预测：
融合容量退化指纹、物理一致性约束与域自适应的电池寿命预后框架
```

核心贡献建议：

1. 将 RUL 逐点回归重构为 cell-level EOL 预测，避免样本伪独立和平均寿命先验问题。
2. 构造早中期容量退化指纹，包括 SOH slope、SOH drop、局部曲线变化和电压片段统计。
3. 引入小样本目标域适配协议，区分 `source_only`、`source+target_adapt`、`target_test blind`。
4. 加入物理一致性：EOL-RUL 一致性、RUL 单调性、SOH 衰退速率约束。
5. 用分 horizon 协议报告 50/100/200/300/500/800/1000 cycles 的性能，明确早期和中期能力边界。

## 7. 后续实验方向

### 优先级 A：必须做

1. 固定严格协议：
   - Batch 1-8：source
   - Batch 9：target_adapt，用于小样本适配和模型选择
   - Batch 10：blind test，只最终评估

2. 补全 horizon 实验：
   - 50、100、200、300、500、800、1000 cycles
   - 每个 horizon 都报告 target_adapt 和 target_test

3. 加 cycle-only / EOL-prior baseline：
   - source mean EOL
   - source median EOL
   - cycle-only regression
   - polynomial base-only

4. 完整模型多项式基座消融：
   - degree=2
   - degree=3
   - degree=4

5. 多种子重复：
   - seed 7、42、123
   - 报告 mean ± std

### 优先级 B：模型改进

1. 从 ExtraTrees baseline 升级为神经 EOL 模型：
   - 输入：前 N 圈 SOH 序列 + 电压曲线 fingerprint
   - 输出：EOL cycle
   - 辅助输出：SOH slope、future SOH at fixed horizon、uncertainty

2. 加物理一致性损失：

```text
predicted_RUL(cycle) = predicted_EOL - current_cycle
RUL should be non-negative and monotonically decreasing.
SOH degradation rate should be physically plausible.
```

3. 加不确定性：
   - quantile regression
   - MC dropout
   - ensemble
   - 报告 80/90/95% prediction interval coverage

4. 加目标域小样本校准：
   - 只用 Batch 9 的前 100/200/500 圈
   - 不用 Batch 9 完整 EOL 做过强监督

### 优先级 C：论文增强

1. 失败案例分析：
   - Battery 10-8 寿命 1400，短寿命电池是否更难
   - Battery 10-6 寿命 2285，长寿命电池是否更易被低估

2. 图表：
   - horizon vs RMSE 曲线
   - predicted EOL vs true EOL scatter
   - per-battery absolute error bar
   - feature importance / SHAP
   - source/target distribution t-SNE or PCA

3. 和 2026 Applied Energy PDF 的衔接：
   - 借鉴其“工程痛点 + 初始化/收敛/鲁棒性”的写法。
   - 我们的对应痛点是“小样本目标域下早中期寿命风险识别”。

## 8. 下次接续建议

下次建议直接从以下任务开始：

1. 把 `cell-level-eol-fingerprint-v1` 改成正式实验套件。
2. 增加严格 target_adapt-only 选模，不允许看 target_test 选 horizon。
3. 输出完整 Excel/Markdown 报告。
4. 实现一个神经版 EOL fingerprint model，与 ExtraTrees 对比。
5. 补 degree=2/3/4 完整模型消融。

当前最有希望的结果是：

```text
500 cycles observed history:
ExtraTrees + source + target_adapt
Batch 10 EOL/RUL RMSE ≈ 77 cycles
```

这是目前最接近一区论文主线的有效结果。
