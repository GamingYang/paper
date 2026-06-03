# RUL 快速训练与审稿式评估日志

- 运行名称：`rul-quick-v1`
- 完成时间：2026-05-28 19:45:45
- 运行环境：`D:\Anaconda\envs\battery_project\python.exe`，`cuda`
- 总耗时：173.4 s
- 数据协议：Batch 1-8 source 监督训练，Batch 9 target_adapt 用于 MMD 和 checkpoint 选择，Batch 10 target_test 只做最终盲测
- 训练设置：最多 20 epoch，每轮 60 step，patience=6；实际早停于第 11 epoch，最佳 epoch 为第 5 epoch

## 一、本轮 RUL 代码改动

新增脚本：

`C:\Users\ROG\Desktop\paper\codex\rul-quick-v1\train_rul_quick_v1.py`

输出文件：

- `C:\Users\ROG\Desktop\paper\codex\rul-quick-v1\rul-quick-v1_metrics.json`
- `C:\Users\ROG\Desktop\paper\codex\rul-quick-v1\rul-quick-v1_history.csv`
- `C:\Users\ROG\Desktop\paper\codex\rul-quick-v1\rul-quick-v1_target_test_predictions.csv`
- `C:\Users\ROG\Desktop\paper\codex\rul-quick-v1\rul-quick-v1_per_battery_metrics.csv`
- `C:\Users\ROG\Desktop\paper\codex\rul-quick-v1\rul-quick-v1_best_by_target_adapt.pt`

模型不是简单把 SOH 标签换成 RUL 标签，而是在现有 SOH 思路上做了 RUL 化改造：

- 保留三次退化物理基座思想，但拟合对象从 SOH 改为归一化 RUL。
- 保留 CNN-LSTM-Attention 时序特征提取结构。
- 保留 source-target_adapt 的生命周期进度自适应 MMD。
- 新增 SOH 辅助头，让共享表征同时学习当前健康状态。
- 新增 RUL 单调递减惩罚，约束同一电池内随 cycle 增加时预测 RUL 不应反向上升。

## 二、训练结果

| Split | RMSE (cycles) | MAE (cycles) | Macro RMSE (cycles) |
|---|---:|---:|---:|
| target_adapt | 168.57 | 137.46 | 159.11 |
| target_test | 268.16 | 228.31 | 254.81 |

分段误差：

| Split | late RUL <=300 RMSE | mid 300-1000 RMSE | early >1000 RMSE |
|---|---:|---:|---:|
| target_adapt | 207.12 | 173.82 | 151.66 |
| target_test | 293.98 | 276.36 | 250.52 |

盲测逐电池误差：

| Battery index | RMSE (cycles) | MAE (cycles) | n |
|---:|---:|---:|---:|
| 69 | 256.93 | 238.67 | 1702 |
| 70 | 353.37 | 345.39 | 1697 |
| 71 | 168.98 | 131.23 | 1848 |
| 72 | 203.41 | 191.92 | 1811 |
| 73 | 97.15 | 64.57 | 2030 |
| 74 | 303.21 | 286.33 | 2285 |
| 75 | 193.43 | 178.35 | 1783 |
| 76 | 462.00 | 455.36 | 1400 |

结论：本轮 RUL 模型已经可运行，并且遵守 Batch 9 选模、Batch 10 盲测的基本协议；但误差仍明显偏大，尤其 Battery 70、74、76 的跨批次泛化失败比较明显。它只能作为 RUL 起步 baseline，不能作为论文主结果。

## 三、RUL 比 SOH 更难还是更简单

RUL 通常比点估计 SOH 更难。SOH 是当前状态回归，RUL 是未来终点距离预测。两个电池在同一 SOH 下可能有不同退化斜率、不同拐点、不同 EOL cycle，因此同样的 SOH 误差会被 RUL 任务放大。早期寿命阶段尤其难，因为这时电池容量曲线还没有显著衰减，未来 EOL 主要取决于尚未暴露的退化速率和工况差异。

当前数据里的 `clean_*_rul.npy` 是根据完整寿命终点反推得到的标签，因此做 RUL 时必须明确在线预测设定：输入只能来自当前 cycle 及以前，不能使用任何由未来曲线计算出的特征。如果论文没有讲清楚这一点，审稿人会直接质疑数据泄漏。

## 四、严苛审稿人判断

以 Applied Energy 这类一区期刊为目标，当前 SOH 方法还没有稳妥达到“强创新”的水平。

正面部分：

- 现有 final-retained SOH 结果在 Batch 10 盲测上 RMSE=0.009499，明显优于物理基座、LSTM、GRU、CNN-LSTM 等内部 baseline。
- 消融结果显示物理基座是有效主贡献：移除物理基座后 target_test RMSE 从 0.009499 上升到 0.020716。
- Attention 有较清晰贡献：移除 Attention 后 RMSE 上升到 0.011677。
- MMD 有小幅正贡献：移除 MMD 后 RMSE 为 0.009677，提升幅度有限但方向正确。
- 后续版本已经采用 target_adapt 选模、target_test 一次性盲测，较早期脚本中使用 target_test 验证的做法更合格。

主要问题：

- 物理基座是三次多项式退化项，属于合理工程先验，但本身不构成强物理创新；如果称为 physics-informed，需要更强机理约束或可解释物理量。
- MMD 提升太小，作为核心创新会被认为贡献不足。它可以保留为跨域增强，但不宜包装成主创新。
- 目前只有同一数据集内 Batch 1-8 到 Batch 9/10 的固定划分，外部泛化证据不足。
- SOH 预测精度高，但期刊审稿会追问：是否因为 cycle position 提供了过强先验？是否能在早期、缺失、噪声、跨工况下稳定？
- 缺少不确定性校准、EOL 风险区间、早期预测能力、跨数据集验证、统计显著性和多种子置信区间。
- 如果图里对预测曲线做显示平滑，必须明确指标使用原始预测值，否则容易被质疑美化结果。

总体判断：当前 SOH 工作可以作为一篇文章的实验基础，但离 Applied Energy 级别还差一个更有说服力的问题定义和机制贡献。最现实的突破点是把 SOH+RUL 做成统一的“健康状态-寿命终点-不确定性”预测框架，而不是只在 RUL 上增加一个常规模型。

## 五、建议的 RUL 创新路线

建议主线：物理约束的 SOH-RUL 联合预测，而不是单任务 RUL 回归。

可以把论文核心改成：

`Physics-guided cross-domain joint SOH-RUL prognostics with monotonic degradation consistency and calibrated EOL uncertainty`

可操作模块：

- SOH-RUL 多任务学习：SOH 头预测当前健康状态，RUL 头预测剩余循环数，共享时序表征。
- EOL 一致性约束：令 `cycle + predicted_RUL` 在同一电池内相对稳定，减少 RUL 曲线漂移。
- 单调约束：同一电池内 `predicted_RUL` 随 cycle 增加应非增。
- 退化速率建模：增加 `dSOH/dcycle` 或局部容量衰减率辅助任务，让模型学未来衰退速度。
- 不确定性输出：使用 MC dropout、深度集成或分位数回归，报告 80/90/95% RUL 区间覆盖率。
- 早期 RUL 协议：只允许使用前 10%、20%、30% cycle 的历史，评估早期寿命预测。这比全寿命逐点 RUL 回归更接近高水平期刊关心的问题。
- 跨数据集验证：至少加入 NASA、CALCE、Oxford 或公开 fast-charging 数据集之一；如果只能用当前 77 个电池，就必须做 leave-batch-out、multi-seed、噪声鲁棒和缺失曲线实验。

## 六、下一轮实验优先级

1. 先做 RUL baseline 套件：linear physical base、RandomForest/XGBoost/SVR、LSTM/GRU/CNN-LSTM、当前 ResidualRULNet。
2. 增加 EOL 一致性损失：`Var(cycle + pred_RUL)` within battery。
3. 增加早期 RUL 协议：10%、20%、30%、50% observed life。
4. 用 3-5 个随机种子重复，报告 mean±std。
5. 对 Battery 70、74、76 做失败案例分析，检查是否为寿命长度分布偏移、容量再生、异常曲线或 Batch 10 特有工况。

## 七、参考对标

- Applied Energy 的期刊范围强调 energy conversion/conservation、optimal energy resources 和 energy systems 的创新与示范应用；单纯模型堆叠很难支撑高水平投稿，需要系统价值和工程解释。
  https://www.sciencedirect.com/journal/applied-energy/about/insights
- 2025 年 Applied Energy 已有 SOH、RUL、短期退化路径联合估计的 physics-informed neural network 工作，说明“SOH+RUL 多任务 + 物理信息”已经是当前强对标方向。
  https://www.sciencedirect.com/science/article/pii/S0306261925011572
- 2025 年 Applied Energy 也已有 physics-informed Bayesian sampling 的不确定性感知电池 prognostics 工作，说明仅报点预测误差已经不够强。
  https://www.sciencedirect.com/science/article/pii/S0306261925016101
- Nature Energy 早期寿命预测工作已经把“early-cycle prediction”设成电池寿命研究的高影响问题之一；如果你的 RUL 部分能转向早期预测，会比普通全寿命逐点回归更有价值。
  https://www.nature.com/articles/s41560-019-0356-8
