# Battery SOH/RUL Prediction Paper Project

这个仓库用于管理锂离子电池健康状态预测相关的代码、实验脚本和结果摘要。当前仓库重点保存可复现实验逻辑的源代码和轻量级结果说明，大体积原始数据、模型权重、中间数据和论文资料没有直接上传到 GitHub。

## 项目内容

- `PYthon训练代码/`
  - SOH 训练与评估脚本。
  - 包含主模型、不同随机种子版本，以及去除注意力、MMD、物理约束、平滑项等消融版本。

- `codex/`
  - 实验自动化脚本、基线实验脚本、补充实验脚本和结果摘要。
  - 已上传的是 `.py`、`.json`、`.md`、少量汇总 `.csv`。
  - 模型权重、详细预测结果、日志、缓存、Word/Excel 报告和图片大多保留在本地。

- `codex_battery_project_smoke_test.py`
  - 环境检查脚本，用于确认 Python、主要依赖包和 Excel 写入能力是否可用。

- `.gitignore`
  - Git 忽略规则，防止误上传数据集、模型权重、缓存、日志和临时文件。

## 未上传到 GitHub 的内容

以下内容保留在本地，不建议直接放入普通 Git 仓库：

- `our_data/`：原始 `.pkl` 数据，体积约 4GB。
- `PYthon训练代码/processed_clean/`：由原始数据处理得到的 `.npy` 和 `.csv` 中间数据。
- `*.pt`、`*.pth`、`*.pkl`：模型权重、checkpoint、序列化 pipeline。
- `*.log`、`__pycache__/`、`.idea/`：运行日志、Python 缓存和本地编辑器配置。
- 大量参考文献 PDF、Word 草稿、Excel 过程表格、绘图素材。

如果后续需要共享大文件，建议使用网盘、学校服务器、对象存储或 Git LFS，而不是直接提交到 GitHub。

## 环境依赖

当前脚本主要依赖：

- Python 3.x
- numpy
- pandas
- torch
- scikit-learn
- matplotlib
- scipy
- openpyxl
- python-docx

可以先运行环境检查：

```powershell
python codex_battery_project_smoke_test.py
```

该脚本会在本地生成检查结果文件；这些结果文件默认不作为核心代码提交。

## 数据准备

训练脚本默认读取 `processed_clean` 数据目录。典型脚本中使用的路径类似：

```python
DATA_DIR = Path("./processed_clean")
```

因此运行 `PYthon训练代码` 下的训练脚本前，需要确保当前工作目录下存在 `processed_clean/`，或者按脚本需要调整 `DATA_DIR`。

`processed_clean/` 至少应包含类似文件：

- `clean_source_x.npy`
- `clean_source_y_nominal.npy`
- `clean_source_p.npy`
- `clean_source_battery_index.npy`
- `clean_target_adapt_x.npy`
- `clean_target_test_x.npy`
- `clean_target_test_meta.csv`
- `protocol.json`

## 常用运行方式

进入项目根目录：

```powershell
cd C:\Users\ROG\Desktop\paper
```

运行环境检查：

```powershell
python codex_battery_project_smoke_test.py
```

运行 SOH 训练脚本时，建议进入训练代码目录，并确认该目录下有 `processed_clean/`：

```powershell
cd C:\Users\ROG\Desktop\paper\PYthon训练代码
python train_soh_processed_clean_checkpoint.py
```

运行评估和绘图脚本：

```powershell
python evaluate_soh_processed_clean_visual_v3.py
```

运行 `codex` 中的实验脚本示例：

```powershell
cd C:\Users\ROG\Desktop\paper
python codex\run_ablation_suite_v1.py
python codex\kernel-baseline-suite-v2\run_kernel_baseline_suite_v2.py
python codex\rul-quick-v1\train_rul_quick_v1.py
```

具体脚本可能依赖本地数据、checkpoint 或已生成的中间结果。运行前需要检查脚本顶部的路径配置。

## Git 使用流程

查看当前状态：

```powershell
git status
```

只添加确定要上传的文件，不建议直接使用 `git add .`：

```powershell
git add README.md
```

提交：

```powershell
git commit -m "Add project README"
```

推送到 GitHub：

```powershell
git push
```

确认本地和 GitHub 同步：

```powershell
git status --short --branch
```

如果显示：

```text
## master...origin/master
```

并且没有 `[ahead 1]`，说明本地和远程仓库已经同步。

## 当前仓库策略

本仓库优先保存：

- 可复现实验的 Python 源码。
- 关键实验配置和轻量级结果摘要。
- 论文写作相关的最终化脚本。
- 项目说明和 Git 管理文件。

本仓库暂不直接保存：

- 大体积原始数据。
- 训练好的模型权重。
- 中间处理数据。
- 大量过程图、过程表格和 Word 草稿。

这样做可以让 GitHub 仓库保持轻量、清晰，方便后续继续维护、回滚和协作。
