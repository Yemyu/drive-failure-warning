# Notebook 说明

[中文](README.md) · [English](README.en.md) · [项目说明](../README.md)

| 文件 | 内容 |
|---|---|
| [`01_disk_failure_warning_zh.ipynb`](01_disk_failure_warning_zh.ipynb) | 中文：任务、评分机制、Q3/Q4 结果与局限 |
| [`01_disk_failure_warning_en.ipynb`](01_disk_failure_warning_en.ipynb) | 同样的计算与结果，使用英文说明 |

两份 Notebook 使用相同的代码和数字。可以直接在 GitHub 阅读已保存输出，也可以在本地执行全部单元。

## 本地运行

先按[环境说明](../environment/README.md)建立独立 Conda 环境或项目 `.venv`，在仓库根目录执行：

```bash
python -m pip install -r environment/requirements.txt
python -m jupyterlab notebooks/01_disk_failure_warning_zh.ipynb
```

选择环境内的 Python 内核，按顺序运行全部单元。代码从仓库根目录或 `notebooks/` 目录定位项目；若更改 Notebook 的存放位置，需要保留仓库结构。

## 内容与输入

Notebook 先解释设备日、机会事件和告警，再介绍七日标签、过去 14 日的资格检查、16 维当前 SMART 特征、每日容量与冷却。合成短流程调用项目中的特征、评分和选单函数，展示记录如何进入名单；输入为手工构造的虚构设备，不用于评价真实模型性能。

随后读取以下已提交文件，展示 Q3/Q4 表格、PR 曲线、概率诊断、告警负担和候选对照：

- [`q3_methods.csv`](../reports/data/q3_methods.csv)：Q3 方法结果。
- [`q3_history_comparison.csv`](../reports/data/q3_history_comparison.csv)：历史特征与当前特征的比较。
- [`q4_summary.json`](../reports/data/q4_summary.json)：固定模型的 Q4 结果。
- [`ml_diagnostics.json`](../reports/data/ml_diagnostics.json)：已有分数的排序和概率诊断。
- [`reports/figures/`](../reports/figures/)：对应的既有结果图。

最后说明 Q3 为探索性比较、Q4 事件量不足只能描述，以及 2024 Q1 因覆盖门停止。Notebook 不下载原始数据、不重新训练，也不生成新的完整季度结论。输入摘要和完整 LR 参数可公开阅读；完整原始 CSV、SQLite 证据与执行记录仍在本地研究工作区。

方法详解见[研究方法](../reports/METHODS.md)，结果总览见[结果汇总](../reports/RESULTS_OVERVIEW.md)。
