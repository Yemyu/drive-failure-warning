# 汇总文件 / Summary files

这些文件是已完成研究的数值导出，供报告、Notebook 和静态看板阅读，不是原始训练数据。

These files contain saved results for reports, notebooks and the static dashboard. They are not raw training data.

| 文件 / File | 内容 / Contents |
|---|---|
| [q3_methods.csv](q3_methods.csv) | Q3 六个基准的默认容量结果 / Six Q3 methods at the default capacity |
| [q3_history_comparison.csv](q3_history_comparison.csv) | 历史 LR 与当前 LR 的设备级配对差值 / Device-level paired comparison of history and current LR |
| [q4_summary.json](q4_summary.json) | Q4 计数、质量门、解释与执行摘要 / Q4 counts, quality gates, interpretation and execution summary |
| [ml_diagnostics.json](ml_diagnostics.json) | Q3/Q4 AP、概率误差与告警负担摘要 / Q3/Q4 ranking, probability error and alert burden |

Q3 CSV 和机器学习诊断沿用已有导出。Q4 JSON 整理了最终结论、数值与来源摘要，省略本地进程编号、文件路径和内部流转字段。原始运行清单保留在本地。Q3 CSV 的 SMART AP 空值表示原表没有计算该指标；事后诊断值在 `ml_diagnostics.json` 中，不把空值补成零。

The Q3 CSV files and machine-learning diagnostics retain the existing exports. The Q4 JSON contains the final interpretation, results and source digests; local process IDs, file paths and internal workflow fields are omitted. Original run manifests remain in the local workspace. Blank SMART AP values in the Q3 CSV mean the original export did not compute AP; post-hoc diagnostic values appear in `ml_diagnostics.json`. Missing values are not zero.

Q4 的 `gain_evidence.final_interpretation` 为 `descriptive_only_insufficient_events`：86 个机会事件少于预设的 100 个，因此只报告描述性差异，不公布确认性区间，也不认定增益获得支持。来源摘要用于核对原产物身份，完整研究数据库不随仓库发布。文字解释见[Q4报告](../R_VALIDATION_RESULTS.md)。

In the Q4 JSON, `gain_evidence.final_interpretation` is `descriptive_only_insufficient_events`: 86 opportunity events fall below the prespecified 100-event requirement. The difference remains descriptive, with no confirmatory interval or supported gain claim. Source digests identify the original artifacts; the full research databases are not published. See the [English result overview](../RESULTS_OVERVIEW_EN.md).
