# 小规模合成回放

[中文](README.md) · [English](README.en.md) · [项目说明](../../README.md)

这个例子用 **12 台虚构设备**演示从每日记录到评分、名单和未来结局评价的完整流程。输入是手工构造的数据，不是 Backblaze 子集。逻辑回归使用已导出的固定参数，不在示例中训练。

## 输入与规则

| 文件 | 用途 |
|---|---|
| [`daily.csv`](daily.csv) | 2023-01-01—02-04，共 35 日的手工记录 |
| [`current_lr.json`](current_lr.json) | 16 维 LR 系数、截距、插补与标准化参数 |
| [`expected.json`](expected.json) | 标签、特征、分数和机会事件的预期检查 |
| [`manifest.json`](manifest.json) | 输入清单与校验值 |

评分期为 01-15—01-28，主事件期为 01-22—01-29，结局截止 02-04。评分资格要求过去 14 个自然日内至少有 12 次观测，且截至评分日没有故障。每种方法每天最多选择 `ceil(N/1000)` 台合格设备，告警后冷却七日，第八日才可以再次入选。

LR 按固定对数几率分数排序。SMART 对照只考虑五项属性中至少有一项非零的设备，按非零项数排序。两种方法分别维护冷却记录，预算均按全部合格设备数计算。

## 运行

使用 Python 3.13 的独立环境，配置方式见[环境说明](../../environment/README.md)。回放依赖 POSIX 信号和资源接口，当前支持 macOS/Linux。现有共享包在导入时需要 NumPy 和 scikit-learn；示例使用冻结参数，不执行拟合。

从仓库根目录运行：

```bash
python -m pip install -r environment/requirements-replay.txt
mkdir -p .tmp/small_replay
python -B tools/run_small_replay.py \
  --output .tmp/small_replay/run_001 \
  --timeout-seconds 120
```

程序会创建 `run_001`，因此这个目录必须尚不存在；父目录需要先建好。再次运行时改用 `run_002` 等新名称。输出必须位于本项目内，不会覆盖已有目录。

默认运行上限为 120 秒，固定输入最多 1,000 行、输出预算 10 MiB、记录的 RSS 上限 256 MiB。RSS 检查是运行时观测，不是操作系统内存隔离。

## 预期结果与输出

| 方法 | 告警 | 捕获机会事件 |
|---|---:|---:|
| `current_lr` | 14 | 1 / 2 |
| `smart_nonzero` | 7 | 2 / 2 |

这些数字只检验手工输入中的执行机制，不能与真实 Q3/Q4 成绩比较，也不是采纳规则模型的依据。

| 输出文件 | 内容 |
|---|---|
| `selection.sqlite` | 截至决策日的特征、分数、预算、冷却和名单；不含未来标签表 |
| `evaluation.json` | 名单完成后计算的结局、机会事件、命中与未知结局 |
| `summary.md` | 合成结果简表 |
| `run_manifest.json` | 完成状态、输入输出哈希、访问边界与资源记录 |

正常退出会打印输出目录。先确认 `run_manifest.json` 中的 `status` 为 `complete`，再打开 `summary.md`。只有部分文件或出现 `error.json` 时，该次运行不能视为成功。

例子包括：A/B 的主窗口故障、C 的窗口外故障、D 的缺日、E 的提前退出、F 的当前 SMART 全缺失、G 的历史不足，以及 H 的既往故障。它们用于区分“不能评分”“结局未知”和“已确认未命中”，避免将这些情况都填成负例。

## 实现入口与复现范围

[`tools/run_small_replay.py`](../../tools/run_small_replay.py) 调用项目的特征与告警选择实现。评分查询限制为日期 `≤t`，名单写完并关闭后才连接未来结局。随访不足仍保留未知，不用未来故障筛选当天的候选。

可在支持平台运行定向检查：

```bash
python -m unittest tests.test_small_replay
```

本例能够复跑这组固定小输入，不能替代完整季度训练与真实数据验证。Notebook 内另有无需 POSIX 回放入口的短流程，适合逐步查看特征和排序。
