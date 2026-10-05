<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

<h1 align="center">硬盘故障提前预警与告警分析</h1>

<p align="center">Backblaze Drive Stats · ST4000DM000 · 七日预警 · 每日检查预算</p>

<p align="center">
  <a href="https://yemyu.github.io/drive-failure-warning/"><b>在线查看</b></a>
  · <a href="./notebooks/01_disk_failure_warning_zh.ipynb">中文 Notebook</a>
  · <a href="./notebooks/01_disk_failure_warning_en.ipynb">English notebook</a>
  · <a href="./reports/R_VALIDATION_RESULTS.md">Q4 验证结果</a>
</p>

## 项目说明

硬盘运行记录每天都会变化，但检查资源有限。本项目使用 Backblaze 的每日记录，为 `ST4000DM000` 硬盘生成检查优先级：在当天可见的设备中排序，每天选出约 0.1%，再回看入选设备是否在接下来七天出现首次观察到的 `failure=1` 标记。

输入是设备编号、日期、型号、容量和 SMART 属性。程序先整理设备日面板、检查历史是否足够，再计算分数和当天的告警名单。未来的故障记录只在名单完成后用于评价。主要比较训练于 2023 Q1/Q2 的逻辑回归 `current_lr` 与无需训练的 SMART 非零规则。

输出包括每日名单、故障事件捕获情况、提前量、重复提醒和未知结局。看板展示这些结果；Notebook 解释计算过程，并用手工构造的小数据演示从记录到告警的步骤。这里的 `failure` 是数据提供者记录的运营标记，未包含机械故障原因或维修干预信息。

## 数据、任务与方法

### 三种统计单位

| 单位 | 含义 | 用于回答 |
|---|---|---|
| 设备日 | 一台设备在某个日期的一次合格评分 | 分数能否把未来七日有故障标记的记录排在前面？ |
| 故障事件 | 一台设备首次观察到的故障标记；有合格预警日才计为“机会事件” | 有多少次故障在发生前收到告警？ |
| 告警 | 某台设备在某天入选检查名单 | 产生多少次检查，多少命中、重复或结局未知？ |

**事件召回 = 捕获的机会事件数 ÷ 全部机会事件数。** 故障前 1–7 日有至少一次告警即算捕获，同一故障只计一次。机会分母由评分资格和事件日期决定，不随某种方法的排名、冷却或实际告警改变。一个事件可能对应多条正标签设备日，也可能被多次告警，因此事件召回和逐告警命中率不能混用。

### 时间与信息边界

| 数据范围 | 用途 | 评分期 / 结局截止 |
|---|---|---|
| 2023 Q1/Q2 | 训练、估计预处理参数 | 01-15—06-23 / 06-30 |
| 2023 Q3 | 开发比较、历史特征和树模型对照 | 07-01—09-23 / 09-30 |
| 2023 Q4 | 固定模型与告警政策的季度验证 | 10-01—12-24 / 12-31 |
| 2024 Q1 | 检查后续季度的数据覆盖 | 01-14 触发预定覆盖停止，未形成效能结果 |

每个评分日要求过去 14 个自然日内至少有 12 次观测，当天及此前没有故障标记。跨季度继承已知设备、近期记录和既往故障。排序只使用日期 `≤t` 的信息，评价窗口为 `t+1…t+7`。能够确认故障的行记为正例；没有故障且随访完整的行记为负例；无法确认结局时保留 `unknown`，不按未来结局删掉评分设备。

Q3 参与过输入检查、协议修订和候选开发，属于探索性比较。Q4 的模型参数、预处理、预算和冷却政策在评价前固定；每日入选分数线仍会随当日设备和排名变化。

### 16 维当前 SMART 逻辑回归

`current_lr` 使用当天的六项 SMART 属性，不把过去 7/14 日的变化直接送入模型。历史记录用于判断评分资格和既往故障。各属性展开为以下 16 列：

| SMART 属性 | 变换 | 列数 |
|---|---|---:|
| 5、187、197、198 | `log1p(raw)`、是否缺失、是否非零 | 各 3 |
| 9 | `log1p(raw)`、是否缺失 | 2 |
| 188 | 是否非零、是否缺失 | 2 |

`log1p` 压缩大计数的尺度；缺失指示保留“没有读数”这一信息。训练仅使用 Q1/Q2 的已知结局行，保留全部正标签日，并按日确定性抽取约 5% 的负标签日。逆抽样概率权重用于拟合，权重均值归一为 1。缺失值插补、均值和标准差都从加权训练样本估计，之后与 L2 逻辑回归系数一起固定。

模型先给出对数几率分数，再按分数排序；分数也可通过 sigmoid 转为概率。本研究的概率尚未校准。完整 16 列、系数、截距与预处理参数见 [`current_lr.json`](examples/small_replay/current_lr.json)。

### SMART 规则与每日名单

`smart_nonzero` 统计 SMART 5、187、188、197、198 中非零的项数。至少一项非零的设备才进入规则候选，按非零项数降序排列；同分使用固定的确定性顺序。它不需要拟合。

两种方法共用每日容量 `ceil(N/1000)`，其中 `N` 是当天全部合格设备数。各方法分别维护七日冷却：一台设备在日期 `t` 告警后，`t+1…t+7` 不再入选，最早 `t+8` 可以再次告警。排序时跳过冷却设备，候选不足时不补齐名额。例如当天有 12,500 台合格设备，最多选 13 台；这个上限不是固定概率阈值。

完整定义与计算过程见[研究方法](reports/METHODS.md)。

## 结果

| 指标 | Q3 `current_lr` | Q3 SMART | Q4 `current_lr` | Q4 SMART |
|---|---:|---:|---:|---:|
| 捕获机会事件 | 65 / 126 | 47 / 126 | 40 / 86 | 35 / 86 |
| 事件召回 | 51.59% | 37.30% | 46.51% | 40.70% |
| 告警条数 | 1,509 | 1,509 | 1,247 | 1,247 |
| 已知结局设备日 AP | 0.1191 | 0.0233 | 0.0825 | 0.0161 |

Q4 中逻辑回归比规则多捕获 5 个事件，召回差为 5.81 个百分点。共有 86 个机会事件，少于预设的 100 个确认门槛，因此只报告描述性差异，不发布确认性区间。Q4 主事件窗口为 10-08—12-25；评分窗口与事件窗口是两个范围。

逻辑回归的 Q4 告警涉及 205 台设备，其中 1,042 条是设备首次告警后的重复提醒；规则涉及 389 台设备，后续告警 858 条。已知结局告警命中率分别为 3.56% 和 3.03%，未知告警分别为 38 和 27 条。逻辑回归捕获的事件更多，同时也更集中地重复提醒少数设备。

概率诊断与排序评价不同。Q4 Brier 分数为 `0.000544`，使用本季度实际发生率的事后常数参照为 `0.000514`；在已知结局设备日上，模型预测的正标签日总量约为实际的 `1.81` 倍。这些结果不支持把输出当作已校准的故障概率。

历史特征逻辑回归与梯度提升树在 Q3 都捕获 72/126。历史模型相对当前模型的差值区间为 −0.79 至 11.38 个百分点，跨过零，未满足采纳要求。后续重复告警重排在 Q4 达到 43/86，净增 3 个事件，低于预设的净增 5 个要求。因此保留 `current_lr`，没有继续搜索权重。2024 Q1 则因连续低覆盖在 01-14 停止，不报告该季度的模型成绩。

报告入口：[结果总览](reports/RESULTS_OVERVIEW.md) · [Q3 研究说明](reports/Q3_RESEARCH_RESULT_BRIEF.md) · [Q4 验证](reports/R_VALIDATION_RESULTS.md) · [排序、概率与告警负担](reports/ML_EVALUATION_DIAGNOSTICS.md) · [当前结论与局限](reports/CURRENT_RESEARCH_CONCLUSIONS.md)。

## 快速开始

### 1. 在线浏览：无需安装

打开 [Drive Signal](https://yemyu.github.io/drive-failure-warning/)。页面先说明项目，再展示季度结果、检查预算、设备案例、模型诊断和数据覆盖。中英文可切换；展示只读取随页面发布的结果。

### 2. 本地运行 Notebook

建议使用独立 Conda 环境。以下命令从克隆开始，不向系统 Python 或 Conda `base` 安装依赖：

```bash
git clone https://github.com/Yemyu/drive-failure-warning.git
cd drive-failure-warning
conda create -n drive-warning python=3.13 pip
conda activate drive-warning
python -m pip install -r environment/requirements.txt
python -m jupyterlab notebooks/01_disk_failure_warning_zh.ipynb
```

选择当前环境的 Python 内核，按顺序运行全部单元。预期看到设备日与事件定义、Q3/Q4 对照表、排序与概率图、告警负担、合成数据短流程和 Q1 覆盖停止说明。英文版使用同目录的 `01_disk_failure_warning_en.ipynb`。

不使用 Conda 时，可按[环境说明](environment/README.md)建立项目 `.venv`。Notebook 读取已提交的 CSV/JSON 与图表，不重建完整季度研究。

### 3. 小规模合成回放：macOS / Linux

在上述环境中、仓库根目录运行：

```bash
python -m pip install -r environment/requirements-replay.txt
mkdir -p .tmp/small_replay
python -B tools/run_small_replay.py \
  --output .tmp/small_replay/run_001 \
  --timeout-seconds 120
```

输出目录必须尚不存在；再次运行时更换 `run_001`。打开输出中的 `summary.md`：冻结 LR 应有 **14 条告警、捕获 1/2 个事件**；SMART 应有 **7 条告警、捕获 2/2 个事件**。输入是 12 台虚构设备的手工数据，这些数字用于检查机制，不能作为真实模型表现。输出还包含名单数据库、评价 JSON 和输入输出清单，详见[示例说明](examples/small_replay/README.md)。

小回放使用 POSIX 信号与资源接口；Windows 可阅读 Notebook 和看板，原生回放不在当前支持范围内。

### 重建与本地浏览看板

公开构建读取仓库内的 `dashboard/signal_v1/data.json`，不需要原始 Backblaze 文件或本地研究数据库：

```bash
python tools/build_signal_dashboard.py
python -m http.server 8000
```

打开 <http://localhost:8000/dashboard/signal_v1/>。直接打开 `index.html` 也可以浏览，需保留 `dashboard/vendor/` 及相对目录结构。macOS/Linux 下的公开页面与小回放检查：

```bash
python -m unittest tests.test_public_dashboard tests.test_small_replay
```

## 仓库目录

```text
configs/                 研究配置
pipeline/                面板、特征、评分、告警与验证实现
tools/                   运行和页面构建入口
examples/small_replay/   12 台虚构设备与冻结 LR 参数
notebooks/               中英文说明与可运行分析
dashboard/signal_v1/     当前看板与展示数据
dashboard/vendor/        Tabler 样式、来源和许可
reports/                 研究报告
reports/data/            Notebook 使用的小型结果摘要
reports/figures/         已有结果图表
tests/                   单元测试和协议检查
environment/             阅读环境与依赖说明
```

[浏览指南](docs/DEMO_GUIDE.md)介绍各页面的指标和读法。多 GB 原始文件、完整 SQLite 产物及执行记录未包含在仓库中；它们仍保留在本地研究工作区。仓库包含冻结 LR 的完整便携参数及合成示例，但仅靠结果摘要不能从头复现完整季度训练与验证。

## 范围与许可

研究只覆盖一种硬盘型号；同一设备可跨训练和评价季度出现，不能据此推断全新设备或其他型号的表现。未知结局可能与退出观察有关。没有维修干预、实际工时或成本收益测量；看板中的检查工时按告警数与假设时长计算。

本仓库的代码、文档和示例采用 [MIT License](LICENSE)。Backblaze 数据的来源及使用条件见[数据页面](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)，仓库许可不重新授权该数据。看板使用 Tabler，来源与许可见[第三方说明](THIRD_PARTY_NOTICES.md)。
