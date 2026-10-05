# 环境配置

[中文](README.md) · [English](README.en.md) · [项目说明](../README.md)

这里配置的是阅读 Notebook、运行合成示例和重建静态看板的环境。看板在线浏览不需要安装软件。完整季度训练涉及未提交的原始数据和本地数据库，不属于这份快速开始。

## 独立 Conda 环境

从仓库根目录执行：

```bash
conda create -n drive-warning python=3.13 pip
conda activate drive-warning
python -m pip install -r environment/requirements.txt
python -c "import sys; print(sys.executable); print(sys.version)"
```

最后一条命令应显示 `drive-warning` 环境中的解释器。安装使用 `python -m pip`，避免把依赖装到另一个 Python 或 Conda `base`。

启动中文 Notebook：

```bash
python -m jupyterlab notebooks/01_disk_failure_warning_zh.ipynb
```

英文文件为 `notebooks/01_disk_failure_warning_en.ipynb`。在 JupyterLab 中选择当前环境的 Python 内核，然后运行全部单元。

## 项目虚拟环境替代方案

已经有 Python 3.13、但不使用 Conda 时，在仓库根目录建立 `.venv`：

macOS / Linux：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python -c "import sys; print(sys.executable)"
python -m jupyterlab notebooks/01_disk_failure_warning_zh.ipynb
```

Windows PowerShell：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment/requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab notebooks/01_disk_failure_warning_zh.ipynb
```

Windows 命令直接调用环境内解释器，不需要改变 PowerShell 执行策略。完整小回放使用 POSIX 接口，当前只支持 macOS/Linux；Notebook 的合成短流程与静态看板不依赖这一入口。

## 依赖用途

| 用途 | 依赖 |
|---|---|
| 显示 Notebook 中的表格、图片和说明 | IPython |
| 打开 Notebook、提供 Python 内核 | JupyterLab、ipykernel |
| 开发检查：读取 Notebook 格式、执行全部单元 | nbformat、nbclient |
| 静态看板构建 | Python 标准库 |
| macOS/Linux 完整合成回放的共享包导入 | NumPy、scikit-learn（单独的 requirements-replay.txt） |

[`requirements.txt`](requirements.txt) 声明这组阅读依赖的兼容版本范围，不是历史训练环境的完整锁文件。[`runtime.txt`](runtime.txt) 记录验证环境版本及用途。双语 Notebook、页面构建和合成 CLI 已在 macOS 的独立项目虚拟环境中验证；未做 Windows 或 Linux 实机验证。Notebook 不要求 pandas；看板不需要 Node、后端服务或外部数据库。

## 验证运行入口

在仓库根目录重建和打开看板：

```bash
python tools/build_signal_dashboard.py
python -m http.server 8000
```

访问 <http://localhost:8000/dashboard/signal_v1/>。公开构建只使用已提交的展示 JSON。运行完整小回放前执行 `python -m pip install -r environment/requirements-replay.txt`；共享包导入需要这些依赖，但示例不拟合模型。小回放的输入、输出目录要求和预期数字见[示例说明](../examples/small_replay/README.md)。

若 Notebook 提示找不到仓库，先在根目录启动 JupyterLab。若导入 IPython 失败，检查实际内核的解释器是否与安装依赖时相同。小回放输出目录已存在时，应改用新的目录名，旧结果不会被覆盖。
