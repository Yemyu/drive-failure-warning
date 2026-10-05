# Environment setup

[中文](README.md) · [English](README.en.md) · [Project overview](../README_EN.md)

This environment supports the notebooks, synthetic example, and static dashboard build. Browsing the dashboard online requires no installation. Full quarterly training depends on raw data and local databases not committed to this repository and is outside this quick start.

## Separate Conda environment

From the repository root:

```bash
conda create -n drive-warning python=3.13 pip
conda activate drive-warning
python -m pip install -r environment/requirements.txt
python -c "import sys; print(sys.executable); print(sys.version)"
```

The final command should show an interpreter inside `drive-warning`. Using `python -m pip` ties installation to that interpreter rather than another Python installation or Conda `base`.

Open the English notebook:

```bash
python -m jupyterlab notebooks/01_disk_failure_warning_en.ipynb
```

The Chinese file is `notebooks/01_disk_failure_warning_zh.ipynb`. Select the current environment's Python kernel in JupyterLab and run all cells.

## Project-local virtual environment

If Python 3.13 is already installed and you do not use Conda, create `.venv` from the repository root.

macOS / Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python -c "import sys; print(sys.executable)"
python -m jupyterlab notebooks/01_disk_failure_warning_en.ipynb
```

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment/requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab notebooks/01_disk_failure_warning_en.ipynb
```

The Windows commands invoke the environment's interpreter directly, without changing PowerShell execution policy. The full small-replay CLI uses POSIX interfaces and currently supports macOS/Linux. The notebook's synthetic walkthrough and static dashboard do not rely on that entry point.

## Dependencies

| Purpose | Packages |
|---|---|
| Notebook tables, figures, and rich text | IPython |
| Notebook interface and Python kernel | JupyterLab, ipykernel |
| Development checks: notebook format and execution | nbformat, nbclient |
| Static dashboard build | Python standard library |
| Shared-package imports for the full POSIX replay | NumPy and scikit-learn (requirements-replay.txt) |

[`requirements.txt`](requirements.txt) specifies compatible ranges for this reading environment, not a complete lock of historical training dependencies. [`runtime.txt`](runtime.txt) records the reference versions and their scope. Both notebooks, the page build and synthetic CLI were checked in a fresh project-local environment on macOS; Windows and Linux were not tested on a separate machine. The notebooks do not require pandas. The dashboard needs no Node installation, backend service, or external database.

## Check the entry points

Build and serve the dashboard from the repository root:

```bash
python tools/build_signal_dashboard.py
python -m http.server 8000
```

Open <http://localhost:8000/dashboard/signal_v1/>. The public build uses committed display JSON only. Install `environment/requirements-replay.txt` before running the full replay; its shared package imports NumPy and scikit-learn, but the example does not fit a model. See the [example guide](../examples/small_replay/README.en.md) for replay inputs, output-directory requirements, and expected results.

If the notebook cannot locate the repository, start JupyterLab from its root. If IPython cannot be imported, check that the selected kernel uses the interpreter where you installed the requirements. If the replay output directory exists, choose a new name; earlier output is never overwritten.
