# Notebook guide

[中文](README.md) · [English](README.en.md) · [Project overview](../README_EN.md)

| File | Contents |
|---|---|
| [`01_disk_failure_warning_en.ipynb`](01_disk_failure_warning_en.ipynb) | Task, scoring mechanism, Q3/Q4 results, and limitations in English |
| [`01_disk_failure_warning_zh.ipynb`](01_disk_failure_warning_zh.ipynb) | The same calculations and results in Chinese |

The notebooks use identical code and numbers. Read the saved outputs on GitHub, or execute all cells locally.

## Run locally

Create a separate Conda environment or project `.venv` using the [environment guide](../environment/README.en.md), then run from the repository root:

```bash
python -m pip install -r environment/requirements.txt
python -m jupyterlab notebooks/01_disk_failure_warning_en.ipynb
```

Select the environment's Python kernel and run cells in order. The code locates the project from its root or `notebooks/` directory. Keep the repository structure if you move files.

## Contents and inputs

The notebook defines device-days, opportunity events, and alerts, then explains seven-day labels, eligibility based on 14 days of history, the 16 current SMART features, daily capacity, and cooldown. A short synthetic walkthrough calls the project's feature, scoring, and selection functions to show how records become an alert list. Its hand-built fictional drives are a mechanism demonstration, not a model-performance test.

The result sections read these committed inputs for Q3/Q4 tables, PR curves, probability diagnostics, alert burden, and candidate comparisons:

- [`q3_methods.csv`](../reports/data/q3_methods.csv): Q3 method results.
- [`q3_history_comparison.csv`](../reports/data/q3_history_comparison.csv): historical versus current features.
- [`q4_summary.json`](../reports/data/q4_summary.json): Q4 results for the fixed model.
- [`ml_diagnostics.json`](../reports/data/ml_diagnostics.json): ranking and probability diagnostics of saved scores.
- [`reports/figures/`](../reports/figures/): the corresponding saved figures.

The final section explains Q3's exploratory role, Q4's descriptive result due to the event-count gate, and the 2024 Q1 coverage stop. The notebook does not download raw data, retrain models, or generate new quarter-level conclusions. Compact inputs and complete LR parameters are public; full raw CSVs, SQLite evidence, and execution records remain in the local research workspace.

Continue with [Methods](../reports/METHODS_EN.md) and the [Results overview](../reports/RESULTS_OVERVIEW_EN.md).
