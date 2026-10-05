<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

<h1 align="center">Hard Drive Failure Warning and Alert Analysis</h1>

<p align="center">Backblaze Drive Stats · ST4000DM000 · Seven-day warning · Daily inspection budget</p>

<p align="center">
  <a href="https://yemyu.github.io/drive-failure-warning/"><b>View dashboard</b></a>
  · <a href="./notebooks/01_disk_failure_warning_en.ipynb">English notebook</a>
  · <a href="./notebooks/01_disk_failure_warning_zh.ipynb">中文 Notebook</a>
  · <a href="./reports/R_VALIDATION_RESULTS.md">Q4 validation results</a>
</p>

## About

Drive records change every day, while inspection capacity is limited. This project uses Backblaze's daily records to prioritize `ST4000DM000` drives for inspection. It ranks devices using information available that day, selects up to about 0.1% of eligible drives, and later checks whether they receive their first observed `failure=1` marker in the following seven days.

The inputs are device ID, date, model, capacity, and SMART attributes. The pipeline builds a device-day panel, checks whether enough recent history is available, and produces scores and daily alert lists. Future failure records enter only after the lists are complete. The main comparison is between `current_lr`, a logistic regression trained on 2023 Q1/Q2, and a SMART non-zero rule that requires no training.

The outputs describe daily selections, captured failure events, warning lead time, repeated alerts, and unknown outcomes. The dashboard displays these results. The notebooks explain the calculations and demonstrate the path from records to alerts with a small hand-built dataset. The public `failure` field is an operational marker; it does not identify a mechanical cause or document a maintenance intervention.

## Data, task, and methods

### Three units of measurement

| Unit | Definition | Question it answers |
|---|---|---|
| Device-day | One eligible scoring observation for one device on one date | Does the score rank records with a failure marker in the next seven days higher? |
| Failure event | A device's first observed failure marker; an “opportunity event” has at least one eligible warning day | How many failures receive a warning before they occur? |
| Alert | A device's selection for inspection on a particular day | How many inspections are triggered, and how many alerts hit, repeat, or have unknown outcomes? |

**Event recall = captured opportunity events / all opportunity events.** An event is captured if it receives at least one alert 1–7 days before failure, and each event counts once. The opportunity denominator depends on scoring eligibility and event dates, independently of a method's ranking, cooldown, or actual alerts. An event can generate several positive-label device-days and several alerts, so event recall and alert hit rate measure different things.

### Time split and available information

| Period | Role | Scoring dates / outcome cutoff |
|---|---|---|
| 2023 Q1/Q2 | Training and preprocessing estimates | Jan 15–Jun 23 / Jun 30 |
| 2023 Q3 | Development comparisons, historical features, and tree baseline | Jul 1–Sep 23 / Sep 30 |
| 2023 Q4 | Quarterly validation with the model and alert policy fixed | Oct 1–Dec 24 / Dec 31 |
| 2024 Q1 | Coverage check for a later quarter | Stopped at the prespecified coverage gate on Jan 14; no performance result |

A scoring day requires at least 12 observations within the preceding 14 calendar days, including that day, and no failure marker on or before it. Known device identities, recent records, and prior failures carry across quarters. Ranking uses only records dated `≤t`; outcomes refer to `t+1…t+7`. A confirmed future failure is positive. Complete follow-up without failure is negative. When the outcome cannot be established, it remains `unknown`, and the device is not removed from scoring based on that future outcome.

Q3 informed input checks, protocol revisions, and candidate development, so its comparison is exploratory. Model coefficients, preprocessing, capacity, and cooldown were fixed before Q4 evaluation. The daily selection cutoff still changes with the eligible population and its scores.

### Logistic regression with 16 current SMART features

`current_lr` uses six SMART attributes measured on the scoring day. It does not feed 7/14-day trends into the model; historical records establish eligibility and prior failure status. The attributes expand into 16 columns:

| SMART attribute | Transformations | Columns |
|---|---|---:|
| 5, 187, 197, 198 | `log1p(raw)`, missing indicator, non-zero indicator | 3 each |
| 9 | `log1p(raw)`, missing indicator | 2 |
| 188 | Non-zero indicator, missing indicator | 2 |

`log1p` reduces the scale of large counts, while missing indicators retain the information that a reading was absent. Training uses known Q1/Q2 outcomes: all positive-label drive-days and a deterministic daily sample of approximately 5% of negative-label drive-days. Inverse sampling probabilities supply fitting weights, normalized to have mean one. Imputation values, means, and standard deviations are estimated from the weighted training sample and then fixed along with the L2 logistic-regression coefficients.

The model produces a log-odds score for ranking. Applying the sigmoid gives a probability, but these probabilities have not been calibrated. The complete feature list, coefficients, intercept, and preprocessing parameters are published in [`current_lr.json`](examples/small_replay/current_lr.json).

### SMART rule and daily selection

`smart_nonzero` counts the non-zero readings among SMART 5, 187, 188, 197, and 198. Drives with at least one non-zero reading become rule candidates, ranked by that count. Ties use a fixed deterministic order. No parameters are fitted.

Both methods share a daily capacity of `ceil(N/1000)`, where `N` is the total number of eligible devices that day. Each method keeps its own seven-day cooldown: after an alert on day `t`, a drive cannot be selected on `t+1…t+7` and becomes available again on `t+8`. Selection skips drives in cooldown and can leave slots unused if too few candidates remain. With 12,500 eligible drives, for example, the maximum is 13 inspections. Capacity is a limit on the daily list, not a fixed probability threshold.

See [Methods](reports/METHODS_EN.md) for the full definitions and calculations.

## Results

| Metric | Q3 `current_lr` | Q3 SMART | Q4 `current_lr` | Q4 SMART |
|---|---:|---:|---:|---:|
| Captured opportunity events | 65 / 126 | 47 / 126 | 40 / 86 | 35 / 86 |
| Event recall | 51.59% | 37.30% | 46.51% | 40.70% |
| Alerts | 1,509 | 1,509 | 1,247 | 1,247 |
| AP on device-days with known outcomes | 0.1191 | 0.0233 | 0.0825 | 0.0161 |

In Q4, logistic regression captures five more events than the rule, a recall difference of 5.81 percentage points. There are 86 opportunity events, below the prespecified 100-event confirmation gate. The difference is therefore descriptive, with no confirmatory interval published. The main event window is Oct 8–Dec 25; it is distinct from the scoring window.

The Q4 logistic-regression alerts concern 205 devices, with 1,042 alerts following a device's first alert. The rule alerts 389 devices, with 858 later alerts. Known-outcome alert hit rates are 3.56% and 3.03%, respectively, while 38 and 27 alerts have unknown outcomes. Logistic regression captures more events and concentrates repeated alerts on fewer drives.

Probability diagnostics address a separate question from ranking. Q4 Brier score is `0.000544`, compared with `0.000514` for a constant reference using the quarter's observed prevalence. On drive-days with known outcomes, the sum of predicted probabilities is about `1.81` times the observed positive-label count. These results do not support treating the outputs as calibrated failure probabilities.

Historical-feature logistic regression and gradient boosting each capture 72/126 events in Q3. The historical model's recall difference interval relative to `current_lr` is −0.79 to 11.38 percentage points, crossing zero and failing the adoption criterion. A later repeat-alert reordering policy reaches 43/86 in Q4, a gain of three events against a required gain of five. `current_lr` is retained, and the candidate search stops. The 2024 Q1 run stops on Jan 14 after consecutive days of low coverage, so no model performance is reported for that quarter.

Start with the [Results overview](reports/RESULTS_OVERVIEW_EN.md). The detailed [Q3 study](reports/Q3_RESEARCH_RESULT_BRIEF.md), [Q4 validation](reports/R_VALIDATION_RESULTS.md), [diagnostics](reports/ML_EVALUATION_DIAGNOSTICS.md), and [conclusions](reports/CURRENT_RESEARCH_CONCLUSIONS.md) are in Chinese; the English overview covers their main findings.

## Quick start

### 1. Browse online: no installation

Open [Drive Signal](https://yemyu.github.io/drive-failure-warning/). Start with the project overview, then visit quarterly results, inspection budgets, device cases, model diagnostics, and data coverage. The page supports Chinese and English and uses the results published with it.

### 2. Run the notebooks locally

Use a separate Conda environment. The commands below clone the repository and install dependencies without changing system Python or Conda `base`:

```bash
git clone https://github.com/Yemyu/drive-failure-warning.git
cd drive-failure-warning
conda create -n drive-warning python=3.13 pip
conda activate drive-warning
python -m pip install -r environment/requirements.txt
python -m jupyterlab notebooks/01_disk_failure_warning_en.ipynb
```

Select the current environment's Python kernel and run all cells in order. Expect definitions, Q3/Q4 comparison tables, ranking and probability plots, alert-burden analysis, a short synthetic walkthrough, and the Q1 coverage stopping result. The Chinese version is `01_disk_failure_warning_zh.ipynb` in the same directory.

For a project-local `.venv` alternative, see the [environment guide](environment/README.en.md). The notebooks consume committed CSV/JSON summaries and figures; they do not rebuild the complete quarterly study.

### 3. Run the synthetic replay: macOS / Linux

From the repository root, in the environment above:

```bash
python -m pip install -r environment/requirements-replay.txt
mkdir -p .tmp/small_replay
python -B tools/run_small_replay.py \
  --output .tmp/small_replay/run_001 \
  --timeout-seconds 120
```

The output directory must not already exist; choose a different name for another run. Open its `summary.md`. The frozen LR should produce **14 alerts and capture 1/2 events**; SMART should produce **7 alerts and capture 2/2 events**. The inputs describe 12 fictional drives. These numbers check the mechanism and are not evidence of real model performance. The output also includes the selection database, evaluation JSON, and an input/output manifest; see the [example guide](examples/small_replay/README.en.md).

The replay uses POSIX signals and resource interfaces. Windows users can read the notebooks and dashboard; native Windows replay is outside the current support scope.

### Rebuild and serve the dashboard

The public build reads the committed `dashboard/signal_v1/data.json`. It does not need raw Backblaze files or the local research databases:

```bash
python tools/build_signal_dashboard.py
python -m http.server 8000
```

Open <http://localhost:8000/dashboard/signal_v1/>. The `index.html` file also works directly if `dashboard/vendor/` and the relative directory structure are preserved. On macOS/Linux, check the public build and synthetic replay with:

```bash
python -m unittest tests.test_public_dashboard tests.test_small_replay
```

## Repository layout

```text
configs/                 Study configurations
pipeline/                Panels, features, scoring, alerts, and validation
tools/                   Run and dashboard-build entry points
examples/small_replay/   12 fictional drives and frozen LR parameters
notebooks/               Chinese and English explanations and analysis
dashboard/signal_v1/     Current dashboard and display data
dashboard/vendor/        Tabler stylesheet, source, and license
reports/                 Research reports
reports/data/            Compact notebook inputs
reports/figures/         Saved result figures
tests/                   Unit and protocol checks
environment/             Reading environment and dependencies
```

The [dashboard guide](docs/DEMO_GUIDE_EN.md) explains the sections and metrics. Multi-GB raw files, full SQLite outputs, and execution records remain in the local research workspace. The repository includes complete portable parameters for the retained LR and a synthetic example, but result summaries alone cannot reproduce the full quarterly training and validation.

## Scope and license

The study covers one drive model. A device can appear in both training and evaluation quarters, so the results do not establish performance on entirely new drives or other models. Unknown outcomes may be related to devices leaving observation. There are no maintenance interventions or measured labor costs; dashboard workload estimates multiply alert counts by an assumed inspection duration.

Code, documentation, and examples use the [MIT License](LICENSE). Backblaze data follows the terms of its [source page](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data); the repository license does not relicense that data. The dashboard uses Tabler; see [third-party notices](THIRD_PARTY_NOTICES.md).
