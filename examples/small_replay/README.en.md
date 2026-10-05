# Small synthetic replay

[中文](README.md) · [English](README.en.md) · [Project overview](../../README_EN.md)

This example uses **12 fictional drives** to demonstrate the full path from daily records to scores, alert lists, and later outcome evaluation. The data is hand-built, not a Backblaze subset. Logistic regression uses exported fixed parameters; nothing is trained here.

## Inputs and rules

| File | Purpose |
|---|---|
| [`daily.csv`](daily.csv) | Hand-built records spanning 35 days, Jan 1–Feb 4, 2023 |
| [`current_lr.json`](current_lr.json) | 16 LR coefficients, intercept, imputation values, and scaling parameters |
| [`expected.json`](expected.json) | Expected checks for labels, features, scores, and opportunity events |
| [`manifest.json`](manifest.json) | Input inventory and hashes |

Scoring runs from Jan 15–28, the main event window is Jan 22–29, and the outcome cutoff is Feb 4. Eligibility requires at least 12 observations in the preceding 14 calendar days, including the scoring day, with no failure on or before it. Each method can select up to `ceil(N/1000)` eligible drives per day, followed by a seven-day cooldown. A drive becomes available again on the eighth day after its alert.

LR ranks by fixed log-odds scores. The SMART rule considers only drives with at least one non-zero reading among its five attributes, ranked by the count of non-zero readings. Each method maintains its own cooldown, while both calculate capacity from the full eligible population.

## Run

Use a separate Python 3.13 environment; see the [environment guide](../../environment/README.en.md). The replay uses POSIX signal and resource interfaces on macOS/Linux. Its shared package imports NumPy and scikit-learn; the example uses fixed parameters and does not fit a model.

From the repository root:

```bash
python -m pip install -r environment/requirements-replay.txt
mkdir -p .tmp/small_replay
python -B tools/run_small_replay.py \
  --output .tmp/small_replay/run_001 \
  --timeout-seconds 120
```

The program creates `run_001`, so that directory must not exist; create its parent first. Choose `run_002` or another new name for another run. Output must remain inside the project. Existing directories are never overwritten.

The default run limit is 120 seconds, with at most 1,000 input rows, a 10 MiB output budget, and a 256 MiB recorded-RSS limit. RSS checking is a runtime observation, not operating-system memory isolation.

## Expected results and outputs

| Method | Alerts | Captured opportunity events |
|---|---:|---:|
| `current_lr` | 14 | 1 / 2 |
| `smart_nonzero` | 7 | 2 / 2 |

These values test the mechanism on hand-built inputs. They cannot be compared with real Q3/Q4 performance and do not justify adopting the rule instead of LR.

| Output | Contents |
|---|---|
| `selection.sqlite` | Features, scores, capacity, cooldown, and selections using information available by each decision day; no future-label table |
| `evaluation.json` | Outcomes, opportunity events, hits, and unknown outcomes computed after selection is complete |
| `summary.md` | Synthetic result table |
| `run_manifest.json` | Completion status, input/output hashes, access boundaries, and resource records |

A successful run prints the output directory. Check that `run_manifest.json` has `status: complete`, then read `summary.md`. Partial files or an `error.json` do not count as a successful run.

The cases include main-window failures A/B, an outside-window failure C, a missing day for D, early exit E, entirely missing current SMART readings for F, initially insufficient history for G, and prior failure H. They distinguish ineligibility, unknown outcomes, and confirmed non-hits rather than coding them all as negatives.

## Implementation and scope

[`tools/run_small_replay.py`](../../tools/run_small_replay.py) calls the project's feature and alert-selection implementation. Scoring queries enforce dates `≤t`. Future outcomes are linked only after the selection database is written and closed. Incomplete follow-up stays unknown, and future failures do not filter the current candidate pool.

On a supported platform, run the targeted checks with:

```bash
python -m unittest tests.test_small_replay
```

This example reproduces a fixed small input, not the full quarterly training and real-data validation. The notebooks also contain a shorter feature-and-ranking walkthrough that does not require the POSIX replay entry point.
