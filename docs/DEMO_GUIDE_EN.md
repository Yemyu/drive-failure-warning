# Dashboard guide

[中文](DEMO_GUIDE.md) · [English](DEMO_GUIDE_EN.md) · [Project overview](../README_EN.md)

Open [Drive Signal](https://yemyu.github.io/drive-failure-warning/) or the local [`index.html`](../dashboard/signal_v1/index.html). The page starts with the project overview. Use the sidebar or previous/next links to move between sections. This is a static display of saved research results, and the device cases are historical reviews.

## Project overview

This section introduces the inputs, two scoring methods, and alert rules. Distinguish device-days, failure events, and alerts: scores belong to device-days; event recall counts each failure once; inspection volume counts alerts. The seven-day prediction horizon covers days after scoring. The seven-day cooldown limits how soon an alerted device can be selected again.

## Quarterly results

Switch between Q3 and Q4 to compare LR and SMART at the same capacity. Q3 captures 65/126 and 47/126 events and is an exploratory development comparison. Q4 captures 40/86 and 35/86, with 1,247 alerts for each method. Its 86 opportunity events fall below the prespecified 100-event gate, so the difference remains descriptive.

Event recall measures how many failures with an eligible warning opportunity were caught. It is not alert accuracy. Alert outcomes are hits, non-hits with complete follow-up, or unknown. Unknown outcomes are not counted as negatives.

## Inspection budget

This section uses three Q3 capacities: 0.05%, 0.1%, and 0.2%. LR captures 55, 65, and 69 events, with 765, 1,509, and 3,018 total alerts. Increasing capacity increases inspections, while additional event capture grows more slowly.

Workload estimates multiply total alerts by an assumed duration per inspection. Changing that duration explores a scenario; it does not measure actual maintenance time or economic benefit. Q4 capacity comparisons that are unavailable are not filled with zeros.

## Device cases

The cases are a post-hoc sample of Q4 failure devices, not a representative quarterly distribution. Search an anonymous ID or filter by capture outcome, then inspect the seven days before failure.

The score curve uses saved model scores. The daily cutoff comes from that day's ranking and capacity, not a fixed probability threshold. A high-scoring drive can remain unselected because of cooldown. Missing scores mean no eligible scoring row was available and should not be replaced with zero. SMART contributions explain how current inputs affect the score, not what caused a failure.

## Model diagnostics

PR curves and AP use device-days with known outcomes to assess ranking across score cutoffs. They use a different unit from event recall. Probability plots and Brier score assess probability error. In Q4, summed predicted probabilities are about 1.81 times the observed number of positive-label days; the model remains uncalibrated.

Historical features and the tree model achieve higher Q3 point estimates but fail the adoption criteria. Later alert reordering also misses the required gain of five Q4 events. These comparisons remain available, while `current_lr` is retained.

## Data coverage

The 2024 Q1 run stops on Jan 14 after consecutive days below the coverage threshold. This section displays the coverage checks and stopping basis, not quarterly model performance. The cause of the population decrease is unresolved; the decrease alone does not establish either failure or source-data corruption.

## Reports and sources

[Methods](../reports/METHODS_EN.md) explains definitions and calculations, and the [Results overview](../reports/RESULTS_OVERVIEW_EN.md) summarizes the quarterly findings. Further reports cover Q4 validation, probability diagnostics, and repeated alerts. Display data is downloadable, and the data source and Tabler license are listed here.

For local execution, use the [environment guide](../environment/README.en.md) and [notebook guide](../notebooks/README.en.md). Full raw data and research databases are not distributed with the static page.
