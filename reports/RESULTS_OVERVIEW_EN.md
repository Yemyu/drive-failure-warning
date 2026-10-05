# Results overview

[中文](RESULTS_OVERVIEW.md) · [Methods and definitions](METHODS_EN.md)

The project ranks drives using daily SMART records, selects an inspection list under a fixed daily capacity, and checks whether the selected drives receive their first public failure marker within the next seven days. It studies Backblaze ST4000DM000 drives. The model was trained on 2023 Q1/Q2, compared with alternatives during Q3 development, and evaluated with fixed parameters and rules in Q4.

## Quarter comparison

Daily capacity is the eligible drive count × 0.1%, rounded up. Each method applies its own seven-day cooldown. Alerts count list entries; events count distinct first failure markers with an eligible warning opportunity.

| Quarter | Method | Captured/opportunity events | Event recall | Alerts | Known-hit alerts | Unknown alerts |
|---|---|---:|---:|---:|---:|---:|
| Q3 | Current SMART LR | 65/126 | 51.59% | 1,509 | 66 | 11 |
| Q3 | SMART non-zero rule | 47/126 | 37.30% | 1,509 | 48 | 8 |
| Q4 | Current SMART LR | 40/86 | 46.51% | 1,247 | 43 | 38 |
| Q4 | SMART non-zero rule | 35/86 | 40.70% | 1,247 | 37 | 27 |

One failure can match several alerts. Event recall is therefore different from alert precision. For Q4 LR, known-outcome alert precision is 43/(1,247−38) = 3.56%, whereas event recall is 40/86 = 46.51%.

Both the history-feature LR and fixed gradient-boosted tree captured 72/126 Q3 events. The history LR's difference from current LR was 5.56 percentage points, with a device-level paired 95% bootstrap interval of −0.79 to 11.38 points. It did not meet the replacement condition that the lower bound exceed zero. Q4 LR exceeded SMART by 5.81 points, but its 86 opportunity events were below this study's prespecified 100-event reporting gate. No confirmatory interval or supported superiority claim is reported. The gate is a study convention, not a universal statistical rule.

## Inspection capacity and repeated alerts

The Q3 scoring period spans 85 decision days, from July 1 to September 23. Counts below are quarter totals.

| Method | 0.05%: alerts/events | 0.1%: alerts/events | 0.2%: alerts/events |
|---|---:|---:|---:|
| Current LR | 765 / 55 | 1,509 / 65 | 3,018 / 69 |
| History LR | 765 / 65 | 1,509 / 72 | 3,018 / 73 |
| History tree | 765 / 63 | 1,509 / 72 | 3,018 / 75 |

All paths use 126 opportunity events. Larger budgets change cooldown histories, so captured-event sets need not be nested. Raising current LR's capacity from 0.1% to 0.2% added 9 captured events but lost 5, for a net increase of 4 at 1,509 additional alerts. There is no measured intervention benefit or actual inspection cost.

Of Q3 LR's alerts, 226 were the first for a drive that quarter and 1,283 were later reminders. Later reminders first captured 37 of the 65 captured events. They consume capacity but cannot all be treated as wasted inspections.

## Ranking, probability error and coverage

Q4 AP is 0.082503 for current LR and 0.016145 for SMART. AP uses all scoring rows with known outcomes; budgeted event recall also depends on selection and cooldown. LR's Q4 sum of predicted positive probabilities was about 1.81 times the observed positive-label-day count. Its Brier score was worse than a post-hoc prevalence constant. The scores are not established as calibrated failure probabilities.

The 2024 Q1 replication stopped on January 14 after three consecutive low-coverage days. Coverage was 9,034/11,386 = 79.34%, below the fixed 80% gate. No Q1 model-performance result was produced. The reason for the observation-count drop remains unknown.

## Scope and source files

The [dashboard](https://yemyu.github.io/drive-failure-warning/) and [notebooks](../notebooks/README.en.md) read committed summaries. A complete current LR parameter file and a hand-built synthetic replay are included. Full raw quarters and local research databases are not included; rebuilding the page or running the synthetic example does not reproduce the full study.

Public `failure` is an operational marker, not a verified mechanical cause. Drives may appear in both training and later quarters. Incomplete follow-up remains unknown. Q3 involved development and protocol amendments, while Q4 supports a descriptive comparison only. The results do not establish benefits from maintenance, transfer to unseen drives, or transfer to other drive models.

Sources: [Q3 methods](data/q3_methods.csv), [history-model comparison](data/q3_history_comparison.csv), [Q4 summary](data/q4_summary.json), [diagnostic summary](data/ml_diagnostics.json), and [dashboard data](../dashboard/signal_v1/data.json). See the [file notes](data/README.md). Detailed quarter reports are in Chinese: [Q3](Q3_RESEARCH_RESULT_BRIEF.md), [Q4](R_VALIDATION_RESULTS.md), and [model diagnostics](ML_EVALUATION_DIAGNOSTICS.md).
