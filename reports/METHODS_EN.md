# Methods and evaluation definitions

[中文](METHODS.md) · [Results overview](RESULTS_OVERVIEW_EN.md)

## Records, inspection lists and outcomes

The input is Backblaze daily drive records: identifier, model, date, `failure`, and six raw SMART fields. The study is limited to ST4000DM000. A decision on day t uses information available through t. Drives with a failure marker on or before t are excluded. Eligibility requires at least 12 observations within the preceding 14 calendar days, including t.

Each method ranks drives and saves its inspection list before joining outcomes from t+1 through t+7. The label is positive if the first public failure marker appears in that interval, negative if follow-up is sufficient and no marker appears, and unknown otherwise. Disappearance from the records is not itself a failure or a negative outcome.

The analysis uses three units:

- A **drive-day** is one scoring record for one drive on one decision day. Drives contribute repeated rows.
- A **failure event** is a drive's first public marker. An **opportunity event** falls within the main event dates and has at least one eligible scoring day in the preceding seven days. This denominator does not depend on a model selecting the drive.
- An **alert** is one entry on a method's daily list. Several alerts may match one failure.

Event recall is captured opportunity events divided by all opportunity events. Known-outcome alert precision is hit alerts divided by all alerts minus unknown alerts. Unknown-outcome bounds assign all unknown alerts to misses or hits; they are not confidence intervals. Lead time is the calendar-day gap from the earliest matching alert to a captured event. Quantiles describe each method's captured events only.

## Training and fixed scoring

The current model is L2-regularised logistic regression. All Q1/Q2 positive-label drive-days were retained and roughly 5% of negative-label drive-days were sampled separately on each date. Inverse-sampling weights were normalised to mean one. Imputation and standardisation used weighted training statistics. All parameters remained fixed in Q4.

Six current raw fields produce 16 transformed inputs:

| SMART field | Transformations | Inputs |
|---|---|---:|
| 5 | log1p, missing indicator, non-zero indicator | 3 |
| 9 | log1p, missing indicator | 2 |
| 187 | log1p, missing indicator, non-zero indicator | 3 |
| 188 | non-zero indicator, missing indicator | 2 |
| 197 | log1p, missing indicator, non-zero indicator | 3 |
| 198 | log1p, missing indicator, non-zero indicator | 3 |

Missing values are not zeros. Imputation and scaling use the fixed training parameters. The final current-value model does not use rolling trends as scoring inputs; history determines eligibility and excludes prior failures. History-feature LR and the tree were separate Q3 comparisons.

The [parameter JSON](../examples/small_replay/current_lr.json) includes all 16 positions, coefficients, the intercept, imputation means and scaling values. The [independent implementation](../pipeline/r_validation/independent_audit.py) and [synthetic replay](../examples/small_replay/README.en.md) show feature transformation and scoring.

The SMART comparator selects only eligible drives with a non-zero value in at least one of fields 5/187/188/197/198. It ranks by the count of non-zero fields and uses a fixed hash tie-break. Zero-signal drives do not fill unused slots. Capacity is based on the full eligible pool for both methods, but insufficient candidates or cooldown can leave slots unused.

## Daily inspection policy

Capacity is ceil(N/1000), where N is the day's eligible drive count. A selected drive is excluded for the following seven days and may be selected again eight days after its previous alert. Cooldown is maintained separately per method. Q4 starts with empty cooldown state but inherits observation history and prior failures.

Parameters, the capacity formula and cooldown are fixed. The realised score cutoff changes with the day's scores, eligible count and cooldown state; it is not a quarter-wide probability threshold. Different budget paths are replayed separately, so their captured-event sets need not be nested.

## Time split and leakage controls

| Period | Role | Scoring dates | Main event dates | Outcome cutoff |
|---|---|---|---|---|
| 2023 Q1/Q2 | Training and preprocessing | Jan 15–Jun 23 | Jan 22–Jun 24 | Jun 30 |
| 2023 Q3 | Development comparison | Jul 1–Sep 23 | Jul 8–Sep 24 | Sep 30 |
| 2023 Q4 | Fixed-model validation | Oct 1–Dec 24 | Oct 8–Dec 25 | Dec 31 |
| 2024 Q1 | Replication attempt | No performance output | No performance output | Stopped at the coverage gate |

The final main event day is one day after the final scoring day because that decision can warn one day ahead. Event dates and scoring dates are different.

Features, missingness, eligibility and scores are computed as of each decision. Future failure, complete lifespan and outcome observability cannot filter inspection lists. Outcome evaluation starts after selection is saved and its database is closed. The original current_lr/SMART comparison does not use Q4 to refit imputation, scaling, coefficients or the selection policy. Later repeat-alert reordering candidates were developed on already-observed Q3/Q4 data and are exploratory, not independently tested.

Devices can occur in multiple quarters. This is temporal holdout evaluation, not evaluation on exclusively unseen drives. Q3 included input-informed protocol amendments and development choices. Q4 used fixed parameters and rules on historical data; it was not a prospective field trial.

## Statistical interpretation

Q3 comparisons use device-level paired bootstrap to account for repeated observations within drives. This does not cover shared calendar shocks. The Q4 gain conditions included at least 100 opportunity events, enough valid resamples, a difference of at least five percentage points, a positive interval lower bound, no more than a two-point increase in unknown-alert proportion, and passed quality/execution checks. Q4 had only 86 events, so no confirmatory interval was published. This is the study's reporting convention, not a general sample-size formula.

AP uses all scoring rows with known outcomes, grouping equal scores. Displayed PR curve points are not integrated to replace AP. Brier score and log loss assess probability error. The post-hoc prevalence constant uses the quarter's labels and is a diagnostic comparator, not a deployable model. No calibrator was fitted.

The Q1 replication monitored device coverage against a rolling reference and stopped after three consecutive days below 80%. January 14 coverage was 79.34%; no model-performance conclusion followed. Operational failure semantics, the single drive model, repeated devices, unknown exits and lack of intervention records limit every result.
