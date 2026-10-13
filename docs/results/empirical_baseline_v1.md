# Empirical baseline v1: model card and results

**Evidence class: EMPIRICAL.** Freddie Mac Single-Family Loan-Level Dataset, Release 47 (July 2026), Standard
Dataset; funded, Freddie Mac-acquired fixed-rate loans only. This is a noncommercial research result and is **not**
validated for lending, pricing or capital use. Contents are aggregate statistics only (every cell covers ≥ 11,000
loans). No loan-level data is included.

Evaluation followed the pre-registered plan in [../evaluation_plan_v1.md](../evaluation_plan_v1.md). Each model was
evaluated on the test split **once**; the audit log records 0 prior evaluations for both.

## 1. Question and target

What is the probability that a newly originated loan reaches **90+ days past due, REO acquisition, or a credit
termination within its first 12 monthly reporting periods**? Full definition:
[../target_specification.md](../target_specification.md). Two targets were evaluated, following the owner's decision to
report both (D-017):

- **Primary** `default_90dpd_12m_v1`: delinquency as reported.
- **Sensitivity** `default_90dpd_12m_exdisaster_v1`: 90+ months flagged as disaster or forbearance are not counted.

## 2. Data and splits (split by first payment date, with a 12-month purge gap)

| Split | Source | First payment | Loans | Events (primary) | Rate | Events (ex-disaster) | Rate |
|---|---|---|---|---|---|---|---|
| Train | full-volume 2015 vintage | 2015-01 – 2016-06 | 1,428,567 | 1,696 | 0.119% | 1,598 | 0.112% |
| Calibration | provider sample, 2017–19 vintages | 2017-07 – 2019-03 | 86,404 | 232 | 0.269% | 178 | 0.206% |
| Test | full-volume 2022 vintage | 2022-01 – 2023-06 | 1,573,718 | 10,814 | 0.687% | 5,744 | 0.366% |

Ingestion reconciled 171,532,275 full-volume lines and 20,730,922 sample performance lines to the source files,
with no rejected rows (one implausible 70% interest-rate value was set to null). Each split comes from a single
source. Calibration windows close before COVID forbearance.

How the population shifted (averages from the origination files):

| Cohort (FPD year) | Note rate | 95th pct rate | Credit score | DTI | LTV | Default rate (primary / ex-disaster) |
|---|---|---|---|---|---|---|
| 2015 (train) | 3.96% | 4.63% | 750 | 33.8 | 73.6 | 0.111% / 0.107% |
| 2016 (train) | 4.03% | 4.75% | 747 | 34.2 | 73.6 | 0.162% / 0.141% |
| 2022 (test) | 4.56% | 6.25% | 743 | 36.5 | 72.9 | 0.678% / 0.350% |
| 2023 (test) | **6.56%** | 7.63% | 746 | 38.0 | 76.1 | 0.780% / 0.530% |

## 3. Model

L2-regularised logistic regression (C = 1.0, not tuned), 17 origination-time inputs (`origination_v1`: credit score,
DTI, LTV, CLTV, note rate, term, MI %, units, log loan amount, multiple borrowers, CLTV−LTV, occupancy, channel,
property type, purpose, first-time-buyer flag, state). Imputation, scaling and category pooling were fitted on train
only, with no class weights. A Platt recalibration was fitted on the calibration split.
Fields known only after origination (servicer, MI cancellation) are excluded by an automated guard.

Coefficient signs (per standard deviation, primary model) are economically consistent: credit score −0.64,
multiple borrowers −0.47, note rate +0.28, DTI +0.26, MI coverage +0.18, term +0.16, LTV/CLTV about +0.06 each.
These are associations, not causal effects.

## 4. Results on the locked test set (2022–23 vintage, 1.57M loans)

| Metric | Primary model | Ex-disaster model |
|---|---|---|
| ROC-AUC (95% CI\*) | **0.739** (0.735–0.744) | **0.765** (0.759–0.771) |
| Gini / KS | 0.479 / 0.355 | 0.530 / 0.399 |
| PR-AUC (prevalence) | 0.019 (0.0069) | 0.014 (0.0037) |
| Brier (prevalence baseline) | 0.00680 (0.00686) | 0.00360 (0.00365) |
| Mean PD vs observed rate | 0.38% vs 0.69% | 0.29% vs 0.37% |
| Observed / expected (95% CI\*) | **1.79** (1.75–1.82) | **1.26** (1.23–1.29) |
| Calibration intercept / slope | +0.59 / 0.73 | +0.24 / 0.77 |
| ROC-AUC against the *other* target | 0.761 | 0.745 |

\*Loan-level bootstrap (200 resamples). This ignores common shocks within a cohort, so the true out-of-time
uncertainty is wider.

Performance decays over time (primary): train AUC 0.839 → calibration (2017–19) 0.772 → test (2022–23) 0.739.

By first-payment year:

| Model | Year | Loans | Events | ROC-AUC | Observed / expected |
|---|---|---|---|---|---|
| Primary | 2022 | 1,434,543 | 9,728 | 0.744 | 2.05 |
| Primary | 2023 | 139,175 | 1,086 | 0.766 | 0.84 |
| Ex-disaster | 2022 | 1,429,823 | 5,008 | 0.764 | 1.40 |
| Ex-disaster | 2023 | 138,825 | 736 | 0.783 | 0.76 |

Reliability by decile of predicted PD (primary; mean PD → observed rate, %):
0.023→0.098 · 0.051→0.130 · 0.080→0.225 · 0.116→0.303 · 0.162→0.421 · 0.223→0.550 · 0.311→0.708 · 0.448→0.887 · 0.705→1.345 · 1.72→2.21.
Observed default rates rise monotonically across deciles: the model ranks risk well but sets the level too low.

## 5. Findings

1. **Rank-ordering transfers across 6–7 years.** Test AUC is 0.74–0.77, with tight CIs far above 0.5, and observed
   default rates increase monotonically by decile.
2. **The absolute PD does not transfer.** On the primary target the model under-predicts the 2022–23 vintage by
   about 1.8×, which is outside the pre-registered PD-use band [0.8, 1.25]. As stated in advance, this is PD
   instability over time, not a coding error.
3. **Disaster/forbearance accounts for a large share of the gap.** Excluding flagged delinquency cuts the
   observed/expected ratio from 1.79 to 1.26 and raises AUC from 0.739 to 0.765. In the full 2022 vintage, 58% of
   primary defaults (5,667 / 9,728) carry disaster or forbearance flags.
4. **The note rate is confounded with the market rate level.** The 2023 cohort's average rate (6.56%) exceeds the
   highest rate in training (5.875%). The model reads the high rate as high borrower risk, so it over-predicts 2023
   (observed/expected 0.84) while under-predicting 2022 (2.05). The calibration slope of 0.73 reflects this
   extrapolation. The next feature version should use the **spread over a contemporaneous market rate**, which needs
   macro data (Phase A3).

## 6. Release decision against pre-registered criteria

| Criterion | Primary | Ex-disaster |
|---|---|---|
| Test AUC CI excludes 0.5 | ✅ | ✅ |
| Parity and replay (automated tests) | ✅ | ✅ |
| Calibration result documented | ✅ (this card) | ✅ |
| **Fit for PD-level use** (observed/expected in [0.8, 1.25]) | ❌ 1.79 | ❌ 1.26 (just outside) |

Both models are **eligible for demonstration scoring only**, for rank-ordering, and only after a named reviewer
records the promotion. Neither is fit for expected-loss, pricing or absolute-PD policy thresholds without
recalibration on recent mature cohorts.

## 7. Limitations

- Population: funded loans acquired by Freddie Mac (Standard Dataset). No rejected applicants and no reject inference.
- The origination state is reconstructed from disclosed values, with no real-time availability evidence. The first
  payment date stands in for the decision date.
- Training covers a single benign period (2015–16) with low default rates. Calibration covers 2017-07 to 2019-03,
  including hurricanes Harvey and Irma.
- Loan-level bootstrap CIs understate cohort-level uncertainty. Only one test cohort period exists.
- No tuning, challenger, segment fairness or proxy analysis yet (Phase C). `property_state` still needs a proxy
  review.
- Reason codes are model attributions, not adverse-action reasons.
- Code version: artefacts were trained from an **uncommitted** working tree. Commit the repository and retrain
  before treating these artefacts as reproducible releases.

## 8. Reproduce

Requires your own Freddie Mac Clarity download of Release 47 (sample 2015–2022, full volume 2015 and 2022);
file checksums are in [../../configs/sources/source_manifest.yaml](../../configs/sources/source_manifest.yaml).

```
uv run crisk unpack          --dataset freddie_mac_historical
uv run crisk ingest          --dataset freddie_mac_historical
uv run crisk ingest          --dataset freddie_mac_sample
uv run crisk build-canonical --dataset freddie_mac_historical   # and freddie_mac_sample
uv run crisk ingest          --dataset freddie_mac_combined     # composes validated files
uv run crisk build-canonical --dataset freddie_mac_combined
uv run crisk build-labels    --dataset freddie_mac_combined
uv run crisk build-features  --dataset freddie_mac_combined
uv run crisk train           --dataset freddie_mac_combined
uv run crisk train           --dataset freddie_mac_combined --target-id default_90dpd_12m_exdisaster_v1
uv run crisk evaluate-test   --dataset freddie_mac_combined --model-id <id>
```

Laptop timings (16 GB RAM, Windows 11): unpack 2.5 min, full-volume ingest 64 min, full-volume canonical 6 min,
labels 13 min, combined build about 22 min, training about 40 s per model.
