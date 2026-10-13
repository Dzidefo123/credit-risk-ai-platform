# Pre-registered evaluation plan: empirical baseline v1

_Written 2026-10-12, **before** any model was trained on `freddie_mac_combined` or the test split was scored._
Changing this plan after the test results are seen must be recorded as a deviation.

## Models

1. **Primary**: L2 logistic regression on `origination_v1`, target `default_90dpd_12m_v1`, Platt-calibrated on the
   calibration split (`configs/models/logistic_v1.yaml`, C = 1.0, no tuning in v1).
2. **Sensitivity**: the same specification, trained and calibrated on `default_90dpd_12m_exdisaster_v1`.
3. **Reference**: a prevalence baseline (train-period default rate for every loan).

The boosted-tree challenger is Phase C and is **not** part of this comparison.

## Data

| Split | Source | First payment | Loans | Events (primary / ex-disaster) |
|---|---|---|---|---|
| train | full volume 2015 vintage | 2015-01..2016-06 | 1,428,567 | 1,696 / 1,598 |
| calibration | provider sample 2017-19 vintages | 2017-07..2019-03 | 86,404 | 232 / 178 |
| test | full volume 2022 vintage | 2022-01..2023-06 | 1,573,718 | 10,814 / 5,744 |

## What will be reported (test split, evaluated once per model; access is audited)

- ROC-AUC, Gini, KS and PR-AUC (with prevalence), Brier score, log loss.
- Calibration-in-the-large: mean PD vs observed rate, calibration intercept and slope, and a decile
  reliability table.
- 95% CIs for ROC-AUC, Brier and the observed/expected ratio from 200 loan-level bootstrap resamples. **Limitation:**
  loan-level resampling ignores common shocks within a cohort, so the CIs are too narrow for out-of-time inference.
- Metrics by first-payment year (2022, 2023).
- Each model scored against **both** target definitions.

## Expectations stated in advance

- **Discrimination** should transfer reasonably, because origination risk ranking (credit score, LTV, DTI) is usually
  stable across vintages. No minimum AUC is imposed.
- **Calibration is expected to fail** on 2022 for the primary target. The test default rate (0.69%) is about 6× the
  train rate (0.12%) and about 2.6× the calibration rate (0.27%), because rates rose, the credit mix changed and
  disaster forbearance increased. A large positive calibration intercept (under-prediction) is the expected
  result. That would be a finding about **PD instability over time**, not a coding error. It would motivate
  recalibration on recent mature cohorts (or a macro/vintage adjustment) before any use.
- The ex-disaster target is expected to show a smaller calibration gap (test 0.37% vs train 0.11%).

## Release criteria (prototype; not an institutional approval)

A model can be **promoted for demonstration scoring** only if all of the following hold:
1. Test ROC-AUC exceeds 0.5 with its bootstrap CI excluding 0.5.
2. Offline/online parity holds and replay matches (automated tests).
3. The model card states the calibration result, whatever it is.

A model is **not fit for PD-level use** (pricing, expected loss, policy thresholds that rely on absolute PD) if the
test observed/expected ratio falls outside [0.8, 1.25]. A rank-ordering use would still be possible.
