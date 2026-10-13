# Model comparison v2: rate-spread features and LightGBM challenger

**Evidence class: EMPIRICAL** (Freddie Mac SFLLD, Release 47). Aggregate results only. Development-data
comparison under the pre-registered plan [../evaluation_plan_v2.md](../evaluation_plan_v2.md). **The locked test has
not been run for these models.** A fresh test (sample 2023–24 vintages) is waiting on the data download.

## Candidates

| ID | Features | Algorithm | Notes |
|---|---|---|---|
| M1 | origination_v1 (raw note rate) | logistic | baseline; tested on 2022–23 in v1 |
| M2 | origination_v2 (note rate − PMMS market rate) | logistic | |
| M3 | origination_v2 | LightGBM | monotone: credit score −, DTI/LTV/CLTV/rate spread + |
| M4 | origination_v1 | LightGBM | same constraints, with the note rate in place of the spread |

LightGBM settings were fixed in advance (15 leaves, min 500 rows per leaf, learning rate 0.03). The number of
boosting rounds came from early stopping on the last 6 months of the training window (2016-01..06, 213k loans):
188 (M3) and 191 (M4) for the primary target. All models were trained on the full-volume 2015 vintage
(1,428,567 loans, 1,696 primary events).

## Out-of-time development results (calibration split: FPD 2017-07 – 2019-03, 86,404 loans; before Platt)

| Model | Train AUC | **Cal AUC** | PR-AUC | Obs/Exp | Cal. slope | Cal. intercept |
|---|---|---|---|---|---|---|
| **Primary target** (232 events) | | | | | | |
| M1 | 0.839 | 0.772 | 0.0148 | 1.59 | 0.94 | +0.47 |
| M2 | 0.837 | 0.782 | 0.0149 | 2.06 | 0.97 | +0.73 |
| M3 | 0.874 | **0.790** | 0.0146 | 2.10 | 0.89 | +0.75 |
| M4 | 0.874 | 0.782 | 0.0137 | 1.74 | 0.89 | +0.56 |
| **Ex-disaster target** (178 events) | | | | | | |
| M1 | 0.840 | 0.802 | 0.0154 | 1.33 | 1.06 | +0.28 |
| M2 | 0.841 | 0.804 | 0.0160 | 1.76 | 1.07 | +0.57 |
| M3 | 0.881 | 0.809 | 0.0137 | 1.71 | 0.93 | +0.54 |
| M4 | 0.874 | 0.810 | 0.0136 | 1.40 | 0.99 | +0.34 |

Paired bootstrap (1,000 resamples) of the AUC difference against M1 on the same loans:

| Target | M2 − M1 | M3 − M1 | M4 − M1 |
|---|---|---|---|
| Primary | +0.010 [+0.003, +0.016] | +0.017 [+0.006, +0.029] | +0.010 [−0.002, +0.021] |
| Ex-disaster | +0.003 [−0.004, +0.009] | +0.007 [−0.006, +0.022] | +0.008 [−0.006, +0.023] |

## Selection (pre-registered rule, applied as written)

The rule: move to a more complex model only for a calibration-split AUC gain of **≥ 0.01** over the currently selected
simpler model, with the calibration slope worsening by no more than 0.1. Complexity order: M1 < M2 < M4 < M3.

- Primary: M2 vs M1, +0.0098 (fails by 0.0002). M4 vs M1, +0.0101, slope −0.05 (passes, so **M4 is selected**).
  M3 vs M4, +0.0073 (fails).
- Ex-disaster: no candidate gains ≥ 0.01 over M1, so **M1 is retained**.

**Flaw in the rule, recorded rather than overridden.** The rule uses a point threshold and ignores uncertainty. The
selected M4's gain is *not* statistically distinguishable from zero (CI includes 0), while M2 and M3 have gains whose
CIs exclude zero. The decision stands, because changing it after seeing results would defeat pre-registration. The
next plan (v3) will pre-register a rule that uses the paired CI, e.g. select only if the lower bound is > 0.

## What the market-rate spread did and did not do

- **Ranking:** the spread modestly improves out-of-time discrimination for the logistic model (+0.010 AUC, CI
  excludes 0) and gives the best challenger (M3).
- **Level:** it does **not** fix calibration drift. Uncalibrated observed/expected *worsens* (1.59 → 2.06), most likely because
  the raw note rate had been partly tracking the rising default level between 2015–16 and 2017–19. The level drift
  is a genuine change in cohort risk (credit mix, hurricanes, economy), not merely a rate artefact. It must be
  handled by **recalibration on recent mature cohorts** or explicit macro/vintage terms, not by this feature alone.
- **Lag noise:** the spread uses the market rate published 3 months before first payment (no look-ahead). When
  rates move fast (2022), that reference lags the actual lock rate. The spread's standard deviation rises from 0.32
  (train) to 0.55 (2022–23), which adds noise exactly when rates move most.

## Next

1. Fresh locked test on sample 2023–24 vintages (FPD 2023-07 – 2025-03) for the selected models (M4 primary,
   M1 ex-disaster), with M1 primary as reference. **Needs `sample_2023.zip` and `sample_2024.zip`.**
2. Pre-register v3 selection using paired CIs; consider a recalibration-on-recent-cohorts design (e.g. Platt
   refitted on the most recent mature 12 months available at each scoring date).
3. Segment, proxy (`property_state`) and fairness review before any model is promoted.
