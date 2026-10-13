# v4 results: through-the-cycle PD level (exploratory; negative)

**Evidence class: EMPIRICAL, exploratory.** Every cohort used here was examined in earlier rounds, so this run can
reject the TTC anchors but could not have independently confirmed them. Plan: [../evaluation_plan_v4.md](../evaluation_plan_v4.md),
committed and pushed before any v4 code was written. Aggregate results only.

## Setup

Fixed ranking models (M1 primary `…b6249e`, M1 ex-disaster `…57741f`); provider sample 2015–2024 (Release 47);
cohorts FPD 2019-04 – 2025-03; 20-month information lag. L0 = status quo. L1 = intercept update on an **expanding**
window (all mature cohorts since 2015-01). L2 = intercept update on a **rolling 60-month** window. The 2025 vintage
is reserved for the confirmatory test: excluded at ingestion, and the backtest refuses any cohort after 2025-04.

## Results

| Target | Method | RMS ln(O/E) across years | Overall O/E | Log loss | Years in [0.8, 1.25] |
|---|---|---|---|---|---|
| Primary | **L0** | **1.550** | 2.15 | **0.06122** | 0/7 |
| | L1 expanding | 1.759 | 0.66 | 0.06652 | 0/7 |
| | L2 rolling 60m | 1.872 | 0.53 | 0.06978 | 0/7 |
| Ex-disaster | **L0** | **0.551** | 0.92 | **0.01986** | 0/7 |
| | L1 expanding | 0.612 | 0.79 | 0.02007 | 1/7 |
| | L2 rolling 60m | 0.652 | 0.72 | 0.02024 | 1/7 |

O/E by first-payment year:

| Year | Primary L0 | Primary L1 | Primary L2 | Ex-disaster L0 | Ex-disaster L1 | Ex-disaster L2 |
|---|---|---|---|---|---|---|
| 2019 | 10.47 | 9.01 | 9.01 | 1.33 | 1.34 | 1.34 |
| 2020 | 16.67 | 15.63 | 15.63 | 2.60 | 2.66 | 2.66 |
| 2021 | 4.50 | 2.00 | 1.99 | 2.10 | 2.09 | 2.09 |
| 2022 | 1.91 | 0.51 | 0.45 | 1.31 | 1.14 | 1.08 |
| 2023 | 0.67 | 0.18 | 0.14 | 0.70 | 0.58 | 0.51 |
| 2024 | 0.66 | 0.19 | 0.15 | 0.70 | 0.57 | 0.51 |
| 2025 (Q1) | 0.55 | 0.19 | 0.15 | 0.61 | 0.53 | 0.50 |

Fitted intercepts: primary L1 rises from +0.11 (2019-04) to +1.34 (2022-06) as the COVID cohorts enter the history;
ex-disaster L1 stays within −0.05 to +0.24.

## Decision (pre-registered rule)

Neither TTC anchor reduces the mean RMS ln(O/E) versus L0 (1.185 and 1.262 vs 1.051), and both raise log loss beyond
the 1% tolerance. **Decision: L0 (status quo).**

## What v3 and v4 together show

1. **In 2015–2025 the 12-month default level is shock-driven.** It shows hurricanes in 2017, COVID forbearance in
   2019–21, and a post-2022 regime with higher rates and more disaster forbearance. A recent window (v3) chases shocks.
   A long-run window (v4) absorbs them and carries them into calmer years. The frozen calibration from a benign
   period (2017-07..2019-03) happens to sit closest to post-2022 conditions.
2. **The scorecard's job is ranking.** Within-year AUC is 0.65–0.82. The absolute level is a separate,
   period-dependent quantity, and lagged outcome data alone cannot set it reliably for an origination PD.
3. **Practical recommendation.** Present model output as relative risk (score or rank bands). Govern the absolute
   level as an explicit, documented overlay, using expert judgement or scenario analysis, with its own review and
   monitoring. Do not auto-recalibrate.
4. **Hypotheses for a future pre-registered round** (not tested; they arose after seeing these results):
   - (a) a TTC anchor that excludes cohorts whose 12-month windows overlap declared crisis periods;
   - (b) a macro-conditional level fitted on a long panel;
   - (c) an early-performance proxy with a shorter lag.

   Each needs a confirmatory test on data not yet seen.

## Confirmatory prospective test (unchanged)

L0 (adopted) will be evaluated once on cohorts with first payment after 2025-04 that mature in Freddie Mac
Release 48+, using the Release 48+ edition of `sample_2025.zip`.
