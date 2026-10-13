# Pre-registered evaluation plan v3: recalibration on recent mature cohorts

_Written 2026-10-13, before any recalibration code was run on real data._

## Problem

Every model so far ranks risk well out of time (AUC 0.74–0.82), but the **absolute PD level drifts**. The
observed/expected ratio ranged from 0.62 to 1.79 on later cohorts. Ranking is a property of the scorecard. The level
depends on the period. v3 tests whether **refreshing only the level, using outcomes that were genuinely known at
scoring time**, keeps PDs usable.

## Fixed inputs

- **Ranking models (unchanged, not refitted):** M1 primary `pd-logit-origination_v1-…b6249e` and M1 ex-disaster
  `pd-logit-origination_v1-sens-…57741f`. Logistic, `origination_v1`, trained on full-volume FPD 2015-01..2016-06,
  commit 71a1aea. M1 was chosen over M4: it has equal holdout discrimination, is simpler and has exact reason codes
  (holdout report).
- **Data:** Freddie Mac **provider sample** vintages 2015–2024 (Release 47), all from one source. Labels are the
  primary and ex-disaster targets.

## Recalibration procedures (the only candidates)

Let *T* be a cohort's first-payment month. The pre-trained model gives `p0` (its current, Platt-calibrated PD).

| ID | Procedure |
|---|---|
| R0 | No recalibration: `p = p0` (status quo) |
| R1 | **Intercept update**: `p = σ(a_T + logit p0)`, with `a_T` fitted by maximum likelihood (offset logistic) on the recalibration window |
| R2 | **Platt refit**: `p = σ(a_T + b_T · logit p0)`, both fitted on the recalibration window |

**Recalibration window for cohort T:** cohorts with FPD in **[T − 31, T − 20] months** (12 monthly cohorts).
Why a 20-month lag: the decision date is about 2 months before T; Freddie Mac releases are quarterly, about 4 months
after the performance cutoff, with up to 3 more months until the next release; and a cohort needs 12 months of
performance to mature. 2 + 4 + 3 + 12 ≈ 20, rounded up conservatively. Only information that would really have been
available at the decision is used.

**Guard:** if the window has fewer than 20 events, carry forward the last fitted parameters, or R0 if none exist yet.

## Evaluation (walk-forward)

Evaluation cohorts: FPD **2019-04 .. 2025-03**, which starts after M1's calibration split (2017-07..2019-03).
Reported for each target and procedure:

- overall and per FPD year: observed/expected, calibration intercept/slope, log loss, Brier, ROC-AUC;
- **share of FPD years with observed/expected in [0.8, 1.25]** (the PD-use band from v1/v2);
- results by period, labelled honestly:
  - P1 2019-04..2021-12: COVID/forbearance cohorts, **never used in any evaluation before**;
  - P2 2022-01..2023-06: overlaps the previously viewed v1 test period (sample loans, not the full-volume test file);
  - P3 2023-07..2025-03: the previously viewed holdout.

## Decision rule (fixed now)

Adopt R1 if, over the full evaluation period and for **both** targets, it (i) lowers log loss vs R0 and (ii) increases
the share of FPD years inside [0.8, 1.25]. Adopt R2 instead of R1 only if it lowers log loss vs R1 by ≥ 1% (relative)
for both targets without reducing the in-band share. Otherwise keep the simpler option.

## Expected in advance

- The lagged recalibration **cannot anticipate shocks**. On the primary target the COVID forbearance spike (2019-20
  cohorts, ~1.8% default) will be badly under-predicted at first, and the window will then over-correct as the spike
  enters the window. The ex-disaster target should behave much better.
- ROC-AUC within a cohort is unchanged by monotone recalibration. Pooled AUC across cohorts can change slightly.

## Confirmatory prospective test (pre-registered, data not yet released)

The adopted procedure will be tested **once** on cohorts that first become mature in Freddie Mac **Release 48 or later**
(FPD after 2025-04, the last cohort mature at the Release 47 cutoff of 2026-03). It will use the sample vintage
containing them (`sample_2025.zip`), with no changes to the model, window, lag or guard. This is the first evaluation
in the project on outcomes that did not exist when the procedure was designed.
