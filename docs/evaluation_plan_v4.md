# Pre-registered evaluation plan v4: through-the-cycle (TTC) PD level

_Written 2026-10-13, before any v4 code was run._

## Motivation (from v3)

Refitting the PD level on the most recent mature cohorts, with an honest 20-month lag, **mistimed shocks**. It missed
the COVID forbearance spike, then over-corrected by 3–7× ([results/recalibration_v3.md](results/recalibration_v3.md)).
Nothing known at origination could have anticipated COVID for a 2019 loan. The defensible origination PD is a
**through-the-cycle** estimate: the ranking from the fixed model, with a level anchored to a *long-run* default rate
rather than the latest year. Point-in-time (PIT) adjustment for a specific economic path belongs in a separate
scenario overlay (Phase J). It is out of scope here.

## Evidence status, stated up front

Every cohort from 2019-04 to 2025-03 has already been examined (v1 test, v2 holdout, v3 backtest). The v4
walk-forward on those cohorts is therefore **exploratory**: it can reject an idea but cannot independently confirm one.
**The decision is confirmed only by the prospective test below.**

## Fixed inputs (unchanged from v3)

Ranking models M1 primary `…b6249e` and M1 ex-disaster `…57741f`. Provider sample vintages 2015–2024 (Release 47),
single source. Information lag of 20 months, as in v3. `p0` is the model's current PD.

## Candidate level procedures

All use the v3 intercept update `p = σ(a_T + logit p0)`, with `a_T` fitted by maximum likelihood (damped Newton,
D-025). They differ only in the window of mature cohorts used:

| ID | Window for cohort T (first-payment months) | Idea |
|---|---|---|
| L0 | none: `p = p0` | status quo (v3 decision) |
| L1 | **expanding**: all available cohorts with FPD ≤ T − 20 (earliest 2015-01) | long-run average; dilutes shocks instead of chasing them |
| L2 | **rolling 60 months**: FPD in [T − 79, T − 20] | about one cycle; adapts slowly |

Guard as in v3: fewer than 20 events in the window means carry forward. **Cohort weighting:** every loan counts
equally (no reweighting toward particular years).

## Metrics (walk-forward FPD 2019-04 – 2025-03; same periods P1–P3 as v3)

- **Primary metric, level stability:** the root-mean-square of ln(observed/expected) across first-payment years,
  `RMSLE_OE`. Lower means a steadier level.
- Log loss and Brier score overall; in-band share (years with O/E in [0.8, 1.25]); O/E by year and by period.

## Decision rule (fixed now)

For each candidate, compute `RMSLE_OE` for both targets. Adopt the candidate with the lowest **mean** `RMSLE_OE` across
the two targets, **if** that mean is at least 10% (relative) below L0's **and** overall log loss is not more than 1%
(relative) above L0's for either target. Otherwise keep L0. Between L1 and L2, prefer L1 (simpler) unless L2's mean
`RMSLE_OE` is at least 10% lower than L1's.

## Expectations stated in advance

- TTC anchors will not make any single shock year accurate. A 2020 O/E far from 1 is expected under every candidate
  for the primary target. The test is whether the *level stays steadier* than either a frozen 2017–19 calibration (L0)
  or a chasing window (v3).
- L1's level in 2022–25 will be pulled up by the COVID cohorts entering its history. Whether that helps or hurts is the
  empirical question.

## Confirmatory prospective test (the decisive evidence)

Run once, with no changes to models, window rules, lag or guard. The adopted candidate and L0 will be evaluated on
cohorts with **first payment after 2025-04** that become mature in **Freddie Mac Release 48 or later**, using the
**Release 48 (or later) edition of `sample_2025.zip`**. The Release 47 edition on disk (performance through 2026-03)
covers only cohorts up to FPD 2025-04 and **must not** be used for this test. It stays untouched until then.

The reported metrics are O/E with a bootstrap CI, log loss and ROC-AUC. The level is judged against the PD-use band
[0.8, 1.25].
