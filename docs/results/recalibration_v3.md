# v3 results: walk-forward recalibration on recent mature cohorts (a negative result)

**Evidence class: EMPIRICAL** (Freddie Mac SFLLD Release 47, provider sample vintages 2015–2024). Aggregate results
only. Run under the pre-registered plan [../evaluation_plan_v3.md](../evaluation_plan_v3.md), which was committed and
pushed before any code ran.

## Setup (as pre-registered)

- Ranking models are fixed: M1 logistic for the primary target (`…b6249e`) and for the ex-disaster target (`…57741f`).
- For each first-payment cohort T from 2019-04 to 2025-03 (72 monthly cohorts, about 295,000 loans), the PD level is
  refreshed using only cohorts with FPD in [T−31, T−20]. That is what would really have been known at the decision
  date, given the 12-month horizon and Freddie Mac's release lag.
- R0 = no recalibration · R1 = intercept update · R2 = Platt refit. The guard (fewer than 20 events) never triggered;
  windows held 59–1,579 events.

## Deviation (recorded)

The first run had a **numerical bug**. The intercept-only MLE used plain Newton steps, which diverge when the window's
default rate is about 15× the model's PDs (the COVID windows). The primary-target R1 intercept reached −1.6 × 10⁸.

The fitter now uses step-halving, so the log-likelihood never decreases, and a regression test covers this case.
The fix leaves every earlier result unchanged: refitting Platt parameters and calibration metrics for M1 and M4 gave
differences of exactly 0.0. Only the primary-target R1 numbers changed between runs, and **the decision did not
change**. Both runs are kept (`backtest__freddie_mac_sample__run1_newton_divergence_bug.json` is local and not
published).

## Results (corrected run)

### Overall, FPD 2019-04 – 2025-03

| Target | Method | Observed/expected | Log loss | Brier | Share of FPD years with O/E in [0.8, 1.25] |
|---|---|---|---|---|---|
| Primary (2,918 events) | **R0** | 2.15 | **0.0612** | **0.00980** | 0/7 |
| | R1 | 0.47 | 0.0702 | 0.01102 | 0/7 |
| | R2 | 0.59 | 0.0665 | 0.01027 | 0/7 |
| Ex-disaster (934 events) | **R0** | 0.92 | **0.01986** | 0.003153 | 0/7 |
| | R1 | 0.51 | 0.02107 | 0.003221 | 1/7 |
| | R2 | 0.90 | 0.01994 | 0.003153 | **3/7** |

### Observed/expected by first-payment year

| Year | Primary R0 | Primary R1 | Primary R2 | Ex-disaster R0 | Ex-disaster R1 | Ex-disaster R2 |
|---|---|---|---|---|---|---|
| 2019 (Apr–Dec) | 10.47 | 5.61 | 5.61 | 1.33 | 0.96 | 0.96 |
| 2020 | 16.67 | 16.97 | 17.50 | 2.60 | 2.58 | 2.51 |
| 2021 | 4.50 | 0.64 | 0.40 | 2.10 | 1.82 | 1.45 |
| 2022 | 1.91 | 0.14 | 0.16 | 1.31 | 0.52 | 0.96 |
| 2023 | 0.67 | 0.15 | 0.22 | 0.70 | 0.32 | 0.83 |
| 2024 | 0.66 | 0.28 | 0.37 | 0.70 | 0.47 | 0.70 |
| 2025 (Jan–Mar) | 0.55 | 0.50 | 0.51 | 0.61 | 0.66 | 0.66 |

Within-year ranking with R0, which recalibration does not change: ROC-AUC 0.65–0.82 on the primary target (weakest
for 2019 cohorts) and 0.67–0.82 on ex-disaster.

## Decision (pre-registered rule, applied as written)

R1 is adopted only if it lowers log loss **and** raises the in-band share versus R0, for **both** targets. It does
neither for the primary target, and it raises log loss for ex-disaster. **Decision: R0. No recalibration procedure is
adopted.**

## What we learned

1. **Lagged outcomes mistime regime shifts.** With an honest 20-month information lag, the recalibration window
   describes conditions about two years before the cohort. The COVID forbearance spike (2019–20 cohorts, primary
   default rate 1.8–2.2%) is missed while it happens (O/E 10–17). Once it enters the window, the procedure
   over-corrects by 3–7× for 2022–23 cohorts (O/E 0.14–0.22). Static calibration (R0) is less wrong on average.
2. **Even without disaster/forbearance delinquency, recalibration only helps the year-level fit.** R2 brings 3 of 7
   years into the band (R0: 0 of 7), but log loss is marginally worse (+0.4%), so the pre-registered rule rejects it.
3. **The ex-disaster target still shows a COVID-era excess** (O/E 2.1–2.6 for 2020–21 cohorts). Not all
   pandemic-related delinquency carries the disaster or forbearance flags.
4. **Implication for the next design:** the PD *level* has to come from information available *at origination*, not
   from lagged 12-month outcomes. Candidates for a v4 plan, not run here:
   - (a) a shorter-lag early-performance signal, e.g. 60+ DPD within 6 months, observable about 6 months sooner;
   - (b) a macro level term such as unemployment or house-price growth, joined point-in-time;
   - (c) explicit through-the-cycle vs point-in-time calibration targets.

## Confirmatory prospective test (unchanged from the plan)

The adopted procedure (R0, the status quo) will be checked once on cohorts that first mature in Freddie Mac
Release 48 or later (FPD after 2025-04), using `sample_2025.zip`.
