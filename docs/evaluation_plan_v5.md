# Pre-registered evaluation plan v5: model refresh on recent vintages

_Written 2026-10-14 and approved by the owner (Dzidefo123) before any v5 model was trained or v5 features were built._

## Why refresh

The champion `…8fd365` was trained on 2015–16 originations. It ranks well out of time (holdout AUC 0.815), but three
open findings trace back to that training period:

- **F-001:** its absolute PD level is unstable across vintages.
- **F-005:** about 90% of 2023–25 loans have note rates above its training maximum (5.875%).
- **F-007:** its credit-score gradient is too steep for 2023–25 cohorts.

v3 and v4 tried to fix only the level, and both were rejected. v5 refits the whole scorecard on a recent,
high-rate period.

## Evidence status, stated up front

Every sample cohort with a first payment up to 2025-03, and the full-volume 2022 vintage, has already been examined
(v1 test, v2 holdout, v3/v4 backtests, monitoring). **Training, calibration and model selection below therefore use
seen data, and that is permitted. Only the confirmatory test on unseen data can support a promotion.**

## Data (Release 47 unless stated)

| Role | Source | First-payment window | Approx. loans / defaults (primary) |
|---|---|---|---|
| Train | full-volume 2022 vintage | 2022-01 .. 2022-12 | ~1.3M / ~9,000 |
| Calibration (Platt) | sample 2024 vintage | 2024-01 .. 2025-03 | ~50k / ~320 |
| Development (selection only) | sample 2023 vintage | 2023-01 .. 2023-12 | ~50k / ~300 |
| **Confirmatory test** | **sample 2025 vintage, Release 48+ edition** (never ingested so far) | mature cohorts only (FPD + 11 ≤ cutoff) | expected ~20k / ~100 at Release 48 |

- The gap between the train and calibration windows is 12 months (the loader rule).
- The confirmatory set is the **same reserved data** as the v4 L0 test. Both tests run in the same single session,
  and neither result can change the other's procedure.
- Primary target `default_90dpd_12m_v1`. The ex-disaster target is reported as a sensitivity.

## Candidates (fixed now; no others)

| ID | Model | Features |
|---|---|---|
| C0 | Current champion `…8fd365`, unchanged (control) | origination_v1_nostate (2015–16 fit) |
| C1 | Logistic + Platt, refit on the windows above | origination_v1_nostate |
| C2 | Logistic + Platt, refit on the windows above | origination_v2_nostate: the note rate replaced by its spread over PMMS (D-019); no state |

There is no boosted-tree candidate. LightGBM added ≤ 0.017 AUC in v2 at the cost of exact reason codes, and fewer
candidates means less selection noise. Hyperparameters follow `logistic_v1` (C = 1.0). There is no tuning.

**Selection (development set):**
- Default to **C2**, because it removes the note-rate extrapolation structurally.
- Choose C1 instead only if C1's AUC exceeds C2's by ≥ 0.005, with the paired-bootstrap 95% CI lower bound above 0.
- The selected model is the **v5 candidate**. C0 and the other refit are reported only.

## Confirmatory test (once; audited)

The v5 candidate passes if **all** of the following hold:

1. **Discrimination:** ROC-AUC 95% CI lower bound ≥ 0.70 (as validation check V06).
2. **No ranking loss vs C0:** paired ΔAUC (candidate − C0) ≥ −0.01.
3. **Level:** observed/expected in [0.8, 1.25] (point estimate), reported with its bootstrap CI.

Reported, but not part of the pass rule:
- O/E by credit-score band (F-007) and by note-rate band (F-005).
- Calibration slope and Brier score.
- The ex-disaster target.

**Consequences:**
- **Pass:** the candidate goes through `crisk validate`. If V07 (level) also passes there, it may be proposed for
  `pd_level` approval, which would be a first. Promotion still needs the named reviewer.
- **Fail on level only:** it may be proposed for `demonstration_rank_ordering`, like the current champion.
- **Fail on ranking:** C0 stays.

**Power warning:** with about 100 events, the AUC CI is about ±0.03 and the O/E CI about ±0.2. A pass at Release 48 is
weak evidence. The same test is pre-committed to be **repeated once at Release 49** on the cumulative mature 2025
cohorts, and that repeat is labelled as a second look.

## Not in scope

Lifetime PD, macro-conditional PIT level, reject inference, geography features.
