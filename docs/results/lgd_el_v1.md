# LGD, EAD and expected loss v1 (Phase J)

**Evidence class: EMPIRICAL.** Aggregate results only. Pre-registered in [../lgd_el_plan_v1.md](../lgd_el_plan_v1.md)
(committed before any severity model was fitted). The expected-loss figures are **illustrative 12-month EL at
origination, not CECL/IFRS 9**.

## Loss data (verified)

The Release 47 loss components reconcile exactly: **actual loss = removal UPB + delinquent accrued interest +
expenses + net sale proceeds + MI recoveries + non-MI recoveries**, with recoveries negative. The difference is under
1e-11 on all 3,405 training liquidations. The older guide formula (which subtracts recoveries) is off by about $347k
per loan on Release 47 data.

## What happens after a default (full-volume 2015 vintage, mature)

Of 1,694 loans that reached 90+ days past due within 12 months:

| Outcome | Loans | Share |
|---|---|---|
| Paid off | 718 | 42% |
| **Liquidated with a loss record** | 498 | **29.4%** |
| Reperforming-loan sale (no loss disclosed) | 245 | 14% |
| Current | 160 | 9% |
| Still delinquent | 41 | 2% |
| Defect repurchase | 32 | 2% |

So **P(liquidation | 90+ default) = 0.294**. EAD at the first 90+ month is 0.95 of the original UPB on average
(median 0.99), because delinquent fixed-rate loans stop amortising. The median workout from first 90+ to liquidation
is 20 months.

## Loss severity (loss given liquidation)

Development (2015 vintage; train = liquidated ≤ 2019, n = 1,642; validation = 2020+, n = 1,763):

| Model | Validation MAE | Spearman (95% CI) |
|---|---|---|
| S0 mean | 0.263 | n/a |
| S1 segments (MI × LTV band) | 0.263, no gain → rejected | 0.20 (0.14–0.24) |
| **S2 fractional logit** (LTV, CLTV, MI %, loan size, credit score, DTI, occupancy, property type, purpose, channel; no geography) | **0.233 (−11%) → selected** | 0.37 (0.33–0.41) |

**Locked test, evaluated once** (sample vintages 2016–2021: different loans and vintages, n = 343):

| | S2 | S0 baseline |
|---|---|---|
| MAE | **0.176** | 0.224 |
| Spearman (95% CI) | **0.36** (0.27–0.44) | n/a |
| Mean predicted vs observed | 0.196 vs 0.173 (conservative) | 0.265 vs 0.173 |
| UPB-weighted predicted vs observed | 0.147 vs 0.128 | 0.265 vs 0.128 |

Severity varies with disposition type (third-party sale 0.08, REO 0.31 observed in the test). That type is not
knowable at origination, so the model predicts a blend. Diagnostic on 2022 full-volume liquidations (early, fast
resolutions): Spearman 0.33, mean predicted 0.161 vs observed 0.184.

**House-price link** (2,222 liquidations with ELTV at the first 90+): severity is about 0.23 at ELTV 60%, 0.26 at
80%, 0.29 at 100% and 0.32 at 120%. Severity rises with negative equity, but modestly. MI and the recovery process
dampen it.

## Expected loss (FPD 2024-04 – 2025-03 sample applications, exposure $15.1B)

PD level: champion deciles mapped to their observed 12-month default rates on the same reference applications
(overlay): 0.04% / 0.02% / 0.09% / 0.19% / 0.26% / 0.32% / 0.37% / 0.73% / 1.18% / **2.58%** for deciles 1–10. The
overlay is estimated on the same loans as the book, so the base-case PD level is in-sample here by construction.

| Scenario | PD | P(liquidation) | House prices | 12-month EL | EL rate |
|---|---|---|---|---|---|
| Base | overlay | 0.294 | 0% | **$2.06M** | **1.4 bps** |
| Adverse | ×2 | 0.394 | −10% | $5.87M | 3.9 bps |
| Severe | ×4 | 0.494 | −20% | $15.8M | 10.5 bps |

The base EL rate by decile ranges from about 0.1 bps (deciles 1–2) to **7.7 bps** (decile 10). Most expected loss
sits in the top two deciles.

## Limitations

- Single-period illustration: no lifetime PD, discounting, staging or macro path, so not CECL/IFRS 9.
- Loss severity is modelled only on liquidated loans. Reperforming-loan sales disclose no loss, and the 2022 vintage's
  unresolved defaults are censored.
- Constant P(liquidation) and EAD ratio. The scenario multipliers are judgement-based, not fitted.
- The book's mean severity (0.11, unweighted across all loans) is below the defaulted-loan average because most of the
  book is low-LTV or MI-covered. The PD-weighted severity is what drives EL.
