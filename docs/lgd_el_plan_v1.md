# Pre-registered plan: LGD, EAD and expected loss v1 (Phase J)

_Written 2026-10-13, before any loss-severity model was fitted. Descriptive counts only were inspected: volumes by
vintage, the 2015-vintage resolution mix, ELTV coverage and the median workout length. No severity values were
inspected by split._

## Verified facts (Release 47 loss fields)

- For dispositions with zero balance codes 02, 03, 09 and 15, **actual loss = removal UPB + delinquent accrued
  interest + total expenses + net sale proceeds + MI recoveries + non-MI recoveries**, with recoveries negative
  (Release 47 sign convention). This reconciles to within 1e-11 on 3,887 records. Losses are not disclosed for codes
  01, 16 or 96.
- Of the 1,696 full-volume 2015-vintage loans that reached the 12-month 90+ target, 509 (30%) were later liquidated
  with a loss record. The rest paid off (718), are current (160), were sold as reperforming loans (245, no loss
  disclosed), were repurchased over defects (32), or are still delinquent (32).
- The median workout from first 90+ to liquidation is 20 months (2015 vintage).

## Structure

`EL(12m, at origination) = PD × P(liquidation | default) × Severity × EAD`. It is **illustrative, single-period, and
not CECL/IFRS 9** (no lifetime horizon, discounting of the PD path, or staging).

| Component | Definition | Estimation |
|---|---|---|
| Severity | actual loss / removal UPB, for liquidated loans (02/03/09/15) with a loss record | model (below) |
| P(liquidation \| default) | share of 12-month 90+ defaults later liquidated with a loss record | mature 2015 vintage; constant; sensitivity 0.20–0.45 |
| EAD | UPB at first 90+ month (EAD ratio = that / original UPB) | mean ratio on 2015-vintage defaults |
| PD | **level overlay**: observed 12-month default rate per current risk decile on the decile-reference applications (FPD 2024-04..2025-03) | the champion is approved for ranking only, so the absolute level comes from a governed overlay (v3/v4 conclusion) |

## Severity models (origination-time features only, so EL can be computed at application)

- **S0**: training mean.
- **S1**: segment means by MI coverage (0 / >0) × LTV band (≤80, 80–90, >90).
- **S2**: fractional-response logistic regression (quasi-binomial) on severity clipped to [0, 1], with inputs orig_ltv,
  orig_cltv, mi_pct, log_orig_upb, credit_score, orig_dti, occupancy, property_type, loan_purpose and channel.
  **No geography** (consistent with D-031).

**Development split (2015 full-volume vintage, 3,405 liquidations):** dev-train = liquidations up to 2019;
dev-validation = liquidations in 2020 and later. **Selection rule:** move to a more complex model (S0 → S1 → S2)
only if dev-validation MAE improves by **≥ 3% (relative)** over the currently selected simpler model. The selected
model is then refitted on all 2015-vintage liquidations.

**Locked test (evaluated once):** liquidations from **sample vintages 2016–2021** (other vintages and loans; about
343). Diagnostic only: full-volume 2022-vintage liquidations, which are early and fast resolutions and so biased.
Metrics: MAE, RMSE, mean predicted vs observed severity (portfolio bias), UPB-weighted mean bias, and Spearman rank
correlation with a bootstrap CI.

## Downturn sensitivity (descriptive)

On 2015-vintage liquidations with ELTV available at the first 90+ month, fit `severity ~ ELTV_at_default`
(fractional logit). This gives a house-price stress: a price decline of x% raises ELTV by a factor 1/(1−x).

## Scenarios (assumptions stated now)

| Scenario | PD overlay multiplier | P(liquidation \| default) | Severity | House-price shock |
|---|---|---|---|---|
| Base | 1.0 | estimate | model | 0% |
| Adverse | 2.0 | estimate + 0.10 | via ELTV link | −10% |
| Severe | 4.0 | estimate + 0.20 | via ELTV link | −20% |

The multipliers are judgement-based assumptions, not fitted macro models. For context, the primary-target 12-month
default rate rose about 9× in the 2019–20 COVID cohorts, but most of those loans cured.
