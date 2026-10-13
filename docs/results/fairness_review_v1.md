# Fairness and geography review v1 (closes finding F-002)

**Evidence class: EMPIRICAL.** Aggregate results only. Pre-registered in
[../fairness_review_plan_v1.md](../fairness_review_plan_v1.md), which was committed before any analysis ran.

**Scope limitation:** the data has no protected characteristics, so this is **not** a disparate-impact test. No
loan was linked to individual-level external data. A clean proxy screen does not establish fairness.

## 1. Does `property_state` add predictive value? No. It reduces out-of-time ranking.

Out-of-time calibration split (FPD 2017-07 – 2019-03, 86,404 loans), AUC before Platt; paired bootstrap of 1,000:

| Target | With state | Without state | Δ (with − without) | 95% CI |
|---|---|---|---|---|
| Primary | 0.772 | 0.784 | **−0.012** | [−0.016, −0.007] |
| Ex-disaster | 0.801 | 0.807 | **−0.005** | [−0.010, −0.001] |

State effects learned on 2015–16 originations (range −0.33 to +0.37 logit: CA lowest, LA highest) do not carry over
to later cohorts. The feature mostly fits period-specific geographic noise.

## 2. Proxy screen: weak

Spearman correlation across 51 jurisdictions (≥ 500 training loans each) between the model's mean state effect and
public state demographics (Census Vintage 2024 estimates for 2022; file `SC-EST2024-ALLDATA6`):

| Demographic share | ρ (primary) | ρ (ex-disaster) |
|---|---|---|
| Not non-Hispanic White alone | −0.09 | −0.14 |
| Black alone | +0.14 | +0.10 |
| Hispanic | −0.04 | −0.14 |

All |ρ| < 0.15 (threshold 0.5), with no statistically meaningful association. A state-level screen cannot detect
within-state proxying.

## 3. Segment errors (selected; full tables in the local JSON report)

Calibration split (primary target), AUC with → without state:

| Segment | Loans | Events | AUC with | AUC without | O/E with | O/E without |
|---|---|---|---|---|---|---|
| South | 30,686 | 121 | 0.741 | 0.763 | 1.40 | 1.41 |
| West | 24,112 | 42 | 0.730 | 0.738 | 0.76 | 0.70 |
| Florida | 5,966 | 64 | 0.709 | 0.711 | **4.21** | 3.52 |
| Credit score < 660 | 4,098 | 59 | 0.636 | 0.644 | 1.47 | 1.41 |
| First-time buyer | 21,670 | 81 | 0.737 | 0.755 | 1.09 | 1.05 |

Holdout 2023–25 (previously viewed; diagnostic): the no-state model is equal or better in AUC in almost every
segment. Calibration by **credit-score band is distorted**: O/E is 1.09 for scores below 660 and 0.26 for 780+. The
credit-score gradient learned in 2015–16 is steeper than in 2023–25 (new finding F-007). Florida's large O/E in the
calibration period reflects hurricane Irma (2017).

## 4. Decision (pre-registered rule)

`property_state` adds no clear value (ΔAUC < 0 for both targets), so it is **removed**, even though the proxy screen
is weak.

| Candidate | Holdout AUC (CI) | O/E | Validation |
|---|---|---|---|
| M1-nostate primary `…8fd365` | **0.815** (0.798–0.835) vs champion 0.810 | 0.63 | 13/14, approved for `demonstration_rank_ordering` |
| M1-nostate ex-disaster `…582606` | **0.826** (0.807–0.842) vs 0.821 | 0.60 | 13/14, approved for `demonstration_rank_ordering` |

Both are clean-commit reproductions (50e98df) of the ablation models, each evaluated once on the holdout. Promotion
requires a named reviewer.

## Process note (recorded)

The ablation models were first trained from an uncommitted working tree, and the first holdout attempt crashed on
a missing feature file *before* any labels were read (0 holdout evaluations recorded). Both models were retrained
from a clean commit (coefficients identical, reproduction audited) and then evaluated once.
