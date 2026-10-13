# Portfolio monitoring v1: historical replay 2023-07 – 2025-03

**Evidence class: EMPIRICAL** (historical replay of the 2023–25 holdout applications, scored by the champion M1
`…b6249e`). Aggregate results only. Thresholds (`configs/monitoring/thresholds_v1.yaml`) were fixed before the first
run.

## What is monitored

Monthly volume; **score PSI** against the champion's rank reference (its decile cut-offs, so 10% per decile is
expected); feature PSI against the 2017–19 reference population; missing-rate shifts; matured outcomes by quarter
(O/E and ROC-AUC, only after the 12-month window closes). Alerts are de-duplicated (one per issue, spanning
first–last period), and each carries a severity, owner and runbook ([../runbooks.md](../runbooks.md)).

## Results

| Alert | Severity | Periods | Interpretation |
|---|---|---|---|
| `score_drift` | **high** | 21/21 months (PSI 0.6–2.4) | 2023–25 scores sit far above the 2017–19 rank reference. 47% of loans land in the old "decile 10" |
| `feature_drift:orig_interest_rate` | medium | 21 | Market rates of 6.5–7.5% vs a 2015–16 training range up to 5.875% (F-005) |
| `feature_drift:orig_dti` | medium | 21 | DTI distribution shifted upward |
| `feature_drift:log_orig_upb`, `loan_purpose` | medium | 15 / 8 | Larger loans; fewer refinances in a high-rate market |
| `low_volume` + single-month feature alerts | low / medium | 2025-03 | Partial final month (small sample), not a real shift |
| `calibration` | info | 7 quarters | Level outside [0.8, 1.25]; informational because the model is approved for rank-ordering only |
| `discrimination_drop` | none | – | Quarterly AUC stays close to the reference: ranking holds |

## Remedy tested (scenario, live champion unchanged)

A decile refresh on **applications only** (FPD 2023-07 – 2024-06, no outcomes used), evaluated out of sample on
2024-07 – 2025-03 with policy 1.0.0:

| | Stale 2017–19 deciles | Refreshed deciles |
|---|---|---|
| Approve / refer / decline | 40.4% / 45.3% / 14.3% | **82.8% / 14.3% / 2.9%** |
| Default rate approved vs declined | – | 0.28% vs 2.59% |
| Contribution (base assumptions) | $27.5M | **$47.3M** |

Referrals return close to the rule's design share (deciles 9–10 of current applicants), and risk separation is
preserved. Applying the refresh to the live champion needs a named reviewer (`crisk refresh-deciles`, RB-02).

## Applied (2026-10-13, D-036)

After the champion changed to M1-nostate, its deciles were refreshed by a named reviewer (Dzidefo123) on applications
with FPD 2024-04 – 2025-03. Re-run monitoring: score PSI ≤ 0.22 in every month (a watch-level alert in 5 months; it
had been high in all 21). Policy 1.0.0 on the whole 2023–25 holdout: approve 76.6% / refer 19.0% / decline 4.4%.
Default rate 0.24% approved vs 2.85% declined. Contribution $105.5M (base assumptions).
