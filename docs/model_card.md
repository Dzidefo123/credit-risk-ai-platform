# Model card: origination PD scorecard (champion) and v5 refresh candidate

_Research prototype on public Freddie Mac data. Not a regulatory model, not validated by an independent function,
and not for credit decisions. Aggregate results only._

## At a glance

| | Champion (live) | v5 candidate (not promoted) |
|---|---|---|
| Model ID | `pd-logit-origination_v1_nostate-20261013T120217-8fd365` | `pd-logit-origination_v2_nostate-20261014T013519-143cb5` |
| Approved use | **`demonstration_rank_ordering`**: risk ranking and deciles; not the PD level | none yet; awaits the confirmatory test |
| Algorithm | L2 logistic regression (C = 1.0) + Platt calibration | same |
| Training data | full-volume 2015 vintage, FPD 2015-01..2016-06: 1,428,567 loans, 1,696 defaults | full-volume 2022 vintage, FPD 2022-02..2022-12: 1,434,542 loans, 9,728 defaults |
| Calibration data | provider sample, FPD 2017-07..2019-03: 86,404 loans, 232 defaults | provider sample, FPD 2024-02..2025-03: 49,729 loans, 302 defaults |
| Promotion | by Dzidefo123, 2026-10-13, after validation 13/14 (V07 PD level failed) | requires: confirmatory pass, then validation, then a named reviewer |
| Code | commit `50e98df` (clean) | commit `06441fd` (clean) |

## Intended use and out-of-scope use

- **Intended use:** at origination, rank fixed-rate conforming mortgage applications by the risk of a serious
  delinquency or credit termination in the first 12 months. The ranking feeds risk deciles, which feed a
  *fictional* demonstration policy, monitoring and an illustrative expected-loss calculation.
- **Out of scope:**
  - Absolute PD for pricing, provisioning or capital (the level is unstable; finding F-001).
  - Adverse-action notices (reason codes are model attributions; F-004).
  - Loans unlike Freddie Mac's funded, acquired population (no reject inference; F-003).
  - Any use with individual-level personal data.

## Target

`default_90dpd_12m_v1` covers months 1–12 from the first payment date. An event is any of:
- 90+ days past due;
- REO acquisition;
- a third-party sale, short sale or REO disposition.

Prepayment counts as a non-default. Censoring removals are excluded, and labels need cohort-level maturity. A
sensitivity target excludes 90+ episodes flagged as disaster- or forbearance-related.
See [target_specification.md](target_specification.md).

## Inputs (origination-time only; leakage-guarded)

- **Common inputs:** credit score, DTI, LTV, CLTV, CLTV − LTV, loan term, MI %, units, log loan size, multiple
  borrowers, occupancy, channel, property type, loan purpose, first-time-buyer flag.
- **Rate input:** the champion uses the **note rate**. The v5 candidate uses the **spread of the note rate over the
  Freddie Mac PMMS market rate** published three months before the first payment. That spread is point-in-time and
  has no revisions (verified with ALFRED vintages).
- **Excluded:**
  - Geography: `property_state` was removed after the pre-registered fairness review, because it lowered
    out-of-time AUC and has a weak proxy screen.
  - Seller and servicer names.
  - Every post-origination field.

See [feature_dictionary.md](feature_dictionary.md).

## Performance (out of time)

| Evaluation | Loans / defaults | ROC-AUC (95% CI) | Observed / expected |
|---|---|---|---|
| Champion, fresh holdout 2023-07..2025-03 (evaluated once) | 83,597 / 505 | **0.815** (0.798–0.835) | **0.63** |
| Champion, 2023 selection cohorts | 42,061 / 251 | 0.803 | 0.66 |
| **v5 candidate**, 2023 selection cohorts (already-examined period) | 42,061 / 251 | **0.815** (0.790–0.837) | **0.98** (0.87–1.09) |

Ranking generalises across seven to ten years (AUC 0.81–0.82). The champion's PD level does not generalise: it
over-predicts 2023–25 default rates (predicted about 1.6 times the observed rate). Two pre-registered fixes were rejected: lagged recalibration (v3) and a
through-the-cycle anchor (v4). The v5 refit restores the level on 2023 cohorts and removes most of the
credit-score distortion (F-007) and the high-rate over-prediction (F-005). This is **selection evidence only**. The
confirmatory test runs once on the unseen Release 48 2025 vintage.

## Fairness

The data has no protected characteristics, so disparate impact cannot be tested directly. The pre-registered review
of `property_state` found:
- it adds no out-of-time value: ΔAUC −0.012, CI excluding 0;
- its state effects correlate weakly with state demographics: |ρ| ≤ 0.15.

The feature was therefore removed. A clean proxy screen does not establish fairness.
See [results/fairness_review_v1.md](results/fairness_review_v1.md).

## Monitoring and controls

- **Monitoring:** score PSI against the champion's decile cut-offs, feature PSI, and matured O/E and AUC by quarter,
  with de-duplicated alerts and runbooks. Decile cut-offs were refreshed by a named reviewer (D-036). Since then score
  drift has been at most PSI 0.22.
- **Controls:**
  - SHA-256-verified artefacts and replay from any score ID.
  - Append-only, hash-chained audit.
  - Role-based API keys and human review events.
  - A policy engine that refuses PD-level rules for this approved use.

## Open findings

F-001 (PD level), F-003 (funded-loan selection), F-004 (reason codes are not adverse-action reasons), F-005 (note-rate
extrapolation) and F-007 (credit-score gradient). The register is in
[configs/validation/findings_register.yaml](../configs/validation/findings_register.yaml). Segregation of duties is
not met: the same person promoted and validated the champion.
