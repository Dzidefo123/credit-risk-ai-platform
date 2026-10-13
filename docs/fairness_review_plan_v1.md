# Pre-registered fairness and geography review (closes finding F-002)

_Written 2026-10-13, before any fairness or geography analysis was run._

## What can and cannot be assessed

- The Freddie Mac SFLLD contains **no protected characteristics** (race, ethnicity, sex, age). Individual-level
  disparate-impact testing is **not possible** with this data. Linking loans to other data to infer such
  characteristics is prohibited by the dataset terms and is **not done**.
- This review therefore covers (1) the **predictive value** of the geographic feature `property_state`,
  (2) a **state-level proxy screen**: whether the model's state effects line up with public state-level
  demographic composition, and (3) **segment-level error analysis** across geography and loan characteristics.
  A clean result is *not* evidence of fairness; it only reduces one specific proxy risk.

## Data

- Model under review: champion M1 `…b6249e` (logistic, `origination_v1`, primary target). The same review is run
  for the ex-disaster M1 `…57741f`.
- Ablation candidate: **M1-nostate**, identical except `property_state` is removed (`origination_v1_nostate`). It is
  trained with the same configuration, seed, training split and Platt step.
- Demographics: U.S. Census Bureau, *SC-EST2024-ALLDATA6* (Vintage 2024 state population estimates by race and
  Hispanic origin, released June 2025). The 2022 estimates are used. Per state: share **not non-Hispanic White
  alone**, share **Black alone**, share **Hispanic**. Public aggregate data; no loan-level linking.
- Development evidence: the out-of-time calibration split (FPD 2017-07 – 2019-03). Diagnostic only: the 2023–25
  holdout. That holdout was viewed earlier for other models, so it is labelled as such.

## Measurements

1. **Value of `property_state`:** calibration-split ROC-AUC (before Platt) of M1 minus M1-nostate, with a paired
   bootstrap 95% CI (1,000 resamples).
2. **Proxy screen:** for each state with at least 500 training loans, take the model's state effect (the mean logit
   contribution of `property_state` on the calibration split). Compute its Spearman correlation across states with
   each demographic share.
3. **Segments** (both models, calibration split and holdout diagnostic): ROC-AUC, observed/expected, and mean PD
   by Census region, the 10 largest states, credit-score band, occupancy, purpose, channel and first-time-buyer flag.
   Segments with fewer than 1,000 loans or fewer than 10 events are not reported.

## Decision rule (fixed now)

`property_state` is **kept** only if **both** hold:
- it adds clear value: ΔAUC ≥ 0.010 **and** the paired CI lower bound is > 0; **and**
- the proxy screen is weak: |Spearman ρ| < 0.5 for **every** demographic share.

Otherwise it is **removed**: M1-nostate becomes the champion candidate. It must pass `crisk validate`, using one
out-of-time evaluation on the 2023–25 holdout (the decision is made without holdout results), and a named reviewer
must promote it. Either way, F-002 is closed with this report as evidence. If `property_state` is kept, segment
monitoring by state becomes a standing control.
