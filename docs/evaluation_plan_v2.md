# Pre-registered evaluation plan: v2 features and LightGBM challenger

_Written 2026-10-13, before any `origination_v2` or LightGBM model was trained on real data._

## Why a new plan

The `origination_v2` feature set (rate spread over the PMMS market rate) was designed **after** seeing the v1
results on the 2022–23 test split. That split is therefore **no longer a clean holdout** for v2 or for a challenger
selected with that knowledge. Results on it are reported only as a *post-hoc diagnostic* and labelled as such.

## Candidate models (fixed in advance; no hyperparameter search)

| ID | Features | Algorithm | Config |
|---|---|---|---|
| M1 | origination_v1 | logistic | logistic_v1 (the existing baseline) |
| M2 | origination_v2 | logistic | logistic_v1 |
| M3 | origination_v2 | LightGBM | lightgbm_v1 (fixed params, monotone constraints, temporal early stopping inside train) |
| M4 | origination_v1 | LightGBM | lightgbm_v1 (separates the feature effect from the algorithm effect) |

Each model is trained for both the primary and the ex-disaster target.

## Model-selection evidence (development data only)

Out-of-time **calibration split** (FPD 2017-07 – 2019-03, sample): discrimination before Platt (ROC-AUC, Gini,
PR-AUC), plus the uncalibrated observed/expected ratio and calibration slope. Platt is fitted on this same split, so
post-Platt calibration here is in-sample and not evidence.

**Selection rule (decided now):** prefer the simplest model (M1 < M2 < M4 < M3) unless a more complex one improves
calibration-split ROC-AUC by **≥ 0.01** *and* does not worsen the uncalibrated calibration slope by more than 0.1.
A gain under 0.01 AUC does not justify the added complexity for this purpose.

## Fresh, locked test for the selected model

A new test split from vintages not yet used: provider **sample 2023 and 2024 vintages**, first payment
**2023-07 – 2025-03**. This starts after the end of the old test window; all 12-month windows are mature by the
2026-03 cutoff. **Blocked until the owner downloads `sample_2023.zip` and `sample_2024.zip`.** Only the selected model,
plus M1 as reference, will be evaluated on it, once each.

Release criteria are unchanged from v1: AUC CI excludes 0.5 for demonstration use; observed/expected in [0.8, 1.25]
for PD-level use.
