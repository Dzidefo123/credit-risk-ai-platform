# Model refresh v5: selection stage (development data)

**Evidence class: EMPIRICAL, already-examined period. This is selection evidence, not confirmation.**
Pre-registered in [../evaluation_plan_v5.md](../evaluation_plan_v5.md) (commit `24e477d`, owner-approved before
training). Both refit models were trained from commit `06441fd`.

## Data

| Role | Source | FPD | Loans | Defaults (primary) |
|---|---|---|---|---|
| Train | full-volume 2022 vintage | 2022-02 .. 2022-12 | 1,434,542 | 9,728 |
| Calibration (Platt) | sample 2024 vintage | 2024-02 .. 2025-03 | 49,729 | 302 |
| Selection | sample 2023 vintage | 2023-02 .. 2023-12 | 42,061 | 251 |

About half of the 2022 vintage's 90+ events are disaster- or forbearance-flagged: 5,008 training defaults under the
ex-disaster definition, against 9,728 under the primary one.

## Selection (pre-registered rule: default C2; C1 only if better by ≥ 0.005 AUC with CI > 0)

| Model | ROC-AUC | vs C0 (paired 95% CI) | Observed / expected |
|---|---|---|---|
| C0 champion `…8fd365` (2015–16 fit) | 0.803 | — | **0.66** |
| C1 refit, note rate `…a924e0` | 0.815 | +0.011 (−0.001, +0.024) | 1.01 |
| **C2 refit, PMMS rate spread `…143cb5` (selected)** | **0.815** (CI 0.790–0.837) | +0.012 (−0.002, +0.025) | **0.98** (CI 0.87–1.09) |

C1 − C2 = −0.0005 (CI −0.003, +0.002), so **C2 is selected** by default.

Ex-disaster sensitivity (202 events): AUC is 0.816 / 0.819 / 0.819. C2's observed/expected is 0.79 for this
narrower event (it was trained on the broader primary target).

## The two open findings, on these loans

| Credit score | Loans | Defaults | O/E C0 | O/E C2 |
|---|---|---|---|---|
| ≤ 660 | 1,676 | 50 | 1.05 | 1.19 |
| 660–700 | 4,930 | 73 | 0.77 | 0.97 |
| 700–740 | 9,268 | 71 | 0.63 | 0.95 |
| 740–780 | 13,653 | 45 | 0.53 | 0.97 |
| > 780 | 12,521 | 12 | **0.30** | **0.67** |

| Note rate | Loans | Defaults | O/E C0 | O/E C2 |
|---|---|---|---|---|
| 5–6% | 8,155 | 45 | 1.26 | 1.23 |
| 6–7% | 23,483 | 129 | 0.67 | 0.94 |
| > 7% | 9,788 | 76 | **0.51** | **0.95** |

The refit removes most of the credit-score gradient distortion (F-007) and the high-rate over-prediction (F-005; O/E 0.51 means predicted PDs were about twice the observed rate)
on these loans. Some residual remains: the extremes (≤ 660 and > 780, with 12 defaults) are still off.

## Why this is not yet evidence for promotion

- The selection set (2023 cohorts) lies in a period already examined, and it sits next to the 2024 calibration
  cohorts. A good level here is therefore expected and proves little about future vintages.
- Gains in AUC over C0 are within noise (CIs include 0).
- **Next:** the pre-registered confirmatory test runs once, on the Release 48 edition of the 2025 vintage, which is
  unseen. The checks are AUC CI lower bound ≥ 0.70, ΔAUC vs C0 ≥ −0.01, and O/E in [0.8, 1.25].
