# Policy engine and historical simulation v1

**Evidence class: EMPIRICAL (simulation); FICTIONAL policies.** Aggregate results only. This is an observational
backtest on funded, Freddie Mac-acquired loans. It is not a causal estimate of a policy change, not a real lender's
policy, and not accounting ECL.

## Engine

- Versioned YAML policies (`configs/policies/`). Each rule is `{id, when, outcome, reason}`.
- Conditions are parsed with a whitelisted expression evaluator, so a rule cannot run code. A test refuses
  `__import__`, attribute access, calls and subscripts.
- Decision precedence: **decline > refer > approve**. Every decision lists the decisive and triggered rules with
  reasons, the policy version and SHA-256, and the model's reason codes as context.
- Missing data never satisfies a threshold. It is routed explicitly (rule D02, referral).
- **Governance guard:** a rule on `pd` is refused unless the serving model is approved for `pd_level`. The demo
  policies therefore use the **risk decile** (rank), consistent with the champion's `demonstration_rank_ordering`
  approval.
- API: `POST /v1/scores/{id}/decision` (roles analyst and administrator) stores an append-only policy decision.
  Decisions are recommendations for human review.

## Simulation (champion M1 `…b6249e`; 2023–25 holdout, 83,597 loans with mature 12-month outcomes)

Assumptions (`configs/policies/simulation_assumptions_v1.yaml`, illustrative): EAD = original UPB, LGD 25%, 12-month
net margin 1.0%, $1,500 cost per funded loan, $250 per manual review. Referred loans are treated as declined in the
base case.

| Policy | Approve | Refer | Decline | Default rate: approved / referred / declined | Defaults declined | Non-defaults declined | Contribution (base) |
|---|---|---|---|---|---|---|---|
| 1.0.0 | 36.5% | 47.4% | 16.1% | 0.09% / 0.74% / 1.36% | 183 | 13,283 | $58.6M (54.8 bps) |
| 1.0.0-tight | 22.5% | 31.9% | 45.6% | 0.06% / 0.47% / 0.97% | 370 | 37,777 | $35.6M (54.8 bps) |
| 1.0.0-loose (no model rules) | 98.7% | 1.1% | 0.1% | 0.60% / 0.21% / 4.13% | 5 | 116 | $103.5M (39.1 bps) |

Sensitivity, policy 1.0.0 (contribution, $M): LGD 15/25/40% × margin 0.5/1.0/1.5% → 6.2/59.7/113.1,
5.2/58.6/112.1, 3.6/57.1/110.6. The result is far more sensitive to the margin assumption than to LGD, because
12-month default rates are low.

## Findings

1. **The model separates risk well.** Approved loans default at 0.09%, against 1.36% for loans declined under
   policy 1.0.0.
2. **Under these assumptions, tighter is not more profitable.** Each declined good loan forgoes margin, and with
   12-month default rates below 1%, the loose policy funds the most profitable book in absolute dollars. The tight
   policy has the best per-dollar quality but the smallest book. This is an illustration of the trade-off, not a
   recommendation: a 12-month horizon understates lifetime losses.
3. **Operational red flag: decile drift.** The champion's decile cut-offs come from the 2017–19 population. On
   2023–25 loans, **47% fall in decile 10 and 63% in deciles 9–10**, mostly because of note-rate extrapolation
   (F-005). Policy 1.0.0 would refer 47% of applications. Rank-based rules need a reference distribution that tracks
   current applicants. Recomputing cut-offs needs **no outcomes**, so there is no information lag. Portfolio
   monitoring (score PSI) is set up to detect this, with a governed decile-refresh procedure as the remedy.
