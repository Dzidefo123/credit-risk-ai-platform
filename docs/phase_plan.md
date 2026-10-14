# Phase plan and acceptance gates

Status legend: **done** (implemented and tested) · **fixture-only** (works on SYNTHETIC data, empirical run
blocked) · **blocked** · **next** · **proposed**.

| Phase | Scope | Status | Acceptance gate |
|---|---|---|---|
| A | Acquisition, contracts, lineage | **fixture-only**; empirical import blocked on Q-1 | Source register ✅ · acquisition steps ✅ · parser/import commands ✅ · fixture path validated ✅ · rejected-record handling ✅ · idempotence test ✅ · official-data reconciliation ⛔ (needs files) · sampling manifest ⛔ (needs files) |
| B | Target and point-in-time features | **fixture-only** | Target spec ✅ · cohort flow ✅ · maturity report ✅ · leakage tests ✅ · split manifest ✅ · feature dictionary ✅ · PIT reconstruction test ✅ (row-wise) · selection-bias note ✅ · real cohort review ⛔ |
| C | Model development and calibration | **partial**: prevalence baseline, logistic, Platt and metrics done | Boosted-tree challenger · temporal CV tuning · bootstrap CIs (time-blocked) · segment and robustness analysis · prepayment-censoring sensitivity · proxy review · model card · champion recommendation. **Empirical only.** |
| D | Independent validation and release gates | proposed | Separate validation runner · pre-registered release criteria · perturbation and edge tests · offline/service parity on empirical data · findings register · promotion/rollback procedure (mechanics ✅) |
| E | Credit policy and economics | proposed | Versioned policy engine with reason codes · deterministic boundary tests · scenario assumptions (LGD/EAD as explicit scenarios) · sensitivity · analyst-readable comparison |
| F | Scoring, workflow, auditability | **partial**: batch/online scoring, idempotency, replay, append-only audit, health/ready done | Authentication · timeouts and retries for dependencies · load test vs declared targets · analyst UI · override/review events · Docker Compose one-command demo |
| G | Document intelligence | **done (v1, SYNTHETIC, mocked LLM)**: seen layouts met all criteria; unseen layout failed safely ([results](results/document_ai_copilot_v1.md)) | Fictional document set with answer keys · extraction benchmark · injection tests · correction workflow |
| H | Analyst copilot | **done (v1, mocked LLM)**: all guarantees met; mock answer quality failed on held-out ([results](results/document_ai_copilot_v1.md)) | Retrieval over versioned policies and model docs · read-only tools · citation/abstention eval vs a plain retrieval baseline |
| I | Monitoring and controlled updates | proposed | Drift (PSI) with justified bins · delayed-outcome joins · alerts and runbooks · candidate-to-promotion walkthrough |
| J | LGD/EAD, early warning, scenarios | proposed | Separate targets and specs; build only after the PD system is validated |

## Backlog (ordered)

1. **[blocked: Q-1]** Import the Freddie Mac sample (2015–2022). Reconcile per-file line counts with ingestion
   rows. Review canonical warnings, label flow by vintage, left-truncation exclusions and the forbearance share.
2. **[blocked: data]** Confirm the empirical split (Q-3). Train the empirical logistic baseline. Write the first
   empirical model report.
3. Phase C: add a LightGBM challenger (same splits and population); time-blocked bootstrap CIs; segment
   metrics by vintage, purpose, occupancy and channel; the prepayment-as-censored sensitivity; a with/without
   forbearance-defaults sensitivity.
4. Phase D: a validation runner separate from training, with pre-registered release criteria in YAML and a
   findings register.
5. Phase F: API authentication (API key → OIDC later), Docker Compose, a minimal Streamlit analyst view with
   SYNTHETIC/EMPIRICAL banners, and a load test.
6. Phase E: a policy engine (approve/decline/review) with versioned YAML policies and boundary tests.
7. Phase I: monitoring (PSI, volumes, latency), delayed-outcome joins and alert tests.
8. Phase A3: FRED/ALFRED vintages, after the baseline exists.
9. Phases G/H, then J.

## Research-integrity rules carried through every phase

- SYNTHETIC results are system tests. They never appear as performance claims.
- The test split is touched only through `crisk evaluate-test`, which is audited.
- Funded-loan selection bias: results describe acquired loans only. No reject inference without an explicit
  model and a sensitivity analysis.
- Historical policy simulations are observational, not causal.
