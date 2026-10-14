# Decision log

Format: ID · date · decision · alternatives considered · rationale · reversibility.

**D-001 · 2026-10-12 · Repository at the workspace root, package `crisk`, project name `credit-risk-ai-platform`.**
Alternative: a nested `credit-risk-ai-platform/` folder. Rationale: the workspace was empty, and the root
avoids a nested repo inside the editor workspace. Reversible (rename the folder or remote).

**D-002 · 2026-10-12 · Stack: Python 3.12, uv lockfile, DuckDB + Parquet, pandas, scikit-learn, FastAPI, SQLite audit store.**
Alternatives: Polars (good, but pandas is needed for sklearn interop anyway); Spark (premature on a laptop);
Postgres (not needed for a single-user prototype). DuckDB parses the pipe files out of core and runs the
validation and label SQL. A Spark/Databricks adapter can reuse the same SQL logic later.

**D-003 · 2026-10-12 · The contract is transcribed from the January 2026 user guide and versioned (`sflld_guide_2026_01`).**
A file with the wrong field count is refused rather than partially loaded. A layout change means adding a
contract version, never editing the existing one in place.

**D-004 · 2026-10-12 · Invalid key fields reject the row; other invalid values are set to NULL and logged as field issues.**
Alternative: reject any row with any invalid field. Rationale: avoids dropping whole loans because of one
non-key anomaly, while keeping every anomaly auditable. Provider sentinels are *not* issues.

**D-005 · 2026-10-12 · File-based model registry with SHA-256 integrity checks instead of MLflow for the first slice.**
Rationale: fewer moving parts. Promotion and rollback are audited and need a named reviewer. MLflow
tracking can be added in Phase C when experiments multiply.

**D-006 · 2026-10-12 · SQLite append-only audit store: UPDATE/DELETE triggers plus a SHA-256 hash chain.**
The triggers stop accidental edits; the chain detects tampering that bypasses them. Production would need
WORM storage and separate credentials (see data_handling_and_security.md).

**D-007 · 2026-10-12 · In-window voluntary prepayment before default is labelled 0.**
Alternative: treat it as censored and exclude it, or use survival methods. Rationale: a standard
fixed-horizon origination PD that answers "did it default before exiting within 12 months?". Sensitivity
analysis is planned for Phase C.

**D-008 · 2026-10-12 · Cohort-level maturity: immature loans are excluded even if they already defaulted.**
Rationale: including only observed early defaults from immature cohorts inflates prevalence.

**D-009 · 2026-10-12 · Temporal splits by FPD with an enforced purge gap ≥ horizon between splits.**
Rationale: the labels of earlier splits must be observable before later cohorts are scored. The config
loader enforces this.

**D-010 · 2026-10-12 · Calibration is a Platt step fitted on the calibration split only; no class weights.**
Rationale: keeps probabilities on the population scale. Metrics on the calibration split after Platt are
in-sample and labelled as such.

**D-011 · 2026-10-12 · Every artefact carries `data_origin` (SYNTHETIC/EMPIRICAL).**
Synthetic loan IDs use product letter `X`, which the provider never issues. Synthetic results are system
tests, never empirical evidence.

**D-012 · 2026-10-12 · The online API accepts canonical values only: "not available" is `null`, and provider sentinel codes are rejected (422).**
Missing required inputs (`orig_upb`, `orig_interest_rate`, `orig_loan_term`) return `status = unavailable`
with no score. Missing optional inputs are imputed and flagged.

**D-013 · 2026-10-12 · Every locked test-set evaluation is written to the audit chain with a count of prior evaluations.**
Rationale: makes repeated peeking at the test set visible to a validator.

**D-014 · 2026-10-12 · Added contract `sflld_r47_2026_07` (Release 47, July 2026) as a new version; the 2026-01 contract is kept.**
The downloaded files have 31 origination / 35 performance fields, which the original contract correctly refused.
The new version is derived from the old one plus the provider's documented changes (header files, file layout
xlsx, Disclosure Changes v1.2). Datasets choose their version via `contract_version`. Tests cover both layouts.

**D-015 · 2026-10-12 · DuckDB work runs in a throw-away on-disk database with a 6 GB memory limit and a spill directory.**
Full-volume quarters are about 3 GB uncompressed each. The in-memory temp tables used for the sample would not
fit in 16 GB RAM at that scale.

**D-016 · 2026-10-12 · Payment-deferral rows are excluded from the "UPB above original" warning.**
On the sample, 3,248 of 3,281 flagged rows carried non-interest-bearing UPB from payment deferrals, which the
provider does not treat as modifications. This is not a data error.

**D-017 · 2026-10-12 · Owner chose "report both" for disaster/forbearance delinquency (Q-3, option C).**
The primary target `default_90dpd_12m_v1` stays as reported. The sensitivity target `default_90dpd_12m_exdisaster_v1`
does not count 90+/RA months flagged as disaster or forbearance as default events. Loans whose only in-window
90+ months are flagged become `excluded_disaster_forbearance`; they are not relabelled 0, because their
credit-driven outcome cannot be known. Credit terminations (02/03/09) still count. Both labels are built from
the same canonical tables, stored side by side in `feature_snapshots`, and every test evaluation reports the
model against both.

**D-018 · 2026-10-12 · Modelling dataset `freddie_mac_combined` = full-volume 2015 (train) + sample 2017-2019 (calibration) + full-volume 2022 (test).**
Alternatives: samples only (about 100 train events for 2015-16, too few for ~40 parameters), or downloading
full-volume 2016-2021 (about 12 GB per year, unnecessary). Each split comes from a single source, so no
reweighting is needed within a split. Calibration cohorts (FPD 2017-07..2019-03) end their windows before COVID
forbearance but include the 2017 hurricanes. FPD 2019-04..2021 cohorts (COVID) are not assigned to any split.
They can serve as a separate stress-period evaluation. Built by composing already-validated interim files
(`crisk ingest --dataset freddie_mac_combined`), with no re-parsing.

**D-019 · 2026-10-13 · Market-rate spread feature from Freddie Mac PMMS (`origination_v2`).**
`rate_spread` = note rate − mean of weekly PMMS rates *published* in month FPD−3 (15-year series for terms ≤ 180,
30-year otherwise). There is no look-ahead: the reference month precedes the note date. ALFRED vintages
(2017, 2020, 2023) showed 0 revised values, so the current series is used with publication date = observation date.
The model bundle embeds the monthly table, so scoring and replay are reproducible. Limitation: a fixed 3-month lag
adds noise when rates move fast.

**D-020 · 2026-10-13 · LightGBM challenger with fixed settings and economically justified monotone constraints.**
Credit score is decreasing; DTI, LTV, CLTV and the note rate or spread are increasing; categoricals are unconstrained.
Boosting rounds come from early stopping on the last 6 months of the training window, then the model is refitted
on all of train. Reason codes use TreeSHAP (log-odds), which, like the logistic reasons, are attributions, not
causal effects.

**D-021 · 2026-10-13 · v2 model selection applied exactly as pre-registered: M4 (LightGBM, v1 features) for the primary target, M1 for ex-disaster.**
The rule's flaw was recorded rather than overridden. M4's AUC gain (+0.0101) cleared the 0.01 point threshold,
but its paired CI includes 0, while M2 and M3 have gains whose CIs exclude 0. The v3 plan will use a CI-based rule.
The 2022–23 test split is treated as *seen* and is not used to select v2 models.

**D-022 · 2026-10-13 · Publication gate `crisk export-results`.**
It exports aggregate JSON and docs only and suppresses rows with fewer than 1,000 loans. It masks event counts
below 10 *together with* any rate from which they could be recomputed (a hole found in review and fixed: 113 rows
were recoverable before the fix). It aborts if any loan-ID-shaped string appears. `publish/` is git-ignored until a
human reviews it.

**D-023 · 2026-10-13 · Repo-local git identity set to the account email; LF line endings pinned via `.gitattributes`.**
JSON/JSONL is now written with LF on every OS, so the byte-compared fixtures match on Linux CI.

**D-024 · 2026-10-13 · v3 walk-forward recalibration: pre-registered rule adopts R0 (no recalibration).**
With a realistic 20-month information lag, intercept or Platt refits on recent mature cohorts mistime regime shifts.
They miss the COVID forbearance spike and then over-correct by 3–7× for 2022–23 cohorts. The negative result is
reported in full (docs/results/recalibration_v3.md). The next step is a level signal available at origination
(early-performance proxy or macro term), which needs a new pre-registered v4 plan.

**D-025 · 2026-10-13 · `fit_logistic_1d` now uses step-halving Newton (bug found by the v3 backtest).**
Plain Newton diverged (intercept −1.6e8) when the offset predictions were about 15× below the observed rate. The fix
changes no earlier result: M1/M4 Platt and calibration metrics were recomputed with a difference of 0.0. A regression
test was added. The first (invalid) backtest run is kept locally as a recorded deviation.

**D-026 · 2026-10-13 · v4 through-the-cycle level: pre-registered rule keeps L0.**
Expanding and 60-month intercept anchors both made the level less stable (mean RMS ln(O/E) 1.19 / 1.26 vs 1.05) and
raised log loss. Together with v3, the conclusion is that the absolute PD level in 2015–2025 is shock-driven and cannot
be set reliably from lagged outcomes. Recommendation: use model output for ranking; govern the absolute level as an
explicit documented overlay. The 2025 sample (Release 47 edition) is excluded from ingestion and reserved for the
confirmatory test on Release 48+ data.

**D-027 · 2026-10-13 · Champion: M1 logistic (`…b6249e`), promoted by Dzidefo123 for `demonstration_rank_ordering` only.**
The promotion record carries the approved use and risk-decile cut-offs from the 86,404-loan calibration population.
Responses show `approved_use` and `risk_decile`; the analyst UI leads with the decile and labels the PD as not
validated. Validated afterwards (`val-…-20261013T113241`): 13/14 checks pass, and V07 PD-level calibration fails
(O/E 0.62), so PD-level use is not approved. Segregation of duties is **not met**: the same person promoted and
validated. This is recorded rather than hidden.

**D-028 · 2026-10-13 · API authentication: hashed API keys with five roles and per-endpoint permissions; human review is append-only.**
Keys use the format `crk_<id>_<secret>`. Only SHA-256 hashes are stored (`artifacts/<ds>/api_keys.json`,
git-excluded), and comparison is constant-time. Creating and revoking keys is audited. No API role can promote models,
change policy or approve loans. Reviews (agree / override / escalate, with rationale) go to `review_events`, which has
the same append-only triggers. An identity provider, TLS and rate limiting are production gaps.

**D-029 · 2026-10-13 · Analyst UI is a pure API client (Streamlit, optional `ui` group).**
The UI never reads model files or the audit store directly, so permissions and audit logging live in one place (the
API). This is decision support only: there is no automated approve/decline, because no policy engine exists yet
(Phase E).

**D-030 · 2026-10-13 · Independent validation workflow with release criteria fixed before its first run; EMPIRICAL promotions require it.**
`crisk validate` runs 14 checks against `release_criteria_v1` and merges the findings register. Promoting an EMPIRICAL
model through the CLI is refused unless the latest validation report approves the requested use.

**D-031 · 2026-10-13 · Fairness and geography review: remove `property_state` (pre-registered rule).**
`property_state` reduces out-of-time AUC (−0.012 primary, −0.005 ex-disaster; both CIs exclude 0), and its
state-level proxy screen against Census demographics is weak (|ρ| ≤ 0.15). The no-state models are clean-commit
reproductions, each evaluated once on the 2023–25 holdout (AUC 0.815 / 0.826) and validated 13/14 for
`demonstration_rank_ordering`. Promotion awaits a named reviewer. F-002 is closed; F-007 (credit-score gradient
drift) is opened.

**D-032 · 2026-10-13 · Policy engine separate from the model; rank-based fictional demo policy; PD rules blocked without pd_level approval.**
A historical simulation on the 2023–25 holdout showed strong risk separation (approved 0.09% vs declined 1.36% default
rate). It also exposed decile drift: 47% of recent loans fall in the 2017–19 "decile 10". The remedy is monitoring
plus a governed decile refresh on recent applicants (no outcomes needed), built in the monitoring step.

**D-033 · 2026-10-13 · Monitoring compares scores with the champion's current rank reference; the drift remedy is a governed decile refresh on applications.**
The 2023–25 replay raised a high `score_drift` alert in all 21 months (PSI 0.6–2.4). Refreshing deciles on 2023-07..2024-06
applications, with no outcomes needed, cut out-of-sample referrals from 45% to 14% under policy 1.0.0 and kept risk
separation. Applying the refresh to the live champion requires a named reviewer.

**D-034 · 2026-10-13 · Docker Compose demo is synthetic-only.**
The `init` service runs the synthetic slice, monitoring and demo key issuance. The API and UI start only after
`init` succeeds. Ports are bound to 127.0.0.1, the container runs as a non-root user, and data, artefacts and
reports are excluded from the image. The demo proves the engineering path without credentials. It is not a
deployment and not evidence about credit risk.

**D-035 · 2026-10-13 · Champion changed to M1-nostate `…8fd365`, promoted by Dzidefo123 for `demonstration_rank_ordering`.**
Promoted following the pre-registered fairness review (D-031), through the validation gate
(`val-…8fd365-20261013T120317`, 13/14; PD-level not approved). The previous champion `…b6249e` is kept as
`previous_model_id`, so it can be restored with `crisk rollback`. The decile cut-offs were reset to the 2017–19
calibration reference at promotion, so the score-drift remedy (`crisk refresh-deciles`, RB-02) still awaits a reviewer.
The API now publishes the model's `input_fields` and the analyst UI sends only those (the no-state model rejects
`property_state`).

**D-036 · 2026-10-13 · Champion decile cut-offs refreshed (RB-02) by Dzidefo123 on applications with FPD 2024-04..2025-03 (46,563 loans, no outcomes used).**
Monitoring after the refresh: score drift fell from a high alert in all 21 months (PSI 0.6–2.4) to a watch-level alert in
5 months (max PSI 0.22). The 2023-07..2024-03 months, outside the reference window, are also stable. Under policy 1.0.0
on the 2023–25 holdout: approve 76.6%, refer 19.0%, decline 4.4% (was 36.5/47.4/16.1). Default rate approved 0.24% vs
declined 2.85%. Input-drift alerts remain, because they are measured against the 2017–19 training population and the
drift is real (finding F-005). The previous cut-offs are kept in `decile_history`.

**D-037 · 2026-10-13 · Phase J v1: severity model S2 selected and tested once; EL = PD overlay × P(liquidation) × severity × EAD.**
The Release 47 loss sign convention was verified empirically. P(liquidation | 90+) = 0.294 and the EAD ratio = 0.95
(2015 vintage). S2 fractional logit was selected by the pre-registered rule (−11% validation MAE). Locked test:
MAE 0.176 vs 0.224 baseline, Spearman 0.36. EL is illustrative 12-month only (base 1.4 bps; severe 10.5 bps), using a
decile-rate PD overlay because the champion is approved for ranking only.

**D-038 · 2026-10-13 · Phases G/H pre-registered before any code: synthetic documents, mocked LLM first.**
Plan [document_ai_copilot_plan_v1.md](document_ai_copilot_plan_v1.md), corpus access rules
(`configs/copilot/corpus_v1.yaml`) and 48 labelled questions (`configs/copilot/eval_questions_v1.yaml`, dev 28 /
held-out 20) were committed before the generator, extractor, retrieval index or copilot existed. Documents are fictional
(SYNTHETIC). The default LLM is a deterministic mock, so results measure the pipeline, not LLM quality. Access filtering
happens before retrieval; the copilot has read-only tools and no write path; unvalidated extractions cannot reach scoring.

**D-039 · 2026-10-13 · Phases G/H v1 evaluated once on held-out (mocked LLM); failures recorded, not re-tuned.**
The system was frozen at `3a473bb`.
- **Documents:** precision 1.0 and seen-layout recall 1.0; every defect type was detected on seen layouts. The unseen
  prose layout yielded no fields (recall 0/24), so cross-document recall (0.60) and the clean-packet false-flag rate
  (0.33) failed. The failure is safe: all four such packets went to review, none was validated, and 0 passed the gate.
- **Copilot:** every guarantee held (permissions, injection, verbatim citations, 0 unsupported claims), and hit@5 and
  abstention recall were 1.0. The extractive mock answerer failed accuracy (0.27 vs dev 0.80), false abstention (0.45)
  and the B0 comparison (−0.18).
- **Conclusion:** the pipeline's controls generalise, but the mock's dev-tuned threshold does not. The next step is a
  separately recorded real-LLM evaluation (needs the user's key and cost approval) and a v2 document set with a
  fresh held-out split.


**D-040 · 2026-10-14 · Real-LLM evaluation (claude-sonnet-5-5, pre-registered addendum A): all held-out criteria met.**
- **Copilot:** accuracy 0.73 (B0 0.45, mock 0.27), false abstention 0.18, abstention recall 1.0. All guarantees held.
- **Documents:** the grounded LLM extractor read the unseen prose layout fully (24/24), with every defect detected,
  0 false flags and 0 gate leaks.
- **Cost:** about $0.94 in total.
- **Open items:** the verifier is over-strict on numbers echoed from the question (v2 change; needs new questions to
  evaluate), and the scenario-vs-applied figure ambiguity in monitoring_v1.md (N08). The owner's API key is held in
  a user environment variable outside the repository; a key pasted in chat was not used and was rotated.
