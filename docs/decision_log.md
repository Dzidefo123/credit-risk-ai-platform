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
