# Assumptions register

Status values: **provisional** (reversible and in use), **confirmed** (verified against a source or by
the owner), **open** (needs a decision or data), **retired**.

| ID | Assumption | Status | Basis / how to verify | Impact if wrong |
|---|---|---|---|---|
| A-001 | Jurisdiction is the US; the product is single-family fixed-rate mortgages in the Freddie Mac Standard Dataset. | provisional | Master prompt §3 | Target, features and policies would change |
| A-002 | The intended use is noncommercial research shown in a public portfolio. | **confirmed by owner 2026-10-12** | The terms allow public, noncommercial research results that cannot recreate the data. Raw and loan-level data stay private | A publication gate is needed before anything is shared |
| A-003 | Origination-file values are treated as known at the decision (reconstructed state). The dataset has no availability timestamps. | provisional | Guide: credit score "generally the score known at acquisition and used to originate"; DTI is computed at delivery | Some features may postdate the decision slightly (e.g. MI%, which is measured at purchase) |
| A-004 | Origin = first payment date month. The note/decision date is undisclosed and is typically 1–2 months earlier. | provisional | Guide field 2 definition | Horizon is offset by about 1–2 months relative to the decision |
| A-005 | 90+ DPD is status ≥ 3 under the MBA method. Status `RA` and zero balance codes 02, 03 and 09 are credit events. | confirmed (codes) / provisional (choice) | Guide p.10–15 | Event rate |
| A-006 | Loans first reported after m = 6 are excluded (left truncation). | provisional | Guide: performance starts at acquisition | Could exclude seasoned loans systematically; review the count on real data |
| A-007 | Up to 2 missing reporting months in the window are tolerated. | provisional | Guide documents reporting gaps (Example 3) | Exclusion rate; small bias if gaps correlate with risk |
| A-008 | Zero balance codes 15, 16 and 96 before any default make the outcome unobservable, so the loan is excluded. | provisional | Guide zero balance codes table | Small. Check counts within 12 months on real data |
| A-009 | Prepayment in the window gives label 0. | provisional (D-007) | Common practice for fixed-horizon origination PD | Prevalence is slightly lower than under censoring. Sensitivity analysis planned |
| A-010 | Borrower identity is unavailable, so borrower-overlap checks across splits are impossible. Relief refinances can be linked through `pre_relief_refi_loan_id`. | confirmed | Guide fields 20 and 27 | Performance may be optimistic if repeat borrowers span splits. Document this; check the relief-refi links |
| A-011 | Sentinel codes (9999, 999, 99, 9, 7) mean "not available", stored as NULL. Missing values are modelled explicitly (median imputation plus indicator, or a `__MISSING__` category). | confirmed (codes) | Guide field definitions | — |
| A-012 | Empirical split design | **open (Q-5)** | Only the 2015 and 2022 vintages were downloaded, so the original 2015-17 / 2019 / 2022 design is impossible. Sample events: 65 (2015 vintage) vs 343 (2022 vintage) | Out-of-time calibration and test design |
| A-013 | `property_state` is an acceptable v1 feature, pending proxy review. | provisional | Geography can proxy protected characteristics | Fair-lending risk; review in Phase C |
| A-014 | Reason codes are model-input attributions, not ECOA/Reg B adverse-action reasons. | confirmed (scope) | Legal mapping not performed | Must not be presented as adverse-action notices |
| A-015 | Local single-user prototype: SQLite audit store and file-based model registry. | provisional (D-005, D-006) | Laptop-first requirement | Needs Postgres/MLflow before multi-user use |
| A-016 | Developer laptop: Windows 11, Python 3.12, 16 GB RAM, a cloud-synced Desktop folder. | confirmed | Environment inspection | uv uses `link-mode = copy`; DuckDB memory limit 6 GB, spills to disk |
| A-017 | The Release 47 (July 2026) layout applies to all downloaded files (31 origination / 35 performance fields). | confirmed | Provider header files and Disclosure Changes v1.2; 0 field issues on 5.6M sample lines | — |
| A-018 | Disaster- and forbearance-related 90+ DPD counts as default in v1 (as reported). | **open (Q-3)** | 54% of 2022-vintage sample defaults (184/343) carry disaster or forbearance flags, concentrated in FL/CA/TX from late 2022, and most return to current | The 2022 test target would partly measure natural-disaster forbearance rather than borrower credit risk |
| A-020 | The market reference rate for a loan is the mean PMMS rate published in month FPD−3, using the 15-year series for terms ≤ 180. | provisional | Note date is about 1–2 months before FPD; rate lock is typically 30–60 days before the note; PMMS is not revised (ALFRED check) | Lags the true lock rate when rates move fast (2022). The spread SD rises from 0.32 to 0.55 |
| A-021 | Full-volume and provider-sample loans may be combined only with each split coming from a single source. | confirmed | The sample is a simple random sample of the full Standard Dataset | Mixing within a split would over-weight full-volume loans by about 25× |
| A-019 | Zero balance code 96 (defect) before default excludes the loan (197 sample loans, all code 96). | provisional | Outcome unobservable after repurchase | Defects may correlate with early delinquency, so excluding them could bias PD down slightly |

## Open questions (blocking only where stated)

| ID | Question | Blocks |
|---|---|---|
| Q-1 | ~~Download data~~ **Done 2026-10-12**: sample and full-volume files for the 2015 and 2022 vintages (Release 47) | — |
| Q-2 | ~~Intended use?~~ **Answered 2026-10-12: noncommercial research for a public portfolio.** Optional: email Freddie Mac to confirm that portfolio use counts as noncommercial | Publication only |
| Q-3 | How should disaster/forbearance delinquency be treated: keep as default, exclude from the event, or report both (primary + sensitivity)? COVID cohorts are absent with 2015 + 2022, but 2022–23 disaster forbearance is present | Final empirical target |
| Q-5 | Download the 2016–2021 vintages too (sample files are about 400 MB each) for a proper out-of-time calibration split, or proceed with 2015 (train + calibration) → 2022 (test)? | Split design |
| Q-4 | Target audience and jurisdiction for governance mapping (e.g. US SR 11-7 style model risk management)? | Phase D/I documentation depth only |
