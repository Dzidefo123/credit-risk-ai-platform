# Project status

_Last updated: 2026-10-12 (session 1, continued)._

## Summary

| Capability | State |
|---|---|
| Repository, uv lock, CI config, ignore rules for restricted data | **implemented** (git initialised, nothing committed yet) |
| SFLLD contracts: January 2026 layout **and Release 47 (July 2026) layout** | **implemented and tested** |
| Ingestion with quarantine and checksum idempotence; nested-zip unpack; dataset composition | **implemented and tested**; run on real data |
| Out-of-core processing (DuckDB on-disk work DB, 6 GB memory cap, spill) | **implemented**; verified on 168M performance rows |
| Canonical build, primary and sensitivity (ex-disaster) targets | **implemented and tested**; run on real data |
| Empirical baseline: logistic + Platt, primary and sensitivity | **trained and evaluated once on locked test**; see [results/empirical_baseline_v1.md](results/empirical_baseline_v1.md) |
| Scoring, replay, audit, API | implemented and tested (synthetic); not yet promoted on empirical data |
| Boosted-tree challenger, tuning, segment/proxy analysis, validation runner, policy engine, UI, monitoring, Docker, auth, macro data | **proposed** |

## Data acquired (Release 47, cutoff 2026-03)

- Sample 2015–2022 (8 × 50,000 loans; 20.7M performance rows): 0 rejects, 0 field issues.
- Full volume 2015 and 2022 (3,055,129 loans; 168,477,146 performance rows): 0 rejects; 1 field issue (a 70% interest rate, set to null).

## Commands run (all succeeded unless noted)

```
uv run crisk ingest/build-canonical/build-labels --dataset freddie_mac_sample
uv run crisk unpack/ingest --dataset freddie_mac_historical            # unpack 2.5 min, ingest 64 min
uv run crisk build-canonical --dataset freddie_mac_historical           # 1st attempt FAILED: DuckDB OOM on a DISTINCT aggregate
                                                                        # -> rewritten (D-015/hash only duplicate keys); rerun 6 min
uv run crisk build-labels --dataset freddie_mac_historical              # 13 min
uv run crisk ingest/build-canonical/build-labels/build-features --dataset freddie_mac_combined
uv run crisk train --dataset freddie_mac_combined [--target-id default_90dpd_12m_exdisaster_v1]
uv run crisk evaluate-test --dataset freddie_mac_combined --model-id <each model>   # once each, audited
uv run pytest -q   # 38 passed
uv run ruff check src tests   # clean
```

Regression check: after the canonical rewrite, the sample dataset reproduced identical canonical checks and label flow.

## Headline empirical results (test = 2022–23 vintage, 1.57M loans)

| | Primary target | Ex-disaster target |
|---|---|---|
| ROC-AUC (95% CI) | 0.739 (0.735–0.744) | 0.765 (0.759–0.771) |
| Observed / expected | 1.79 | 1.26 |

The model ranks risk well across 6–7 years, but the absolute PD is not stable; disaster forbearance and note-rate
extrapolation explain much of the gap. Neither model is fit for PD-level use under the pre-registered criteria.

## Known issues

- Artefacts were trained from an uncommitted tree (`code_version = uncommitted`). Commit, then retrain for a reproducible release.
- The note rate acts as a proxy for the market rate level (2023 rates exceed the training range). This needs a rate-spread feature, which depends on macro data.
- No empirical model has been promoted; that needs a named reviewer.
- A publication gate (aggregate-only export with a minimum cell size) is not implemented yet.

## Next actionable tasks

1. Commit the repository; retrain both models from the committed code (same seed) so artefacts carry a commit hash.
2. Phase A3: add the Freddie Mac PMMS 30-year rate (FRED, weekly, published in real time) and build `origination_v2`
   with the rate spread; re-run the same evaluation protocol on a new, pre-registered test design.
3. Phase C: LightGBM challenger, time-blocked uncertainty, segment and proxy analysis, prepayment-censoring sensitivity.
4. Publication gate plus a portfolio write-up based on `docs/results/empirical_baseline_v1.md`.
