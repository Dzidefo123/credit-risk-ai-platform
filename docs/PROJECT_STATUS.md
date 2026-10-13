# Project status

_Last updated: 2026-10-13 (session 1, continued)._

## Summary

| Capability | State |
|---|---|
| Repository: committed on `main`, LF pinned, CI config, data/artefacts/publish git-excluded | **implemented** |
| SFLLD contracts (January 2026 and Release 47 layouts); ingestion, quarantine, unpack, compose; out-of-core DuckDB | **implemented and tested**; run on 192.7M raw lines (189.2M monthly performance records) |
| Primary and ex-disaster targets | **implemented and tested**; run on real data |
| Market-rate feature: PMMS point-in-time join, `origination_v2` | **implemented and tested** |
| Models: logistic (M1/M2), LightGBM challenger (M3/M4), Platt calibration, TreeSHAP/linear reasons | **implemented and tested**; trained on real data |
| v1 baseline evaluated once on the locked 2022–23 test | **done**: [results/empirical_baseline_v1.md](results/empirical_baseline_v1.md) |
| v2 / challenger comparison on out-of-time development data | **done**: [results/model_comparison_v2.md](results/model_comparison_v2.md) |
| Fresh locked holdout (sample 2023–24, FPD 2023-07..2025-03) | **done**, once per model: [results/holdout_2023_2024.md](results/holdout_2023_2024.md). AUC 0.81–0.82; observed/expected 1.78 (M4) / 0.62 (M1) / 0.67 (M1 ex-disaster) |
| v3 walk-forward recalibration (pre-registered) | **done**: [results/recalibration_v3.md](results/recalibration_v3.md). Negative result: R0 kept. A Newton-divergence bug was found and fixed (D-025) |
| v4 through-the-cycle level (pre-registered; exploratory) | **done**: [results/ttc_level_v4.md](results/ttc_level_v4.md). Negative result: L0 kept. 2025 vintage reserved for the Release 48+ confirmatory test |
| Publication gate `crisk export-results` | **implemented and tested**; two owner-reviewed exports published |
| Scoring, replay, append-only audit, API | implemented and tested; no empirical model promoted (needs a named reviewer) |
| Segment/proxy/fairness review, validation runner, policy engine, UI, monitoring, Docker, auth | **proposed** |

## This session's commands (all succeeded unless noted)

```
git commit (initial + 4 follow-ups)          # repo-local identity kalom <calormenu@gmail.com>
uv run crisk train --dataset freddie_mac_combined [--target-id ...]   # retrain from commit 71a1aea: coefficients identical (diff 0.0)
rm data/raw/freddie_mac/historical/extracted/*.txt                    # freed 19 GB; zips + _unpack_manifest.json kept
curl FRED MORTGAGE30US/MORTGAGE15US + 3 ALFRED vintages + PMMS xlsx   # vintage check: 0 revisions
uv run crisk build-features --dataset freddie_mac_combined --feature-config configs/features/origination_v2.yaml
uv run crisk train ... (M2, M3, M4 x primary/ex-disaster)            # about 1 min each
uv run crisk export-results --dataset freddie_mac_combined --model-id <10 ids>
uv run pytest -q   # 44 passed
uv run ruff check src tests   # clean
```

Failures and fixes this session:
- JSON was written with CRLF on Windows, so byte-compared fixtures would fail on Linux CI. Fixed: LF writes plus `.gitattributes`.
- Publication gate v1: masked event counts were recoverable from `observed_rate × n` (113 rows). Fixed, and covered by a test.
- A stray `cat` in a background command hung; it was stopped.

## Key results

- v1 test (2022–23, 1.57M loans): AUC 0.739 primary / 0.765 ex-disaster; observed/expected 1.79 / 1.26. Rank-ordering
  works; absolute PD is not stable.
- v2 development comparison (2017–19 calibration split): the rate spread improves logistic AUC by +0.010
  (CI [+0.003, +0.016]). LightGBM with the spread gives +0.017 (CI [+0.006, +0.029]). Neither fixes level drift.
  Pre-registered selection: M4 (primary), M1 (ex-disaster). The rule's flaw (point threshold) is recorded, not overridden.

## Next actionable tasks

1. Done: pushed to https://github.com/Dzidefo123/credit-risk-ai-platform (CI green). Both owner-reviewed exports
   are published. New results (v3) are in `docs/results/` only; export them via the gate after review if wanted.
2. When Freddie Mac publishes Release 48+, download its edition of `sample_2025.zip` and run the pre-registered
   confirmatory test of L0 (once). The Release 47 edition on disk is excluded from ingestion.
3. Optional future round (needs its own pre-registration and unseen data): crisis-excluded TTC anchor, macro-conditional
   level, or early-performance proxy. Segment and proxy review (`property_state`).
4. Promote M1 (logistic) for demonstration scoring with a named reviewer. Phase D validation runner; Phase F hardening
   (auth, Docker, analyst UI).
