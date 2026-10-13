# credit-risk-ai-platform

A research and engineering prototype for **origination-time mortgage probability of default (PD)**, built on
the Freddie Mac Single-Family Loan-Level Dataset (SFLLD). It includes auditable, replayable scoring.

> **Status:** the pipeline runs end to end on Freddie Mac SFLLD Release 47: about 3.4M distinct loans and 189.2M monthly
> records from the 2015–2022 samples plus the full 2015 and 2022 vintages. Empirical results:
> [baseline model card](docs/results/empirical_baseline_v1.md) (test AUC 0.74–0.77; PD level not stable over time)
> and [v2/challenger comparison](docs/results/model_comparison_v2.md).
> Research prototype only: not production-ready, not a regulatory model, not a validated credit decision system.

## Evidence classes

| Label | Meaning |
|---|---|
| **SYNTHETIC** | Fictional fixtures (loan IDs start with `X`). Tests system behaviour only. Metrics are not evidence about credit risk. |
| **EMPIRICAL** | Freddie Mac SFLLD research results on funded, acquired loans. *None produced yet.* |

Every table, model artefact, score and API response carries `data_origin`.

## What works now

Raw pipe files are processed as follows: contract validation → quarantine → canonical tables →
12-month 90+ DPD labels (with maturity and censoring) → point-in-time features → temporal splits →
logistic baseline + Platt calibration → audited test evaluation → audited promotion → batch/online scoring
→ replay from score ID. Every step writes to an append-only, hash-chained audit store.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q                                      # 34 tests, synthetic data only
uv run crisk run-slice --dataset synthetic_fixture    # full slice (~45 s), writes reports/synthetic_fixture/slice_summary.json
uv run crisk serve --dataset synthetic_fixture        # http://127.0.0.1:8000/docs  (no auth: localhost only)
```

Step-by-step commands (the same steps work for `--dataset freddie_mac_sample` once files are present):

```bash
uv run crisk fixtures        --dataset synthetic_fixture --n-random 20000
uv run crisk ingest          --dataset synthetic_fixture
uv run crisk build-canonical --dataset synthetic_fixture
uv run crisk build-labels    --dataset synthetic_fixture
uv run crisk build-features  --dataset synthetic_fixture
uv run crisk train           --dataset synthetic_fixture
uv run crisk evaluate-test   --dataset synthetic_fixture --model-id <id>     # audited access to locked test split
uv run crisk promote         --dataset synthetic_fixture --model-id <id> --reviewer <name> --rationale "<why>"
uv run crisk score-batch     --dataset synthetic_fixture --split test --limit 200
uv run crisk replay          --dataset synthetic_fixture --score-id <score_id>
uv run crisk verify-audit    --dataset synthetic_fixture
uv run crisk rollback        --dataset synthetic_fixture --reviewer <name> --rationale "<why>"
```

Real data (after downloading under Freddie Mac's terms; see the acquisition doc):

```bash
uv run crisk unpack          --dataset freddie_mac_historical   # nested full-volume zips -> extracted/
uv run crisk ingest          --dataset freddie_mac_historical   # also: freddie_mac_sample
uv run crisk ingest          --dataset freddie_mac_combined     # composes validated files (train/cal/test sources)
uv run crisk build-features  --dataset freddie_mac_combined --feature-config configs/features/origination_v2.yaml
uv run crisk train           --dataset freddie_mac_combined --feature-config configs/features/origination_v2.yaml \
                             --model-config configs/models/lightgbm_v1.yaml [--target-id default_90dpd_12m_exdisaster_v1]
uv run crisk export-results  --dataset freddie_mac_combined --model-id <id1,id2,...>   # aggregate-only publication gate
```

## Documentation

- [Project status](docs/PROJECT_STATUS.md): what is implemented, tested, blocked or proposed
- [Phase plan and acceptance gates](docs/phase_plan.md)
- [Architecture](docs/architecture.md)
- [Target specification](docs/target_specification.md) · [Feature dictionary](docs/feature_dictionary.md)
- [Assumptions register and open questions](docs/assumptions_register.md) · [Decision log](docs/decision_log.md)
- [Freddie Mac acquisition steps](docs/data_acquisition_freddie_mac.md) · [Source manifest](configs/sources/source_manifest.yaml)
- [Data handling and security](docs/data_handling_and_security.md)

## Data terms

Freddie Mac SFLLD data is used under Freddie Mac's terms for internal research. Raw data, derived loan-level
data, model artefacts and audit stores are git-excluded and must not be redistributed. Any attempt to
identify individuals is prohibited. Source: <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>.
