# credit-risk-ai-platform

A research and engineering prototype for **origination-time mortgage probability of default (PD)**, built on
the Freddie Mac Single-Family Loan-Level Dataset (SFLLD). It includes auditable, replayable scoring.

> **Status:** the pipeline runs end to end on Freddie Mac SFLLD Release 47: about 3.4M distinct loans and 189.2M monthly
> records from the 2015–2022 samples plus the full 2015 and 2022 vintages. Empirical results:
> [baseline model card](docs/results/empirical_baseline_v1.md) (2022–23 test AUC 0.74–0.77),
> [v2/challenger comparison](docs/results/model_comparison_v2.md), and the
> [fresh 2023–24 holdout](docs/results/holdout_2023_2024.md) (AUC 0.81–0.82; absolute PD not stable over time), and a
> pre-registered [recalibration backtest](docs/results/recalibration_v3.md) and
> [through-the-cycle study](docs/results/ttc_level_v4.md) (both negative: the PD level is shock-driven; use the model for ranking),
> plus [fairness review](docs/results/fairness_review_v1.md), [policy simulation](docs/results/policy_simulation_v1.md),
> [monitoring](docs/results/monitoring_v1.md) and [LGD/expected loss](docs/results/lgd_el_v1.md).
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

**Documents and copilot (Phases G/H, mocked LLM first).** Fictional underwriting packets (SYNTHETIC) are extracted
with page and quote references, then checked for arithmetic, cross-document and completeness problems and for
embedded instructions. They go through a human correction workflow, and only validated packets pass a feature gate.
The analyst copilot answers from approved, versioned project documents and read-only tools. It filters by role
before retrieval, cites verbatim quotes, verifies numbers, refuses injection and abstains without evidence. The LLM
is provider-neutral: a deterministic mock by default, with an optional Claude backend (`uv sync --group llm`, key in
the environment). Results, including the failures: [docs/results/document_ai_copilot_v1.md](docs/results/document_ai_copilot_v1.md).

```
uv run crisk copilot-ask --dataset freddie_mac_combined --role analyst --question "What is the base scenario 12-month expected loss rate?"
uv run crisk docai-eval --dataset synthetic_fixture --split dev
```

## One-command demo (Docker, SYNTHETIC data, no credentials)

```bash
docker compose up --build        # first run: about 5-10 minutes (pipeline + model + monitoring)
docker compose logs init         # prints the demo analyst and auditor API keys
```

Open the analyst UI at http://localhost:8501 (paste the analyst key) and the API docs at http://localhost:8000/docs.
The demo trains and promotes a model on the deterministic synthetic fixture. Freddie Mac data is never in the image.
Stop with `docker compose down` (add `-v` to delete the demo volumes).

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

Serving, review and validation (champion = M1 logistic without `property_state`, approved for **rank-ordering demonstrations only**):

```bash
uv run crisk create-api-key --dataset freddie_mac_combined --name alice --role analyst   # key shown once
uv run crisk serve          --dataset freddie_mac_combined                              # http://127.0.0.1:8000/docs
CRISK_API_URL=http://127.0.0.1:8000 uv run --group ui streamlit run src/crisk/ui/app.py  # analyst interface
uv run crisk validate       --dataset freddie_mac_combined --model-id <id> --validator <name>
```

Roles: `analyst` (score, read, review), `auditor` (read, replay), `validator` (read, replay), `modeller` (score),
`administrator` (all API actions). No role can promote models or approve loans through the API.

## Documentation

- [Project status](docs/PROJECT_STATUS.md): what is implemented, tested, blocked or proposed
- [Phase plan and acceptance gates](docs/phase_plan.md)
- [Architecture](docs/architecture.md)
- [Target specification](docs/target_specification.md) · [Feature dictionary](docs/feature_dictionary.md)
- [Assumptions register and open questions](docs/assumptions_register.md) · [Decision log](docs/decision_log.md)
- [Freddie Mac acquisition steps](docs/data_acquisition_freddie_mac.md) · [Source manifest](configs/sources/source_manifest.yaml)
- [Data handling and security](docs/data_handling_and_security.md)
- [Document AI and copilot plan](docs/document_ai_copilot_plan_v1.md) · [results](docs/results/document_ai_copilot_v1.md)

## Data terms

Freddie Mac SFLLD data is used under Freddie Mac's terms for internal research. Raw data, derived loan-level
data, model artefacts and audit stores are git-excluded and must not be redistributed. Any attempt to
identify individuals is prohibited. Source: <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>.
