# Architecture (current slice)

Laptop-first, single process, no external services. Modules are separated by responsibility. There are no
microservices yet.

```mermaid
flowchart LR
  subgraph Sources
    FM[Freddie Mac SFLLD<br/>sample files<br/>(manual download)]
    SYN[Synthetic fixture<br/>generator (seeded)]
  end
  subgraph Ingestion["ingestion + contracts"]
    P[DuckDB structural parse<br/>32-field contract] --> T[typing, sentinels,<br/>field rules]
    T -->|valid| I[(interim parquet<br/>+ lineage cols)]
    T -->|rejects/issues| Q[(quarantine)]
  end
  subgraph Quality["quality"]
    C[dedup, conflicts,<br/>orphans, warnings] --> CAN[(loan_origination<br/>loan_monthly_performance)]
  end
  subgraph Modelling
    L[labels:<br/>default_90dpd_12m_v1] --> O[(loan_outcomes)]
    F[features origination_v1<br/>+ temporal splits] --> FS[(feature_snapshots<br/>+ split manifest)]
    TR[train: baseline, logistic,<br/>Platt on calibration split] --> REG[(model registry<br/>sha256 + metadata)]
  end
  subgraph Serving
    SVC[ScoringService<br/>compute_frame] --> API[FastAPI /v1/score]
    SVC --> B[batch scoring]
    R[replay] --> SVC
  end
  AUD[(append-only audit<br/>SQLite + hash chain)]
  FM --> P
  SYN --> P
  I --> C
  CAN --> L
  CAN --> F
  O --> F
  FS --> TR
  REG -->|verified load| SVC
  SVC --> AUD
  TR --> AUD
  REG -->|promote / rollback<br/>named reviewer| AUD
```

## Module map

| Package | Responsibility | Phase |
|---|---|---|
| `crisk.contracts` | Versioned SFLLD layouts, sentinels and availability classes | A |
| `crisk.ingestion` | Raw → typed interim Parquet, quarantine, checksum idempotence | A |
| `crisk.quality` | Canonical tables, dedup, relationships, warning checks | A |
| `crisk.fixtures` | Deterministic SYNTHETIC files and answer key | A |
| `crisk.labels` | Target specification (SQL) | B |
| `crisk.features` | Row-wise feature builder, leakage guard, temporal splits, snapshots | B |
| `crisk.evaluation` | Discrimination and calibration metrics | C |
| `crisk.models` | Model bundle, training, locked test evaluation, registry, promotion/rollback | C/D |
| `crisk.scoring` | Contract-derived schemas, service, replay, batch, API | F |
| `crisk.audit` | Append-only store and hash chain | F |
| `crisk.macro` | PMMS market-rate table, point-in-time join | A3 |
| `crisk.security` | API keys, roles, permissions | F |
| `crisk.ui` | Analyst interface (Streamlit) and its stdlib API client | F |
| `crisk.validation` | Independent validation runner, release criteria, findings | D |
| `crisk.publish` | Publication gate for aggregate results | — |
| *(planned)* `crisk.policies`, `crisk.monitoring`, `crisk.documents`, `crisk.copilot` | | E, I, G, H |

## Data locations (all git-excluded)

```
data/raw/<dataset>/                 unmodified source files
data/interim/<dataset>/             typed parquet per file, _ingestion manifests + log, _quarantine
data/processed/<dataset>/           canonical tables, loan_outcomes, feature_snapshots, reports (json)
artifacts/<dataset>/models/<id>/    model.joblib + metadata.json; champion.json
artifacts/<dataset>/audit.sqlite    scoring events + audit chain
reports/<dataset>/                  test evaluation, batch scores, slice summary
```

## Offline/online parity

There is one feature function (`build_feature_frame`) and one scoring function (`compute_frame`), used by
training, batch scoring, the API and replay. `test_offline_online_parity` checks that batch predictions from
feature snapshots equal service predictions from raw records, to within 1e-12.
