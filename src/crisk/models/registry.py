"""File-based model registry with content hashes and a governed champion pointer.

Layout: ``artifacts/<dataset>/models/<model_id>/{model.joblib, metadata.json}`` and
``artifacts/<dataset>/models/champion.json``. A model is only usable for scoring after it
has been loaded with a verified SHA-256. The champion pointer changes only through
``promote``/``rollback``, which require a named reviewer and rationale and are written
to the audit chain. (MLflow is a candidate replacement; see decision D-005.)
"""

from __future__ import annotations

from pathlib import Path

import joblib

from crisk.audit.store import AuditStore
from crisk.config import DatasetConfig
from crisk.models.bundle import ModelBundle
from crisk.utils import read_json, sha256_file, utc_now, write_json


class ModelIntegrityError(RuntimeError):
    pass


def model_dir(cfg: DatasetConfig, model_id: str) -> Path:
    return cfg.models_dir / model_id


def save_bundle(cfg: DatasetConfig, bundle: ModelBundle, model_id: str, metadata: dict) -> dict:
    d = model_dir(cfg, model_id)
    d.mkdir(parents=True, exist_ok=False)
    path = d / "model.joblib"
    joblib.dump(bundle, path)
    metadata = {**metadata, "model_id": model_id, "artifact": "model.joblib", "artifact_sha256": sha256_file(path)}
    write_json(d / "metadata.json", metadata)
    return metadata


def load_metadata(cfg: DatasetConfig, model_id: str) -> dict:
    return read_json(model_dir(cfg, model_id) / "metadata.json")


def load_bundle(cfg: DatasetConfig, model_id: str) -> tuple[ModelBundle, dict]:
    meta = load_metadata(cfg, model_id)
    path = model_dir(cfg, model_id) / meta["artifact"]
    actual = sha256_file(path)
    if actual != meta["artifact_sha256"]:
        raise ModelIntegrityError(f"{model_id}: artefact hash {actual} != registered {meta['artifact_sha256']}")
    return joblib.load(path), meta


def list_models(cfg: DatasetConfig) -> list[str]:
    if not cfg.models_dir.exists():
        return []
    return sorted(p.name for p in cfg.models_dir.iterdir() if (p / "metadata.json").exists())


def champion(cfg: DatasetConfig) -> dict | None:
    p = cfg.models_dir / "champion.json"
    return read_json(p) if p.exists() else None


APPROVED_USES = {
    "demonstration_rank_ordering": "Demonstration only: use risk deciles / relative ranking; absolute PD is not "
                                   "validated for pricing, expected loss or PD thresholds.",
    "pd_level": "Absolute PD approved for use (requires passing the PD-level validation criteria).",
}


def reference_deciles(bundle: ModelBundle, X) -> list[float]:
    """Nine cut-offs of the calibrated PD on a reference population; decile 10 = highest risk."""
    import numpy as np

    return [float(v) for v in np.quantile(bundle.predict(X), np.linspace(0.1, 0.9, 9))]


def promote(cfg: DatasetConfig, model_id: str, reviewer: str, rationale: str, store: AuditStore,
            action: str = "model_promoted", approved_use: str = "demonstration_rank_ordering",
            extra: dict | None = None) -> dict:
    if not reviewer.strip() or not rationale.strip():
        raise ValueError("Promotion requires a named reviewer and a rationale")
    if approved_use not in APPROVED_USES:
        raise ValueError(f"approved_use must be one of {sorted(APPROVED_USES)}")
    load_bundle(cfg, model_id)  # integrity check before promotion
    meta = load_metadata(cfg, model_id)
    previous = champion(cfg)
    pointer = {"model_id": model_id, "artifact_sha256": meta["artifact_sha256"], "promoted_at": utc_now(),
               "reviewer": reviewer, "rationale": rationale, "data_origin": meta["data_origin"],
               "approved_use": approved_use, "approved_use_text": APPROVED_USES[approved_use],
               "previous_model_id": previous["model_id"] if previous else None, **(extra or {})}
    event = store.append_event(actor=reviewer, action=action, entity_type="model", entity_id=model_id,
                               details=pointer)
    pointer["audit_event_id"] = event["event_id"]
    write_json(cfg.models_dir / "champion.json", pointer)
    return pointer


def rollback(cfg: DatasetConfig, reviewer: str, rationale: str, store: AuditStore) -> dict:
    cur = champion(cfg)
    if not cur or not cur.get("previous_model_id"):
        raise ValueError("No previous champion to roll back to")
    return promote(cfg, cur["previous_model_id"], reviewer, rationale, store, action="model_rolled_back",
                   approved_use=cur.get("approved_use", "demonstration_rank_ordering"))


def refresh_deciles(cfg: DatasetConfig, cutoffs: list[float], reference: str, reviewer: str, rationale: str,
                    store: AuditStore) -> dict:
    """Replace the champion's risk-decile cut-offs (rank reference only; model and calibration unchanged).

    Ranking against the current applicant population needs no outcomes, so it can track drift without the
    information lag that defeats PD recalibration. A named reviewer is required and the change is audited; earlier
    cut-offs are kept in ``decile_history``.
    """
    if not reviewer.strip() or not rationale.strip():
        raise ValueError("A decile refresh requires a named reviewer and a rationale")
    if len(cutoffs) != 9 or any(b <= a for a, b in zip(cutoffs, cutoffs[1:], strict=False)):
        raise ValueError("Expected 9 strictly increasing cut-offs")
    cur = champion(cfg)
    if cur is None:
        raise ValueError("No champion to update")
    history = [*cur.get("decile_history", []), {"cutoffs": cur.get("risk_decile_cutoffs"),
                                                 "reference": cur.get("risk_decile_reference"),
                                                 "replaced_at": utc_now()}]
    event = store.append_event(actor=reviewer, action="decile_cutoffs_refreshed", entity_type="model",
                               entity_id=cur["model_id"], details={"reference": reference, "cutoffs": cutoffs,
                                                                   "rationale": rationale})
    pointer = {**cur, "risk_decile_cutoffs": [float(c) for c in cutoffs], "risk_decile_reference": reference,
               "decile_refreshed_at": utc_now(), "decile_refresh_reviewer": reviewer,
               "decile_refresh_audit_event_id": event["event_id"], "decile_history": history}
    write_json(cfg.models_dir / "champion.json", pointer)
    return pointer
