"""Scoring service: one code path for online, batch and replay.

``compute_frame`` is pure (no I/O): canonical inputs -> features -> PD, flags and reasons.
``ScoringService.score`` adds identifiers, idempotency and the append-only audit record.
``replay`` recomputes a stored score from its input snapshot and the registered artefact.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crisk.audit.store import AuditStore, IdempotencyConflict
from crisk.config import DatasetConfig, load_yaml
from crisk.features.origination import DERIVED_INPUTS, FeatureSpec, build_feature_frame
from crisk.labels.default_horizon import TargetSpec
from crisk.models.bundle import ModelBundle
from crisk.models.registry import champion, load_bundle
from crisk.models.train import EVIDENCE_NOTE
from crisk.scoring.schemas import REQUIRED_FOR_SCORING
from crisk.utils import canonical_json, code_version, sha256_text, utc_now

REASON_CODES_CONFIG = "configs/reason_codes_v1.yaml"
REPLAY_TOLERANCE = 1e-12


class ModelUnavailableError(RuntimeError):
    pass


@dataclass
class Computed:
    status: str
    pd: float | None
    pd_uncalibrated: float | None
    features: dict
    dq_flags: list[str]
    reason_codes: list[dict]


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def compute_frame(bundle: ModelBundle, records: pd.DataFrame, reason_table: dict, max_reasons: int) -> list[Computed]:
    spec = bundle.feature_spec
    records = records.reset_index(drop=True)
    for c in spec.raw_inputs:
        if c not in records.columns:
            records[c] = None
    req_missing = records[list(REQUIRED_FOR_SCORING)].isna()
    feats = build_feature_frame(records, spec)
    results: list[Computed | None] = [None] * len(records)

    ok = ~req_missing.any(axis=1)
    if ok.any():
        X = feats[ok]
        pd_raw = bundle.predict_raw(X)
        pd_cal = bundle.predict(X)
        contrib = bundle.contributions(X)
        unseen = {c: set(v) for c, v in bundle.unseen_categories(X).items()}
        for pos, i in enumerate(X.index):
            flags = [f"missing_imputed:{c}" for c in spec.raw_inputs
                     if c not in REQUIRED_FOR_SCORING and pd.isna(records.at[i, c])]
            flags += [f"unseen_category:{c}={X.at[i, c]}" for c in unseen if str(X.at[i, c]) in unseen[c]]
            row = contrib.loc[i]
            reasons = []
            for feat, val in row[row > 0].sort_values(ascending=False).head(max_reasons).items():
                entry = reason_table.get(feat, {"code": "RC99", "text": feat, "missing_text": feat})
                raw_missing = all(pd.isna(records.at[i, r]) for r in _inputs_of(feat, spec))
                reasons.append({"code": entry["code"], "feature": feat,
                                "text": entry["missing_text"] if raw_missing else entry["text"],
                                "contribution_logit": round(float(val), 10)})
            results[i] = Computed("scored", float(pd_cal[pos]), float(pd_raw[pos]),
                                  {k: _clean(v) for k, v in X.loc[i].items()}, sorted(flags), reasons)
    for i in records.index[~ok]:
        missing = [c for c in REQUIRED_FOR_SCORING if req_missing.at[i, c]]
        results[i] = Computed("unavailable", None, None, {}, [f"required_missing:{c}" for c in missing], [])
    return results  # type: ignore[return-value]


def _inputs_of(feature: str, spec: FeatureSpec) -> tuple[str, ...]:
    return DERIVED_INPUTS.get(feature, (feature,))


class ScoringService:
    def __init__(self, cfg: DatasetConfig, store: AuditStore, model_id: str | None = None):
        self.cfg = cfg
        self.store = store
        if model_id is None:
            ch = champion(cfg)
            if ch is None:
                raise ModelUnavailableError("No champion model promoted; pass model_id explicitly or promote one")
            model_id = ch["model_id"]
        self.bundle, self.meta = load_bundle(cfg, model_id)
        self.model_id = model_id
        self.tspec = TargetSpec.load(cfg.target_config)
        rc = load_yaml(REASON_CODES_CONFIG)
        self.reason_table = rc["codes"]
        self.max_reasons = int(self.meta["model_config"]["reason_codes"]["max_reasons"])
        self._code_version = canonical_json(code_version())

    # ---- shared fields ---------------------------------------------------------------------------
    def _envelope(self) -> dict:
        return {
            "data_origin": self.meta["data_origin"],
            "evidence_note": EVIDENCE_NOTE[self.meta["data_origin"]],
            "target": {"target_id": self.tspec.target_id, "horizon_months": self.tspec.horizon_months,
                       "definition": " ".join(self.tspec.description.split())},
            "model": {"model_id": self.model_id, "artifact_sha256": self.meta["artifact_sha256"],
                      "calibrated": bool(self.bundle.calibrated),
                      "feature_set_id": self.bundle.feature_spec.feature_set_id},
        }

    def compute(self, applications: list[dict]) -> list[Computed]:
        return compute_frame(self.bundle, pd.DataFrame(applications), self.reason_table, self.max_reasons)

    # ---- online / batch --------------------------------------------------------------------------
    def score(self, application: dict, idempotency_key: str | None = None,
              correlation_id: str | None = None) -> dict:
        return self.score_many([application], [idempotency_key], correlation_id)[0]

    def score_many(self, applications: list[dict], idempotency_keys: list[str | None] | None = None,
                   correlation_id: str | None = None) -> list[dict]:
        keys = idempotency_keys or [None] * len(applications)
        out: list[dict | None] = [None] * len(applications)
        todo = []
        for i, (app, key) in enumerate(zip(applications, keys, strict=True)):
            rh = sha256_text(canonical_json(app))
            if key:
                prev = self.store.find_by_idempotency_key(key)
                if prev:
                    if prev["request_hash"] != rh:
                        raise IdempotencyConflict(key)
                    out[i] = {**_json(prev["response"]), "idempotent_replay": True}
                    continue
            todo.append((i, app, key, rh))
        if todo:
            computed = self.compute([t[1] for t in todo])
            env = self._envelope()
            for (i, app, key, rh), c in zip(todo, computed, strict=True):
                resp = {"score_id": uuid.uuid4().hex, "status": c.status, "scored_at": utc_now(),
                        "correlation_id": correlation_id, **env, "pd": c.pd, "pd_uncalibrated": c.pd_uncalibrated,
                        "dq_flags": c.dq_flags, "reason_codes": c.reason_codes, "idempotent_replay": False}
                self.store.insert_score({
                    "score_id": resp["score_id"], "idempotency_key": key, "request_hash": rh,
                    "correlation_id": correlation_id, "scored_at": resp["scored_at"], "status": c.status,
                    "data_origin": env["data_origin"], "model_id": self.model_id,
                    "model_sha256": self.meta["artifact_sha256"],
                    "feature_set_id": env["model"]["feature_set_id"], "target_id": self.tspec.target_id,
                    "horizon_months": self.tspec.horizon_months, "pd": c.pd, "pd_uncalibrated": c.pd_uncalibrated,
                    "input_snapshot": canonical_json(app), "feature_vector": canonical_json(c.features),
                    "dq_flags": canonical_json(c.dq_flags), "reason_codes": canonical_json(c.reason_codes),
                    "response": canonical_json(resp), "code_version": self._code_version,
                })
                out[i] = resp
        return out  # type: ignore[return-value]


def _json(s: str):
    return json.loads(s)


def replay(cfg: DatasetConfig, store: AuditStore, score_id: str, actor: str = "auditor") -> dict:
    """Recompute a stored score from its snapshot with the exact registered artefact."""
    ev = store.get_score(score_id)
    if ev is None:
        raise KeyError(f"score_id {score_id} not found")
    report: dict = {"score_id": score_id, "model_id": ev["model_id"], "replayed_at": utc_now()}
    if ev["status"] != "scored":
        report.update(match=True, note="unavailable result; nothing to recompute")
    else:
        svc = ScoringService(cfg, store, model_id=ev["model_id"])
        report["artifact_hash_match"] = svc.meta["artifact_sha256"] == ev["model_sha256"]
        c = svc.compute([_json(ev["input_snapshot"])])[0]
        report["stored_pd"] = ev["pd"]
        report["replayed_pd"] = c.pd
        report["abs_diff"] = abs(c.pd - ev["pd"])
        report["features_match"] = canonical_json(c.features) == ev["feature_vector"]
        report["reasons_match"] = canonical_json(c.reason_codes) == ev["reason_codes"]
        report["match"] = bool(report["artifact_hash_match"] and report["abs_diff"] <= REPLAY_TOLERANCE
                               and report["features_match"] and report["reasons_match"])
    store.append_event(actor=actor, action="score_replayed", entity_type="score", entity_id=score_id,
                       details={k: v for k, v in report.items() if k != "replayed_at"})
    return report


def batch_score_split(cfg: DatasetConfig, store: AuditStore, split: str, limit: int | None,
                      model_id: str | None = None) -> pd.DataFrame:
    """Score canonical origination records of one split through the same service path."""
    import duckdb

    svc = ScoringService(cfg, store, model_id=model_id)
    cols = ", ".join(["o.loan_id", *[f"o.{c}" for c in svc.bundle.feature_spec.raw_inputs]])
    lim = f"LIMIT {int(limit)}" if limit else ""
    with duckdb.connect() as con:
        df = con.execute(f"""
            SELECT {cols} FROM read_parquet('{(cfg.processed_dir / 'loan_origination.parquet').as_posix()}') o
            JOIN read_parquet('{(cfg.processed_dir / 'feature_snapshots.parquet').as_posix()}') s USING (loan_id)
            WHERE s.split = '{split}' ORDER BY o.loan_id {lim}""").df()
    apps = [{k: _clean(v) for k, v in r.items()} for r in df.to_dict(orient="records")]
    keys = [f"batch:{svc.model_id}:{a['loan_id']}" for a in apps]
    res = svc.score_many(apps, keys, correlation_id=f"batch-{uuid.uuid4().hex[:8]}")
    return pd.DataFrame({"loan_id": df.loan_id, "score_id": [r["score_id"] for r in res],
                         "status": [r["status"] for r in res], "pd": [r["pd"] for r in res]})
