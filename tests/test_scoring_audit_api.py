"""End-to-end behaviour on SYNTHETIC data: training artefacts, scoring, parity, replay, audit, API."""

import sqlite3

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from crisk.audit.store import AuditStore, IdempotencyConflict
from crisk.models.registry import ModelIntegrityError, champion, load_bundle, promote, rollback
from crisk.models.train import evaluate_on_test, train_model
from crisk.scoring.api import create_app
from crisk.scoring.service import ScoringService, batch_score_split, replay

APP = dict(credit_score=700, orig_dti=40, orig_ltv=90, orig_cltv=90, orig_interest_rate=4.5, orig_loan_term=360,
           mi_pct=25, num_units=1, orig_upb=300000.0, num_borrowers=1, occupancy_status="I", channel="B",
           property_type="SF", loan_purpose="C", first_time_homebuyer_flag=None, property_state="TX")


@pytest.fixture(scope="module")
def promoted(trained_ws):
    cfg, store, meta = trained_ws["cfg"], trained_ws["store"], trained_ws["meta"]
    promote(cfg, meta["model_id"], "test-reviewer", "unit test", store)
    return trained_ws


def test_model_metadata_records_lineage(trained_ws):
    meta = trained_ws["meta"]
    assert meta["data_origin"] == "SYNTHETIC" and "not evidence" in meta["evidence_note"]
    for k in ("seed", "feature_columns", "model_config", "code_version", "environment", "artifact_sha256",
              "split_manifest", "training_data"):
        assert k in meta
    assert meta["test_set_evaluated"] is False
    assert "test" not in meta["metrics"]  # training never touches the locked test split
    assert not any(c in meta["feature_columns"] for c in ("servicer_name", "seller_name"))


def test_tampered_artifact_is_refused(trained_ws, tmp_path):
    cfg, meta = trained_ws["cfg"], trained_ws["meta"]
    path = cfg.models_dir / meta["model_id"] / "model.joblib"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"x")
        with pytest.raises(ModelIntegrityError):
            load_bundle(cfg, meta["model_id"])
    finally:
        path.write_bytes(original)


def test_offline_online_parity(promoted):
    """Batch predictions from feature snapshots == service predictions from raw application records."""
    cfg = promoted["cfg"]
    bundle, _ = load_bundle(cfg, promoted["meta"]["model_id"])
    snap = pd.read_parquet(cfg.processed_dir / "feature_snapshots__origination_v1.parquet")
    snap = snap[snap.split == "test"].sort_values("loan_id").head(50)
    offline = bundle.predict(snap)
    scored = batch_score_split(cfg, promoted["store"], "test", 50)
    assert list(scored.loan_id) == list(snap.loan_id)
    np.testing.assert_allclose(scored.pd.to_numpy(dtype=float), offline, rtol=0, atol=1e-12)


def test_scoring_record_and_replay(promoted):
    cfg, store = promoted["cfg"], promoted["store"]
    svc = ScoringService(cfg, store)
    r = svc.score(APP, correlation_id="cid-1")
    assert r["status"] == "scored" and 0 < r["pd"] < 1
    assert r["target"]["horizon_months"] == 12 and r["data_origin"] == "SYNTHETIC"
    assert r["model"]["artifact_sha256"] == promoted["meta"]["artifact_sha256"]
    assert r["reason_codes"] and all(rc["contribution_logit"] > 0 for rc in r["reason_codes"])
    rep = replay(cfg, store, r["score_id"])
    assert rep["match"] and rep["abs_diff"] == 0.0
    assert store.events(action="score_replayed", entity_id=r["score_id"])


def test_missing_required_input_returns_unavailable_not_a_score(promoted):
    svc = ScoringService(promoted["cfg"], promoted["store"])
    r = svc.score({**APP, "orig_upb": None})
    assert r["status"] == "unavailable" and r["pd"] is None
    assert "required_missing:orig_upb" in r["dq_flags"]


def test_missing_optional_input_is_imputed_and_flagged(promoted):
    svc = ScoringService(promoted["cfg"], promoted["store"])
    r = svc.score({**APP, "credit_score": None})
    assert r["status"] == "scored" and "missing_imputed:credit_score" in r["dq_flags"]


def test_idempotency(promoted):
    svc = ScoringService(promoted["cfg"], promoted["store"])
    n0 = promoted["store"].count_scores()
    a = svc.score(APP, idempotency_key="idem-1")
    b = svc.score(APP, idempotency_key="idem-1")
    assert a["score_id"] == b["score_id"] and b["idempotent_replay"]
    assert promoted["store"].count_scores() == n0 + 1
    with pytest.raises(IdempotencyConflict):
        svc.score({**APP, "orig_dti": 20}, idempotency_key="idem-1")


def test_audit_store_is_append_only_and_tamper_evident(promoted):
    store = promoted["store"]
    store.append_event("tester", "noop", "test", "x")
    con = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE audit_events SET actor = 'mallory'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM scoring_events")
    assert store.verify_chain()[0]
    # Bypass the triggers (direct file tampering) -> hash chain detects it.
    con.execute("DROP TRIGGER audit_events_no_update")
    con.execute("UPDATE audit_events SET actor = 'mallory' WHERE seq = 1")
    con.commit()
    con.close()
    assert store.verify_chain()[0] is False


def test_promotion_requires_reviewer_and_rollback_restores_previous(trained_ws, tmp_path):
    cfg = trained_ws["cfg"]
    store = AuditStore(tmp_path / "audit.sqlite")
    with pytest.raises(ValueError):
        promote(cfg, trained_ws["meta"]["model_id"], "", "no reviewer", store)
    first = trained_ws["meta"]["model_id"]
    second = train_model(cfg, store)["model_id"]
    promote(cfg, first, "reviewer-a", "first", store)
    promote(cfg, second, "reviewer-b", "second", store)
    assert champion(cfg)["model_id"] == second
    rollback(cfg, "reviewer-c", "calibration issue", store)
    assert champion(cfg)["model_id"] == first
    assert [e["action"] for e in store.events()][-3:] == ["model_promoted", "model_promoted", "model_rolled_back"]


def test_test_set_access_is_audited(trained_ws, tmp_path):
    store = AuditStore(tmp_path / "audit2.sqlite")
    r1 = evaluate_on_test(trained_ws["cfg"], trained_ws["meta"]["model_id"], store)
    r2 = evaluate_on_test(trained_ws["cfg"], trained_ws["meta"]["model_id"], store)
    assert r1["prior_test_evaluations"] == 0 and r2["prior_test_evaluations"] == 1


def _client(cfg, tmp_path, store=None):
    """App + one key per role (keys live in a temp key store)."""
    from crisk.security.auth import ROLES, KeyStore

    ks = KeyStore(tmp_path / "keys.json")
    st = store or AuditStore(cfg.audit_db)
    hdr = {r: {"Authorization": f"Bearer {ks.create(f'{r}-user', r, st)[0]}"} for r in ROLES}
    return TestClient(create_app(cfg, keys=ks)), hdr, ks


def test_api_contract(promoted, tmp_path):
    client, H, _ = _client(promoted["cfg"], tmp_path)
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["data_origin"] == "SYNTHETIC"

    r = client.post("/v1/score", json={"application": APP}, headers={**H["analyst"], "X-Correlation-ID": "abc"})
    assert r.status_code == 200 and r.headers["X-Correlation-ID"] == "abc"
    body = r.json()
    assert body["status"] == "scored" and body["correlation_id"] == "abc" and body["requested_by"] == "analyst-user"

    post = lambda app, **kw: client.post("/v1/score", json={"application": app, **kw}, headers=H["analyst"])  # noqa: E731
    bad = post({**APP, "credit_score": 1200})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_request"
    assert post({**APP, "occupancy_status": "Z"}).status_code == 422
    # Provider sentinel codes are not accepted online: "not available" is sent as null.
    assert post({**APP, "first_time_homebuyer_flag": "9"}).status_code == 422
    assert post({**APP, "servicer_name": "X"}).status_code == 422  # post-origination fields are not accepted
    un = post({**APP, "orig_interest_rate": None})
    assert un.status_code == 200 and un.json()["status"] == "unavailable" and un.json()["pd"] is None
    post(APP, idempotency_key="api-1")
    assert post({**APP, "orig_dti": 10}, idempotency_key="api-1").status_code == 409


def test_api_authentication_and_roles(promoted, tmp_path):
    client, H, ks = _client(promoted["cfg"], tmp_path)
    # no key / bad key / malformed key -> 401
    assert client.post("/v1/score", json={"application": APP}).status_code == 401
    assert client.get("/v1/scores", headers={"Authorization": "Bearer crk_deadbeef_nope"}).status_code == 401
    assert client.get("/v1/scores", headers={"X-API-Key": "not-a-key"}).json()["error"]["code"] == "unauthenticated"
    # role permissions
    assert client.post("/v1/score", json={"application": APP}, headers=H["auditor"]).status_code == 403
    assert client.post("/v1/score", json={"application": APP}, headers=H["validator"]).status_code == 403
    sid = client.post("/v1/score", json={"application": APP}, headers=H["modeller"]).json()["score_id"]
    assert client.get("/v1/scores", headers=H["modeller"]).status_code == 403
    assert client.get(f"/v1/scores/{sid}", headers=H["auditor"]).status_code == 200
    assert client.post(f"/v1/scores/{sid}/replay", headers=H["analyst"]).status_code == 403
    rep = client.post(f"/v1/scores/{sid}/replay", headers=H["auditor"])
    assert rep.status_code == 200 and rep.json()["match"]
    assert client.get("/v1/model", headers=H["analyst"]).json()["caller"]["role"] == "analyst"
    # revoked key -> 401; key store never contains the plain key
    kid = H["analyst"]["Authorization"].split("_")[1]
    ks.revoke(kid, AuditStore(promoted["cfg"].audit_db))
    assert client.get("/v1/scores", headers=H["analyst"]).status_code == 401
    raw = (tmp_path / "keys.json").read_text()
    assert H["auditor"]["Authorization"][7:] not in raw and "sha256" in raw


def test_review_workflow(promoted, tmp_path):
    client, H, _ = _client(promoted["cfg"], tmp_path)
    sid = client.post("/v1/score", json={"application": APP}, headers=H["analyst"]).json()["score_id"]
    listed = client.get("/v1/scores?limit=500", headers=H["analyst"]).json()["items"]
    assert any(i["score_id"] == sid and i["review_status"] == "pending" for i in listed)
    bad = client.post(f"/v1/scores/{sid}/reviews", json={"action": "override", "rationale": "short"},
                      headers=H["analyst"])
    assert bad.status_code == 422  # rationale too short and override decision missing
    ok = client.post(f"/v1/scores/{sid}/reviews", headers=H["analyst"],
                     json={"action": "override", "override_decision": "refer",
                           "rationale": "Income documentation inconsistent; refer to senior underwriter."})
    assert ok.status_code == 201 and ok.json()["reviewer"] == "analyst-user"
    assert client.post(f"/v1/scores/{sid}/reviews", headers=H["auditor"],
                       json={"action": "agree", "rationale": "auditors cannot review decisions"}).status_code == 403
    detail = client.get(f"/v1/scores/{sid}", headers=H["auditor"]).json()
    assert detail["review_status"] == "override" and detail["override_decision"] == "refer"
    assert len(detail["reviews"]) == 1
    # reviews are append-only
    import sqlite3

    con = sqlite3.connect(promoted["cfg"].audit_db)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE review_events SET action = 'agree'")
    con.close()


def test_api_without_model_reports_not_ready(trained_ws, tmp_path):
    cfg = trained_ws["cfg"].with_root(tmp_path)  # empty workspace: no champion
    client, H, _ = _client(cfg, tmp_path)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    r = client.post("/v1/score", json={"application": APP}, headers=H["analyst"])
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_publication_gate(trained_ws, tmp_path):
    import dataclasses
    import json

    from crisk.publish.export import PublicationGateError, export_results

    cfg, mid = trained_ws["cfg"], trained_ws["meta"]["model_id"]
    with pytest.raises(PublicationGateError, match="EMPIRICAL"):
        export_results(cfg, [mid], out_root=tmp_path)
    emp = dataclasses.replace(cfg, data_origin="EMPIRICAL")
    res = export_results(emp, [mid], out_root=tmp_path / "ok")
    out = __import__("pathlib").Path(res["out_dir"])
    assert (out / "export_manifest.json").exists()
    assert not list(out.rglob("*.parquet")) and not list(out.rglob("*.joblib"))
    label = json.loads((out / "label_report.json").read_text())
    assert any("suppressed" in row for row in label["by_fpd_year"])  # small synthetic cohorts suppressed
    # a masked event count cannot be recovered from a rate shown next to it
    from crisk.publish.export import _suppress

    row = _suppress({"bin": 0, "n": 8641, "defaults": 2, "observed_rate": 2 / 8641, "mean_pd": 0.0001},
                    {"suppressed_rows": 0, "masked_event_counts": 0})
    assert row["defaults"] == "<10" and isinstance(row["observed_rate"], str) and row["mean_pd"] == 0.0001
    # a loan-identifier-shaped string anywhere aborts the export
    bad = cfg.reports_dir / mid
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "leak.json").write_text(json.dumps({"example": "X19Q10000001"}))
    with pytest.raises(PublicationGateError, match="loan-identifier"):
        export_results(emp, [mid], out_root=tmp_path / "bad")
    (bad / "leak.json").unlink()


def test_holdout_evaluation_is_audited_separately(trained_ws, tmp_path):
    from crisk.models.train import evaluate_on_test

    store = AuditStore(tmp_path / "audit_holdout.sqlite")
    cfg, mid = trained_ws["cfg"], trained_ws["meta"]["model_id"]
    r = evaluate_on_test(cfg, mid, store, holdout_cfg=cfg)
    assert r["evaluation_dataset"] == cfg.dataset_id and r["prior_test_evaluations"] == 0
    assert [e["action"] for e in store.events()] == ["holdout_evaluated"]
    assert (cfg.reports_dir / mid / f"evaluation_holdout__{cfg.dataset_id}.json").exists()


def test_ui_client_against_api(promoted, tmp_path):
    """The analyst UI's client talks to the real API (in-process transport) and surfaces API errors."""
    from crisk.ui.client import ApiClient, ApiError

    client, H, _ = _client(promoted["cfg"], tmp_path)

    def transport(method, url, headers, body):
        path = url.split("http://testserver", 1)[1]
        r = client.request(method, path, headers=headers, content=body)
        return r.status_code, r.content

    analyst = ApiClient("http://testserver", H["analyst"]["Authorization"][7:], transport)
    assert analyst.model()["caller"]["role"] == "analyst"
    sid = analyst.score(APP)["score_id"]
    assert any(i["score_id"] == sid for i in analyst.scores(limit=500))
    analyst.review(sid, "escalate", "Unusual combination of DTI and LTV; second opinion.")
    assert analyst.score_detail(sid)["review_status"] == "escalate"
    with pytest.raises(ApiError) as e:
        analyst.replay(sid)
    assert e.value.status == 403
    with pytest.raises(ApiError) as e:
        ApiClient("http://testserver", "crk_bad_key", transport).model()
    assert e.value.status == 401
