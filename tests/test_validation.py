"""Independent validation workflow and the promotion gate (SYNTHETIC data)."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from crisk.audit.store import AuditStore
from crisk.models.train import evaluate_on_test, train_model
from crisk.validation.runner import latest_validation, validate_model


@pytest.fixture(scope="module")
def fresh_model(trained_ws):
    cfg = trained_ws["cfg"]
    store = AuditStore(cfg.audit_db)
    return cfg, store, train_model(cfg, store)["model_id"]


def test_validation_without_evidence_approves_nothing(fresh_model):
    cfg, store, mid = fresh_model
    r = validate_model(cfg, mid, "val-a", store)
    ids = [c["id"] for c in r["checks"]]
    assert ids == [f"V{i:02d}" for i in range(1, 15)]
    st = {c["id"]: c["status"] for c in r["checks"]}
    assert st["V05"] == "fail" and r["approved_for"] == []
    for cid in ("V01", "V03", "V04", "V09", "V10", "V11", "V12", "V13", "V14"):
        assert st[cid] == "pass", (cid, next(c for c in r["checks"] if c["id"] == cid)["evidence"])
    assert any(f["id"] == "AUTO-V05" for f in r["findings"])
    out = cfg.reports_dir / "validation" / mid
    assert (out / "validation_report.json").exists() and (out / "validation_report.md").exists()
    assert store.events(action="model_validated", entity_id=mid)


def test_validation_uses_out_of_time_evidence_and_access_discipline(fresh_model):
    cfg, store, mid = fresh_model
    evaluate_on_test(cfg, mid, store)
    r = validate_model(cfg, mid, "val-a", store)
    st = {c["id"]: c for c in r["checks"]}
    assert st["V05"]["status"] == "pass" and st["V08"]["status"] == "pass"
    evaluate_on_test(cfg, mid, store)  # a second look at the locked test violates access discipline
    r2 = validate_model(cfg, mid, "val-a", store)
    assert {c["id"]: c["status"] for c in r2["checks"]}["V08"] == "fail"
    assert r2["approved_for"] == []


def test_promotion_gate_for_empirical_models(fresh_model, monkeypatch):
    from crisk import cli

    cfg, store, mid = fresh_model
    emp = dataclasses.replace(cfg, data_origin="EMPIRICAL")
    args = SimpleNamespace(model_id=mid, reviewer="rev", rationale="demo", approved_use="demonstration_rank_ordering")
    with pytest.raises(SystemExit, match="Promotion refused"):
        cli.cmd_promote(emp, args)
    # with a validation report approving the use, promotion proceeds and records the validation id
    rep = latest_validation(cfg, mid)
    rep["approved_for"] = ["demonstration_rank_ordering"]
    (cfg.reports_dir / "validation" / mid / "validation_report.json").write_text(json.dumps(rep))
    out = cli.cmd_promote(emp, args)
    assert out["validation_id"] == rep["validation_id"] and out["approved_use"] == "demonstration_rank_ordering"
