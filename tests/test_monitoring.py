"""Monitoring: PSI, de-duplicated alerts, delayed outcomes, governed decile refresh, API access."""

import numpy as np
import pandas as pd
import pytest

from crisk.audit.store import AuditStore
from crisk.monitoring.monitor import monitor, psi

TH = {"psi_watch": 0.10, "psi_alert": 0.25, "missing_rate_shift_alert": 0.05, "min_monthly_volume": 100,
      "auc_drop_alert": 0.05, "observed_expected_band": [0.8, 1.25]}


def test_psi_basics():
    e = np.array([0.1] * 10)
    assert psi(e, e) == pytest.approx(0.0, abs=1e-12)
    assert psi(e, np.array([0.05] * 9 + [0.55])) > 0.25


def _frame(n, months, rng, pd_shift=0.0, label_rate=None):
    rows = []
    for m in months:
        x = rng.normal(0, 1, n)
        p = 1 / (1 + np.exp(-(-5 + 0.8 * x + pd_shift)))
        y = (rng.random(n) < (p if label_rate is None else label_rate)).astype(float)
        rows.append(pd.DataFrame({"month": m, "pd": p, "label": y, "f": x, "c": rng.choice(["A", "B"], n)}))
    return pd.concat(rows, ignore_index=True)


def test_stable_stream_raises_no_drift_alerts():
    rng = np.random.default_rng(1)
    ref = _frame(20000, ["2020-01"], rng)
    stream = _frame(5000, ["2021-01", "2021-02", "2021-03"], rng)
    r = monitor(ref, stream, ["f", "c"], ["f"], ref_auc=0.75, approved_use="demonstration_rank_ordering",
                thresholds=TH)
    assert all(m["score_psi"] < 0.10 for m in r["months"])
    assert not [a for a in r["alerts"] if a["alert_id"] == "score_drift"]


def test_drift_alert_is_deduplicated_and_calibration_severity_follows_approved_use():
    rng = np.random.default_rng(2)
    ref = _frame(20000, ["2020-01"], rng)
    stream = _frame(5000, ["2021-01", "2021-02", "2021-03", "2021-04"], rng, pd_shift=1.5, label_rate=0.002)
    r = monitor(ref, stream, ["f", "c"], ["f"], 0.75, "demonstration_rank_ordering", thresholds=TH)
    drift = [a for a in r["alerts"] if a["alert_id"] == "score_drift"]
    assert len(drift) == 1 and drift[0]["periods"] == 4 and drift[0]["severity"] == "high"
    assert (drift[0]["first_period"], drift[0]["last_period"]) == ("2021-01", "2021-04")
    cal = next(a for a in r["alerts"] if a["alert_id"] == "calibration")
    assert cal["severity"] == "info"  # rank-ordering approval: level drift is informational
    r2 = monitor(ref, stream, ["f", "c"], ["f"], 0.75, "pd_level", thresholds=TH)
    assert next(a for a in r2["alerts"] if a["alert_id"] == "calibration")["severity"] == "high"
    assert any(a["alert_id"] == "discrimination_drop" for a in r["alerts"])  # random labels -> AUC ~0.5


def test_refresh_deciles_is_governed(promoted, tmp_path):
    from crisk.models.registry import champion, refresh_deciles

    cfg = promoted["cfg"]
    store = AuditStore(tmp_path / "a.sqlite")
    cut = list(np.linspace(0.001, 0.009, 9))
    with pytest.raises(ValueError):
        refresh_deciles(cfg, cut, "ref", "", "why", store)
    with pytest.raises(ValueError):
        refresh_deciles(cfg, cut[::-1], "ref", "rev", "why", store)
    before = champion(cfg).get("risk_decile_cutoffs")
    out = refresh_deciles(cfg, cut, "recent applications", "rev", "score drift alert RB-02", store)
    assert out["risk_decile_cutoffs"] == cut and out["decile_history"][-1]["cutoffs"] == before
    assert store.events(action="decile_cutoffs_refreshed")


def test_monitoring_endpoint(promoted, tmp_path):
    import json

    from fastapi.testclient import TestClient

    from crisk.scoring.api import create_app
    from crisk.security.auth import KeyStore

    cfg = promoted["cfg"]
    ks = KeyStore(tmp_path / "k.json")
    st = AuditStore(cfg.audit_db)
    key = ks.create("mon", "analyst", st)[0]
    client = TestClient(create_app(cfg, keys=ks))
    path = cfg.reports_dir / "monitoring" / "latest.json"
    if path.exists():
        path.unlink()
    assert client.get("/v1/monitoring", headers={"X-API-Key": key}).status_code == 404
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"alerts": [], "months": []}))
    assert client.get("/v1/monitoring", headers={"X-API-Key": key}).json() == {"alerts": [], "months": []}
    assert client.get("/v1/monitoring").status_code == 401
