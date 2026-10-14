"""LGD / EAD / expected loss (SYNTHETIC fixture data)."""

import numpy as np
import pandas as pd
import pytest

from crisk.lgd.data import liquidations
from crisk.lgd.el import SCENARIOS, expected_loss, stressed_severity
from crisk.lgd.models import SeverityModel, severity_metrics


def test_synthetic_liquidations_reconcile_with_release47_signs(trained_ws):
    liq = liquidations(trained_ws["cfg"])
    assert len(liq) > 0 and liq.reconciles.all()
    assert (liq.net_sale_proceeds <= 0).all()  # recoveries are negative in Release 47
    assert liq.severity.between(0.15, 0.6).all()
    assert liq.first_90dpd.notna().all() and (liq.workout_months >= 0).all()


def _frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    ltv = rng.integers(50, 98, n)
    sev = np.clip(0.05 + 0.004 * (ltv - 50) + rng.normal(0, 0.08, n), 0, 1)
    return pd.DataFrame({"orig_ltv": ltv, "orig_cltv": ltv, "mi_pct": np.where(ltv > 80, 25, 0),
                         "orig_upb": rng.integers(100, 600, n) * 1000.0, "credit_score": rng.integers(620, 820, n),
                         "orig_dti": rng.integers(10, 50, n), "occupancy_status": "P", "property_type": "SF",
                         "loan_purpose": "P", "channel": "R", "severity": sev})


def test_severity_models_fit_and_rank():
    df = _frame()
    tr, te = df.iloc[:1000], df.iloc[1000:]
    preds = {k: SeverityModel(k).fit(tr).predict(te) for k in ("S0", "S1", "S2")}
    assert np.allclose(preds["S0"], tr.severity.mean())
    assert all(((p >= 0) & (p <= 1)).all() for p in preds.values())
    m2 = severity_metrics(te.severity, preds["S2"], te.orig_upb)
    m0 = severity_metrics(te.severity, preds["S0"], te.orig_upb)
    assert m2["mae"] < m0["mae"] and m2["spearman"] > 0.5


def test_expected_loss_arithmetic_and_stress_ordering():
    book = pd.DataFrame({"risk_decile": [1, 10], "severity": [0.2, 0.3], "orig_upb": [100000.0, 200000.0],
                         "orig_ltv": [80.0, 95.0]})
    overlay = {1: 0.001, 10: 0.02}
    out = expected_loss(book, overlay, p_liq=0.3, ead_ratio=0.95, eltv_slope=0.0)
    base = out["base"]["expected_loss_usd"]
    assert base == pytest.approx(0.001 * 0.3 * 0.2 * 95000 + 0.02 * 0.3 * 0.3 * 190000)
    assert out["base"]["exposure_usd"] == pytest.approx(285000)
    out2 = expected_loss(book, overlay, 0.3, 0.95, eltv_slope=0.756)
    el = {k: out2[k]["expected_loss_usd"] for k in ("base", "adverse", "severe")}
    assert el["base"] < el["adverse"] < el["severe"]
    assert SCENARIOS["severe"]["pd_multiplier"] == 4.0


def test_house_price_shock_raises_severity_monotonically():
    sev, ltv = np.array([0.2, 0.2]), np.array([60.0, 95.0])
    s10 = stressed_severity(sev, ltv, 0.756, 0.10)
    s20 = stressed_severity(sev, ltv, 0.756, 0.20)
    assert (s20 > s10).all() and (s10 > sev).all() and s10[1] > s10[0]
    assert np.array_equal(stressed_severity(sev, ltv, 0.756, 0.0), sev)
