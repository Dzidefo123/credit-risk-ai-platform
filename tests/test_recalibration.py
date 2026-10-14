import numpy as np
import pandas as pd
import pytest

from crisk.evaluation.metrics import logit, sigmoid
from crisk.models.recalibration import RecalSpec, decide, walk_forward, window_for


def _panel(n_per_month=4000, start="2015-01", end="2020-12", shift=0.7, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    m = np.repeat(months, n_per_month)
    p0 = sigmoid(rng.normal(-5.0, 0.8, len(m)))
    y = (rng.random(len(m)) < sigmoid(logit(p0) + shift)).astype(float)
    return pd.Series(m), p0, y


def test_window_is_lagged_and_twelve_months():
    lo, hi = window_for(pd.Period("2020-06", "M"), RecalSpec())
    assert (str(lo), str(hi)) == ("2017-11", "2018-10")


def test_intercept_update_recovers_level_shift():
    months, p0, y = _panel(shift=0.7)
    preds, params = walk_forward(months, p0, y, [pd.Period("2019-06", "M")])
    assert params.loc[0, "R1_a"] == pytest.approx(0.7, abs=0.1)
    assert params.loc[0, "R2_b"] == pytest.approx(1.0, abs=0.15)
    assert preds.R1.mean() == pytest.approx(preds.y.mean(), rel=0.15)


def test_no_look_ahead():
    """Changing outcomes of cohorts inside the lag (or of cohort T itself) must not change T's parameters."""
    months, p0, y = _panel()
    t = pd.Period("2019-06", "M")
    _, params = walk_forward(months, p0, y, [t])
    y2 = y.copy()
    lo, hi = window_for(t, RecalSpec())
    y2[(months > hi).to_numpy()] = 1.0  # every loan after the window (incl. cohort T) defaults
    _, params2 = walk_forward(months, p0, y2, [t])
    pd.testing.assert_frame_equal(params, params2)


def test_guard_carries_forward_when_window_is_thin():
    months, p0, y = _panel(start="2015-01", end="2016-12")
    ts = [pd.Period("2016-12", "M"), pd.Period("2019-12", "M")]  # second window 2017-05..2018-04: no data
    _, params = walk_forward(months, p0, y, ts, RecalSpec(min_events=20))
    assert params.loc[0, "refitted"] and not params.loc[1, "refitted"]
    assert params.loc[1, "R1_a"] == params.loc[0, "R1_a"]


def test_decision_rule():
    def s(ll0, ll1, ll2, b0, b1, b2):
        return {"overall": {"R0": {"log_loss": ll0}, "R1": {"log_loss": ll1}, "R2": {"log_loss": ll2}},
                "in_band_share": {"R0": b0, "R1": b1, "R2": b2}}
    assert decide({"a": s(1.0, 0.9, 0.85, 0.2, 0.6, 0.6), "b": s(1.0, 0.9, 0.88, 0.2, 0.6, 0.6)})["adopted"] == "R2"
    assert decide({"a": s(1.0, 0.9, 0.895, 0.2, 0.6, 0.6), "b": s(1.0, 0.9, 0.85, 0.2, 0.6, 0.6)})["adopted"] == "R1"
    assert decide({"a": s(1.0, 0.9, 0.85, 0.2, 0.2, 0.2), "b": s(1.0, 0.9, 0.85, 0.2, 0.6, 0.6)})["adopted"] == "R0"


def test_expanding_window_starts_at_earliest_month():
    lo, hi = window_for(pd.Period("2020-06", "M"), RecalSpec(window_months=None), pd.Period("2015-01", "M"))
    assert (str(lo), str(hi)) == ("2015-01", "2018-10")
    with pytest.raises(ValueError):
        window_for(pd.Period("2020-06", "M"), RecalSpec(window_months=None))


def test_ttc_predictions_and_rule():
    from crisk.models.recalibration import decide_ttc, ttc_predictions

    months, p0, y = _panel(start="2015-01", end="2021-12", shift=0.4)
    preds, params = ttc_predictions(months, p0, y, list(pd.period_range("2019-04", "2019-06", freq="M")))
    np.testing.assert_allclose(preds.L0.to_numpy(), p0[preds.row.to_numpy()])
    assert params["L1"][0]["window_start"] == "2015-01" and params["L2"][0]["window_start"] == "2012-09"

    def s(r0, r1, r2, ll0=1.0, ll1=1.0, ll2=1.0):
        return {"rmsle_oe": {"L0": r0, "L1": r1, "L2": r2},
                "overall": {"L0": {"log_loss": ll0}, "L1": {"log_loss": ll1}, "L2": {"log_loss": ll2}}}
    assert decide_ttc({"a": s(1.0, 0.5, 0.48), "b": s(1.0, 0.5, 0.48)})["adopted"] == "L1"   # L2 gain < 10% vs L1
    assert decide_ttc({"a": s(1.0, 0.5, 0.3), "b": s(1.0, 0.5, 0.3)})["adopted"] == "L2"
    assert decide_ttc({"a": s(1.0, 0.95, 0.95), "b": s(1.0, 0.95, 0.95)})["adopted"] == "L0"  # < 10% vs L0
    assert decide_ttc({"a": s(1.0, 0.5, 0.6, ll1=1.02), "b": s(1.0, 0.5, 0.6)})["adopted"] == "L0"  # log loss guard


def test_v5_compare_selection_rule_and_single_confirmatory(trained_ws, tmp_path):
    """Plan v5 mechanics on SYNTHETIC data: identical models tie, so the default (C2) is chosen; the
    confirmatory stage runs once."""
    from crisk.audit.store import AuditStore
    from crisk.models.refresh_v5 import compare

    cfg, mid = trained_ws["cfg"], trained_ws["meta"]["model_id"]
    store = AuditStore(tmp_path / "a.sqlite")
    models = {"C0": (cfg, mid), "C1": (cfg, mid), "C2": (cfg, mid)}
    r = compare(models, cfg, "selection", store, cfg, actor="test")
    assert r["selection"]["chosen"] == "C2" and r["paired"]["C1-C2"]["delta_auc"] == 0.0
    assert r["loans"] > 0 and set(r["credit_score_bands"]) == {"C0", "C1", "C2"}
    c = compare({"C0": (cfg, mid), "C2": (cfg, mid)}, cfg, "confirmatory", store, cfg, actor="test", candidate="C2")
    assert c["confirmatory"]["checks"]["no_ranking_loss_vs_c0"] and "pass" in c["confirmatory"]
    with pytest.raises(RuntimeError, match="already run"):
        compare({"C0": (cfg, mid), "C2": (cfg, mid)}, cfg, "confirmatory", store, cfg, actor="test")
