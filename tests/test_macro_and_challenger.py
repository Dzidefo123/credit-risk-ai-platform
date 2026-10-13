"""Market-rate feature (point-in-time) and the LightGBM challenger, on SYNTHETIC data."""

import numpy as np
import pandas as pd
import pytest

from crisk.features.origination import FeatureSpec, build_feature_frame
from crisk.features.snapshots import build_feature_snapshots, snapshot_path
from crisk.macro.pmms import MarketRateTable
from crisk.models.registry import load_bundle
from crisk.models.train import train_model
from crisk.scoring.service import ScoringService, replay

V2 = "configs/features/origination_v2.yaml"
LGBM = "configs/models/lightgbm_v1.yaml"


def write_fred_csvs(d, start="2014-01-02", end="2024-12-26"):
    """Synthetic weekly PMMS-like series (Thursdays); 15y = 30y - 0.7. Includes FRED's '.' missing marker."""
    d.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start, end, freq="W-THU")
    r30 = 3.5 + 0.02 * np.arange(len(dates)) % 3.0
    for sid, vals in (("MORTGAGE30US", r30), ("MORTGAGE15US", r30 - 0.7)):
        df = pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"), sid: np.round(vals, 2).astype(str)})
        df.loc[5, sid] = "."
        df.to_csv(d / f"fred_{sid}_current.csv", index=False)
    return d


@pytest.fixture(scope="module")
def macro_dir(tmp_path_factory):
    return write_fred_csvs(tmp_path_factory.mktemp("pmms"))


def test_reference_rate_is_point_in_time(macro_dir):
    t = MarketRateTable.from_fred_csv(macro_dir)
    lut = t.monthly.set_index("month")
    fpd = pd.Series(pd.to_datetime(["2020-04-01", "2020-04-01", "2030-01-01"]))
    term = pd.Series([360, 180, 360])
    r = t.reference_rate(fpd, term)
    # FPD 2020-04 -> values published in 2020-01 (three months earlier), never later
    assert r[0] == pytest.approx(lut.loc["2020-01", "MORTGAGE30US"])
    assert r[1] == pytest.approx(lut.loc["2020-01", "MORTGAGE15US"])
    assert np.isnan(r[2])  # beyond the published table -> missing, not extrapolated
    # only weeks published inside the reference month are averaged
    weekly = pd.read_csv(macro_dir / "fred_MORTGAGE30US_current.csv")
    weekly["v"] = pd.to_numeric(weekly.iloc[:, 1], errors="coerce")
    jan = weekly[weekly.observation_date.str.startswith("2020-01")]["v"].mean()
    assert r[0] == pytest.approx(jan)


def test_rate_spread_feature_requires_macro_and_is_rowwise(macro_dir):
    spec = FeatureSpec.load(V2)
    assert "orig_interest_rate" not in spec.model_columns and "rate_spread" in spec.model_columns
    rec = pd.DataFrame([dict(credit_score=720, orig_dti=30, orig_ltv=80, orig_cltv=80, orig_interest_rate=5.0,
                             orig_loan_term=360, mi_pct=0, num_units=1, orig_upb=2e5, num_borrowers=2,
                             occupancy_status="P", channel="R", property_type="SF", loan_purpose="P",
                             first_time_homebuyer_flag="N", property_state="TX",
                             first_payment_date=pd.Timestamp("2020-04-01"))])
    with pytest.raises(ValueError, match="market-rate"):
        build_feature_frame(rec, spec)
    t = MarketRateTable.from_fred_csv(macro_dir)
    f = build_feature_frame(rec, spec, macro=t)
    assert f.loc[0, "rate_spread"] == pytest.approx(5.0 - t.monthly.set_index("month").loc["2020-01", "MORTGAGE30US"])


@pytest.fixture(scope="module")
def challenger(trained_ws, macro_dir):
    mp = pytest.MonkeyPatch()
    mp.setenv("CRISK_PMMS_DIR", str(macro_dir))
    cfg, store = trained_ws["cfg"], trained_ws["store"]
    build_feature_snapshots(cfg, feature_config=V2)
    meta = train_model(cfg, store, feature_config=V2, model_config=LGBM)
    yield {"cfg": cfg, "store": store, "meta": meta}
    mp.undo()


def test_lightgbm_challenger_trains_and_records_lineage(challenger):
    meta = challenger["meta"]
    assert meta["model_family"] == "lightgbm" and meta["feature_set_id"] == "origination_v2"
    assert meta["fit_info"]["best_iteration"] >= 1
    assert meta["macro"]["reference_lag_months"] == 3
    assert meta["fit_info"]["monotone_constraints"]["credit_score"] == -1


def test_lightgbm_monotone_in_credit_score(challenger):
    cfg = challenger["cfg"]
    bundle, _ = load_bundle(cfg, challenger["meta"]["model_id"])
    snap = pd.read_parquet(snapshot_path(cfg, "origination_v2")).head(200)
    lo, hi = snap.copy(), snap.copy()
    lo["credit_score"], hi["credit_score"] = 620.0, 800.0
    assert np.all(bundle.predict_raw(hi) <= bundle.predict_raw(lo) + 1e-12)


def test_lightgbm_scoring_reasons_and_replay(challenger):
    cfg, store = challenger["cfg"], challenger["store"]
    svc = ScoringService(cfg, store, model_id=challenger["meta"]["model_id"])
    app = dict(credit_score=650, orig_dti=45, orig_ltv=95, orig_cltv=95, orig_interest_rate=6.0,
               orig_loan_term=360, mi_pct=30, num_units=1, orig_upb=300000.0, num_borrowers=1,
               occupancy_status="P", channel="B", property_type="SF", loan_purpose="P",
               first_time_homebuyer_flag="Y", property_state="TX", first_payment_date="2019-06-01")
    r = svc.score(app)
    assert r["status"] == "scored" and 0 < r["pd"] < 1 and r["reason_codes"]
    assert replay(cfg, store, r["score_id"])["match"]
    late = svc.score({**app, "first_payment_date": "2031-01-01"})
    assert "market_rate_unavailable:rate_spread_imputed" in late["dq_flags"]
