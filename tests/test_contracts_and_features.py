import numpy as np
import pandas as pd
import pytest

from crisk.contracts.freddie_mac import ORIGINATION, ORIGINATION_R47, PERFORMANCE, PERFORMANCE_R47
from crisk.features.origination import MISSING_CATEGORY, FeatureSpec, build_feature_frame
from crisk.features.splits import SplitSpec

FEATURES = "configs/features/origination_v1.yaml"


def test_contract_shapes_match_guide():
    assert len(ORIGINATION.fields) == 32
    assert len(PERFORMANCE.fields) == 32
    assert ORIGINATION.by_name("loan_id").position == 20
    assert PERFORMANCE.by_name("zero_balance_code").position == 9


def test_release47_layout_matches_provider_headers():
    # Order from file_headers_july_2026.zip (origination 31, performance 35 fields).
    assert len(ORIGINATION_R47.fields) == 31 and len(PERFORMANCE_R47.fields) == 35
    assert ORIGINATION_R47.names[-1] == "vantage_score_4"
    assert ORIGINATION_R47.by_name("loan_id").position == 20
    assert ORIGINATION_R47.by_name("seller_name").position == 24
    assert ORIGINATION_R47.by_name("io_indicator").position == 30
    assert "servicer_name" not in ORIGINATION_R47.names
    assert PERFORMANCE_R47.names[-3:] == ["mi_cancellation_indicator", "servicer_name", "bankruptcy_cramdown_costs"]
    assert PERFORMANCE_R47.by_name("servicer_name").availability == "performance"


def test_feature_set_uses_only_origination_time_inputs():
    spec = FeatureSpec.load(FEATURES)
    for leak in ("servicer_name", "mi_cancellation_indicator", "seller_name", "loan_id"):
        assert leak not in spec.raw_inputs
    for name in spec.raw_inputs:
        assert ORIGINATION_R47.by_name(name).availability in ("origination", "acquisition")


def test_availability_guard_rejects_post_origination_field():
    spec = FeatureSpec.load(FEATURES)
    bad = FeatureSpec(spec.feature_set_id, spec.numeric, spec.derived_numeric,
                      (*spec.categorical, "servicer_name"), spec.acquisition_fields_accepted)
    with pytest.raises(ValueError, match="servicer_name"):
        bad.check_availability()


def _record(**kw):
    base = dict(credit_score=720, orig_dti=30, orig_ltv=80, orig_cltv=85, orig_interest_rate=4.0,
                orig_loan_term=360, mi_pct=0, num_units=1, orig_upb=200000.0, num_borrowers=1,
                occupancy_status="P", channel="R", property_type="SF", loan_purpose="P",
                first_time_homebuyer_flag="N", property_state="TX")
    base.update(kw)
    return base


def test_feature_frame_is_rowwise_point_in_time():
    """A loan's features depend only on its own origination record (no cross-row information)."""
    spec = FeatureSpec.load(FEATURES)
    rows = [_record(), _record(credit_score=None, occupancy_status=None, num_borrowers=3),
            _record(orig_upb=500000.0)]
    batch = build_feature_frame(pd.DataFrame(rows), spec)
    for i, r in enumerate(rows):
        single = build_feature_frame(pd.DataFrame([r]), spec)
        pd.testing.assert_frame_equal(single.reset_index(drop=True), batch.iloc[[i]].reset_index(drop=True))
    assert np.isnan(batch.loc[1, "credit_score"])
    assert batch.loc[1, "occupancy_status"] == MISSING_CATEGORY
    assert batch.loc[1, "multiple_borrowers"] == 1.0
    assert batch.loc[0, "cltv_minus_ltv"] == 5
    assert batch.loc[2, "log_orig_upb"] == pytest.approx(np.log(500000.0))


def test_split_gap_is_enforced(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "split_id: bad\n"
        "date_column: first_payment_date\n"
        "min_gap_months: 12\n"
        "splits:\n"
        '  train: {start: "2016-01", end: "2018-12"}\n'
        '  calibration: {start: "2019-06", end: "2019-12"}\n'
        '  test: {start: "2022-01", end: "2022-12"}\n')
    with pytest.raises(ValueError, match="calibration starts"):
        SplitSpec.load(str(p), horizon_months=12)
    p.write_text(p.read_text().replace("min_gap_months: 12", "min_gap_months: 6"))
    with pytest.raises(ValueError, match="horizon"):
        SplitSpec.load(str(p), horizon_months=12)


def test_split_assignment():
    s = SplitSpec.load("configs/splits/synthetic_temporal_v1.yaml", 12)
    d = pd.Series(pd.to_datetime(["2016-01-01", "2018-12-01", "2019-06-01", "2020-03-01", "2022-12-01"]))
    assert list(s.assign(d)) == ["train", "train", "unassigned", "calibration", "test"]


def test_split_source_filter(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "split_id: src\n"
        "date_column: first_payment_date\n"
        "min_gap_months: 12\n"
        "splits:\n"
        '  train: {start: "2016-01", end: "2018-12", source_id: full}\n'
        '  calibration: {start: "2020-01", end: "2020-12", source_id: sample}\n'
        '  test: {start: "2022-01", end: "2022-12"}\n')
    s = SplitSpec.load(str(p), 12)
    d = pd.Series(pd.to_datetime(["2016-05-01", "2016-05-01", "2020-03-01", "2020-03-01", "2022-02-01"]))
    src = pd.Series(["full", "sample", "full", "sample", "sample"])
    assert list(s.assign(d, src)) == ["train", "unassigned", "unassigned", "calibration", "test"]
    with pytest.raises(ValueError):
        s.assign(d)
