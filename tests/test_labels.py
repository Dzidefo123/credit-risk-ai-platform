import duckdb
import pandas as pd

from crisk.labels.default_horizon import TargetSpec, label_sql


def _con(cfg):
    con = duckdb.connect()
    pdir = cfg.processed_dir.as_posix()
    con.execute(f"CREATE VIEW orig AS SELECT * FROM read_parquet('{pdir}/loan_origination.parquet')")
    con.execute(f"CREATE TABLE perf AS SELECT * FROM read_parquet('{pdir}/loan_monthly_performance.parquet')")
    return con


def test_every_scenario_matches_answer_key(scenario_ws):
    cfg, key = scenario_ws["cfg"], scenario_ws["key"]
    out = pd.read_parquet(cfg.processed_dir / "loan_outcomes.parquet").set_index("loan_id")
    for loan_id, expected in key["expected_outcomes"].items():
        assert out.loc[loan_id, "outcome"] == expected, key["descriptions"][loan_id]
    for loan_id, m in key["expected_default_month"].items():
        assert out.loc[loan_id, "default_month_index"] == m, key["descriptions"][loan_id]


def test_label_values_and_exclusions(scenario_ws):
    out = pd.read_parquet(scenario_ws["cfg"].processed_dir / "loan_outcomes.parquet")
    assert set(out.loc[out.outcome == "default", "label"]) == {1}
    assert set(out.loc[out.outcome.isin(["non_default", "non_default_prepaid"]), "label"]) == {0}
    assert out.loc[out.outcome.str.startswith("excluded"), "label"].isna().all()


def test_forbearance_flag_recorded(scenario_ws):
    out = pd.read_parquet(scenario_ws["cfg"].processed_dir / "loan_outcomes.parquet").set_index("loan_id")
    assert bool(out.loc["X19Q10000014", "forbearance_in_window"])
    assert bool(out.loc["X19Q10000014", "disaster_flag_in_window"])


def test_labels_ignore_information_after_the_window(scenario_ws):
    """Leakage test: post-window delinquency and termination must not change any label."""
    cfg = scenario_ws["cfg"]
    spec = TargetSpec.load(cfg.target_config)
    con = _con(cfg)
    base = con.execute(label_sql(spec, cfg.performance_cutoff)).df().set_index("loan_id")
    con.execute("""
        INSERT INTO perf BY NAME
        SELECT p.loan_id, CAST(o.first_payment_date + to_months(12) AS DATE) AS reporting_period,
               '6' AS delinquency_status, '03' AS zero_balance_code
        FROM (SELECT DISTINCT loan_id FROM perf) p JOIN orig o USING (loan_id)
        WHERE NOT EXISTS (SELECT 1 FROM perf q WHERE q.loan_id = p.loan_id
                          AND q.reporting_period = CAST(o.first_payment_date + to_months(12) AS DATE))""")
    after = con.execute(label_sql(spec, cfg.performance_cutoff)).df().set_index("loan_id")
    pd.testing.assert_series_equal(base["outcome"].sort_index(), after["outcome"].sort_index())


def test_immature_cohort_excluded_even_with_early_default(scenario_ws):
    """Maturity is checked before default, so early defaults of immature cohorts do not inflate prevalence."""
    cfg = scenario_ws["cfg"]
    spec = TargetSpec.load(cfg.target_config)
    con = _con(cfg)
    con.execute("UPDATE perf SET delinquency_status = '3' "
                "WHERE loan_id = 'X19Q10000006' AND reporting_period = DATE '2023-07-01'")
    out = con.execute(label_sql(spec, cfg.performance_cutoff)).df().set_index("loan_id")
    assert out.loc["X19Q10000006", "outcome"] == "excluded_immature"


def test_sensitivity_target_excludes_only_flagged_delinquency(scenario_ws):
    """Option C: the ex-disaster target differs from the primary only where 90+ was flagged."""
    cfg = scenario_ws["cfg"]
    primary = pd.read_parquet(cfg.processed_dir / "loan_outcomes.parquet").set_index("loan_id")
    sens_file = cfg.processed_dir / "loan_outcomes__default_90dpd_12m_exdisaster_v1.parquet"
    sens = pd.read_parquet(sens_file).set_index("loan_id")
    assert primary.loc["X19Q10000014", "outcome"] == "default"
    assert sens.loc["X19Q10000014", "outcome"] == "excluded_disaster_forbearance"
    assert pd.isna(sens.loc["X19Q10000014", "label"])
    changed = primary.index[primary.outcome != sens.outcome]
    assert list(changed) == ["X19Q10000014"]
    report = scenario_ws["labels"]["sensitivity_targets"]["default_90dpd_12m_exdisaster_v1"]
    assert report["defaults"] == scenario_ws["labels"]["defaults"] - 1
