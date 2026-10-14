"""Loss data extraction for LGD/EAD (pre-registered: docs/lgd_el_plan_v1.md).

Release 47 sign convention (verified to within 1e-11 on 3,887 dispositions):
    actual_loss = removal UPB + delinquent accrued interest + total expenses
                  + net sale proceeds + MI recoveries + non-MI recoveries     (recoveries are negative)
"""

from __future__ import annotations

import duckdb
import pandas as pd

from crisk.config import DatasetConfig

LIQUIDATION_CODES = ("02", "03", "09", "15")
ORIG_FEATURES = ("orig_ltv", "orig_cltv", "mi_pct", "orig_upb", "credit_score", "orig_dti", "occupancy_status",
                 "property_type", "loan_purpose", "channel", "first_payment_date", "orig_interest_rate")


def _paths(cfg: DatasetConfig) -> tuple[str, str]:
    return ((cfg.processed_dir / "loan_monthly_performance.parquet").as_posix(),
            (cfg.processed_dir / "loan_origination.parquet").as_posix())


def liquidations(cfg: DatasetConfig) -> pd.DataFrame:
    """One row per liquidated loan with a loss record: severity, components, workout and origination features."""
    perf, orig = _paths(cfg)
    codes = ", ".join(f"'{c}'" for c in LIQUIDATION_CODES)
    feats = ", ".join(f"o.{c}" for c in ORIG_FEATURES)
    sql = f"""
    WITH liq AS (
        SELECT loan_id, reporting_period AS liquidation_period, zero_balance_code, zero_balance_removal_upb AS removal_upb,
               coalesce(delinquent_accrued_interest, 0) AS delinquent_interest, coalesce(total_expenses, 0) AS expenses,
               coalesce(net_sale_proceeds, 0) AS net_sale_proceeds, coalesce(mi_recoveries, 0) AS mi_recoveries,
               coalesce(non_mi_recoveries, 0) AS non_mi_recoveries, actual_loss, source_id
        FROM read_parquet('{perf}')
        WHERE zero_balance_code IN ({codes}) AND actual_loss IS NOT NULL AND zero_balance_removal_upb > 0
    ),
    d90 AS (
        SELECT p.loan_id, min(p.reporting_period) AS first_90dpd
        FROM read_parquet('{perf}') p JOIN liq USING (loan_id)
        WHERE TRY_CAST(p.delinquency_status AS INT) >= 3 OR p.delinquency_status = 'RA'
        GROUP BY 1
    ),
    at_default AS (
        SELECT p.loan_id, arg_max(p.current_actual_upb, p.reporting_period) AS upb_at_default,
               arg_max(p.estimated_ltv, p.reporting_period) FILTER (WHERE p.estimated_ltv IS NOT NULL) AS eltv_at_default
        FROM read_parquet('{perf}') p JOIN d90 USING (loan_id)
        WHERE p.reporting_period <= d90.first_90dpd
        GROUP BY 1
    )
    SELECT liq.*, d90.first_90dpd, at_default.upb_at_default, at_default.eltv_at_default,
           datediff('month', d90.first_90dpd, liq.liquidation_period) AS workout_months, {feats},
           '20' || substr(liq.loan_id, 2, 2) AS vintage
    FROM liq LEFT JOIN d90 USING (loan_id) LEFT JOIN at_default USING (loan_id)
    JOIN read_parquet('{orig}') o USING (loan_id)
    """
    with duckdb.connect() as con:
        df = con.execute(sql).df()
    recon = (df.removal_upb + df.delinquent_interest + df.expenses + df.net_sale_proceeds + df.mi_recoveries
             + df.non_mi_recoveries - df.actual_loss).abs()
    df["reconciles"] = recon < 0.01
    df["severity"] = df.actual_loss / df.removal_upb
    df["liquidation_year"] = pd.to_datetime(df.liquidation_period).dt.year
    return df


def default_resolution(cfg: DatasetConfig, vintages: tuple[str, ...]) -> pd.DataFrame:
    """For 12-month 90+ defaults (primary label), the loan's eventual outcome; also EAD ratio at first 90+."""
    perf, orig = _paths(cfg)
    outc = (cfg.processed_dir / "loan_outcomes.parquet").as_posix()
    vin = ", ".join(f"'{v[-2:]}'" for v in vintages)
    codes = ", ".join(f"'{c}'" for c in LIQUIDATION_CODES)
    sql = f"""
    WITH d AS (SELECT loan_id FROM read_parquet('{outc}') WHERE label = 1 AND substr(loan_id, 2, 2) IN ({vin})),
    last AS (
        SELECT p.loan_id, arg_max(p.zero_balance_code, p.reporting_period) AS zbc,
               arg_max(p.delinquency_status, p.reporting_period) AS last_status,
               max(p.actual_loss) AS actual_loss
        FROM read_parquet('{perf}') p JOIN d USING (loan_id) GROUP BY 1),
    first90 AS (
        SELECT p.loan_id, arg_min(p.current_actual_upb, p.reporting_period) AS upb_at_90
        FROM read_parquet('{perf}') p JOIN d USING (loan_id)
        WHERE TRY_CAST(p.delinquency_status AS INT) >= 3 OR p.delinquency_status = 'RA' GROUP BY 1)
    SELECT last.loan_id, '20' || substr(last.loan_id, 2, 2) AS vintage,
           CASE WHEN zbc IN ({codes}) AND actual_loss IS NOT NULL THEN 'liquidated_with_loss'
                WHEN zbc = '01' THEN 'paid_off' WHEN zbc = '16' THEN 'reperforming_sale'
                WHEN zbc = '96' THEN 'defect_repurchase'
                WHEN zbc IS NULL AND TRY_CAST(last_status AS INT) = 0 THEN 'active_current'
                ELSE 'active_delinquent' END AS resolution,
           first90.upb_at_90, o.orig_upb, first90.upb_at_90 / o.orig_upb AS ead_ratio
    FROM last JOIN first90 USING (loan_id) JOIN read_parquet('{orig}') o USING (loan_id)
    """
    with duckdb.connect() as con:
        return con.execute(sql).df()
