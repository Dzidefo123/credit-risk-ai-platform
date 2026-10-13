"""Mature-cohort default labels over a fixed horizon from first payment.

Unit of observation: one loan (Loan Sequence Number). Origin: the first payment date (FPD)
month. The note/decision date is NOT disclosed; it precedes the FPD by an unknown interval
(typically one to two months), so the origin is a documented proxy, not the decision date.

Month index: m = months(reporting_period - FPD) + 1, so m = 1 is the FPD month. Computed
from dates instead of the provider's Loan Age, which resets after modifications.

Outcome classes (evaluated in this order):
  excluded_ineligible          product outside the target population (e.g. ARM)
  excluded_no_performance      no monthly records
  excluded_immature            FPD + horizon - 1 is after the performance cutoff (cohort-level
                               maturity: early defaults from immature cohorts are excluded too,
                               to avoid inflating prevalence)
  excluded_late_acquisition    first record after max_first_observed_month (left truncation)
  default                      default event in [1, H] not preceded by a censoring removal
  excluded_censored_removal    whole-loan sale / RPL securitisation / defect before any default
  non_default_prepaid          voluntary payoff in-window, no default, coverage OK until payoff
  non_default                  observed through m = H with no default and coverage OK
  excluded_incomplete_reporting  everything else (reporting stops or too many gaps)

Labels: default -> 1; non_default / non_default_prepaid -> 0; all excluded -> NULL.
Treating in-window prepayment as 0 is a documented choice (decision D-007); a censored
treatment is a sensitivity analysis for Phase C.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crisk.config import DatasetConfig, load_yaml
from crisk.duck import work_connection
from crisk.utils import utc_now, write_json


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    horizon_months: int
    default_delinquency_status_min: int
    default_status_codes: tuple[str, ...]
    default_zero_balance_codes: tuple[str, ...]
    censoring_zero_balance_codes: tuple[str, ...]
    prepay_zero_balance_codes: tuple[str, ...]
    eligible_amortization_types: tuple[str, ...]
    max_first_observed_month: int
    max_missing_months_in_window: int
    description: str = ""
    # Sensitivity variant: a 90+/RA status reported while the loan carries a disaster flag or is in a
    # forbearance plan is not a default event; loans whose only in-window 90+ months are flagged are
    # excluded (outcome unknowable), not relabelled as non-defaults. Credit terminations still count.
    exclude_disaster_forbearance_events: bool = False

    @classmethod
    def load(cls, path: str) -> TargetSpec:
        raw = load_yaml(path)
        tuples = {k: tuple(str(x) for x in v) for k, v in raw.items() if isinstance(v, list)}
        return cls(**{**raw, **tuples})


def _in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join("'" + v + "'" for v in values) + ")" if values else "(NULL)"


def label_sql(spec: TargetSpec, cutoff: str) -> str:
    H = spec.horizon_months
    status_event = (f"(TRY_CAST(p.delinquency_status AS INT) >= {spec.default_delinquency_status_min} "
                    f"OR p.delinquency_status IN {_in(spec.default_status_codes)})")
    termination_event = f"(p.zero_balance_code IN {_in(spec.default_zero_balance_codes)})"
    flagged = ("(coalesce(p.delinquency_due_to_disaster, '') = 'Y' "
               "OR coalesce(p.borrower_assistance_status, '') = 'F')")
    if spec.exclude_disaster_forbearance_events:
        is_default = f"(({status_event} AND NOT {flagged}) OR {termination_event})"
        flagged_branch = "WHEN flagged_dlq_m IS NOT NULL THEN 'excluded_disaster_forbearance'"
    else:
        is_default = f"({status_event} OR {termination_event})"
        flagged_branch = ""
    return f"""
    WITH o AS (
        SELECT loan_id, first_payment_date AS fpd, amortization_type, data_origin FROM orig
    ),
    pm AS (
        SELECT p.*, datediff('month', o.fpd, p.reporting_period) + 1 AS m, {is_default} AS is_default,
               ({status_event} AND {flagged}) AS is_flagged_dlq
        FROM perf p JOIN o USING (loan_id)
    ),
    agg AS (
        SELECT loan_id,
            min(m) AS first_obs_m,
            max(m) AS last_obs_m,
            min(m) FILTER (WHERE m BETWEEN 1 AND {H} AND is_default) AS default_m,
            min(m) FILTER (WHERE m BETWEEN 1 AND {H} AND is_flagged_dlq) AS flagged_dlq_m,
            min(m) FILTER (WHERE m BETWEEN 1 AND {H} AND zero_balance_code IN {_in(spec.censoring_zero_balance_codes)}) AS censor_m,
            min(m) FILTER (WHERE m BETWEEN 1 AND {H} AND zero_balance_code IN {_in(spec.prepay_zero_balance_codes)}) AS prepay_m
        FROM pm GROUP BY loan_id
    ),
    cov AS (
        -- reporting coverage: distinct observed months between the first expected month and the end
        -- of the observable window (exit month or H)
        SELECT a.loan_id,
               greatest(a.first_obs_m, 1) AS cov_start,
               least({H}, coalesce(a.prepay_m, {H})) AS cov_end,
               -- (loan_id, reporting_period) is unique after the canonical build, so count(*) counts months
               count(*) FILTER (
                   WHERE pm.m BETWEEN greatest(a.first_obs_m, 1) AND least({H}, coalesce(a.prepay_m, {H}))
               ) AS observed_months
        FROM agg a LEFT JOIN pm USING (loan_id)
        GROUP BY a.loan_id, a.first_obs_m, a.prepay_m
    ),
    flags AS (
        SELECT pm.loan_id,
               bool_or(pm.delinquency_due_to_disaster = 'Y') AS disaster_flag_before_default,
               bool_or(pm.borrower_assistance_status = 'F') AS forbearance_before_default,
               bool_or(pm.modification_flag IN ('Y', 'P')) AS modified_in_window
        FROM pm JOIN agg a USING (loan_id)
        WHERE pm.m BETWEEN 1 AND coalesce(a.default_m, {H})
        GROUP BY pm.loan_id
    ),
    j AS (
        SELECT o.*, a.first_obs_m, a.last_obs_m, a.default_m, a.censor_m, a.prepay_m, a.flagged_dlq_m,
               c.cov_start, c.cov_end, c.observed_months,
               greatest(c.cov_end - c.cov_start + 1, 0) - coalesce(c.observed_months, 0) AS missing_months,
               f.disaster_flag_before_default, f.forbearance_before_default, f.modified_in_window,
               CAST(o.fpd + to_months({H} - 1) AS DATE) AS window_end_period
        FROM o LEFT JOIN agg a USING (loan_id) LEFT JOIN cov c USING (loan_id) LEFT JOIN flags f USING (loan_id)
    )
    SELECT loan_id, fpd AS first_payment_date, window_end_period, data_origin,
        CASE
            WHEN amortization_type IS NULL OR amortization_type NOT IN {_in(spec.eligible_amortization_types)}
                THEN 'excluded_ineligible'
            WHEN first_obs_m IS NULL THEN 'excluded_no_performance'
            WHEN window_end_period > DATE '{cutoff}-01' THEN 'excluded_immature'
            WHEN first_obs_m > {spec.max_first_observed_month} THEN 'excluded_late_acquisition'
            WHEN default_m IS NOT NULL AND (censor_m IS NULL OR default_m <= censor_m) THEN 'default'
            {flagged_branch}
            WHEN censor_m IS NOT NULL THEN 'excluded_censored_removal'
            WHEN missing_months > {spec.max_missing_months_in_window} THEN 'excluded_incomplete_reporting'
            WHEN prepay_m IS NOT NULL THEN 'non_default_prepaid'
            WHEN last_obs_m >= {H} THEN 'non_default'
            ELSE 'excluded_incomplete_reporting'
        END AS outcome,
        default_m AS default_month_index,
        coalesce(prepay_m, censor_m) AS exit_month_index,
        first_obs_m AS first_observed_month_index,
        missing_months AS missing_months_in_window,
        coalesce(disaster_flag_before_default, FALSE) AS disaster_flag_in_window,
        coalesce(forbearance_before_default, FALSE) AS forbearance_in_window,
        coalesce(modified_in_window, FALSE) AS modified_in_window
    FROM j
    """


def outcomes_path(cfg: DatasetConfig, target_id: str | None = None) -> Path:
    """Primary target -> loan_outcomes.parquet; sensitivity targets -> loan_outcomes__<target_id>.parquet."""
    return cfg.processed_dir / ("loan_outcomes.parquet" if target_id is None else f"loan_outcomes__{target_id}.parquet")


def _build_one(cfg: DatasetConfig, spec: TargetSpec, con, out: Path) -> dict:
    con.execute(f"CREATE OR REPLACE TABLE outcomes AS {label_sql(spec, cfg.performance_cutoff)}")
    con.execute(f"""
        CREATE OR REPLACE TABLE outcomes_final AS
        SELECT *,
            CASE WHEN outcome = 'default' THEN 1
                 WHEN outcome IN ('non_default', 'non_default_prepaid') THEN 0 END AS label,
            '{spec.target_id}' AS target_id,
            {spec.horizon_months} AS horizon_months,
            '{cfg.performance_cutoff}' AS performance_cutoff
        FROM outcomes ORDER BY loan_id""")
    con.execute(f"COPY outcomes_final TO '{out.as_posix()}' (FORMAT parquet)")

    flow = con.execute("SELECT outcome, count(*) FROM outcomes_final GROUP BY 1 ORDER BY 1").fetchall()
    by_cohort = con.execute("""
        SELECT strftime(first_payment_date, '%Y') AS fpd_year, count(*) AS loans,
               count(label) AS labelled, sum(label) AS defaults,
               round(avg(label), 5) AS default_rate_labelled,
               sum(CASE WHEN outcome = 'excluded_immature' THEN 1 ELSE 0 END) AS immature,
               sum(CASE WHEN outcome = 'default' AND (disaster_flag_in_window OR forbearance_in_window)
                        THEN 1 ELSE 0 END) AS defaults_with_disaster_or_forbearance
        FROM outcomes_final GROUP BY 1 ORDER BY 1""").df()
    maturity = con.execute(f"""
        SELECT strftime(first_payment_date, '%Y-Q') || CAST(quarter(first_payment_date) AS VARCHAR) AS fpd_quarter,
               max(window_end_period) AS last_window_end,
               bool_and(window_end_period <= DATE '{cfg.performance_cutoff}-01') AS mature
        FROM outcomes_final GROUP BY 1 ORDER BY 1""").df()
    labelled = sum(n for o, n in flow if o in ("default", "non_default", "non_default_prepaid"))
    defaults = sum(n for o, n in flow if o == "default")
    return {
        "dataset_id": cfg.dataset_id,
        "data_origin": cfg.data_origin,
        "target_id": spec.target_id,
        "built_at": utc_now(),
        "performance_cutoff": cfg.performance_cutoff,
        "cohort_flow": dict(flow),
        "labelled_loans": labelled,
        "defaults": defaults,
        "prevalence_labelled": (defaults / labelled) if labelled else None,
        "by_fpd_year": by_cohort.to_dict(orient="records"),
        "maturity_by_fpd_quarter": maturity.astype(str).to_dict(orient="records"),
    }


def build_labels(cfg: DatasetConfig) -> dict:
    """Build the primary target and every configured sensitivity target from the same canonical tables."""
    primary = TargetSpec.load(cfg.target_config)
    sensitivities = [TargetSpec.load(p) for p in cfg.sensitivity_target_configs]
    pdir = cfg.processed_dir
    with work_connection(cfg.interim_dir / "_work", "labels") as con:
        con.execute(f"CREATE OR REPLACE VIEW orig AS SELECT * FROM read_parquet('{(pdir / 'loan_origination.parquet').as_posix()}')")
        con.execute(f"CREATE OR REPLACE VIEW perf AS SELECT * FROM read_parquet('{(pdir / 'loan_monthly_performance.parquet').as_posix()}')")
        report = _build_one(cfg, primary, con, outcomes_path(cfg))
        report["sensitivity_targets"] = {}
        for spec in sensitivities:
            r = _build_one(cfg, spec, con, outcomes_path(cfg, spec.target_id))
            write_json(pdir / f"label_report__{spec.target_id}.json", r)
            report["sensitivity_targets"][spec.target_id] = {
                k: r[k] for k in ("cohort_flow", "labelled_loans", "defaults", "prevalence_labelled")}
    write_json(pdir / "label_report.json", report)
    return report
