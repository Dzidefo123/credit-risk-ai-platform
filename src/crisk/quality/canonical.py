"""Build canonical ``loan_origination`` and ``loan_monthly_performance`` tables.

Runs after per-file ingestion. Applies dataset-level rules that need all files:

Blocking (records quarantined):
- origination loan_id duplicated with conflicting content -> all copies quarantined
- performance (loan_id, reporting_period) duplicated with conflicting content -> all copies quarantined
  (the month becomes a reporting gap, which the label logic treats explicitly)
- performance rows whose loan_id has no clean origination record (orphans)
Resolved (logged):
- exact duplicates -> one copy kept
Warnings (counted; records kept, see report):
- maturity_date before first_payment_date; orig_loan_term inconsistent with dates
- CLTV < LTV (the provider says this is masked to Not Available, so it should not occur)
- loan_id year/quarter far from first_payment_date (heuristic: FPD normally 1-3 months after note)
- more than one zero balance record per loan; records after the zero balance period
- delinquency status jumps > 1 month between consecutive reported months (the guide documents
  irregular progressions, so this is informational)
- reporting period after the declared performance cutoff
- origination loans without any performance records
"""

from __future__ import annotations

from pathlib import Path

from crisk.config import DatasetConfig
from crisk.contracts.freddie_mac import get_contract
from crisk.duck import work_connection
from crisk.utils import utc_now, write_json


def _glob(p: Path) -> str:
    return (p / "*.parquet").as_posix()


def build_canonical(cfg: DatasetConfig) -> dict:
    out_dir = cfg.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    q_dir = cfg.quarantine_dir / "canonical"
    q_dir.mkdir(parents=True, exist_ok=True)
    ORIGINATION = get_contract(cfg.contract_version, "origination")
    PERFORMANCE = get_contract(cfg.contract_version, "performance")
    with work_connection(cfg.interim_dir / "_work", "canonical") as con:
        o_cols = ", ".join(ORIGINATION.names)
        p_cols = ", ".join(PERFORMANCE.names)
        con.execute(f"CREATE OR REPLACE VIEW o_in AS SELECT * FROM read_parquet('{_glob(cfg.interim_dir / 'origination')}')")
        con.execute(f"CREATE OR REPLACE VIEW p_in AS SELECT * FROM read_parquet('{_glob(cfg.interim_dir / 'performance')}')")

        # --- origination de-duplication -----------------------------------------------------------
        con.execute(f"""
            CREATE OR REPLACE TABLE o_h AS
            SELECT *, md5(concat_ws('|', {', '.join(f"coalesce(CAST({c} AS VARCHAR), '<null>')" for c in ORIGINATION.names)})) AS content_hash
            FROM o_in""")
        con.execute("""
            CREATE OR REPLACE TABLE o_dup AS
            SELECT loan_id, count(*) AS n_copies, count(DISTINCT content_hash) AS n_versions
            FROM o_h GROUP BY loan_id HAVING count(*) > 1""")
        con.execute(f"""
            CREATE OR REPLACE TABLE o_clean AS
            SELECT {o_cols}, source_file, source_sha256, contract_version, source_id, data_origin, ingested_at
            FROM (
                SELECT *, row_number() OVER (PARTITION BY loan_id ORDER BY source_file) AS rn FROM o_h
            ) WHERE rn = 1
              AND loan_id NOT IN (SELECT loan_id FROM o_dup WHERE n_versions > 1)""")

        # --- performance de-duplication (scales to 10^8+ rows) ----------------------------------------
        # Only keys that occur more than once are hashed; the full table is never materialised in the
        # work DB. Plain GROUP BY / SEMI / ANTI joins can spill to disk, DISTINCT aggregates over all rows cannot.
        lineage = "source_file, source_sha256, contract_version, source_id, data_origin, ingested_at"
        p_hash = ", ".join(f"coalesce(CAST({c} AS VARCHAR), '<null>')" for c in PERFORMANCE.names)
        con.execute("""
            CREATE OR REPLACE TABLE p_dupkeys AS
            SELECT loan_id, reporting_period, count(*) AS n_copies FROM p_in GROUP BY 1, 2 HAVING count(*) > 1""")
        con.execute(f"""
            CREATE OR REPLACE TABLE p_duprows AS
            SELECT p.*, md5(concat_ws('|', {p_hash})) AS content_hash
            FROM p_in p SEMI JOIN p_dupkeys k USING (loan_id, reporting_period)""")
        con.execute("""
            CREATE OR REPLACE TABLE p_dup AS
            SELECT loan_id, reporting_period, count(*) AS n_copies, count(DISTINCT content_hash) AS n_versions
            FROM p_duprows GROUP BY 1, 2""")
        con.execute("CREATE OR REPLACE TABLE o_keys AS SELECT loan_id FROM o_clean")
        p_out = out_dir / "loan_monthly_performance.parquet"
        con.execute(f"""
            COPY (
                SELECT {p_cols}, {lineage} FROM p_in
                ANTI JOIN p_dupkeys USING (loan_id, reporting_period)
                SEMI JOIN o_keys USING (loan_id)
                UNION ALL
                SELECT {p_cols}, {lineage} FROM (
                    SELECT *, row_number() OVER (PARTITION BY loan_id, reporting_period ORDER BY source_file) AS rn
                    FROM p_duprows
                    SEMI JOIN (SELECT loan_id, reporting_period FROM p_dup WHERE n_versions = 1)
                         USING (loan_id, reporting_period)
                ) SEMI JOIN o_keys USING (loan_id)
                WHERE rn = 1
            ) TO '{p_out.as_posix()}' (FORMAT parquet)""")
        con.execute(f"CREATE OR REPLACE VIEW p_clean AS SELECT * FROM read_parquet('{p_out.as_posix()}')")

        # --- quarantine outputs -------------------------------------------------------------------
        con.execute(f"""COPY (SELECT h.*, 'origination_duplicate_conflict' AS reason FROM o_h h
                     JOIN o_dup d USING (loan_id) WHERE d.n_versions > 1)
                     TO '{(q_dir / 'origination_conflicts.parquet').as_posix()}' (FORMAT parquet)""")
        con.execute(f"""COPY (
                SELECT h.* EXCLUDE (content_hash), 'performance_duplicate_conflict' AS reason FROM p_duprows h
                JOIN p_dup d USING (loan_id, reporting_period) WHERE d.n_versions > 1
                UNION ALL BY NAME
                SELECT *, 'orphan_performance' AS reason FROM p_in ANTI JOIN o_keys USING (loan_id)
            ) TO '{(q_dir / 'performance_rejects.parquet').as_posix()}' (FORMAT parquet)""")

        cutoff = cfg.performance_cutoff + "-01"
        checks: dict[str, int] = {}

        def count(name: str, sql: str) -> None:
            checks[name] = int(con.execute(sql).fetchone()[0])

        count("origination_input_rows", "SELECT count(*) FROM o_in")
        count("origination_exact_duplicates_removed",
              "SELECT coalesce(sum(n_copies - 1), 0) FROM o_dup WHERE n_versions = 1")
        count("origination_conflicting_loans_quarantined", "SELECT count(*) FROM o_dup WHERE n_versions > 1")
        count("origination_clean_loans", "SELECT count(*) FROM o_clean")
        count("origination_loan_id_unique",
              "SELECT CASE WHEN count(*) = count(DISTINCT loan_id) THEN 1 ELSE 0 END FROM o_clean")
        count("performance_input_rows", "SELECT count(*) FROM p_in")
        count("performance_exact_duplicates_removed",
              "SELECT coalesce(sum(n_copies - 1), 0) FROM p_dup WHERE n_versions = 1")
        count("performance_conflicting_loan_months_quarantined", "SELECT count(*) FROM p_dup WHERE n_versions > 1")
        # orphans counted on de-duplicated keys so the number is comparable with the small-data path
        count("performance_orphan_rows_quarantined", """
              SELECT count(*) FROM (SELECT DISTINCT loan_id, reporting_period FROM p_dupkeys
                                    UNION ALL
                                    SELECT loan_id, reporting_period FROM p_in ANTI JOIN p_dupkeys USING (loan_id, reporting_period))
              ANTI JOIN o_keys USING (loan_id)""")
        count("performance_clean_rows", "SELECT count(*) FROM p_clean")
        count("performance_loan_month_unique", """
              SELECT CASE WHEN count(*) = 0 THEN 1 ELSE 0 END FROM (
                SELECT loan_id, reporting_period FROM p_clean GROUP BY 1, 2 HAVING count(*) > 1)""")
        # warnings
        count("warn_maturity_before_first_payment",
              "SELECT count(*) FROM o_clean WHERE maturity_date < first_payment_date")
        count("warn_loan_term_inconsistent_with_dates",
              "SELECT count(*) FROM o_clean WHERE maturity_date IS NOT NULL AND orig_loan_term IS NOT NULL "
              "AND datediff('month', first_payment_date, maturity_date) + 1 <> orig_loan_term")
        count("warn_cltv_below_ltv", "SELECT count(*) FROM o_clean WHERE orig_cltv < orig_ltv")
        count("warn_fpd_far_from_loan_id_quarter", """
              SELECT count(*) FROM (
                SELECT first_payment_date,
                       make_date(CASE WHEN CAST(substr(loan_id, 2, 2) AS INT) >= 90 THEN 1900 ELSE 2000 END
                                 + CAST(substr(loan_id, 2, 2) AS INT),
                                 (CAST(substr(loan_id, 5, 1) AS INT) - 1) * 3 + 1, 1) AS q_start
                FROM o_clean)
              WHERE datediff('month', q_start, first_payment_date) NOT BETWEEN 0 AND 6""")
        count("warn_origination_without_performance",
              "SELECT count(*) FROM o_keys ANTI JOIN (SELECT DISTINCT loan_id FROM p_clean) USING (loan_id)")
        con.execute("""CREATE OR REPLACE TABLE zb AS
                       SELECT loan_id, reporting_period, zero_balance_effective_date FROM p_clean
                       WHERE zero_balance_code IS NOT NULL""")
        count("warn_multiple_zero_balance_records",
              "SELECT count(*) FROM (SELECT loan_id FROM zb GROUP BY loan_id HAVING count(*) > 1)")
        count("warn_records_after_zero_balance", """
              SELECT count(*) FROM p_clean p JOIN (
                SELECT loan_id, min(reporting_period) AS zb_period FROM zb GROUP BY loan_id) z USING (loan_id)
              WHERE p.reporting_period > z.zb_period""")
        count("warn_zero_balance_code_without_date",
              "SELECT count(*) FROM zb WHERE zero_balance_effective_date IS NULL")
        # status rose by more than one month between consecutive months: only rows with status >= 2 can qualify
        con.execute("""CREATE OR REPLACE TABLE dq AS
                       SELECT loan_id, reporting_period, TRY_CAST(delinquency_status AS INT) AS s FROM p_clean
                       WHERE TRY_CAST(delinquency_status AS INT) >= 2""")
        count("warn_delinquency_jump_gt_1", """
              SELECT count(*) FROM dq d JOIN p_clean p
                ON p.loan_id = d.loan_id AND p.reporting_period = CAST(d.reporting_period - INTERVAL 1 MONTH AS DATE)
              WHERE d.s - TRY_CAST(p.delinquency_status AS INT) > 1""")
        count("warn_period_after_declared_cutoff",
              f"SELECT count(*) FROM p_clean WHERE reporting_period > DATE '{cutoff}'")
        count("warn_upb_above_original_no_mod_or_deferral", """
              SELECT count(*) FROM p_clean p JOIN o_clean o USING (loan_id)
              WHERE p.modification_flag IS NULL AND p.payment_deferral_flag IS NULL
                AND coalesce(p.current_non_interest_bearing_upb, 0) = 0
                AND p.current_actual_upb > o.orig_upb + 1000""")

        observed = con.execute("SELECT min(reporting_period), max(reporting_period) FROM p_clean").fetchone()
        fpd_range = con.execute("SELECT min(first_payment_date), max(first_payment_date) FROM o_clean").fetchone()

        con.execute(f"COPY (SELECT * FROM o_clean ORDER BY loan_id) TO "
                    f"'{(out_dir / 'loan_origination.parquet').as_posix()}' (FORMAT parquet)")

    blocking_ok = checks["origination_loan_id_unique"] == 1 and checks["performance_loan_month_unique"] == 1
    report = {
        "dataset_id": cfg.dataset_id,
        "data_origin": cfg.data_origin,
        "built_at": utc_now(),
        "declared_performance_cutoff": cfg.performance_cutoff,
        "observed_reporting_period_range": [str(observed[0]), str(observed[1])],
        "first_payment_date_range": [str(fpd_range[0]), str(fpd_range[1])],
        "checks": checks,
        "uniqueness_ok": blocking_ok,
    }
    write_json(out_dir / "canonical_build_report.json", report)
    if not blocking_ok:
        raise RuntimeError("Canonical uniqueness checks failed; see canonical_build_report.json")
    return report
