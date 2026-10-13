from pathlib import Path

import duckdb
import pandas as pd
import pytest

from crisk.config import load_dataset_config
from crisk.fixtures.synthetic import generate, orig_line
from crisk.ingestion.raw_files import LayoutMismatchError, ingest_dataset

COMMITTED = Path(__file__).parent / "fixtures" / "sflld_synthetic_scenarios"


def test_fixture_generation_is_deterministic(tmp_path):
    generate(tmp_path / "a", n_random=200)
    generate(tmp_path / "b", n_random=200)
    for f in sorted((tmp_path / "a").iterdir()):
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes(), f.name


def test_committed_scenario_fixture_matches_generator(tmp_path):
    generate(tmp_path, n_random=0)
    committed = sorted(COMMITTED.iterdir())
    assert committed, "committed scenario fixture missing"
    for f in committed:
        assert f.read_bytes() == (tmp_path / f.name).read_bytes(), f"regenerate fixtures: {f.name} differs"


def test_ingestion_counts_match_answer_key(scenario_ws):
    res = {r.kind: r for r in scenario_ws["ingest"]}
    exp = scenario_ws["key"]["expected_quarantine"]
    assert res["origination"].row_rejects == exp["origination_rejected_rows"]
    assert res["performance"].structural_rejects == exp["performance_structural_rejects"]


def test_sentinels_and_invalid_values_become_null_with_issue(scenario_ws):
    cfg = scenario_ws["cfg"]
    orig = pd.read_parquet(cfg.processed_dir / "loan_origination.parquet").set_index("loan_id")
    assert pd.isna(orig.loc["X19Q10000024", "credit_score"])  # 9999 sentinel
    assert pd.isna(orig.loc["X19Q10000024", "orig_dti"])  # 999 sentinel
    assert pd.isna(orig.loc["X19Q10000023", "credit_score"])  # 1200 invalid
    issues = pd.concat(pd.read_parquet(p) for p in (cfg.quarantine_dir / "origination").glob("*field_issues*"))
    assert ((issues.loan_id_raw == "X19Q10000023") & (issues.field == "credit_score")).any()
    assert not ((issues.loan_id_raw == "X19Q10000024") & (issues.field == "credit_score")).any()  # sentinel != error
    perf_issues = pd.concat(pd.read_parquet(p) for p in (cfg.quarantine_dir / "performance").glob("*field_issues*"))
    assert ((perf_issues.loan_id_raw == "X19Q10000022") & (perf_issues.field == "delinquency_status")).any()
    assert {"source_file", "source_sha256", "contract_version", "data_origin"} <= set(orig.columns)
    assert set(orig.data_origin) == {"SYNTHETIC"}


def test_canonical_dedup_conflicts_and_orphans(scenario_ws):
    c = scenario_ws["canonical"]["checks"]
    exp = scenario_ws["key"]["expected_quarantine"]
    assert c["origination_exact_duplicates_removed"] == exp["origination_exact_duplicates_removed"]
    assert c["performance_exact_duplicates_removed"] == exp["performance_exact_duplicates_removed"]
    assert c["origination_conflicting_loans_quarantined"] == len(exp["origination_conflicting_loans"])
    assert c["performance_conflicting_loan_months_quarantined"] == exp["performance_conflicting_loan_months"]
    assert c["origination_loan_id_unique"] == 1 and c["performance_loan_month_unique"] == 1
    rej = pd.read_parquet(scenario_ws["cfg"].quarantine_dir / "canonical" / "performance_rejects.parquet")
    assert sorted(rej.loc[rej.reason == "orphan_performance", "loan_id"].unique()) == exp["orphan_loans"]


def _mtimes(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): p.stat().st_mtime_ns for p in d.rglob("*.parquet")}


def test_ingestion_is_idempotent_and_handles_corrections(tmp_path):
    cfg = load_dataset_config("synthetic_scenarios").with_root(tmp_path, raw_dir=tmp_path / "raw")
    generate(cfg.raw_dir, n_random=0)
    first = ingest_dataset(cfg)
    before = _mtimes(cfg.interim_dir)
    second = ingest_dataset(cfg)
    assert {r.status for r in second} == {"skipped_unchanged"}
    assert _mtimes(cfg.interim_dir) == before

    def n_rows() -> int:
        glob = (cfg.interim_dir / "origination").as_posix() + "/*.parquet"
        return duckdb.sql(f"select count(*) from read_parquet('{glob}')").fetchone()[0]

    n1 = n_rows()
    # Source correction: a new valid loan appears in the origination file.
    orig_file = next(cfg.raw_dir.glob("sample_orig_*.txt"))
    with orig_file.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(orig_line("X19Q18888888", "2019-03") + "\n")
    third = {r.kind: r for r in ingest_dataset(cfg)}
    assert third["origination"].status == "reloaded_changed"
    assert third["performance"].status == "skipped_unchanged"
    assert n_rows() == n1 + 1  # replaced, not appended twice
    log = (cfg.interim_dir / "_ingestion" / "ingestion_log.jsonl").read_text().strip().splitlines()
    assert len(log) == len(first) + 1
    orig_first = next(r for r in first if r.kind == "origination")
    assert orig_first.sha256 in log[-1]  # supersedes previous checksum


def test_previous_layout_still_parses_with_its_own_contract(tmp_path):
    """Files from releases before R47 load with the 2026-01 contract, and are refused under R47."""
    from dataclasses import replace

    from crisk.contracts.freddie_mac import CONTRACT_VERSION

    base = load_dataset_config("synthetic_scenarios").with_root(tmp_path, raw_dir=tmp_path / "raw")
    generate(base.raw_dir, n_random=0, contract_version=CONTRACT_VERSION)
    with pytest.raises(LayoutMismatchError):
        ingest_dataset(base)  # R47 contract vs 32-field files
    old = replace(base, contract_version=CONTRACT_VERSION)
    res = {r.kind: r for r in ingest_dataset(old)}
    assert res["origination"].row_rejects == 2 and res["performance"].structural_rejects == 1


def test_layout_mismatch_refuses_whole_file(tmp_path):
    cfg = load_dataset_config("synthetic_scenarios").with_root(tmp_path, raw_dir=tmp_path / "raw")
    generate(cfg.raw_dir, n_random=0)
    f = next(cfg.raw_dir.glob("sample_orig_*.txt"))
    raw_before = f.read_text()
    # Simulate a different layout version: 31 fields on every line.
    f.write_text("\n".join("|".join(line.split("|")[:-1]) for line in raw_before.splitlines()) + "\n")
    with pytest.raises(LayoutMismatchError):
        ingest_dataset(cfg)
    out = cfg.interim_dir / "origination"
    assert not out.exists() or not list(out.glob("*.parquet"))
