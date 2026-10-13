"""Parse SFLLD pipe-delimited files into typed interim Parquet with quarantine.

Per source file the pipeline is:

1. Structural parse with DuckDB (all VARCHAR, exact column count). Lines with the wrong
   number of fields go to quarantine with their line number and raw text. If the
   structural reject rate exceeds the dataset threshold the whole file is refused: that
   pattern indicates a layout/version mismatch, not a few bad rows.
2. Typing driven by the contract: blank and sentinel codes -> NULL; values that fail
   type, allowed-value, range or pattern checks are recorded as field-level issues.
   Invalid key/required fields reject the row; other invalid values are set to NULL.
3. Valid rows are written to ``interim/<kind>/<file>.parquet`` with lineage columns.

Idempotence: each file's SHA-256 is recorded. Re-running with an unchanged file is a
no-op; a changed file (source correction) is reprocessed and its outputs replaced, and
the ingestion log keeps the superseded checksum. Raw files are never modified.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb

from crisk.config import DatasetConfig
from crisk.contracts.freddie_mac import (
    LOAN_ID_PATTERN_EMPIRICAL,
    LOAN_ID_PATTERN_SYNTHETIC,
    Field,
    FileContract,
    get_contract,
)
from crisk.duck import work_connection
from crisk.utils import append_jsonl, read_json, sha256_file, utc_now, write_json


class LayoutMismatchError(RuntimeError):
    """Raised when a file's structure does not match the contract version."""


# Pattern checks applied in addition to dtype/allowed/range: the contract's own field patterns,
# plus the loan-ID pattern, which depends on whether the dataset is empirical or synthetic.
def _patterns(contract: FileContract, profile: str) -> dict[str, str]:
    loan_id = LOAN_ID_PATTERN_EMPIRICAL if profile == "empirical" else LOAN_ID_PATTERN_SYNTHETIC
    out = {f.name: f.pattern for f in contract.fields if f.pattern}
    out.update({n: loan_id for n in ("loan_id", "pre_relief_refi_loan_id") if n in contract.names})
    return out


# Normalisations applied before validation (documented provider formatting variants).
_NORMALISE = {
    # Zero balance codes are 2-character codes ("01"); tolerate an unpadded "1".
    "zero_balance_code": "lpad({v}, 2, '0')",
}


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _clean_expr(f: Field, col: str) -> str:
    v = f"nullif(trim({col}), '')"
    if f.name in _NORMALISE:
        v = _NORMALISE[f.name].format(v=v)
    if f.sentinels:
        v = f"(CASE WHEN {v} IN ({', '.join(_q(s) for s in f.sentinels)}) THEN NULL ELSE {v} END)"
    return v


def _typed_expr(f: Field, clean: str) -> str:
    if f.dtype in ("str", "code"):
        return clean
    if f.dtype == "int":
        return f"TRY_CAST({clean} AS BIGINT)"
    if f.dtype == "float":
        return f"TRY_CAST({clean} AS DOUBLE)"
    if f.dtype == "yyyymm":
        return (f"(CASE WHEN regexp_full_match({clean}, '[0-9]{{6}}') "
                f"THEN TRY_CAST(try_strptime({clean} || '01', '%Y%m%d') AS DATE) END)")
    raise ValueError(f.dtype)


def _invalid_expr(f: Field, clean: str, typed: str, patterns: dict[str, str]) -> str:
    """TRUE when a non-missing value violates the contract."""
    conds = [f"{typed} IS NULL"]
    if f.allowed:
        conds.append(f"{clean} NOT IN ({', '.join(_q(a) for a in f.allowed)})")
    if f.min_value is not None and f.dtype in ("int", "float"):
        conds.append(f"{typed} < {f.min_value}")
    if f.max_value is not None and f.dtype in ("int", "float"):
        conds.append(f"{typed} > {f.max_value}")
    if f.name in patterns:
        conds.append(f"NOT regexp_full_match({clean}, {_q(patterns[f.name])})")
    return f"({clean} IS NOT NULL AND ({' OR '.join(conds)}))"


@dataclass
class FileResult:
    kind: str
    file: str
    sha256: str
    status: str  # loaded | skipped_unchanged | reloaded_changed
    lines: int = 0
    structural_rejects: int = 0
    row_rejects: int = 0
    rows_loaded: int = 0
    field_issues: int = 0


def _raw_columns(n: int) -> list[str]:
    return [f"c{i:02d}" for i in range(1, n + 1)]


def parse_file(con: duckdb.DuckDBPyConnection, path: Path, contract: FileContract, cfg: DatasetConfig,
               sha: str, run_ts: str) -> tuple[FileResult, dict[str, Path]]:
    """Parse one raw file; returns stats and the temporary output paths (not yet published)."""
    kind = contract.kind
    cols = _raw_columns(len(contract.fields))
    col_spec = "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in cols) + "}"
    p = path.as_posix()
    con.execute("DROP TABLE IF EXISTS reject_errors; DROP TABLE IF EXISTS reject_scans;")
    con.execute(f"""
        CREATE OR REPLACE TABLE raw_in AS
        SELECT * FROM read_csv({_q(p)}, delim='|', header=false, quote='', escape='', auto_detect=false,
                               columns={col_spec}, store_rejects=true, null_padding=false,
                               parallel=false)
    """)
    n_ok = con.execute("SELECT count(*) FROM raw_in").fetchone()[0]
    n_struct = con.execute("SELECT count(*) FROM reject_errors").fetchone()[0]
    total = n_ok + n_struct
    res = FileResult(kind=kind, file=path.name, sha256=sha, status="loaded", lines=total,
                     structural_rejects=n_struct)
    if total == 0:
        raise LayoutMismatchError(f"{path.name}: file is empty")
    if n_struct / total > cfg.max_structural_reject_rate:
        raise LayoutMismatchError(
            f"{path.name}: {n_struct}/{total} lines have the wrong field count for contract "
            f"{contract.version} ({len(cols)} fields). Refusing to load; check the layout version.")

    patterns = _patterns(contract, cfg.loan_id_profile)
    selects, issue_selects = [], []
    key_name = "loan_id"
    for f, c in zip(contract.fields, cols, strict=True):
        clean = _clean_expr(f, c)
        typed = _typed_expr(f, clean)
        inv = _invalid_expr(f, clean, typed, patterns)
        selects.append(f"{typed} AS __t_{f.name}")
        selects.append(f"{inv} AS __inv_{f.name}")
        selects.append(f"{c} AS __raw_{f.name}")
    con.execute(f"CREATE OR REPLACE TABLE typed_in AS SELECT {', '.join(selects)} FROM raw_in")

    reject_conds = []
    for f in contract.fields:
        if f.on_invalid == "reject":
            reject_conds.append(f"__inv_{f.name}")
        if f.required:
            reject_conds.append(f"__t_{f.name} IS NULL")
    reject_pred = " OR ".join(reject_conds) or "FALSE"

    # Field-level issue log (long format). Locator = loan_id (+ period) raw values.
    period_raw = "__raw_reporting_period" if kind == "performance" else "NULL"
    for f in contract.fields:
        action = "reject_row" if f.on_invalid == "reject" else "set_null"
        issue_selects.append(
            f"SELECT __raw_{key_name} AS loan_id_raw, {period_raw} AS period_raw, {_q(f.name)} AS field, "
            f"__raw_{f.name} AS raw_value, 'contract_violation' AS rule, {_q(action)} AS action "
            f"FROM typed_in WHERE __inv_{f.name}")
        if f.required:
            issue_selects.append(
                f"SELECT __raw_{key_name}, {period_raw}, {_q(f.name)}, __raw_{f.name}, 'required_missing', "
                f"'reject_row' FROM typed_in WHERE __t_{f.name} IS NULL AND NOT __inv_{f.name}")
    con.execute(f"CREATE OR REPLACE TABLE issues AS {' UNION ALL '.join(issue_selects)}")

    lineage = (f"{_q(path.name)} AS source_file, {_q(sha)} AS source_sha256, "
               f"{_q(contract.version)} AS contract_version, {_q(cfg.source_id)} AS source_id, "
               f"{_q(cfg.data_origin)} AS data_origin, {_q(run_ts)} AS ingested_at")
    out_cols = ", ".join(
        f"(CASE WHEN __inv_{f.name} THEN NULL ELSE __t_{f.name} END) AS {f.name}" for f in contract.fields)
    raw_cols = ", ".join(f"__raw_{f.name} AS {f.name}" for f in contract.fields)

    stem = path.stem
    out = {
        "valid": cfg.interim_dir / kind / f"{stem}.parquet",
        "rejected_rows": cfg.quarantine_dir / kind / f"{stem}__rejected_rows.parquet",
        "structural": cfg.quarantine_dir / kind / f"{stem}__structural.parquet",
        "issues": cfg.quarantine_dir / kind / f"{stem}__field_issues.parquet",
    }
    tmp = {k: v.with_suffix(".parquet.tmp") for k, v in out.items()}
    for v in out.values():
        v.parent.mkdir(parents=True, exist_ok=True)

    con.execute(f"COPY (SELECT {out_cols}, {lineage} FROM typed_in WHERE NOT ({reject_pred})) "
                f"TO {_q(tmp['valid'].as_posix())} (FORMAT parquet)")
    con.execute(f"COPY (SELECT {raw_cols}, {lineage} FROM typed_in WHERE {reject_pred}) "
                f"TO {_q(tmp['rejected_rows'].as_posix())} (FORMAT parquet)")
    con.execute(f"COPY (SELECT line, error_type, error_message, csv_line, {lineage} FROM reject_errors) "
                f"TO {_q(tmp['structural'].as_posix())} (FORMAT parquet)")
    con.execute(f"COPY (SELECT *, {lineage} FROM issues) TO {_q(tmp['issues'].as_posix())} (FORMAT parquet)")

    res.row_rejects = con.execute(f"SELECT count(*) FROM typed_in WHERE {reject_pred}").fetchone()[0]
    res.rows_loaded = n_ok - res.row_rejects
    res.field_issues = con.execute("SELECT count(*) FROM issues").fetchone()[0]
    return res, {k: (tmp[k], out[k]) for k in out}  # type: ignore[return-value]


def _manifest_path(cfg: DatasetConfig, kind: str, stem: str) -> Path:
    return cfg.interim_dir / "_ingestion" / f"{kind}__{stem}.json"


def discover_files(cfg: DatasetConfig) -> dict[str, list[Path]]:
    excluded = set(cfg.exclude_files)
    return {
        "origination": sorted(p for p in cfg.raw_dir.glob(cfg.origination_glob) if p.name not in excluded),
        "performance": sorted(p for p in cfg.raw_dir.glob(cfg.performance_glob) if p.name not in excluded),
    }


def ingest_dataset(cfg: DatasetConfig) -> list[FileResult]:
    files = discover_files(cfg)
    if not files["origination"] or not files["performance"]:
        raise FileNotFoundError(
            f"No raw files matching {cfg.origination_glob!r}/{cfg.performance_glob!r} in {cfg.raw_dir}")
    # Guard against filename-pattern overlap (e.g. a glob that matches both file types).
    overlap = set(files["origination"]) & set(files["performance"])
    if overlap:
        raise ValueError(f"Files matched both origination and performance globs: {sorted(overlap)}")

    run_ts = utc_now()
    results: list[FileResult] = []
    with work_connection(cfg.interim_dir / "_work", "ingest") as con:
        for kind, paths in files.items():
            contract = get_contract(cfg.contract_version, kind)
            for path in paths:
                sha = sha256_file(path)
                mpath = _manifest_path(cfg, kind, path.stem)
                prior = read_json(mpath) if mpath.exists() else None
                valid_out = cfg.interim_dir / kind / f"{path.stem}.parquet"
                if (prior and prior["sha256"] == sha and prior["contract_version"] == contract.version
                        and valid_out.exists()):
                    results.append(FileResult(kind=kind, file=path.name, sha256=sha, status="skipped_unchanged",
                                              rows_loaded=prior["rows_loaded"]))
                    continue
                res, outs = parse_file(con, path, contract, cfg, sha, run_ts)
                for tmp, final in outs.values():
                    os.replace(tmp, final)  # publish only after all outputs were written
                if prior:
                    res.status = "reloaded_changed"
                entry = {
                    **res.__dict__, "contract_version": contract.version, "source_id": cfg.source_id,
                    "data_origin": cfg.data_origin, "raw_path": str(path), "size_bytes": path.stat().st_size,
                    "ingested_at": run_ts, "supersedes_sha256": prior["sha256"] if prior else None,
                }
                write_json(mpath, entry)
                append_jsonl(cfg.interim_dir / "_ingestion" / "ingestion_log.jsonl", entry)
                results.append(res)
    return results


def loan_id_quarter(loan_id: str) -> tuple[int, int] | None:
    """Origination (note) year and quarter encoded in the loan sequence number."""
    m = re.match(r"^[A-Z]([0-9]{2})Q([1-4])", loan_id)
    if not m:
        return None
    yy = int(m.group(1))
    return (1900 + yy if yy >= 90 else 2000 + yy), int(m.group(2))
