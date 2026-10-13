"""DuckDB connections that can process files larger than memory.

Work happens in a throw-away database file under ``interim/<dataset>/_work`` with an explicit
memory limit and spill directory, so full-volume SFLLD quarters (several GB uncompressed each)
are processed out of core on a laptop. Set ``CRISK_DUCKDB_MEMORY`` (e.g. "4GB") to change the limit.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

DEFAULT_MEMORY_LIMIT = "6GB"
DEFAULT_THREADS = 4  # fewer threads = less memory per operator on a 16 GB laptop


@contextmanager
def work_connection(work_dir: Path, name: str) -> Iterator[duckdb.DuckDBPyConnection]:
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / f"{name}.duckdb"
    spill = work_dir / f"{name}_spill"
    for p in (db, db.with_suffix(".duckdb.wal")):
        p.unlink(missing_ok=True)
    con = duckdb.connect(str(db))
    con.execute(f"SET memory_limit = '{os.environ.get('CRISK_DUCKDB_MEMORY', DEFAULT_MEMORY_LIMIT)}'")
    con.execute(f"SET temp_directory = '{spill.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET threads = {int(os.environ.get('CRISK_DUCKDB_THREADS', DEFAULT_THREADS))}")
    try:
        yield con
    finally:
        con.close()
        for p in (db, db.with_suffix(".duckdb.wal")):
            p.unlink(missing_ok=True)
        shutil.rmtree(spill, ignore_errors=True)
