"""Small shared helpers: hashing, atomic writes, run metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from crisk.config import REPO_ROOT


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing and audit snapshots."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical_json(obj) + "\n")


def code_version() -> dict[str, Any]:
    """Git commit and dirty flag; 'uncommitted' when the repo has no commits yet."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": "uncommitted", "git_dirty": True}


def environment_info() -> dict[str, Any]:
    pkgs = {}
    for name in ("duckdb", "pandas", "numpy", "pyarrow", "scikit-learn", "joblib", "fastapi", "pydantic"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs}


def month_add(yyyymm: str, months: int) -> str:
    """'2019-01' + 3 -> '2019-04'. Accepts 'YYYY-MM' or 'YYYYMM'."""
    s = yyyymm.replace("-", "")
    y, m = int(s[:4]), int(s[4:6])
    idx = y * 12 + (m - 1) + months
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def month_diff(a: str, b: str) -> int:
    """Months from b to a (a - b) for 'YYYY-MM' strings."""
    sa, sb = a.replace("-", ""), b.replace("-", "")
    return (int(sa[:4]) * 12 + int(sa[4:6])) - (int(sb[:4]) * 12 + int(sb[4:6]))
