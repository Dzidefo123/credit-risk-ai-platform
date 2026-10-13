"""Append-only audit store (SQLite) for scoring events and governance actions.

Prototype protections:
- UPDATE/DELETE on every table is blocked by triggers.
- ``audit_events`` form a SHA-256 hash chain; ``verify_chain`` detects edits made by
  bypassing the triggers (e.g. editing the file directly).
Production would add: WORM/object-lock storage or a managed ledger, separate write
credentials, encryption at rest, retention schedules and access logging. See docs/security.md.

Scoring events store the input snapshot. For EMPIRICAL Freddie Mac data the snapshot holds
provider fields only (no personal data is disclosed in the dataset); the store must stay in
the git-excluded artefacts directory because loan-level data may not be redistributed.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from crisk.utils import canonical_json, sha256_text, utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS scoring_events (
    score_id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE,
    request_hash TEXT NOT NULL,
    correlation_id TEXT,
    scored_at TEXT NOT NULL,
    status TEXT NOT NULL,
    data_origin TEXT NOT NULL,
    model_id TEXT,
    model_sha256 TEXT,
    feature_set_id TEXT,
    target_id TEXT,
    horizon_months INTEGER,
    pd REAL,
    pd_uncalibrated REAL,
    input_snapshot TEXT NOT NULL,
    feature_vector TEXT,
    dq_flags TEXT NOT NULL,
    reason_codes TEXT NOT NULL,
    response TEXT NOT NULL,
    code_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL
);
"""

_TABLES = ("scoring_events", "audit_events")


def _triggers() -> str:
    out = []
    for t in _TABLES:
        for op in ("UPDATE", "DELETE"):
            out.append(f"""CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t}
                BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;""")
    return "\n".join(out)


class IdempotencyConflict(RuntimeError):
    """Same idempotency key reused with a different request payload."""


GENESIS = "0" * 64


class AuditStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(SCHEMA + _triggers())

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.row_factory = sqlite3.Row
        return con

    def healthcheck(self) -> bool:
        try:
            with closing(self._connect()) as con:
                con.execute("SELECT 1 FROM audit_events LIMIT 1").fetchall()
            return True
        except sqlite3.Error:
            return False

    # ---- audit events ----------------------------------------------------------------------------
    def append_event(self, actor: str, action: str, entity_type: str, entity_id: str,
                     details: dict[str, Any] | None = None) -> dict:
        details_json = canonical_json(details or {})
        with closing(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                row = con.execute("SELECT event_hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
                prev = row["event_hash"] if row else GENESIS
                ts = utc_now()
                body = canonical_json({"ts": ts, "actor": actor, "action": action, "entity_type": entity_type,
                                       "entity_id": entity_id, "details": details_json, "prev_hash": prev})
                h = sha256_text(body)
                event = {"event_id": h[:32], "ts": ts, "actor": actor, "action": action,
                         "entity_type": entity_type, "entity_id": entity_id, "details": details_json,
                         "prev_hash": prev, "event_hash": h}
                con.execute("INSERT INTO audit_events (event_id, ts, actor, action, entity_type, entity_id, "
                            "details, prev_hash, event_hash) VALUES (:event_id, :ts, :actor, :action, "
                            ":entity_type, :entity_id, :details, :prev_hash, :event_hash)", event)
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
        return event

    def events(self, action: str | None = None, entity_id: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM audit_events WHERE 1=1", []
        if action:
            sql += " AND action = ?"
            args.append(action)
        if entity_id:
            sql += " AND entity_id = ?"
            args.append(entity_id)
        with closing(self._connect()) as con:
            rows = con.execute(sql + " ORDER BY seq", args).fetchall()
        return [{**dict(r), "details": json.loads(r["details"])} for r in rows]

    def verify_chain(self) -> tuple[bool, str]:
        prev = GENESIS
        with closing(self._connect()) as con:
            for r in con.execute("SELECT * FROM audit_events ORDER BY seq"):
                body = canonical_json({"ts": r["ts"], "actor": r["actor"], "action": r["action"],
                                       "entity_type": r["entity_type"], "entity_id": r["entity_id"],
                                       "details": r["details"], "prev_hash": prev})
                if r["prev_hash"] != prev or sha256_text(body) != r["event_hash"]:
                    return False, f"chain broken at seq {r['seq']}"
                prev = r["event_hash"]
        return True, "ok"

    # ---- scoring events --------------------------------------------------------------------------
    def find_by_idempotency_key(self, key: str) -> dict | None:
        with closing(self._connect()) as con:
            r = con.execute("SELECT * FROM scoring_events WHERE idempotency_key = ?", (key,)).fetchone()
        return dict(r) if r else None

    def get_score(self, score_id: str) -> dict | None:
        with closing(self._connect()) as con:
            r = con.execute("SELECT * FROM scoring_events WHERE score_id = ?", (score_id,)).fetchone()
        return dict(r) if r else None

    def insert_score(self, rec: dict) -> None:
        cols = ", ".join(rec)
        ph = ", ".join(f":{k}" for k in rec)
        try:
            with closing(self._connect()) as con:
                con.execute(f"INSERT INTO scoring_events ({cols}) VALUES ({ph})", rec)
        except sqlite3.IntegrityError as e:
            if "idempotency_key" in str(e):
                raise IdempotencyConflict(rec.get("idempotency_key")) from e
            raise

    def count_scores(self) -> int:
        with closing(self._connect()) as con:
            return con.execute("SELECT count(*) FROM scoring_events").fetchone()[0]
