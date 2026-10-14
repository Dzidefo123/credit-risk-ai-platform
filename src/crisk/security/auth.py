"""API-key authentication and role-based authorisation (prototype).

Keys look like ``crk_<key_id>_<secret>``. Only the SHA-256 of the full key is stored, in
``artifacts/<dataset>/api_keys.json`` (git-excluded). A key is shown once, when created. Lookup is by
``key_id``, and the hash comparison is constant-time. Creating and revoking keys is written to the
audit chain.

Roles follow the governance roles in the master prompt. Permissions are deliberately narrow: no role can
approve loans, promote models or change policy through the API.

Production would use an identity provider (OIDC/SAML) with short-lived tokens, a secrets manager, rate
limiting and per-tenant isolation. See docs/data_handling_and_security.md.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from pathlib import Path

from crisk.audit.store import AuditStore
from crisk.utils import read_json, utc_now, write_json

ROLES = ("modeller", "validator", "analyst", "administrator", "auditor")

PERMISSIONS: dict[str, set[str]] = {
    "score:create": {"analyst", "modeller", "administrator"},
    "score:read": {"analyst", "auditor", "validator", "administrator"},
    "score:replay": {"auditor", "validator", "administrator"},
    "review:create": {"analyst", "administrator"},
    "decision:create": {"analyst", "administrator"},
    "monitoring:read": {"analyst", "auditor", "validator", "modeller", "administrator"},
    "model:read": set(ROLES),
    # Phase G/H: synthetic document intake and review; copilot questions (read-only answers, role-filtered sources).
    "document:create": {"analyst", "administrator"},
    "document:read": {"analyst", "validator", "auditor", "administrator"},
    "document:review": {"analyst", "administrator"},
    "copilot:ask": set(ROLES),
}


@dataclass(frozen=True)
class Principal:
    key_id: str
    name: str
    role: str

    def can(self, permission: str) -> bool:
        return self.role in PERMISSIONS.get(permission, set())


class AuthError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KeyStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> dict:
        return read_json(self.path) if self.path.exists() else {"keys": {}}

    def create(self, name: str, role: str, store: AuditStore, actor: str = "administrator-cli") -> tuple[str, dict]:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if not name.strip():
            raise ValueError("a key needs a holder name")
        data = self._load()
        key_id = secrets.token_hex(4)
        while key_id in data["keys"]:
            key_id = secrets.token_hex(4)
        key = f"crk_{key_id}_{secrets.token_urlsafe(24)}"
        entry = {"key_id": key_id, "name": name.strip(), "role": role, "sha256": _hash(key),
                 "created_at": utc_now(), "revoked_at": None}
        data["keys"][key_id] = entry
        write_json(self.path, data)
        store.append_event(actor=actor, action="api_key_created", entity_type="api_key", entity_id=key_id,
                           details={"name": entry["name"], "role": role})
        return key, {k: v for k, v in entry.items() if k != "sha256"}

    def revoke(self, key_id: str, store: AuditStore, actor: str = "administrator-cli") -> dict:
        data = self._load()
        if key_id not in data["keys"]:
            raise KeyError(key_id)
        data["keys"][key_id]["revoked_at"] = utc_now()
        write_json(self.path, data)
        store.append_event(actor=actor, action="api_key_revoked", entity_type="api_key", entity_id=key_id,
                           details={"name": data["keys"][key_id]["name"]})
        return {k: v for k, v in data["keys"][key_id].items() if k != "sha256"}

    def list(self) -> list[dict]:
        return [{k: v for k, v in e.items() if k != "sha256"} for e in self._load()["keys"].values()]

    def authenticate(self, presented: str | None) -> Principal:
        if not presented:
            raise AuthError(401, "unauthenticated", "Missing API key (Authorization: Bearer <key>)")
        parts = presented.split("_", 2)
        if len(parts) != 3 or parts[0] != "crk":
            raise AuthError(401, "unauthenticated", "Malformed API key")
        entry = self._load()["keys"].get(parts[1])
        # Compare against a dummy hash for unknown ids so timing does not reveal which key ids exist.
        expected = entry["sha256"] if entry else _hash("crk_unknown_" + secrets.token_hex(8))
        if not hmac.compare_digest(_hash(presented), expected) or entry is None:
            raise AuthError(401, "unauthenticated", "Invalid API key")
        if entry.get("revoked_at"):
            raise AuthError(401, "unauthenticated", "API key revoked")
        return Principal(key_id=entry["key_id"], name=entry["name"], role=entry["role"])


def authorise(principal: Principal, permission: str) -> None:
    if not principal.can(permission):
        raise AuthError(403, "forbidden", f"Role '{principal.role}' lacks permission '{permission}'")


def key_store_path(cfg) -> Path:
    return cfg.artifacts_dir / "api_keys.json"
