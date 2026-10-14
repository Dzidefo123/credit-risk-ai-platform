"""Minimal API client used by the analyst interface (standard library only).

The interface never touches model files or the audit store directly. Every action goes through the
authenticated API with the user's own key, so permissions and audit logging are enforced in one place.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

Transport = Callable[[str, str, dict, bytes | None], tuple[int, bytes]]


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


def urllib_transport(timeout: float = 10.0) -> Transport:
    def send(method: str, url: str, headers: dict, body: bytes | None) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - URL is operator-configured
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
    return send


@dataclass
class ApiClient:
    base_url: str
    api_key: str
    transport: Transport | None = None

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")
        status, raw = (self.transport or urllib_transport())(method, self.base_url.rstrip("/") + path, headers, body)
        data = json.loads(raw.decode("utf-8") or "{}") if raw else {}
        if status >= 400:
            err = data.get("error", {})
            raise ApiError(status, err.get("code", "http_error"), err.get("message", str(data)))
        return data

    def model(self) -> dict:
        return self._call("GET", "/v1/model")

    def score(self, application: dict, idempotency_key: str | None = None) -> dict:
        return self._call("POST", "/v1/score", {"application": application, "idempotency_key": idempotency_key})

    def scores(self, limit: int = 50, offset: int = 0) -> list[dict]:
        return self._call("GET", f"/v1/scores?limit={int(limit)}&offset={int(offset)}")["items"]

    def score_detail(self, score_id: str) -> dict:
        return self._call("GET", f"/v1/scores/{score_id}")

    def review(self, score_id: str, action: str, rationale: str, override_decision: str | None = None) -> dict:
        return self._call("POST", f"/v1/scores/{score_id}/reviews",
                          {"action": action, "rationale": rationale, "override_decision": override_decision})

    def replay(self, score_id: str) -> dict:
        return self._call("POST", f"/v1/scores/{score_id}/replay")

    def monitoring(self) -> dict:
        return self._call("GET", "/v1/monitoring")

    # ---- documents (SYNTHETIC) and copilot ------------------------------------------------------------------------
    def submit_documents(self, packet: dict) -> dict:
        return self._call("POST", "/v1/documents", packet)

    def documents(self, limit: int = 50) -> list[dict]:
        return self._call("GET", f"/v1/documents?limit={int(limit)}")["items"]

    def document(self, packet_id: str, include_documents: bool = False) -> dict:
        flag = "true" if include_documents else "false"
        return self._call("GET", f"/v1/documents/{packet_id}?include_documents={flag}")

    def correct_document(self, packet_id: str, field: str, value: str, rationale: str) -> dict:
        return self._call("POST", f"/v1/documents/{packet_id}/corrections",
                          {"field": field, "value": value, "rationale": rationale})

    def review_document(self, packet_id: str, action: str, rationale: str) -> dict:
        return self._call("POST", f"/v1/documents/{packet_id}/review", {"action": action, "rationale": rationale})

    def verified_inputs(self, packet_id: str) -> dict:
        return self._call("GET", f"/v1/documents/{packet_id}/verified-inputs")

    def ask(self, question: str) -> dict:
        return self._call("POST", "/v1/copilot/ask", {"question": question})
