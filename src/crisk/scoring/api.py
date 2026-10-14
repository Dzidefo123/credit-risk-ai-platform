"""FastAPI scoring service with API-key authentication and role-based permissions (prototype).

No `from __future__ import annotations` here: FastAPI resolves the locally built request model
from real annotations.

Endpoints (auth = ``Authorization: Bearer <key>`` or ``X-API-Key``; see crisk.security.auth):
  GET  /health, /ready                     no auth (liveness/readiness only, no data)
  GET  /v1/model                           model:read    champion, approved use, evidence links
  POST /v1/score                           score:create
  GET  /v1/scores?limit=&offset=           score:read    recent scoring events (summary)
  GET  /v1/scores/{score_id}               score:read    full record incl. input snapshot and reviews
  POST /v1/scores/{score_id}/replay        score:replay
  POST /v1/scores/{score_id}/reviews       review:create human review: agree | override | escalate
  POST /v1/scores/{score_id}/decision      decision:create policy recommendation (versioned policy engine)
  GET  /v1/monitoring                      monitoring:read latest monitoring report (drift, outcomes, alerts)
  POST /v1/documents                       document:create submit a SYNTHETIC document packet (extract + validate)
  GET  /v1/documents                       document:read   recent packets
  GET  /v1/documents/{packet_id}           document:read   extraction, flags, status, reviewer actions
  POST /v1/documents/{packet_id}/corrections  document:review correct a field (rationale required)
  POST /v1/documents/{packet_id}/review    document:review accept open acceptable flags, or reject
  GET  /v1/documents/{packet_id}/verified-inputs  document:read  only for validated packets (409 otherwise)
  POST /v1/copilot/ask                     copilot:ask     cited answer from role-permitted sources + read-only tools
Errors use one JSON shape: {"error": {"code", "message", "correlation_id"}}.
Still a prototype: no TLS termination, rate limiting or identity provider. Keep it on localhost.
"""

import json
import logging
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from crisk.audit.store import AuditStore, IdempotencyConflict
from crisk.config import DatasetConfig
from crisk.copilot.copilot import Caller, Copilot, Tools
from crisk.copilot.corpus import Corpus
from crisk.docai.schema import PacketIn
from crisk.docai.service import DocumentService, UnvalidatedExtractionError, WorkflowError, verified_inputs
from crisk.features.origination import FeatureSpec
from crisk.llm import get_llm
from crisk.models.registry import champion
from crisk.policies.engine import Policy, PolicyError
from crisk.scoring.schemas import ScoreResponse, application_model
from crisk.scoring.service import ModelUnavailableError, ScoringService, replay
from crisk.security.auth import AuthError, KeyStore, Principal, authorise, key_store_path
from crisk.utils import utc_now

log = logging.getLogger("crisk.api")


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["agree", "override", "escalate"]
    override_decision: Literal["approve", "decline", "refer"] | None = None
    rationale: str = Field(min_length=10, max_length=2000)

    @model_validator(mode="after")
    def _override_needs_decision(self):
        if (self.action == "override") != (self.override_decision is not None):
            raise ValueError("override_decision is required for (and only for) action 'override'")
        return self


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(min_length=3, max_length=64, description="doc_type.field, e.g. paystub.ytd_gross")
    value: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=10, max_length=2000)


class DocReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["accept", "reject"]
    rationale: str = Field(min_length=10, max_length=2000)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=1000)


def _error(status: int, code: str, message: str, cid: str | None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
                                                                "correlation_id": cid}},
                        headers={"X-Correlation-ID": cid or ""})


def _summary(rec: dict, latest_review: dict | None) -> dict:
    resp = json.loads(rec["response"])
    return {"score_id": rec["score_id"], "scored_at": rec["scored_at"], "status": rec["status"], "pd": rec["pd"],
            "risk_decile": resp.get("risk_decile"), "model_id": rec["model_id"],
            "requested_by": resp.get("requested_by"),
            "approved_use": resp.get("approved_use"), "data_origin": rec["data_origin"],
            "review_status": latest_review["action"] if latest_review else "pending",
            "override_decision": latest_review.get("override_decision") if latest_review else None}


DEFAULT_POLICY = "configs/policies/demo_policy_v1.yaml"


def create_app(cfg: DatasetConfig, model_id: str | None = None, keys: KeyStore | None = None,
               policy_path: str = DEFAULT_POLICY, llm=None, corpus: Corpus | None = None) -> FastAPI:
    store = AuditStore(cfg.audit_db)
    docs = DocumentService(store)
    copilot_state: dict = {"copilot": None, "error": None}
    try:
        copilot_state["copilot"] = Copilot(corpus or Corpus.load(), llm or get_llm(), Tools(cfg, store), store)
    except Exception as e:  # the scoring service still starts; copilot endpoints report 503
        copilot_state["error"] = f"{type(e).__name__}: {e}"
        log.error("copilot unavailable: %s", copilot_state["error"])
    keys = keys or KeyStore(key_store_path(cfg))
    app = FastAPI(title="Credit risk PD scoring (prototype)", version="0.2.0")
    state: dict = {"service": None, "load_error": None}
    try:
        state["service"] = ScoringService(cfg, store, model_id=model_id)
    except Exception as e:  # service still starts so /health and /ready report the failure
        state["load_error"] = f"{type(e).__name__}: {e}"
        log.error("model load failed: %s", state["load_error"])

    # Request schema follows the served model's feature set (falls back to the dataset default).
    # Built with create_model: a locally defined class annotation would be a forward reference here.
    spec = state["service"].bundle.feature_spec if state["service"] else FeatureSpec.load(cfg.feature_config)
    Application = application_model(spec)
    ScoreRequest = create_model("ScoreRequest", __config__=ConfigDict(extra="forbid"),
                                application=(Application, ...), idempotency_key=(str | None, None))

    def principal(request: Request) -> Principal:
        auth = request.headers.get("Authorization", "")
        presented = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("X-API-Key")
        return keys.authenticate(presented)

    def needs(permission: str):
        def dep(p: Principal = Depends(principal)) -> Principal:  # noqa: B008
            authorise(p, permission)
            return p
        return dep

    @app.middleware("http")
    async def correlation(request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        request.state.cid = cid
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    @app.exception_handler(AuthError)
    async def auth_handler(request: Request, exc: AuthError):
        return _error(exc.status, exc.code, exc.message, getattr(request.state, "cid", None))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return _error(422, "invalid_request", details, getattr(request.state, "cid", None))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        svc = state["service"]
        ok = svc is not None and store.healthcheck()
        body = {"ready": ok, "model_id": svc.model_id if svc else None,
                "data_origin": svc.meta["data_origin"] if svc else None, "load_error": state["load_error"]}
        return JSONResponse(status_code=200 if ok else 503, content=body)

    @app.get("/v1/model")
    def model_info(p: Principal = Depends(needs("model:read"))):  # noqa: B008
        svc: ScoringService | None = state["service"]
        ch = champion(cfg)
        if svc is None:
            return {"model_id": None, "load_error": state["load_error"], "champion": ch}
        return {"model_id": svc.model_id, "model_family": svc.meta["model_family"],
                "feature_set_id": svc.meta["feature_set_id"], "target": svc._envelope()["target"],
                "input_fields": svc.bundle.feature_spec.raw_inputs,
                "data_origin": svc.meta["data_origin"], "evidence_note": svc._envelope()["evidence_note"],
                "approved_use": svc.approved_use,
                "approved_use_text": (ch or {}).get("approved_use_text") if svc.approved_use != "not_promoted" else
                "Model is not the promoted champion.",
                "champion": {k: (ch or {}).get(k) for k in ("promoted_at", "reviewer", "rationale",
                                                            "risk_decile_reference")},
                "caller": {"name": p.name, "role": p.role}}

    create_dep = Depends(needs("score:create"))

    @app.post("/v1/score", response_model=ScoreResponse)
    def score(req: ScoreRequest, request: Request, p: Principal = create_dep):  # type: ignore[valid-type]
        cid = request.state.cid
        svc: ScoringService | None = state["service"]
        if svc is None:
            return _error(503, "model_unavailable", "No validated model is loaded; no score was produced.", cid)
        try:
            return svc.score(req.application.model_dump(), idempotency_key=req.idempotency_key, correlation_id=cid,
                             requested_by=p.name)
        except IdempotencyConflict:
            return _error(409, "idempotency_conflict",
                          "idempotency_key was already used with a different application payload", cid)
        except ModelUnavailableError as e:
            return _error(503, "model_unavailable", str(e), cid)
        except Exception:
            log.exception("scoring failed cid=%s", cid)  # no payload in logs
            return _error(500, "internal_error", "Scoring failed; no score was produced.", cid)

    @app.get("/v1/scores")
    def list_scores(limit: int = 50, offset: int = 0, p: Principal = Depends(needs("score:read"))):  # noqa: B008
        limit = max(1, min(int(limit), 500))
        latest = {}
        for r in store.reviews():
            latest[r["score_id"]] = r  # ordered by time: last one wins
        return {"items": [_summary(rec, latest.get(rec["score_id"])) for rec in store.list_scores(limit, offset)],
                "limit": limit, "offset": offset}

    @app.get("/v1/scores/{score_id}")
    def get_score(score_id: str, request: Request, p: Principal = Depends(needs("score:read"))):  # noqa: B008
        rec = store.get_score(score_id)
        if rec is None:
            return _error(404, "not_found", "Unknown score_id", request.state.cid)
        reviews = store.reviews(score_id)
        return {**_summary(rec, reviews[-1] if reviews else None), "response": json.loads(rec["response"]),
                "input_snapshot": json.loads(rec["input_snapshot"]), "feature_vector": json.loads(rec["feature_vector"]
                                                                                              or "{}"),
                "model_sha256": rec["model_sha256"], "code_version": json.loads(rec["code_version"]),
                "reviews": reviews, "policy_decisions": store.decisions(score_id)}

    @app.post("/v1/scores/{score_id}/replay")
    def replay_score(score_id: str, request: Request, p: Principal = Depends(needs("score:replay"))):  # noqa: B008
        if store.get_score(score_id) is None:
            return _error(404, "not_found", "Unknown score_id", request.state.cid)
        return replay(cfg, store, score_id, actor=f"{p.name} ({p.role})")

    @app.post("/v1/scores/{score_id}/reviews", status_code=201)
    def add_review(score_id: str, body: ReviewRequest, request: Request,
                   p: Principal = Depends(needs("review:create"))):  # noqa: B008
        rec = store.get_score(score_id)
        if rec is None:
            return _error(404, "not_found", "Unknown score_id", request.state.cid)
        if rec["status"] != "scored" and body.action == "agree":
            return _error(409, "not_scored", "Cannot agree with an unavailable score; escalate or override",
                          request.state.cid)
        review = {"review_id": uuid.uuid4().hex, "score_id": score_id, "reviewed_at": utc_now(),
                  "reviewer": p.name, "reviewer_role": p.role, "action": body.action,
                  "override_decision": body.override_decision, "rationale": body.rationale}
        store.add_review(review)
        store.append_event(actor=f"{p.name} ({p.role})", action=f"review_{body.action}", entity_type="score",
                           entity_id=score_id, details={"review_id": review["review_id"],
                                                        "override_decision": body.override_decision})
        return review

    @app.get("/v1/monitoring")
    def monitoring(request: Request, p: Principal = Depends(needs("monitoring:read"))):  # noqa: B008
        latest = cfg.reports_dir / "monitoring" / "latest.json"
        if not latest.exists():
            return _error(404, "not_found", "No monitoring report yet (run `crisk monitor`)", request.state.cid)
        return json.loads(latest.read_text(encoding="utf-8"))

    @app.post("/v1/scores/{score_id}/decision", status_code=201)
    def decide(score_id: str, request: Request, p: Principal = Depends(needs("decision:create"))):  # noqa: B008
        rec = store.get_score(score_id)
        if rec is None:
            return _error(404, "not_found", "Unknown score_id", request.state.cid)
        try:
            policy = Policy.load(policy_path)
            resp = json.loads(rec["response"])
            out = policy.evaluate(json.loads(rec["input_snapshot"]), resp, resp.get("approved_use", "not_promoted"))
        except PolicyError as e:
            return _error(409, "policy_not_applicable", str(e), request.state.cid)
        row = {"decision_id": uuid.uuid4().hex, "score_id": score_id, "decided_at": utc_now(),
               "requested_by": f"{p.name} ({p.role})", "policy_id": out["policy_id"],
               "policy_version": out["policy_version"], "policy_sha256": out["policy_sha256"],
               "decision": out["decision"], "detail": json.dumps(out, default=str)}
        store.add_decision(row)
        store.append_event(actor=row["requested_by"], action="policy_decision", entity_type="score",
                           entity_id=score_id, details={"decision_id": row["decision_id"], "decision": out["decision"],
                                                        "policy_version": out["policy_version"]})
        return {**out, "decision_id": row["decision_id"], "decided_at": row["decided_at"],
                "note": "Policy recommendation for human review; FICTIONAL demonstration policy."}

    # ---- Phase G: documents (SYNTHETIC only) ----------------------------------------------------------------------
    @app.post("/v1/documents", status_code=201)
    def submit_documents(body: PacketIn, request: Request,
                         p: Principal = Depends(needs("document:create"))):  # noqa: B008
        try:
            return docs.submit(body.model_dump(), actor=p.name, role=p.role)
        except WorkflowError as e:
            return _error(422, "invalid_packet", str(e), request.state.cid)

    @app.get("/v1/documents")
    def list_documents(limit: int = 50, p: Principal = Depends(needs("document:read"))):  # noqa: B008
        items = []
        for row in store.doc_packets(max(1, min(int(limit), 200))):
            st = docs.state(row["packet_id"])
            items.append({**row, "status": st["status"], "open_flags": st["open_flags"]})
        return {"items": items}

    @app.get("/v1/documents/{packet_id}")
    def get_document(packet_id: str, request: Request, include_documents: bool = False,
                     p: Principal = Depends(needs("document:read"))):  # noqa: B008
        try:
            return docs.state(packet_id, include_documents=include_documents)
        except KeyError:
            return _error(404, "not_found", "Unknown packet_id", request.state.cid)

    @app.post("/v1/documents/{packet_id}/corrections")
    def correct_document(packet_id: str, body: CorrectionRequest, request: Request,
                         p: Principal = Depends(needs("document:review"))):  # noqa: B008
        try:
            return docs.correct(packet_id, body.field, body.value, body.rationale, actor=p.name, role=p.role)
        except KeyError:
            return _error(404, "not_found", "Unknown packet_id", request.state.cid)
        except WorkflowError as e:
            return _error(409, "workflow_conflict", str(e), request.state.cid)

    @app.post("/v1/documents/{packet_id}/review")
    def review_document(packet_id: str, body: DocReviewRequest, request: Request,
                        p: Principal = Depends(needs("document:review"))):  # noqa: B008
        try:
            fn = docs.accept if body.action == "accept" else docs.reject
            return fn(packet_id, body.rationale, actor=p.name, role=p.role)
        except KeyError:
            return _error(404, "not_found", "Unknown packet_id", request.state.cid)
        except WorkflowError as e:
            return _error(409, "workflow_conflict", str(e), request.state.cid)

    @app.get("/v1/documents/{packet_id}/verified-inputs")
    def document_inputs(packet_id: str, request: Request,
                        p: Principal = Depends(needs("document:read"))):  # noqa: B008
        try:
            return verified_inputs(docs.state(packet_id))
        except KeyError:
            return _error(404, "not_found", "Unknown packet_id", request.state.cid)
        except UnvalidatedExtractionError as e:
            return _error(409, "not_validated", str(e), request.state.cid)

    # ---- Phase H: copilot ------------------------------------------------------------------------------------------
    @app.post("/v1/copilot/ask")
    def ask(body: AskRequest, request: Request, p: Principal = Depends(needs("copilot:ask"))):  # noqa: B008
        cp: Copilot | None = copilot_state["copilot"]
        if cp is None:
            return _error(503, "copilot_unavailable", copilot_state["error"] or "copilot not loaded", request.state.cid)
        return cp.ask(body.question, Caller(name=p.name, role=p.role, can=p.can))

    return app
