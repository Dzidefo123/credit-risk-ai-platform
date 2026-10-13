"""FastAPI scoring endpoint (prototype).

No `from __future__ import annotations` here: FastAPI resolves the locally built request model
from real annotations.

Endpoints: GET /health (liveness), GET /ready (model + audit store), POST /v1/score.
Errors use one JSON shape: {"error": {"code", "message", "correlation_id"}}.
Authentication is not implemented yet (Phase F); do not expose this service beyond localhost.
"""


import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, create_model

from crisk.audit.store import AuditStore, IdempotencyConflict
from crisk.config import DatasetConfig
from crisk.features.origination import FeatureSpec
from crisk.scoring.schemas import ScoreResponse, application_model
from crisk.scoring.service import ModelUnavailableError, ScoringService

log = logging.getLogger("crisk.api")


def _error(status: int, code: str, message: str, cid: str | None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
                                                                "correlation_id": cid}},
                        headers={"X-Correlation-ID": cid or ""})


def create_app(cfg: DatasetConfig, model_id: str | None = None) -> FastAPI:
    store = AuditStore(cfg.audit_db)
    app = FastAPI(title="Credit risk PD scoring (prototype)", version="0.1.0")
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

    @app.middleware("http")
    async def correlation(request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        request.state.cid = cid
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return _error(422, "invalid_request", details, getattr(request.state, "cid", None))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        svc = state["service"]
        ok = svc is not None and store.healthcheck()
        body = {"ready": ok, "model_id": svc.model_id if svc else None,
                "data_origin": svc.meta["data_origin"] if svc else None, "load_error": state["load_error"]}
        return JSONResponse(status_code=200 if ok else 503, content=body)

    @app.post("/v1/score", response_model=ScoreResponse)
    def score(req: ScoreRequest, request: Request):  # type: ignore[valid-type]
        cid = request.state.cid
        svc: ScoringService | None = state["service"]
        if svc is None:
            return _error(503, "model_unavailable", "No validated model is loaded; no score was produced.", cid)
        try:
            return svc.score(req.application.model_dump(), idempotency_key=req.idempotency_key, correlation_id=cid)
        except IdempotencyConflict:
            return _error(409, "idempotency_conflict",
                          "idempotency_key was already used with a different application payload", cid)
        except ModelUnavailableError as e:
            return _error(503, "model_unavailable", str(e), cid)
        except Exception:
            log.exception("scoring failed cid=%s", cid)  # no payload in logs
            return _error(500, "internal_error", "Scoring failed; no score was produced.", cid)

    return app
