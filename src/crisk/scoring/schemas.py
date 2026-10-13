"""Request/response contracts for origination PD scoring.

The application schema is generated from the SFLLD contract (ranges, allowed codes) for the
raw inputs the feature set reads, so API validation cannot drift from ingestion validation.
All inputs are optional at the schema level: a missing *required* input yields an
``unavailable`` result (no score) rather than a validation error, and a missing optional
input is imputed and flagged.
"""

from __future__ import annotations

import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from crisk.contracts.freddie_mac import CONTRACT_SETS, DEFAULT_CONTRACT_VERSION
from crisk.features.origination import FeatureSpec

REQUIRED_FOR_SCORING = ("orig_upb", "orig_interest_rate", "orig_loan_term")
ORIGINATION = CONTRACT_SETS[DEFAULT_CONTRACT_VERSION]["origination"]


def _field_type(name: str) -> tuple[Any, Any]:
    f = ORIGINATION.by_name(name)
    kw: dict[str, Any] = {"default": None, "description": f.formal_name}
    if f.dtype == "yyyymm":
        typ: Any = datetime.date
    elif f.dtype == "int":
        typ = int
    elif f.dtype == "float":
        typ = float
    elif f.allowed:
        typ = Literal[tuple(f.allowed)]  # type: ignore[valid-type]
    else:
        typ = str
    if f.min_value is not None and f.dtype in ("int", "float"):
        kw["ge"] = f.min_value
    if f.max_value is not None and f.dtype in ("int", "float"):
        kw["le"] = f.max_value
    if name == "property_state":
        kw["pattern"] = r"^[A-Z]{2}$"
    return (typ | None, Field(**kw))


def application_model(spec: FeatureSpec) -> type[BaseModel]:
    fields = {name: _field_type(name) for name in spec.raw_inputs}
    fields["loan_id"] = (str | None, Field(default=None, description="Reference only; never a model input"))
    return create_model("OriginationApplication", __config__=ConfigDict(extra="forbid"), **fields)


class ReasonCode(BaseModel):
    code: str
    feature: str
    text: str
    contribution_logit: float


class TargetInfo(BaseModel):
    target_id: str
    horizon_months: int
    definition: str


class ModelInfo(BaseModel):
    model_id: str
    artifact_sha256: str
    calibrated: bool
    feature_set_id: str


class ScoreResponse(BaseModel):
    score_id: str
    status: Literal["scored", "unavailable"]
    scored_at: str
    correlation_id: str | None = None
    data_origin: Literal["SYNTHETIC", "EMPIRICAL"]
    evidence_note: str
    target: TargetInfo
    model: ModelInfo
    pd: float | None = None
    pd_uncalibrated: float | None = None
    dq_flags: list[str] = []
    reason_codes: list[ReasonCode] = []
    idempotent_replay: bool = False
    approved_use: str = "not_promoted"
    risk_decile: int | None = None
    requested_by: str | None = None
