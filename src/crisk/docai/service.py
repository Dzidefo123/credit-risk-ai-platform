"""Document intake, review workflow and the feature gate.

State is rebuilt from append-only rows: the packet (untrusted content, stored as data), its extraction, and reviewer
actions (``correct``, ``accept``, ``reject``), each with a rationale and audited. Status:

- ``quarantined``: an embedded instruction was found and not yet accepted by a reviewer (runbook RB-05);
- ``needs_review``: any other open flag;
- ``validated``: no open flags (accepted exceptions are listed);
- ``rejected``: a reviewer rejected the packet.

The feature gate (:func:`verified_inputs`) only accepts ``validated`` packets. Current PD models accept no document
inputs at all (their request schemas forbid extra fields); verified inputs are for underwriter comparison only.
"""

from __future__ import annotations

import json
import uuid

from crisk.audit.store import AuditStore
from crisk.docai.extract import llm_extract_document, rule_extract_document
from crisk.docai.schema import FIELDS, PERIODS_PER_YEAR, FieldValue, PacketIn, parse_value
from crisk.docai.validate import validate
from crisk.security.untrusted import clean_text, screen_text
from crisk.utils import canonical_json, sha256_text, utc_now


class UnvalidatedExtractionError(RuntimeError):
    """An extraction that is not validated was offered as an input."""


class WorkflowError(ValueError):
    """A reviewer action is not allowed in the packet's current state."""


def _fv(d: dict) -> FieldValue:
    return FieldValue(**d)


class DocumentService:
    def __init__(self, store: AuditStore, llm=None):
        """``llm=None`` uses the rule-based extractor; otherwise the LLM extractor with the grounding check."""
        self.store, self.llm = store, llm

    @property
    def extractor_name(self) -> str:
        return "rule-v1" if self.llm is None else f"llm-grounded-v1:{self.llm.name}"

    # ---- intake ------------------------------------------------------------------------------------------------
    def submit(self, packet: dict, actor: str, role: str) -> dict:
        p = PacketIn.model_validate(packet)
        types = [d.doc_type for d in p.documents]
        if len(set(types)) != len(types):
            raise WorkflowError("at most one document of each type per packet")
        docs = [{"doc_type": d.doc_type, "pages": [clean_text(pg) for pg in d.pages]} for d in p.documents]
        packet_id = uuid.uuid4().hex
        self.store.insert("doc_packets", {"packet_id": packet_id, "received_at": utc_now(),
                                          "submitted_by": f"{actor} ({role})", "data_origin": p.data_origin,
                                          "packet_ref": p.packet_ref,
                                          "content_sha256": sha256_text(canonical_json(docs)),
                                          "documents": json.dumps(docs)})
        fields, screen = {}, {}
        for d in docs:
            ext = (rule_extract_document(d["doc_type"], d["pages"]) if self.llm is None
                   else llm_extract_document(self.llm, d["doc_type"], d["pages"]))
            fields[d["doc_type"]] = {f: v.as_dict() for f, v in ext.items()}
            screen[d["doc_type"]] = [{"pattern": h.pattern, "line": h.line, "excerpt": h.excerpt, "page": pno}
                                     for pno, pg in enumerate(d["pages"], start=1) for h in screen_text(pg)]
        self.store.insert("doc_extractions", {"extraction_id": uuid.uuid4().hex, "packet_id": packet_id,
                                              "created_at": utc_now(), "extractor": self.extractor_name,
                                              "fields": json.dumps(fields, default=str), "screen": json.dumps(screen)})
        state = self.state(packet_id)
        self.store.append_event(actor=f"{actor} ({role})", action="document_packet_submitted",
                                entity_type="document_packet", entity_id=packet_id,
                                details={"packet_ref": p.packet_ref, "extractor": self.extractor_name,
                                         "status": state["status"], "flags": [f["code"] for f in state["flags"]]})
        return state

    # ---- state -------------------------------------------------------------------------------------------------
    def state(self, packet_id: str, include_documents: bool = False) -> dict:
        pk = self.store.doc_packet(packet_id)
        if pk is None:
            raise KeyError(packet_id)
        exts = self.store.doc_rows("doc_extractions", packet_id)
        ext_row = exts[-1]
        fields = {doc: {f: _fv(v) for f, v in m.items()} for doc, m in json.loads(ext_row["fields"]).items()}
        screen = json.loads(ext_row["screen"])
        actions = self.store.doc_rows("doc_actions", packet_id)
        accepted: set[str] = set()
        rejected = None
        for a in actions:
            if a["action"] == "correct":
                doc, f = a["field"].split(".", 1)
                fields.setdefault(doc, {})[f] = FieldValue(value=json.loads(a["value"]), raw=None, page=None,
                                                           quote=None, method=f"human:{a['actor']}")
            elif a["action"] == "accept":
                accepted |= set(json.loads(a["accepted_flags"]))
            elif a["action"] == "reject":
                rejected = a
        flags = validate(fields, screen)
        open_flags = [f for f in flags if f.code not in accepted]
        if rejected:
            status = "rejected"
        elif any(f.code == "injection" for f in open_flags):
            status = "quarantined"
        elif open_flags:
            status = "needs_review"
        else:
            status = "validated"
        out = {"packet_id": packet_id, "packet_ref": pk["packet_ref"], "data_origin": pk["data_origin"],
               "received_at": pk["received_at"], "submitted_by": pk["submitted_by"], "extractor": ext_row["extractor"],
               "status": status, "fields": {doc: {f: v.as_dict() for f, v in m.items()} for doc, m in fields.items()},
               "flags": [{**f.as_dict(), "accepted": f.code in accepted} for f in flags],
               "open_flags": [f.code for f in open_flags], "screen": screen,
               "actions": [{k: a[k] for k in ("created_at", "actor", "actor_role", "action", "field", "rationale")}
                           for a in actions],
               "note": "SYNTHETIC documents. Extracted values are not model inputs; see verified_inputs."}
        if include_documents:
            out["documents"] = json.loads(pk["documents"])
        return out

    # ---- reviewer actions --------------------------------------------------------------------------------------
    def _act(self, packet_id: str, actor: str, role: str, action: str, rationale: str, **extra) -> dict:
        if len(rationale.strip()) < 10:
            raise WorkflowError("a rationale of at least 10 characters is required")
        st = self.state(packet_id)
        if st["status"] == "rejected":
            raise WorkflowError("packet was rejected; submit a new packet")
        rec = {"action_id": uuid.uuid4().hex, "packet_id": packet_id, "created_at": utc_now(), "actor": actor,
               "actor_role": role, "action": action, "field": extra.get("field"), "value": extra.get("value"),
               "accepted_flags": extra.get("accepted_flags"), "rationale": rationale.strip()}
        self.store.insert("doc_actions", rec)
        new = self.state(packet_id)
        self.store.append_event(actor=f"{actor} ({role})", action=f"document_{action}", entity_type="document_packet",
                                entity_id=packet_id, details={"field": rec["field"], "status": new["status"],
                                                              "accepted_flags": extra.get("accepted_flags")})
        return new

    def correct(self, packet_id: str, field: str, value: str, rationale: str, actor: str, role: str) -> dict:
        doc, _, f = field.partition(".")
        if doc not in FIELDS or f not in FIELDS[doc]:
            raise WorkflowError(f"unknown field {field!r} (use doc_type.field)")
        try:
            v = parse_value(FIELDS[doc][f], str(value))
        except ValueError as e:
            raise WorkflowError(str(e)) from e
        return self._act(packet_id, actor, role, "correct", rationale, field=field, value=json.dumps(v))

    def accept(self, packet_id: str, rationale: str, actor: str, role: str) -> dict:
        st = self.state(packet_id)
        flags = {f["code"]: f for f in st["flags"]}
        blocking = [c for c in st["open_flags"] if not flags[c]["acceptable"]]
        if blocking:
            raise WorkflowError(f"these flags must be corrected, not accepted: {blocking}")
        if not st["open_flags"]:
            raise WorkflowError("nothing to accept")
        return self._act(packet_id, actor, role, "accept", rationale, accepted_flags=json.dumps(st["open_flags"]))

    def reject(self, packet_id: str, rationale: str, actor: str, role: str) -> dict:
        return self._act(packet_id, actor, role, "reject", rationale)


def verified_inputs(state: dict) -> dict:
    """The only path from documents to any downstream use. Raises unless the packet is validated."""
    if state.get("status") != "validated":
        raise UnvalidatedExtractionError(f"packet {state.get('packet_id')} is {state.get('status')!r}, not validated")
    f = state["fields"]
    gross, freq = f["paystub"]["gross_pay_period"]["value"], f["paystub"]["pay_frequency"]["value"]
    income = gross * PERIODS_PER_YEAR[freq]
    debt = f["application"]["monthly_debt_payments"]["value"]
    return {"packet_id": state["packet_id"], "source": "document_ai_v1 (SYNTHETIC)",
            "documented_annual_income": round(income, 2), "monthly_debt_payments": debt,
            "other_debt_to_income_pct": round(100 * debt / (income / 12), 1),  # excludes the proposed housing payment
            "accepted_exceptions": [x["code"] for x in state["flags"] if x["accepted"]],
            "note": "For underwriter comparison with the application. Not a PD model input."}
