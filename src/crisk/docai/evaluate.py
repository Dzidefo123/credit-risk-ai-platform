"""Evaluation of document extraction and validation against the generator's answer keys (plan v1 criteria).

The held-out split is evaluated once per extractor: the run is recorded in the audit chain and a second run is
refused (``force`` exists for test workspaces only and is itself audited).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from crisk.audit.store import AuditStore
from crisk.docai.extract import grounded
from crisk.docai.generator import load_split
from crisk.docai.schema import FIELDS, values_equal
from crisk.docai.service import DocumentService, UnvalidatedExtractionError, verified_inputs

CRITERIA = {"field_precision": 0.98, "field_recall_seen_layouts": 0.95, "reference_correctness": 1.0,
            "recall_injection": 1.0, "recall_missing_field": 1.0, "recall_arithmetic": 0.95,
            "recall_cross_document": 0.95, "false_flag_rate_clean_max": 0.10, "gate_leaks_max": 0}
_GROUP = {"arithmetic_bank": "arithmetic", "arithmetic_paystub": "arithmetic", "income_mismatch": "cross_document",
          "employer_mismatch": "cross_document", "injection": "injection"}


def _group(code: str) -> str:
    return "missing_field" if code.startswith("missing_field") else _GROUP.get(code, code.split(":")[0])


def _ratio(a: int, b: int) -> float | None:
    return None if b == 0 else a / b


def evaluate_documents(fixture_dir: Path, split: str, store: AuditStore, llm=None, force: bool = False,
                       actor: str = "evaluator") -> dict:
    svc = DocumentService(store, llm=llm)
    if split == "heldout":
        done = [e for e in store.events(action="docai_heldout_evaluated") if e["details"]["extractor"] ==
                svc.extractor_name]
        if done and not force:
            raise RuntimeError(f"held-out already evaluated for {svc.extractor_name} ({done[0]['ts']})")
    fields = Counter()
    by_layout = defaultdict(Counter)
    flags = defaultdict(Counter)
    clean = Counter()
    gate_leaks, status_counts, per_packet = 0, Counter(), []
    for packet, key in load_split(fixture_dir, split):
        st = svc.submit(packet, actor=actor, role="evaluator")
        status_counts[st["status"]] += 1
        pages = {d["doc_type"]: d["pages"] for d in packet["documents"]}
        for doc, kinds in FIELDS.items():
            layout = key["layouts"][doc]
            for f, kind in kinds.items():
                truth = key["fields"][doc][f]
                got = st["fields"][doc][f]
                if got["value"] is not None:
                    fields["extracted"] += 1
                    ok = values_equal(kind, got["value"], truth["value"])
                    fields["correct"] += ok
                    ref_ok = (grounded(kind, got["value"], got["raw"], got["page"], got["quote"], pages[doc])
                              and got["page"] == truth["page"])
                    fields["ref_ok"] += ref_ok
                if truth["value"] is not None:
                    seen = "unseen" if layout == "C" else "seen"
                    by_layout[seen]["present"] += 1
                    by_layout[seen]["recalled"] += got["value"] is not None and values_equal(kind, got["value"],
                                                                                              truth["value"])
        expected = set(key["expected_flags"])
        raised = {f["code"] for f in st["flags"]}
        for code in expected:
            flags[_group(code)]["expected"] += 1
            flags[_group(code)]["detected"] += code in raised
        unexpected = raised - expected
        if not expected:
            clean["packets"] += 1
            clean["false_flagged"] += bool(raised)
        try:
            verified_inputs(st)
            gate_leaks += st["status"] != "validated"
        except UnvalidatedExtractionError:
            pass
        per_packet.append({"packet_ref": key["packet_ref"], "status": st["status"], "expected": sorted(expected),
                           "raised": sorted(raised), "unexpected": sorted(unexpected),
                           "layout_application": key["layouts"]["application"], "degraded": key["degraded"]})
    m = {"field_precision": _ratio(fields["correct"], fields["extracted"]),
         "field_recall_seen_layouts": _ratio(by_layout["seen"]["recalled"], by_layout["seen"]["present"]),
         "field_recall_unseen_layout": _ratio(by_layout["unseen"]["recalled"], by_layout["unseen"]["present"]),
         "reference_correctness": _ratio(fields["ref_ok"], fields["extracted"]),
         **{f"recall_{g}": _ratio(c["detected"], c["expected"]) for g, c in flags.items()},
         "false_flag_rate_clean": _ratio(clean["false_flagged"], clean["packets"]), "gate_leaks": gate_leaks}
    counts = {"packets": len(per_packet), "fields_extracted": fields["extracted"],
              "fields_present_seen": by_layout["seen"]["present"],
              "fields_present_unseen": by_layout["unseen"]["present"],
              "flags_expected": {g: c["expected"] for g, c in flags.items()}, "clean_packets": clean["packets"],
              "status": dict(status_counts)}
    checks = {}
    for k, thr in CRITERIA.items():
        name = k.removesuffix("_max")
        v = m.get(name)
        if v is None:
            checks[k] = "not_applicable"
        elif k.endswith("_max"):
            checks[k] = "pass" if v <= thr else "fail"
        else:
            checks[k] = "pass" if v >= thr - 1e-12 else "fail"
    report = {"split": split, "extractor": svc.extractor_name, "data_origin": "SYNTHETIC", "metrics": m,
              "counts": counts, "criteria": CRITERIA, "checks": checks, "packets": per_packet,
              "label": "SYNTHETIC documents; pipeline evaluation (rule extractor or mock LLM is not LLM evidence)"}
    if split == "heldout":
        store.append_event(actor=actor, action="docai_heldout_evaluated", entity_type="docai_eval",
                           entity_id=svc.extractor_name, details={"extractor": svc.extractor_name, "forced": force,
                                                                  "checks": checks})
    return report
