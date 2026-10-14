"""Deterministic stand-in for an LLM (no network). Results obtained with it measure the pipeline, not an LLM.

``answer``: extractive. It splits the evidence into units (sentences, list items, YAML lines, table rows with their
header), scores each by IDF-weighted coverage of the question's content words, and returns the best unit verbatim
as the answer, citing it. A table row is rendered as ``header: value`` pairs and cites both the header row and the
data row. It abstains when the best coverage is below ``min_coverage``.

``extract``: delegates to the rule-based extractor, so the LLM extraction path (schema, grounding check, routing to
review) can be exercised end to end without a provider.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from crisk.llm.adapter import AnswerDraft, Citation, Evidence, Usage
from crisk.llm.text import content_tokens, display

_SEP_ROW = re.compile(r"^\|[\s:\-|]+\|$")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*`(\[])")
_ITEM = re.compile(r"^(\s*[-*]\s|\s*\d+\.\s)")


@dataclass
class _Unit:
    evidence_id: str
    rank: int
    pos: int
    tokens: set[str]
    answer: str
    quotes: list[str]


def _table_units(ev: Evidence, rank: int, lines: list[str], start: int) -> list[_Unit]:
    header, out = None, []
    for i, line in enumerate(lines):
        if _SEP_ROW.match(line.strip()):
            continue
        if header is None:
            header = line
            continue
        hcells = [c.strip() for c in header.strip().strip("|").split("|")]
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        pairs = [f"{display(h)}: {display(v)}" if display(h) else display(v)
                 for h, v in zip(hcells, cells, strict=False) if display(v) not in ("", "-", "–")]
        toks = set(content_tokens(header + " " + line + " " + ev.title))
        out.append(_Unit(ev.evidence_id, rank, start + i, toks, "; ".join(pairs), [header.strip(), line.strip()]))
    return out


def _units(ev: Evidence, rank: int) -> list[_Unit]:
    lines = ev.text.split("\n")
    units: list[_Unit] = []
    yaml_like = ev.source_id.startswith(("policy.", "monitoring.", "validation.findings"))
    block: list[str] = []
    table: list[str] = []

    def flush_block(pos: int) -> None:
        if not block:
            return
        joined = " ".join(s.strip() for s in block)
        for j, sent in enumerate(s for s in _SENT.split(joined) if s.strip()):
            units.append(_Unit(ev.evidence_id, rank, pos * 100 + j, set(content_tokens(sent + " " + ev.title)),
                               display(sent), [sent.strip()]))
        block.clear()

    for pos, line in enumerate(lines):
        if line.strip().startswith("|"):
            flush_block(pos)
            table.append(line)
            continue
        if table:
            units.extend(_table_units(ev, rank, table, pos * 100))
            table = []
        if line.lstrip().startswith("#") and not yaml_like:
            flush_block(pos)
            head = line.lstrip("#").strip()
            if head:
                units.append(_Unit(ev.evidence_id, rank, pos * 100, set(content_tokens(head + " " + ev.title)),
                                   display(head), [head]))
            continue
        if not line.strip():
            flush_block(pos)
            continue
        if yaml_like or _ITEM.match(line):
            flush_block(pos)
        block.append(line)
        if yaml_like:
            flush_block(pos)
    if table:
        units.extend(_table_units(ev, rank, table, len(lines) * 100))
    flush_block(len(lines))
    return units


class MockLLM:
    name = "mock-extractive-v1"

    def __init__(self, min_coverage: float = 0.5):
        self.min_coverage = float(min_coverage)
        self.usage = Usage(provider="mock", model=self.name)

    def answer(self, question: str, evidence: list[Evidence]) -> AnswerDraft:
        self.usage.calls += 1
        units = [u for rank, ev in enumerate(evidence) for u in _units(ev, rank)]
        q = set(content_tokens(question))
        if not units or not q:
            return AnswerDraft(abstain=True, reason="no_evidence")
        n = len(units)
        df = {t: sum(t in u.tokens for u in units) for t in q}
        idf = {t: math.log(1 + n / max(df[t], 1)) if df[t] else math.log(1 + n) for t in q}
        total = sum(idf.values())

        def score(u: _Unit) -> float:
            return sum(idf[t] for t in q if t in u.tokens) / total

        best = max(units, key=lambda u: (round(score(u), 9), -u.rank, -u.pos))
        cov = score(best)
        if cov < self.min_coverage:
            return AnswerDraft(abstain=True, reason=f"low_coverage:{cov:.2f}")
        return AnswerDraft(abstain=False, answer=best.answer,
                           citations=[Citation(best.evidence_id, q_) for q_ in best.quotes])

    def extract(self, doc_type: str, pages: list[str], fields: dict[str, str]) -> dict[str, dict]:
        from crisk.docai.extract import rule_extract_document

        self.usage.calls += 1
        out = rule_extract_document(doc_type, pages)
        return {f: {"value": None if v.value is None else str(v.raw), "page": v.page, "quote": v.quote}
                for f, v in out.items() if f in fields}
