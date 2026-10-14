"""Field extraction with page and quote references.

``rule_extract_document``: label-based extraction for ``label: value`` lines and markdown table rows. Labels are
matched after OCR-tolerant normalisation (0/o, 1/l/i, spacing). It never guesses: if a label is not found, or the
value does not parse, or the same field has two different values, the field is left empty and routed to review.

``llm_extract_document``: any provider through the LLM adapter, followed by a grounding check. A value is kept only
if its quote appears verbatim on the cited page and the value appears in the quote. Ungrounded values are dropped.
"""

from __future__ import annotations

import re

from crisk.docai.schema import DESCRIPTIONS, FIELDS, FieldValue, parse_value
from crisk.llm.text import norm

LABELS: dict[str, dict[str, list[str]]] = {
    "application": {
        "applicant_name": ["applicant name", "borrower", "applicant"],
        "employer": ["employer", "current employer"],
        "stated_annual_income": ["annual income", "gross annual income (usd)", "gross annual income"],
        "monthly_debt_payments": ["monthly debt payments", "total monthly debts (usd)", "total monthly debts"],
        "loan_amount": ["loan amount", "requested loan (usd)", "requested loan"],
        "property_value": ["property value", "estimated property value (usd)", "estimated property value"],
    },
    "paystub": {
        "employee_name": ["employee", "employee name"],
        "employer": ["employer", "company"],
        "pay_date": ["pay date", "check date"],
        "pay_frequency": ["pay frequency", "frequency"],
        "gross_pay_period": ["gross pay this period"],
        "ytd_gross": ["year-to-date gross"],
        "periods_ytd": ["pay periods year-to-date", "periods paid ytd"],
    },
    "bank_statement": {
        "account_holder": ["account holder"],
        "statement_period": ["statement period"],
        "opening_balance": ["opening balance", "beginning balance"],
        "total_deposits": ["total deposits", "credits"],
        "total_withdrawals": ["total withdrawals", "debits"],
        "closing_balance": ["closing balance", "ending balance"],
    },
}
# Table rows with several value columns: label -> fields in column order.
MULTI_COLUMN = {"paystub": {"gross earnings": ["gross_pay_period", "ytd_gross"]}}
_SEP = re.compile(r"^\|[\s:\-|]+\|$")


def canon_label(s: str) -> str:
    """OCR-tolerant label key: lower case, 0->o, 1/i->l, collapsed spaces."""
    s = s.lower().replace("0", "o").replace("1", "l").replace("i", "l")
    return re.sub(r"\s+", " ", s).strip(" :")


_INDEX = {doc: {canon_label(lbl): f for f, lbls in m.items() for lbl in lbls} for doc, m in LABELS.items()}
_MULTI = {doc: {canon_label(k): v for k, v in m.items()} for doc, m in MULTI_COLUMN.items()}


def _candidates(doc_type: str, pages: list[str]):
    """Yield (field, raw value, page, quote) for every labelled value in the document."""
    index, multi = _INDEX.get(doc_type, {}), _MULTI.get(doc_type, {})
    for pno, page in enumerate(pages, start=1):
        for line in page.split("\n"):
            s = line.strip()
            if s.startswith("|"):
                if _SEP.match(s):
                    continue
                cells = [c.strip() for c in s.strip("|").split("|")]
                key = canon_label(cells[0])
                if key in multi:
                    for f, raw in zip(multi[key], cells[1:], strict=False):
                        yield f, raw, pno, s
                elif key in index and len(cells) >= 2:
                    yield index[key], cells[1], pno, s
            elif ":" in s:
                label, raw = s.split(":", 1)
                key = canon_label(label)
                if key in index:
                    yield index[key], raw.strip(), pno, s


def rule_extract_document(doc_type: str, pages: list[str]) -> dict[str, FieldValue]:
    kinds = FIELDS[doc_type]
    found: dict[str, list[FieldValue]] = {}
    for f, raw, pno, quote in _candidates(doc_type, pages):
        try:
            v = parse_value(kinds[f], raw)
        except ValueError:
            continue
        found.setdefault(f, []).append(FieldValue(value=v, raw=raw, page=pno, quote=quote, method="rule"))
    out: dict[str, FieldValue] = {}
    for f in kinds:
        vals = found.get(f, [])
        distinct = {repr(v.value) for v in vals}
        out[f] = vals[0] if len(distinct) == 1 else FieldValue(method="rule" if not vals else "rule:conflict")
    return out


def grounded(kind: str, value, raw: str | None, page: int | None, quote: str | None, pages: list[str]) -> bool:
    """The quote must be verbatim on the cited page and must contain the value as written."""
    if value is None or not raw or not quote or not page or not 1 <= page <= len(pages):
        return False
    return norm(quote) in norm(pages[page - 1]) and norm(raw) in norm(quote)


def llm_extract_document(llm, doc_type: str, pages: list[str]) -> dict[str, FieldValue]:
    kinds = FIELDS[doc_type]
    resp = llm.extract(doc_type, pages, {f: DESCRIPTIONS[f] for f in kinds})
    out: dict[str, FieldValue] = {}
    for f, kind in kinds.items():
        r = resp.get(f) or {}
        raw, page, quote = r.get("value"), r.get("page"), r.get("quote")
        try:
            v = parse_value(kind, raw) if raw else None
        except ValueError:
            v = None
        if grounded(kind, v, raw, page, quote, pages):
            out[f] = FieldValue(value=v, raw=raw, page=int(page), quote=quote, method=f"llm:{llm.name}")
        else:
            out[f] = FieldValue(method=f"llm:{llm.name}:ungrounded" if raw else f"llm:{llm.name}")
    return out
