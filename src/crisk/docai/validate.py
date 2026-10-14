"""Arithmetic, cross-document and completeness checks on an extraction, and the resulting packet status.

Flags that a reviewer may ACCEPT with a rationale (``acceptable``): income, employer and name mismatches (there can be
legitimate explanations) and a suspected embedded instruction (after confirming it was ignored). Missing fields and
arithmetic errors cannot be accepted: the value must be corrected (with a rationale) or the packet rejected.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from crisk.docai.schema import FIELDS, PERIODS_PER_YEAR, FieldValue

INCOME_TOLERANCE = 0.10      # relative difference between stated and documented annual income
CENT = 0.01
YTD_TOLERANCE = 1.00         # USD: cents rounding over a year of pay periods


@dataclass(frozen=True)
class Flag:
    code: str
    severity: str            # critical | high | medium
    detail: str
    acceptable: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _v(ext: dict, doc: str, field: str):
    fv = ext.get(doc, {}).get(field)
    return None if fv is None else fv.value


def _entity(s: str | None) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").casefold()).strip()


def validate(ext: dict[str, dict[str, FieldValue]], screen: dict[str, list[dict]]) -> list[Flag]:
    flags: list[Flag] = []
    for doc, kinds in FIELDS.items():
        if doc not in ext:
            flags.append(Flag(f"missing_document:{doc}", "high", f"No {doc} in the packet", False))
            continue
        for f in kinds:
            fv = ext[doc].get(f)
            if fv is None or fv.value is None:
                why = "two different values found" if fv is not None and fv.method.endswith("conflict") else (
                    "value not grounded in the document" if fv is not None and fv.method.endswith("ungrounded")
                    else "not found")
                flags.append(Flag(f"missing_field:{doc}.{f}", "high", f"{doc}.{f}: {why}", False))

    o, d, w, c = (_v(ext, "bank_statement", f) for f in ("opening_balance", "total_deposits", "total_withdrawals",
                                                        "closing_balance"))
    if None not in (o, d, w, c) and abs(o + d - w - c) > CENT:
        flags.append(Flag("arithmetic_bank", "high",
                          f"opening {o:,.2f} + deposits {d:,.2f} - withdrawals {w:,.2f} = {o + d - w:,.2f}, "
                          f"but closing balance is {c:,.2f}", False))
    g, y, n = (_v(ext, "paystub", f) for f in ("gross_pay_period", "ytd_gross", "periods_ytd"))
    if None not in (g, y, n) and abs(g * n - y) > YTD_TOLERANCE:
        flags.append(Flag("arithmetic_paystub", "high",
                          f"gross {g:,.2f} x {n} periods = {g * n:,.2f}, but year-to-date gross is {y:,.2f}", False))
    stated, freq = _v(ext, "application", "stated_annual_income"), _v(ext, "paystub", "pay_frequency")
    if None not in (stated, g, freq):
        documented = g * PERIODS_PER_YEAR[freq]
        if abs(stated - documented) / documented > INCOME_TOLERANCE:
            flags.append(Flag("income_mismatch", "medium",
                              f"stated annual income {stated:,.2f} vs documented {documented:,.2f} "
                              f"({g:,.2f} x {PERIODS_PER_YEAR[freq]}), difference "
                              f"{(stated - documented) / documented:+.1%}", True))
    e1, e2 = _v(ext, "application", "employer"), _v(ext, "paystub", "employer")
    if None not in (e1, e2) and _entity(e1) != _entity(e2):
        flags.append(Flag("employer_mismatch", "medium", f"application employer {e1!r} vs pay stub {e2!r}", True))
    names = {k: _v(ext, doc, f) for k, doc, f in (("application", "application", "applicant_name"),
                                                 ("paystub", "paystub", "employee_name"),
                                                 ("bank_statement", "bank_statement", "account_holder"))}
    present = {k: _entity(v) for k, v in names.items() if v is not None}
    if len(set(present.values())) > 1:
        flags.append(Flag("name_mismatch", "medium", f"names differ across documents: {names}", True))
    hits = [(doc, h) for doc, hs in screen.items() for h in hs]
    if hits:
        where = "; ".join(f"{doc} line {h['line']}: {h['excerpt'][:80]!r}" for doc, h in hits[:3])
        flags.append(Flag("injection", "critical",
                          f"Instruction-like text aimed at an AI system was found and ignored ({where}). "
                          "Treat the packet as suspicious.", True))
    return flags
