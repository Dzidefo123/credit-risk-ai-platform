"""Typed extraction schema for underwriting documents.

Every extracted field carries the value as parsed, the text as written, the 1-based page and a verbatim quote from
that page, plus the method that produced it (rule, llm or human).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocType = Literal["application", "paystub", "bank_statement"]

FIELDS: dict[str, dict[str, str]] = {
    "application": {"applicant_name": "text", "employer": "text", "stated_annual_income": "money",
                    "monthly_debt_payments": "money", "loan_amount": "money", "property_value": "money"},
    "paystub": {"employee_name": "text", "employer": "text", "pay_date": "date", "pay_frequency": "frequency",
                "gross_pay_period": "money", "ytd_gross": "money", "periods_ytd": "int"},
    "bank_statement": {"account_holder": "text", "statement_period": "month", "opening_balance": "money",
                       "total_deposits": "money", "total_withdrawals": "money", "closing_balance": "money"},
}

DESCRIPTIONS: dict[str, str] = {
    "applicant_name": "full name of the applicant", "employer": "employer name",
    "stated_annual_income": "gross annual income stated by the applicant (USD)",
    "monthly_debt_payments": "total monthly payments on other debts (USD)",
    "loan_amount": "requested loan amount (USD)", "property_value": "estimated property value (USD)",
    "employee_name": "employee name on the pay stub", "pay_date": "pay or check date",
    "pay_frequency": "pay frequency (weekly, biweekly, semimonthly or monthly)",
    "gross_pay_period": "gross pay for this pay period (USD)", "ytd_gross": "year-to-date gross pay (USD)",
    "periods_ytd": "number of pay periods paid year to date", "account_holder": "bank account holder name",
    "statement_period": "statement month", "opening_balance": "opening or beginning balance (USD)",
    "total_deposits": "total deposits or credits (USD)", "total_withdrawals": "total withdrawals or debits (USD)",
    "closing_balance": "closing or ending balance (USD)",
}

PERIODS_PER_YEAR = {"weekly": 52, "biweekly": 26, "semimonthly": 24, "monthly": 12}
_FREQ = {"weekly": "weekly", "biweekly": "biweekly", "bi-weekly": "biweekly", "every two weeks": "biweekly",
         "semimonthly": "semimonthly", "semi-monthly": "semimonthly", "twice monthly": "semimonthly",
         "monthly": "monthly"}
_MONEY = re.compile(r"^\(?-?\s*(?:usd)?\s*\$?\s*-?[\d,]+(?:\.\d{1,2})?\)?\s*(?:usd)?$", re.IGNORECASE)


def parse_value(kind: str, raw: str):
    """Parse text as written into a typed value; raises ValueError if it does not fit the kind."""
    s = raw.strip()
    if not s:
        raise ValueError("empty")
    if kind == "money":
        if not _MONEY.match(s):
            raise ValueError(f"not an amount: {raw!r}")
        neg = s.startswith("(") or "-" in s
        v = float(re.sub(r"[^\d.]", "", s))
        return round(-v if neg else v, 2)
    if kind == "int":
        if not re.fullmatch(r"\d{1,3}", s):
            raise ValueError(f"not a count: {raw!r}")
        return int(s)
    if kind == "date":
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d %B %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except ValueError:
                continue
        raise ValueError(f"not a date: {raw!r}")
    if kind == "month":
        for fmt in ("%Y-%m", "%B %Y", "%b %Y", "%m/%Y"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m")
            except ValueError:
                continue
        raise ValueError(f"not a month: {raw!r}")
    if kind == "frequency":
        k = s.lower().strip(" .")
        if k not in _FREQ:
            raise ValueError(f"not a frequency: {raw!r}")
        return _FREQ[k]
    if kind == "text":
        if len(s) > 120 or not re.search(r"[A-Za-z]", s):
            raise ValueError(f"not a name: {raw!r}")
        return re.sub(r"\s+", " ", s)
    raise ValueError(kind)


def values_equal(kind: str, a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if kind == "money":
        return abs(float(a) - float(b)) < 0.005
    if kind == "text":
        return str(a).casefold() == str(b).casefold()
    return a == b


@dataclass
class FieldValue:
    value: object = None
    raw: str | None = None
    page: int | None = None
    quote: str | None = None
    method: str = "rule"

    def as_dict(self) -> dict:
        return {"value": self.value, "raw": self.raw, "page": self.page, "quote": self.quote, "method": self.method}


# ---- API input -------------------------------------------------------------------------------------------------------
class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_type: DocType
    pages: list[str] = Field(min_length=1, max_length=10)


class PacketIn(BaseModel):
    """Prototype intake accepts fictional documents only: real underwriting documents contain personal data that
    this prototype is not built to protect (no encryption at rest, retention schedule or access logging)."""
    model_config = ConfigDict(extra="forbid")
    data_origin: Literal["SYNTHETIC"]
    packet_ref: str | None = Field(default=None, max_length=64)
    documents: list[DocumentIn] = Field(min_length=1, max_length=5)


def today() -> str:
    return date.today().isoformat()
