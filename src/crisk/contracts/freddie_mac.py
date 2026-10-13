"""Freddie Mac Single-Family Loan-Level Dataset (SFLLD) file contracts.

Two layout versions are defined (files are pipe-delimited with no header row):

- ``sflld_guide_2026_01``: Releases up to 46. Source: General User Guide, January 2026
  (https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf, retrieved 2026-10-12).
- ``sflld_r47_2026_07``: Release 47 (July 2026) onwards, 31 origination / 35 performance fields.
  Sources (retrieved 2026-10-12): file_layout_july_2026.xlsx, file_headers_july_2026.zip and
  "SFLLD Disclosure Changes Effective July 2026" v1.2 (6/2026). Built from the 2026-01 contract by
  the documented moves, additions and enumeration changes, so unchanged definitions stay identical.

Canonical column names are this project's own snake_case names; the provider's formal
names are kept in ``formal_name`` for traceability. If Freddie Mac changes the layout,
add a new contract version instead of editing this one in place.

``availability`` records *when* a value is known relative to the lending decision. It is
used by the feature layer to keep post-origination information out of origination models:

- ``identifier``: keys/links, never model features.
- ``origination``: describes the loan/borrower at the note date (reconstructed state;
  the dataset gives no real-time availability timestamps).
- ``acquisition``: as of Freddie Mac's purchase, which is after origination.
- ``post_origination``: updated after origination or derived from later aggregates.
- ``performance``: monthly servicing outcome data; never an origination feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CONTRACT_VERSION = "sflld_guide_2026_01"
GUIDE_URL = "https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf"

DType = Literal["str", "code", "int", "float", "yyyymm"]
Availability = Literal["identifier", "origination", "acquisition", "post_origination", "performance"]
OnInvalid = Literal["reject", "null"]

LOAN_ID_PATTERN_EMPIRICAL = r"^[FA][0-9]{2}Q[1-4][0-9]{7}$"
# Synthetic fixtures use product letter "X", which the provider never issues, so fixture
# IDs cannot collide with real Loan Sequence Numbers.
LOAN_ID_PATTERN_SYNTHETIC = r"^X[0-9]{2}Q[1-4][0-9]{7}$"


@dataclass(frozen=True)
class Field:
    position: int
    name: str
    formal_name: str
    dtype: DType
    availability: Availability
    sentinels: tuple[str, ...] = ()
    allowed: tuple[str, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    required: bool = False
    on_invalid: OnInvalid = "null"
    pattern: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class FileContract:
    kind: Literal["origination", "performance"]
    version: str
    fields: tuple[Field, ...] = field(default_factory=tuple)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.fields]

    def by_name(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)


ORIGINATION = FileContract(
    kind="origination",
    version=CONTRACT_VERSION,
    fields=(
        Field(1, "credit_score", "CREDIT SCORE", "int", "origination", sentinels=("9999",),
              min_value=300, max_value=850,
              notes="Generally the score known at acquisition and used to originate; <300 or >850 shown as 9999."),
        Field(2, "first_payment_date", "FIRST PAYMENT DATE", "yyyymm", "origination", required=True,
              on_invalid="reject",
              notes="First scheduled payment due date; not the application or note date."),
        Field(3, "first_time_homebuyer_flag", "FIRST TIME HOMEBUYER FLAG", "code", "origination",
              sentinels=("9",), allowed=("Y", "N")),
        Field(4, "maturity_date", "MATURITY DATE", "yyyymm", "origination"),
        Field(5, "msa_code", "METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION", "str",
              "origination", pattern=r"^[0-9]{5}$", notes="Null = not in an MSA/MD or unknown."),
        Field(6, "mi_pct", "MORTGAGE INSURANCE PERCENTAGE (MI %)", "int", "acquisition",
              sentinels=("999",), min_value=0, max_value=55,
              notes="0 = no MI; measured at Freddie Mac purchase."),
        Field(7, "num_units", "NUMBER OF UNITS", "int", "origination", sentinels=("99",),
              min_value=1, max_value=4),
        Field(8, "occupancy_status", "OCCUPANCY STATUS", "code", "origination", sentinels=("9",),
              allowed=("P", "I", "S")),
        Field(9, "orig_cltv", "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)", "int", "origination",
              sentinels=("999",), min_value=1, max_value=998,
              notes="2018Q1 and prior valid range 6-200; 2018Q2+ 1-998. Widest range enforced here."),
        Field(10, "orig_dti", "ORIGINAL DEBT-TO-INCOME (DTI) RATIO", "int", "origination",
              sentinels=("999",), min_value=0, max_value=65,
              notes=">65% and all HARP loans disclosed as 999."),
        Field(11, "orig_upb", "ORIGINAL UPB", "float", "origination", required=True, min_value=0,
              notes="USD, rounded to nearest $1,000 for disclosure."),
        Field(12, "orig_ltv", "ORIGINAL LOAN-TO-VALUE (LTV)", "int", "origination", sentinels=("999",),
              min_value=1, max_value=998,
              notes="2018Q1 and prior valid range 6-105; 2018Q2+ 1-998. Widest range enforced here."),
        Field(13, "orig_interest_rate", "ORIGINAL INTEREST RATE", "float", "origination", required=True,
              min_value=0, max_value=25, notes="Percent. Upper bound is a plausibility check, not provider-defined."),
        Field(14, "channel", "CHANNEL", "code", "origination", sentinels=("9",),
              allowed=("R", "B", "C", "T")),
        Field(15, "ppm_flag", "PREPAYMENT PENALTY MORTGAGE (PPM) FLAG", "code", "origination",
              allowed=("Y", "N")),
        Field(16, "amortization_type", "AMORTIZATION TYPE", "code", "origination", allowed=("FRM", "ARM")),
        Field(17, "property_state", "PROPERTY STATE", "str", "origination", pattern=r"^[A-Z]{2}$"),
        Field(18, "property_type", "PROPERTY TYPE", "code", "origination", sentinels=("99",),
              allowed=("CO", "PU", "MH", "SF", "CP")),
        Field(19, "postal_code", "POSTAL CODE", "str", "origination", sentinels=("00", "00000"),
              pattern=r"^[0-9]{3}00$",
              notes="Masked to first three digits + '00'. '00' = unknown (treated with '00000')."),
        Field(20, "loan_id", "LOAN SEQUENCE NUMBER", "str", "identifier", required=True, on_invalid="reject",
              notes="PYYQnXXXXXXX; P = product, YYQn = origination year/quarter."),
        Field(21, "loan_purpose", "LOAN PURPOSE", "code", "origination", sentinels=("9",),
              allowed=("P", "C", "N", "R")),
        Field(22, "orig_loan_term", "ORIGINAL LOAN TERM", "int", "origination", required=True,
              min_value=1, max_value=600, notes="Months; maturity - first payment + 1."),
        Field(23, "num_borrowers", "NUMBER OF BORROWERS", "int", "origination", sentinels=("99",),
              min_value=1, max_value=10, notes="2018Q1 and prior: 02 means '>1 borrowers'."),
        Field(24, "seller_name", "SELLER NAME", "str", "acquisition",
              notes="Masked to 'Other Sellers' using a quarter-level 1% UPB rule computed after the fact."),
        Field(25, "servicer_name", "SERVICER NAME", "str", "post_origination",
              notes="Servicer as of the LAST reported period: leaks future information."),
        Field(26, "super_conforming_flag", "SUPER CONFORMING FLAG", "code", "origination", allowed=("Y",)),
        Field(27, "pre_relief_refi_loan_id", "PRE-RELIEF REFINANCE LOAN SEQUENCE NUMBER", "str", "identifier"),
        Field(28, "special_eligibility_program", "SPECIAL ELIGIBILITY PROGRAM", "code", "origination",
              sentinels=("9",), allowed=("H", "F", "R")),
        Field(29, "relief_refi_indicator", "RELIEF REFINANCE INDICATOR", "code", "origination", allowed=("Y",)),
        Field(30, "property_valuation_method", "PROPERTY VALUATION METHOD", "code", "origination",
              sentinels=("7",), allowed=("1", "2", "3", "4"), notes="Populated for loans originated 2017+."),
        Field(31, "io_indicator", "INTEREST ONLY INDICATOR (I/O INDICATOR)", "code", "origination",
              allowed=("Y", "N")),
        Field(32, "mi_cancellation_indicator", "MI CANCELLATION INDICATOR", "code", "post_origination",
              sentinels=("7", "9"), allowed=("Y", "N"),
              notes="Reports MI cancellation after purchase: post-origination information."),
    ),
)

_MONEY = {"dtype": "float", "availability": "performance"}

PERFORMANCE = FileContract(
    kind="performance",
    version=CONTRACT_VERSION,
    fields=(
        Field(1, "loan_id", "LOAN SEQUENCE NUMBER", "str", "identifier", required=True, on_invalid="reject"),
        Field(2, "reporting_period", "MONTHLY REPORTING PERIOD", "yyyymm", "performance", required=True,
              on_invalid="reject"),
        Field(3, "current_actual_upb", "CURRENT ACTUAL UPB", "float", "performance", min_value=0),
        Field(4, "delinquency_status", "CURRENT LOAN DELINQUENCY STATUS", "str", "performance",
              pattern=r"^([0-9]{1,3}|RA)$",
              notes="'0','1','2',... months delinquent (MBA method); 'RA' = REO acquisition."),
        Field(5, "loan_age", "LOAN AGE", "int", "performance", min_value=-12, max_value=600,
              notes="Resets for modified loans; do not use as calendar time since first payment."),
        Field(6, "remaining_months_to_maturity", "REMAINING MONTHS TO LEGAL MATURITY", "int", "performance"),
        Field(7, "defect_settlement_date", "DEFECT SETTLEMENT DATE", "yyyymm", "performance"),
        Field(8, "modification_flag", "MODIFICATION FLAG", "code", "performance", allowed=("Y", "P")),
        Field(9, "zero_balance_code", "ZERO BALANCE CODE", "code", "performance",
              allowed=("01", "02", "03", "09", "15", "16", "96")),
        Field(10, "zero_balance_effective_date", "ZERO BALANCE EFFECTIVE DATE", "yyyymm", "performance"),
        Field(11, "current_interest_rate", "CURRENT INTEREST RATE", "float", "performance", min_value=0,
              max_value=25),
        Field(12, "current_non_interest_bearing_upb", "CURRENT NON-INTEREST BEARING UPB", **_MONEY),
        Field(13, "ddlpi", "DUE DATE OF LAST PAID INSTALLMENT (DDLPI)", "yyyymm", "performance"),
        Field(14, "mi_recoveries", "MI RECOVERIES", **_MONEY),
        Field(15, "net_sale_proceeds", "NET SALE PROCEEDS", sentinels=("U",), **_MONEY,
              notes="'U' = unknown amount."),
        Field(16, "non_mi_recoveries", "NON MI RECOVERIES", **_MONEY),
        Field(17, "total_expenses", "TOTAL EXPENSES", **_MONEY),
        Field(18, "legal_costs", "LEGAL COSTS", **_MONEY),
        Field(19, "maintenance_preservation_costs", "MAINTENANCE AND PRESERVATION COSTS", **_MONEY),
        Field(20, "taxes_insurance", "TAXES AND INSURANCE", **_MONEY),
        Field(21, "misc_expenses", "MISCELLANEOUS EXPENSES", **_MONEY),
        Field(22, "actual_loss", "ACTUAL LOSS CALCULATION", **_MONEY),
        Field(23, "cumulative_modification_cost", "CUMULATIVE MODIFICATION COST", **_MONEY),
        Field(24, "step_modification_flag", "INTEREST RATE STEP INDICATOR", "code", "performance",
              allowed=("Y", "N")),
        Field(25, "payment_deferral_flag", "PAYMENT DEFERRAL FLAG", "code", "performance", allowed=("Y", "P")),
        Field(26, "estimated_ltv", "ESTIMATED LOAN TO VALUE (ELTV)", "int", "performance", sentinels=("999",),
              min_value=1, max_value=998),
        Field(27, "zero_balance_removal_upb", "ZERO BALANCE REMOVAL UPB", **_MONEY),
        Field(28, "delinquent_accrued_interest", "DELINQUENT ACCRUED INTEREST", **_MONEY),
        Field(29, "delinquency_due_to_disaster", "DELINQUENCY DUE TO DISASTER", "code", "performance",
              allowed=("Y",)),
        Field(30, "borrower_assistance_status", "BORROWER ASSISTANCE STATUS CODE", "code", "performance",
              allowed=("F", "R", "T")),
        Field(31, "current_month_modification_cost", "CURRENT MONTH MODIFICATION COST", **_MONEY),
        Field(32, "interest_bearing_upb", "INTEREST BEARING UPB", **_MONEY),
    ),
)

# ---------------------------------------------------------------------------------------------
# Release 47 (July 2026) layout
# ---------------------------------------------------------------------------------------------
CONTRACT_VERSION_R47 = "sflld_r47_2026_07"


def _derive(base: FileContract, version: str, order: list[str], changes: dict[str, dict],
            new: dict[str, Field]) -> FileContract:
    fields = []
    for pos, name in enumerate(order, start=1):
        f = new[name] if name in new else base.by_name(name)
        fields.append(Field(**{**f.__dict__, "position": pos, **changes.get(name, {})}))
    return FileContract(kind=base.kind, version=version, fields=tuple(fields))


ORIGINATION_R47 = _derive(
    ORIGINATION, CONTRACT_VERSION_R47,
    order=[n for n in ORIGINATION.names if n not in ("servicer_name", "mi_cancellation_indicator")]
    + ["vantage_score_4"],
    changes={
        "credit_score": {"formal_name": "CLASSIC FICO"},
        "first_time_homebuyer_flag": {"formal_name": "FIRST TIME HOMEBUYER INDICATOR"},
        "ppm_flag": {"formal_name": "PREPAYMENT PENALTY INDICATOR"},
        "loan_id": {"formal_name": "LOAN IDENTIFIER"},
        "postal_code": {"sentinels": ("000",), "pattern": r"^[0-9]{3}$",
                        "notes": "First three digits of the 5-digit postal code; '000' = unknown (R47)."},
        "num_borrowers": {"notes": "R47 drops zero padding. 2018Q1 and prior: 2 means '>1 borrowers'."},
        "seller_name": {"notes": "'OTHER' below a quarter-level 1% UPB threshold computed after the quarter."},
        "super_conforming_flag": {"allowed": ("Y", "N")},
        "pre_relief_refi_loan_id": {"formal_name": "PRE-HARP LOAN SEQUENCE NUMBER"},
        "special_eligibility_program": {"sentinels": ("9",), "notes": "R47: Null = not available/applicable."},
        "relief_refi_indicator": {"formal_name": "HARP INDICATOR", "allowed": ("Y", "N")},
    },
    new={
        "vantage_score_4": Field(31, "vantage_score_4", "VANTAGESCORE 4.0", "int", "origination",
                                 sentinels=("9999",), min_value=300, max_value=850,
                                 notes="New in R47. Check coverage by vintage before using as a feature."),
    },
)

PERFORMANCE_R47 = _derive(
    PERFORMANCE, CONTRACT_VERSION_R47,
    order=PERFORMANCE.names + ["mi_cancellation_indicator", "servicer_name", "bankruptcy_cramdown_costs"],
    changes={
        "reporting_period": {"formal_name": "PERIOD"},
        "delinquency_status": {"sentinels": ("XX",),
                               "notes": "R47: '00','01',... (capped at 99); 'RA' = REO acquisition; 'XX' = not "
                                        "available; '00' is also used for cycles before the first payment date."},
        "defect_settlement_date": {"formal_name": "UNDERWRITING DEFECT AND MAJOR SERVICING DEFECT SETTLEMENT DATE"},
        "payment_deferral_flag": {"allowed": ("C", "P"), "notes": "R47: C = current period (previously Y)."},
        "estimated_ltv": {"notes": "R47: 999 = data not available."},
        "borrower_assistance_status": {"formal_name": "BORROWER ASSISTANCE PLAN"},
        "net_sale_proceeds": {"notes": "R47 sign convention: recoveries/gains negative, expenses/losses positive."},
    },
    new={
        "mi_cancellation_indicator": Field(33, "mi_cancellation_indicator",
                                           "MORTGAGE INSURANCE CANCELLATION INDICATOR", "code", "performance",
                                           sentinels=("7",), allowed=("Y", "N")),
        "servicer_name": Field(34, "servicer_name", "SERVICER NAME", "str", "performance"),
        "bankruptcy_cramdown_costs": Field(35, "bankruptcy_cramdown_costs", "BANKRUPTCY CRAMDOWN COSTS", "float",
                                           "performance"),
    },
)

CONTRACT_SETS: dict[str, dict[str, FileContract]] = {
    CONTRACT_VERSION: {"origination": ORIGINATION, "performance": PERFORMANCE},
    CONTRACT_VERSION_R47: {"origination": ORIGINATION_R47, "performance": PERFORMANCE_R47},
}
DEFAULT_CONTRACT_VERSION = CONTRACT_VERSION_R47


def get_contract(version: str, kind: str) -> FileContract:
    if version not in CONTRACT_SETS:
        raise KeyError(f"Unknown contract version {version!r}; known: {sorted(CONTRACT_SETS)}")
    return CONTRACT_SETS[version][kind]


def _check_contract(c: FileContract) -> None:
    positions = [f.position for f in c.fields]
    assert positions == list(range(1, len(c.fields) + 1)), f"{c.kind}: positions not contiguous"
    assert len(set(c.names)) == len(c.names), f"{c.kind}: duplicate names"


for _set in CONTRACT_SETS.values():
    for _c in _set.values():
        _check_contract(_c)
