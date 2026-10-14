"""Deterministic SYNTHETIC fixtures in the Freddie Mac SFLLD raw file format.

Everything produced here is fictional and exists only to test system behaviour.
Loan IDs use product letter "X", which Freddie Mac never issues, so fixture records
cannot be mistaken for, or collide with, real Loan Sequence Numbers. Model metrics
computed on these fixtures are NOT evidence about real credit risk.

Two parts:
1. Scenario loans: hand-specified edge cases with an answer key (expected label outcome,
   ingestion/quarantine expectations). Used by tests.
2. Random loans (optional): a simple simulated portfolio so the training/scoring slice has
   enough rows to exercise. The default mechanism is a known logistic formula; this is a
   simulation choice, not a claim about how mortgages default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from crisk.contracts.freddie_mac import CONTRACT_VERSION, CONTRACT_VERSION_R47, get_contract
from crisk.utils import month_add, month_diff, write_json

FIXTURE_VERSION = "synthetic_sflld_v1"
DEFAULT_SEED = 20261012
DEFAULT_CUTOFF = "2023-12"

_ORIG_DEFAULTS = {
    "credit_score": "740", "first_time_homebuyer_flag": "N", "msa_code": "", "mi_pct": "0", "num_units": "1",
    "occupancy_status": "P", "orig_cltv": "75", "orig_dti": "35", "orig_upb": "250000", "orig_ltv": "75",
    "orig_interest_rate": "4.25", "channel": "R", "ppm_flag": "N", "amortization_type": "FRM",
    "property_state": "TX", "property_type": "SF", "postal_code": "75000", "loan_purpose": "P",
    "orig_loan_term": "360", "num_borrowers": "02", "seller_name": "Synthetic Seller A",
    "servicer_name": "Synthetic Servicer A", "super_conforming_flag": "", "pre_relief_refi_loan_id": "",
    "special_eligibility_program": "9", "relief_refi_indicator": "", "property_valuation_method": "2",
    "io_indicator": "N", "mi_cancellation_indicator": "7", "vantage_score_4": "9999",
}

# Layout used when writing fixture files; set by ``generate``.
_LAYOUT = {"version": CONTRACT_VERSION_R47}


def _orig_to_r47(vals: dict) -> dict:
    """Release 47 enumeration changes (Disclosure Changes v1.2, 6/2026)."""
    v = dict(vals)
    if v["postal_code"] in ("00", "00000"):
        v["postal_code"] = "000"
    elif len(v["postal_code"]) == 5:
        v["postal_code"] = v["postal_code"][:3]
    if v["num_borrowers"].isdigit() and v["num_borrowers"] != "99":
        v["num_borrowers"] = str(int(v["num_borrowers"]))
    v["super_conforming_flag"] = v["super_conforming_flag"] or "N"
    v["relief_refi_indicator"] = v["relief_refi_indicator"] or "N"
    if v["special_eligibility_program"] == "9":
        v["special_eligibility_program"] = ""
    if v["seller_name"] == "Other Sellers":
        v["seller_name"] = "OTHER"
    return v


def _perf_to_r47(vals: dict) -> dict:
    v = dict(vals)
    if v["delinquency_status"].isdigit():
        v["delinquency_status"] = f"{int(v['delinquency_status']):02d}"
    if v["payment_deferral_flag"] == "Y":
        v["payment_deferral_flag"] = "C"
    v.setdefault("mi_cancellation_indicator", "7")
    v["mi_cancellation_indicator"] = v["mi_cancellation_indicator"] or "7"
    v["servicer_name"] = v.get("servicer_name") or "Synthetic Servicer A"
    return v


def _ymd(yyyy_mm: str) -> str:
    return yyyy_mm.replace("-", "")


def orig_line(loan_id: str, fpd: str, **overrides: str) -> str:
    term = int(overrides.get("orig_loan_term", _ORIG_DEFAULTS["orig_loan_term"]))
    vals = dict(_ORIG_DEFAULTS)
    vals.update(loan_id=loan_id, first_payment_date=_ymd(fpd), maturity_date=_ymd(month_add(fpd, term - 1)))
    vals.update(overrides)
    contract = get_contract(_LAYOUT["version"], "origination")
    if _LAYOUT["version"] == CONTRACT_VERSION_R47:
        vals = _orig_to_r47(vals)
    return "|".join(str(vals[n]) for n in contract.names)


def perf_line(loan_id: str, fpd: str, m: int, *, upb: float, status: str = "0", rate: str = "4.25",
              term: int = 360, **overrides: str) -> str:
    period = month_add(fpd, m - 1)
    contract = get_contract(_LAYOUT["version"], "performance")
    vals = {n: "" for n in contract.names}
    s_int = int(status) if status.isdigit() else 0
    vals.update(
        loan_id=loan_id, reporting_period=_ymd(period),
        current_actual_upb=f"{round(upb, -3) if m <= 6 else upb:.2f}", delinquency_status=status,
        loan_age=str(m), remaining_months_to_maturity=str(term - m), current_interest_rate=rate,
        current_non_interest_bearing_upb="0", ddlpi=_ymd(month_add(fpd, m - 1 - s_int)),
        interest_bearing_upb=f"{upb:.2f}",
    )
    vals.update(overrides)
    if _LAYOUT["version"] == CONTRACT_VERSION_R47:
        vals = _perf_to_r47(vals)
    return "|".join(str(vals[n]) for n in contract.names)


@dataclass
class Scenario:
    key: str
    loan_id: str
    fpd: str
    description: str
    expected_outcome: str | None  # None => loan must not reach the outcomes table
    months: list[int] = field(default_factory=list)
    status: dict[int, str] = field(default_factory=dict)
    zero_balance: tuple[int, str] | None = None
    month_overrides: dict[int, dict[str, str]] = field(default_factory=dict)
    orig_overrides: dict[str, str] = field(default_factory=dict)
    expected_default_month: int | None = None


def _sid(n: int) -> str:
    return f"X19Q1{n:07d}"


def scenarios() -> list[Scenario]:
    r = lambda a, b: list(range(a, b + 1))  # noqa: E731
    S = Scenario
    return [
        S("S01_clean", _sid(1), "2019-03", "Current throughout the window", "non_default", r(0, 20)),
        S("S02_90dpd", _sid(2), "2019-03", "Misses payments from month 2; status 3 at m=5", "default", r(0, 15),
          status={3: "1", 4: "2", 5: "3", 6: "4", 7: "5", 8: "6", 9: "7", 10: "8", 11: "9", 12: "10", 13: "11",
                  14: "12", 15: "13"}, expected_default_month=5),
        S("S03_prepaid", _sid(3), "2019-03", "Voluntary payoff at m=7", "non_default_prepaid", r(1, 7),
          zero_balance=(7, "01")),
        S("S04_reo", _sid(4), "2019-03", "REO acquisition reported as RA at m=6", "default", r(1, 8),
          status={3: "1", 4: "2", 5: "2", 6: "RA", 7: "RA", 8: "RA"}, expected_default_month=6),
        S("S05_short_sale", _sid(5), "2019-03", "Short sale (ZB 03) at m=6 while 60 days delinquent", "default",
          r(1, 6), status={4: "1", 5: "2", 6: "2"}, zero_balance=(6, "03"), expected_default_month=6),
        S("S06_immature", _sid(6), "2023-03", "Window ends 2024-02, after the 2023-12 cutoff", "excluded_immature",
          r(1, 10)),
        S("S07_reporting_stops", _sid(7), "2019-03", "Reports end at m=8 without termination",
          "excluded_incomplete_reporting", r(1, 8)),
        S("S08_one_gap", _sid(8), "2019-03", "Single missing month (m=5) is tolerated", "non_default",
          [m for m in r(1, 14) if m != 5]),
        S("S09_long_gap", _sid(9), "2019-03", "Four missing months (m=4..7)", "excluded_incomplete_reporting",
          [m for m in r(1, 14) if m not in (4, 5, 6, 7)]),
        S("S10_late_acquisition", _sid(10), "2019-03", "First reported at m=9", "excluded_late_acquisition",
          r(9, 16)),
        S("S11_defect", _sid(11), "2019-03", "Defect settlement (ZB 96) at m=6", "excluded_censored_removal",
          r(1, 6), zero_balance=(6, "96"), month_overrides={6: {"defect_settlement_date": "201908"}}),
        S("S12_late_default", _sid(12), "2019-03", "90+ DPD first at m=15, outside the window", "non_default",
          r(1, 16), status={13: "1", 14: "2", 15: "3", 16: "4"}),
        S("S13_modified", _sid(13), "2019-03", "Modified at m=6, never 90+", "non_default", r(1, 14),
          status={4: "1", 5: "2"}, month_overrides={6: {"modification_flag": "Y"},
                                                     **{m: {"modification_flag": "P"} for m in range(7, 15)}}),
        S("S14_forbearance", _sid(14), "2019-11", "90+ while in disaster forbearance (2020)", "default", r(1, 14),
          status={6: "1", 7: "2", 8: "3", 9: "4"}, expected_default_month=8,
          month_overrides={m: {"delinquency_due_to_disaster": "Y", "borrower_assistance_status": "F"}
                           for m in range(5, 10)}),
        S("S15_no_performance", _sid(15), "2019-03", "Origination record without performance rows",
          "excluded_no_performance", []),
        S("S16_irregular", _sid(16), "2019-03", "Irregular progression 2,1,3 (status 3 at m=6)", "default",
          r(1, 14), status={3: "1", 4: "2", 5: "1", 6: "3", 7: "1"}, expected_default_month=6),
        S("S17_boundary_m13", _sid(17), "2019-03", "Status 3 first reached at m=13", "non_default", r(1, 14),
          status={11: "1", 12: "2", 13: "3", 14: "4"}),
        S("S18_boundary_m12", _sid(18), "2019-03", "Status 3 first reached at m=12", "default", r(1, 14),
          status={10: "1", 11: "2", 12: "3", 13: "4", 14: "5"}, expected_default_month=12),
        S("S19_arm", _sid(19), "2019-03", "ARM product: outside the FRM target population", "excluded_ineligible",
          r(1, 14), orig_overrides={"amortization_type": "ARM"}),
        S("S20_whole_loan_sale", _sid(20), "2019-03", "Whole loan sale (ZB 15) at m=10, no prior default",
          "excluded_censored_removal", r(1, 10), zero_balance=(10, "15")),
        S("S21_conflicting_month", _sid(21), "2019-03", "Month 5 reported twice with different values -> gap",
          "non_default", r(1, 14)),
        S("S22_bad_status", _sid(22), "2019-03", "Unparseable delinquency status 'ZZ' at m=4 is nulled",
          "non_default", r(1, 14), month_overrides={4: {"delinquency_status": "ZZ"}}),
        S("S23_bad_credit_score", _sid(23), "2019-03", "Credit score 1200 is invalid -> NULL", "non_default",
          r(1, 14), orig_overrides={"credit_score": "1200"}),
        S("S24_missing_score", _sid(24), "2019-03", "Credit score sentinel 9999 -> NULL", "non_default", r(1, 14),
          orig_overrides={"credit_score": "9999", "orig_dti": "999"}),
        S("S25_prepay_then_window", _sid(25), "2019-03", "Payoff at m=3 after reporting from m=1",
          "non_default_prepaid", r(1, 3), zero_balance=(3, "01")),
    ]


# Records that must be quarantined or de-duplicated (answer key for ingestion tests).
CONFLICT_LOAN = _sid(9000001)
BAD_FPD_LOAN = _sid(9000002)
ORPHAN_LOAN = _sid(9999999)


def _scenario_lines() -> tuple[list[str], list[str], dict]:
    orig, perf = [], []
    for sc in scenarios():
        orig.append(orig_line(sc.loan_id, sc.fpd, **sc.orig_overrides))
        upb = float(_ORIG_DEFAULTS["orig_upb"])
        for m in sc.months:
            ov = dict(sc.month_overrides.get(m, {}))
            status = ov.pop("delinquency_status", sc.status.get(m, "0"))
            if sc.zero_balance and sc.zero_balance[0] == m:
                ov.update(zero_balance_code=sc.zero_balance[1],
                          zero_balance_effective_date=_ymd(month_add(sc.fpd, m - 1)),
                          zero_balance_removal_upb=f"{upb:.2f}")
                perf.append(perf_line(sc.loan_id, sc.fpd, m, upb=0.0, status=status, **ov))
            else:
                perf.append(perf_line(sc.loan_id, sc.fpd, m, upb=upb, status=status, **ov))
            upb = max(upb - 400.0, 0.0)

    # Exact duplicates (resolved by keeping one copy).
    orig.append(orig[0])
    perf.append(perf[3])
    # Conflicting month for S21 (both copies quarantined -> reporting gap at m=5).
    s21 = next(s for s in scenarios() if s.key == "S21_conflicting_month")
    perf.append(perf_line(s21.loan_id, s21.fpd, 5, upb=123456.0))
    # Conflicting origination duplicates -> both quarantined, performance becomes orphaned.
    orig.append(orig_line(CONFLICT_LOAN, "2019-03", credit_score="700"))
    orig.append(orig_line(CONFLICT_LOAN, "2019-03", credit_score="701"))
    perf += [perf_line(CONFLICT_LOAN, "2019-03", m, upb=250000.0) for m in range(1, 4)]
    # Invalid keys -> row rejected.
    orig.append(orig_line("BADID123", "2019-03"))
    orig.append(orig_line(BAD_FPD_LOAN, "2019-03").replace("|201903|", "|2019AB|", 1))
    perf += [perf_line(BAD_FPD_LOAN, "2019-03", m, upb=250000.0) for m in range(1, 4)]
    # Orphan performance rows.
    perf += [perf_line(ORPHAN_LOAN, "2019-03", m, upb=250000.0) for m in range(1, 4)]
    # Structural error: one field missing.
    perf.append("|".join(perf_line(_sid(1), "2019-03", 99, upb=1.0).split("|")[:-1]))

    key = {
        "fixture_version": FIXTURE_VERSION,
        "data_origin": "SYNTHETIC",
        "performance_cutoff": DEFAULT_CUTOFF,
        "expected_outcomes": {s.loan_id: s.expected_outcome for s in scenarios()},
        "expected_default_month": {s.loan_id: s.expected_default_month for s in scenarios()
                                   if s.expected_default_month is not None},
        "descriptions": {s.loan_id: f"{s.key}: {s.description}" for s in scenarios()},
        "expected_quarantine": {
            "origination_rejected_rows": 2,  # BADID123, bad FPD
            "origination_conflicting_loans": [CONFLICT_LOAN],
            "performance_structural_rejects": 1,
            "performance_conflicting_loan_months": 1,
            "orphan_loans": sorted([CONFLICT_LOAN, BAD_FPD_LOAN, ORPHAN_LOAN]),
            "origination_exact_duplicates_removed": 1,
            "performance_exact_duplicates_removed": 1,
        },
        "expected_field_issues": {
            _sid(23): "credit_score",
            _sid(22): "delinquency_status",
        },
    }
    return orig, perf, key


_STATES = ["TX", "CA", "FL", "NY", "IL", "OH", "GA", "NC", "AZ", "WA"]
_BASE_RATE = {2015: 3.9, 2016: 3.7, 2017: 4.1, 2018: 4.6, 2019: 4.1, 2020: 3.1, 2021: 3.0, 2022: 5.0, 2023: 6.6}


def _random_lines(n: int, seed: int, cutoff: str, fpd_start: str, fpd_end: str) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    span = month_diff(fpd_end, fpd_start) + 1
    orig, perf = [], []
    for i in range(n):
        fpd = month_add(fpd_start, int(rng.integers(0, span)))
        note = month_add(fpd, -2)  # synthetic convention: note date two months before first payment
        yy, q = int(note[2:4]), (int(note[5:7]) - 1) // 3 + 1
        loan_id = f"X{yy:02d}Q{q}{1_000_000 + i:07d}"
        year = int(fpd[:4])
        cs = int(np.clip(rng.normal(745, 45), 580, 830))
        ltv = int(rng.integers(40, 98))
        cltv = ltv + (int(rng.integers(5, 15)) if rng.random() < 0.1 else 0)
        dti = int(np.clip(rng.normal(36, 9), 5, 50))
        upb = float(np.clip(round(rng.lognormal(np.log(250_000), 0.45), -3), 50_000, 900_000))
        base = _BASE_RATE.get(year, 4.0)
        rate = round(base + rng.normal(0, 0.3) + 0.004 * (760 - cs) / 10, 3)
        term = int(rng.choice([360, 180, 240], p=[0.85, 0.12, 0.03]))
        occ = str(rng.choice(["P", "I", "S"], p=[0.9, 0.06, 0.04]))
        chan = str(rng.choice(["R", "C", "B", "T"], p=[0.5, 0.3, 0.18, 0.02]))
        purpose = str(rng.choice(["P", "C", "N"], p=[0.5, 0.22, 0.28]))
        ptype = str(rng.choice(["SF", "PU", "CO", "MH"], p=[0.65, 0.25, 0.09, 0.01]))
        state = str(rng.choice(_STATES))
        nb = str(rng.choice(["01", "02"]))
        mi = str(int(rng.choice([6, 12, 25, 30, 35]))) if ltv > 80 else "0"
        fthb = ("Y" if rng.random() < 0.3 else "N") if purpose == "P" else "9"

        z = (-4.3 - 0.013 * (cs - 745) + 0.035 * (ltv - 75) + 0.03 * (dti - 36) + 0.4 * (occ == "I")
             + 0.25 * (purpose == "C") + 0.2 * (chan == "B") + 0.25 * (nb == "01") + 0.4 * (rate - base)
             + (0.35 if year == 2020 else 0.0))
        p12 = 1 / (1 + np.exp(-z))

        raw_cs = "9999" if rng.random() < 0.015 else str(cs)
        raw_dti = "999" if rng.random() < 0.02 else str(dti)
        orig.append(orig_line(
            loan_id, fpd, credit_score=raw_cs, orig_ltv=str(ltv), orig_cltv=str(cltv), orig_dti=raw_dti,
            orig_upb=f"{upb:.0f}", orig_interest_rate=f"{rate:.3f}", orig_loan_term=str(term), occupancy_status=occ,
            channel=chan, loan_purpose=purpose, property_type=ptype, property_state=state, num_borrowers=nb,
            mi_pct=mi, first_time_homebuyer_flag=fthb, mi_cancellation_indicator="N" if mi != "0" else "7",
            postal_code=f"{int(rng.integers(100, 999))}00",
            property_valuation_method="2" if year >= 2017 else "7",
            seller_name=str(rng.choice(["Synthetic Seller A", "Synthetic Seller B", "Other Sellers"])),
            servicer_name=str(rng.choice(["Synthetic Servicer A", "Other Servicers"])),
        ))

        m_cut = month_diff(cutoff, fpd) + 1
        m0 = int(rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1]))
        if rng.random() < 0.005:
            m0 = int(rng.integers(8, 11))
        d_start = None
        if rng.random() < p12:
            d_start = int(rng.integers(1, 10))  # status 3 reached at d_start + 3 <= 12
        elif rng.random() < 0.6 * p12:
            d_start = int(rng.integers(10, 37))
        cures = rng.random() < 0.25
        hazard = 0.006 + 0.003 * max(rate - base, 0)
        prepay_m = int(rng.geometric(hazard))
        blip = int(rng.integers(2, 30)) if (d_start is None and rng.random() < 0.05) else None
        gap = int(rng.integers(2, 12)) if rng.random() < 0.02 else None
        covid = rng.random() < 0.03

        end = min(m_cut, prepay_m)
        term_code = "01" if prepay_m <= m_cut else None
        if d_start is not None and not cures:
            m_term = d_start + 9
            if m_term < end or (m_term == end and term_code is None):
                end, term_code = m_term, str(rng.choice(["03", "09"]))
            elif d_start + 1 <= prepay_m:
                # a seriously delinquent loan does not voluntarily prepay in this simulation
                end, term_code = min(m_cut, m_term), ("03" if m_term <= m_cut else None)
        bal = upb
        for m in range(m0, end + 1):
            if gap is not None and m == gap:
                continue
            status = "0"
            if d_start is not None and m > d_start:
                s = m - d_start
                if cures and s > 3:
                    s = 0
                status = str(s)
                if term_code == "09" and s >= 7:
                    status = "RA"
            elif blip is not None and m == blip:
                status = "1"
            ov: dict[str, str] = {}
            period = month_add(fpd, m - 1)
            if covid and "2020-04" <= period <= "2020-12" and (d_start is None):
                months_in = month_diff(period, "2020-04")
                status = str(min(months_in + 1, 4)) if months_in < 6 else "0"
                ov = {"delinquency_due_to_disaster": "Y", "borrower_assistance_status": "F"}
            if covid and period == "2021-01" and d_start is None:
                ov = {"payment_deferral_flag": "Y"}
            if m == end and term_code is not None:
                ov.update(zero_balance_code=term_code, zero_balance_effective_date=_ymd(period),
                          zero_balance_removal_upb=f"{bal:.2f}")
                if term_code in ("03", "09"):
                    # recovery rate varies deterministically by loan (no extra random draws)
                    proceeds = round(bal * (0.55 + 0.30 * ((i % 7) / 6)), 2)
                    expenses = round(bal * 0.08, 2)
                    if _LAYOUT["version"] == CONTRACT_VERSION_R47:  # recoveries disclosed as negatives
                        nsp, loss = -proceeds, bal + expenses - proceeds
                    else:
                        nsp, loss = proceeds, bal - proceeds + expenses
                    ov.update(net_sale_proceeds=f"{nsp:.2f}", total_expenses=f"{expenses:.2f}",
                              mi_recoveries="0", non_mi_recoveries="0", delinquent_accrued_interest="0",
                              actual_loss=f"{loss:.2f}")
                perf.append(perf_line(loan_id, fpd, m, upb=0.0, status=status, rate=f"{rate:.3f}", term=term, **ov))
            else:
                perf.append(perf_line(loan_id, fpd, m, upb=bal, status=status, rate=f"{rate:.3f}", term=term, **ov))
            if status == "0" and m >= 1:
                bal = max(bal - upb / term * 0.8, 0.0)
    return orig, perf


def generate(out_dir: Path, n_random: int = 0, seed: int = DEFAULT_SEED, cutoff: str = DEFAULT_CUTOFF,
             fpd_start: str = "2016-01", fpd_end: str = "2023-06",
             contract_version: str = CONTRACT_VERSION_R47) -> dict:
    """Write SYNTHETIC raw files + answer key. Output is byte-identical for the same arguments."""
    if contract_version not in (CONTRACT_VERSION, CONTRACT_VERSION_R47):
        raise ValueError(contract_version)
    _LAYOUT["version"] = contract_version
    out_dir.mkdir(parents=True, exist_ok=True)
    orig, perf, key = _scenario_lines()
    if n_random:
        o2, p2 = _random_lines(n_random, seed, cutoff, fpd_start, fpd_end)
        orig += o2
        perf += p2
    # Group by origination year from the loan id (like provider vintage files); unparseable IDs go to 2019.
    def year_of(line: str, pos: int) -> str:
        lid = line.split("|")[pos]
        return f"20{lid[1:3]}" if lid[:1] == "X" and lid[1:3].isdigit() else "2019"

    files: dict[str, list[str]] = {}
    for line in orig:
        files.setdefault(f"sample_orig_{year_of(line, 19)}.txt", []).append(line)
    for line in perf:
        files.setdefault(f"sample_svcg_{year_of(line, 0)}.txt", []).append(line)
    for name, lines in sorted(files.items()):
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    key.update({"seed": seed, "n_random": n_random, "files": sorted(files), "contract_version": contract_version,
                "warning": "SYNTHETIC fixture. Not empirical evidence about credit risk."})
    write_json(out_dir / "answer_key.json", key)
    return key
