"""Freddie Mac Primary Mortgage Market Survey (PMMS) market rates, joined point-in-time.

Source: Freddie Mac PMMS via FRED (MORTGAGE30US, MORTGAGE15US). Weekly, published Thursdays at
12:00 ET, covering applications from the prior Thursday to Wednesday. Attribution required:
"Freddie Mac, Primary Mortgage Market Survey". Retrieved 2026-10-12.

Revision check (2026-10-12): ALFRED vintages of MORTGAGE30US as of 2017-01-05, 2020-01-02 and
2023-01-05 match the current series exactly on all overlapping weeks (0 revised values), so each
weekly value is treated as known on its publication date (= the FRED observation date, a Thursday).

Point-in-time rule (assumption A-020): a loan's market reference rate is the mean of the weekly
values *published* in the calendar month ``first_payment_date - REFERENCE_LAG_MONTHS``. With a lag
of 3 months, every value used was published before the note date (usually 1-2 months before the
first payment) and close to typical rate-lock timing. Terms <= 180 months use the 15-year series;
longer terms use the 30-year series.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crisk.config import REPO_ROOT
from crisk.utils import sha256_file

PMMS_DIR = REPO_ROOT / "data" / "raw" / "macro" / "pmms"  # override with env CRISK_PMMS_DIR (tests)
SERIES = {"MORTGAGE30US": "fred_MORTGAGE30US_current.csv", "MORTGAGE15US": "fred_MORTGAGE15US_current.csv"}
REFERENCE_LAG_MONTHS = 3
SHORT_TERM_MAX_MONTHS = 180


@dataclass
class MarketRateTable:
    """Monthly averages of published weekly PMMS values, keyed by publication month (YYYY-MM)."""

    monthly: pd.DataFrame  # columns: month (str YYYY-MM), MORTGAGE30US, MORTGAGE15US, weeks_30, weeks_15
    source_sha256: dict[str, str]
    last_publication_date: str

    @classmethod
    def from_fred_csv(cls, directory: Path | None = None) -> MarketRateTable:
        directory = Path(directory or os.environ.get("CRISK_PMMS_DIR") or PMMS_DIR)
        frames, shas = [], {}
        for sid, fname in SERIES.items():
            path = Path(directory) / fname
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; see docs/data_acquisition_macro.md")
            shas[sid] = sha256_file(path)
            df = pd.read_csv(path)
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
            df = df.dropna()
            df["series"] = sid
            frames.append(df)
        weekly = pd.concat(frames)
        weekly["month"] = pd.to_datetime(weekly["date"]).dt.strftime("%Y-%m")
        agg = weekly.groupby(["month", "series"])["value"].agg(["mean", "count"]).unstack("series")
        monthly = pd.DataFrame({
            "month": agg.index,
            "MORTGAGE30US": agg[("mean", "MORTGAGE30US")].to_numpy(),
            "MORTGAGE15US": agg[("mean", "MORTGAGE15US")].to_numpy(),
            "weeks_30": agg[("count", "MORTGAGE30US")].to_numpy(),
            "weeks_15": agg[("count", "MORTGAGE15US")].to_numpy(),
        }).reset_index(drop=True)
        return cls(monthly=monthly, source_sha256=shas, last_publication_date=str(weekly["date"].max()))

    def reference_rate(self, first_payment_date: pd.Series, loan_term: pd.Series) -> pd.Series:
        """Market reference rate per loan (NaN when the reference month is not covered)."""
        fpd = pd.to_datetime(first_payment_date)
        ref_month = (fpd - pd.DateOffset(months=REFERENCE_LAG_MONTHS)).dt.strftime("%Y-%m")
        lut = self.monthly.set_index("month")
        r30 = ref_month.map(lut["MORTGAGE30US"])
        r15 = ref_month.map(lut["MORTGAGE15US"])
        term = pd.to_numeric(loan_term, errors="coerce")
        out = np.where(term <= SHORT_TERM_MAX_MONTHS, r15, r30)
        return pd.Series(out, index=first_payment_date.index, dtype="float64")

    def metadata(self) -> dict:
        return {"source": "Freddie Mac PMMS via FRED", "series": list(SERIES), "source_sha256": self.source_sha256,
                "last_publication_date": self.last_publication_date, "reference_lag_months": REFERENCE_LAG_MONTHS,
                "short_term_max_months": SHORT_TERM_MAX_MONTHS, "months": len(self.monthly)}
