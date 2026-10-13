"""Temporal train / calibration / test assignment with purge gaps."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crisk.config import load_yaml
from crisk.utils import month_diff

SPLIT_ORDER = ("train", "calibration", "test")


@dataclass(frozen=True)
class SplitSpec:
    split_id: str
    date_column: str
    min_gap_months: int
    ranges: dict[str, tuple[str, str]]
    # optional: restrict a split to one source_id, so a split never mixes full-volume and sample loans
    sources: dict[str, str] | None = None

    @classmethod
    def load(cls, path: str, horizon_months: int) -> SplitSpec:
        raw = load_yaml(path)
        ranges = {k: (str(v["start"]), str(v["end"])) for k, v in raw["splits"].items()}
        sources = {k: v["source_id"] for k, v in raw["splits"].items() if "source_id" in v}
        spec = cls(raw["split_id"], raw["date_column"], int(raw["min_gap_months"]), ranges, sources or None)
        spec.validate(horizon_months)
        return spec

    def validate(self, horizon_months: int) -> None:
        if tuple(self.ranges) != SPLIT_ORDER:
            raise ValueError(f"splits must be exactly {SPLIT_ORDER} in order, got {tuple(self.ranges)}")
        if self.min_gap_months < horizon_months:
            raise ValueError(f"min_gap_months ({self.min_gap_months}) must be >= target horizon ({horizon_months}) "
                             "so earlier labels are observable before later cohorts are scored")
        prev_end = None
        for name in SPLIT_ORDER:
            start, end = self.ranges[name]
            if month_diff(end, start) < 0:
                raise ValueError(f"{name}: end before start")
            if prev_end is not None and month_diff(start, prev_end) < self.min_gap_months + 1:
                raise ValueError(f"{name} starts {start}; must be > {self.min_gap_months} months after previous "
                                 f"split end {prev_end}")
            prev_end = end

    def assign(self, dates: pd.Series, source_ids: pd.Series | None = None) -> pd.Series:
        ym = pd.to_datetime(dates).dt.strftime("%Y-%m")
        out = pd.Series("unassigned", index=dates.index, dtype="object")
        for name, (start, end) in self.ranges.items():
            mask = (ym >= start) & (ym <= end)
            if self.sources and name in self.sources:
                if source_ids is None:
                    raise ValueError(f"split {name!r} filters on source_id but none was provided")
                mask &= source_ids == self.sources[name]
            out[mask] = name
        return out
