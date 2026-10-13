"""Origination-time feature set: one definition shared by offline training and online scoring.

``build_feature_frame`` is a pure, row-wise function from canonical origination fields to
model inputs. It learns nothing from data, so it is safe to apply to any split. All learned
transformations (imputation, scaling, category pooling) live in the sklearn preprocessing
pipeline in ``crisk.models`` and are fitted on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crisk.config import load_yaml
from crisk.contracts.freddie_mac import CONTRACT_SETS, DEFAULT_CONTRACT_VERSION

ALLOWED_AVAILABILITY = {"origination"}
# Explicit token for missing/not-available categorical values (provider sentinels are already NULL).
MISSING_CATEGORY = "__MISSING__"
# Raw inputs used to compute derived features.
DERIVED_INPUTS = {
    "log_orig_upb": ("orig_upb",),
    "multiple_borrowers": ("num_borrowers",),
    "cltv_minus_ltv": ("orig_cltv", "orig_ltv"),
}


@dataclass(frozen=True)
class FeatureSpec:
    feature_set_id: str
    numeric: tuple[str, ...]
    derived_numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    acquisition_fields_accepted: dict[str, str] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> FeatureSpec:
        raw = load_yaml(path)
        spec = cls(
            feature_set_id=raw["feature_set_id"],
            numeric=tuple(raw["numeric"]),
            derived_numeric=tuple(raw.get("derived_numeric", {}).keys()),
            categorical=tuple(raw["categorical"]),
            acquisition_fields_accepted=dict(raw.get("acquisition_fields_accepted", {})),
            excluded=dict(raw.get("excluded", {})),
        )
        spec.check_availability()
        return spec

    @property
    def model_columns(self) -> list[str]:
        return [*self.numeric, *self.derived_numeric, *self.categorical]

    @property
    def numeric_columns(self) -> list[str]:
        return [*self.numeric, *self.derived_numeric]

    @property
    def raw_inputs(self) -> list[str]:
        """Canonical origination fields the feature set reads."""
        cols = [*self.numeric, *self.categorical]
        for d in self.derived_numeric:
            cols += list(DERIVED_INPUTS[d])
        return sorted(set(cols))

    def check_availability(self) -> None:
        """Leakage guard: every raw input must be known at origination (or explicitly accepted)."""
        origination = CONTRACT_SETS[DEFAULT_CONTRACT_VERSION]["origination"]
        for name in self.raw_inputs:
            if name not in origination.names:
                raise ValueError(f"Feature input {name!r} is not an origination-file field in "
                                 f"{DEFAULT_CONTRACT_VERSION} (post-origination or performance data)")
            f = origination.by_name(name)
            if f.availability in ALLOWED_AVAILABILITY:
                continue
            if f.availability == "acquisition" and name in self.acquisition_fields_accepted:
                continue
            raise ValueError(f"Feature input {name!r} has availability {f.availability!r}; not allowed "
                             f"in origination feature set {self.feature_set_id!r}")
        unknown = set(self.derived_numeric) - set(DERIVED_INPUTS)
        if unknown:
            raise ValueError(f"Derived features without an implementation: {sorted(unknown)}")


def build_feature_frame(records: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Map canonical origination records to model input columns (missing stays missing)."""
    missing = [c for c in spec.raw_inputs if c not in records.columns]
    if missing:
        raise KeyError(f"Missing canonical input columns: {missing}")
    out = pd.DataFrame(index=records.index)
    for c in spec.numeric:
        out[c] = pd.to_numeric(records[c], errors="coerce").astype("float64")
    upb = pd.to_numeric(records["orig_upb"], errors="coerce").astype("float64")
    nb = pd.to_numeric(records["num_borrowers"], errors="coerce").astype("float64")
    cltv = pd.to_numeric(records["orig_cltv"], errors="coerce").astype("float64")
    ltv = pd.to_numeric(records["orig_ltv"], errors="coerce").astype("float64")
    derived = {
        "log_orig_upb": np.log(upb.where(upb > 0)),
        "multiple_borrowers": np.where(nb.isna(), np.nan, (nb >= 2).astype("float64")),
        "cltv_minus_ltv": cltv - ltv,
    }
    for d in spec.derived_numeric:
        out[d] = pd.Series(derived[d], index=records.index, dtype="float64")
    for c in spec.categorical:
        s = records[c].astype("object")
        present = s.notna() & (s.astype(str).str.strip() != "")
        out[c] = pd.Series(np.where(present, s.astype(str).str.strip(), MISSING_CATEGORY), index=records.index,
                           dtype="object")
    return out[spec.model_columns]
