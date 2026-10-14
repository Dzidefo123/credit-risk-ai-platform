"""Loss-severity models (loss given liquidation) using origination-time inputs.

S0 = mean; S1 = segment means (MI coverage x LTV band); S2 = fractional-response logistic regression. S2 is fitted
as quasi-likelihood logistic regression: each loan enters twice, as a "loss" row weighted by its severity and a
"no loss" row weighted by 1 - severity. Severity is clipped to [0, 1] for fitting; evaluation uses the raw values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["orig_ltv", "orig_cltv", "mi_pct", "log_orig_upb", "credit_score", "orig_dti"]
CATEGORICAL = ["occupancy_status", "property_type", "loan_purpose", "channel"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["log_orig_upb"] = np.log(pd.to_numeric(X["orig_upb"], errors="coerce").clip(lower=1))
    for c in CATEGORICAL:
        X[c] = X[c].astype("object").where(X[c].notna(), "__MISSING__").astype(str)
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def ltv_band(ltv: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(ltv, errors="coerce"), [-np.inf, 80, 90, np.inf],
                  labels=["<=80", "80-90", ">90"]).astype(str)


def segment_key(df: pd.DataFrame) -> pd.Series:
    mi = np.where(pd.to_numeric(df["mi_pct"], errors="coerce").fillna(0) > 0, "MI", "noMI")
    return pd.Series(mi, index=df.index) + "|" + ltv_band(df["orig_ltv"])


@dataclass
class SeverityModel:
    kind: str  # S0 | S1 | S2
    mean_: float = 0.0
    segments_: dict = field(default_factory=dict)
    pipeline: Pipeline | None = None

    def fit(self, df: pd.DataFrame) -> SeverityModel:
        y = df["severity"].to_numpy(dtype="float64")
        self.mean_ = float(np.mean(y))
        if self.kind == "S1":
            self.segments_ = df.assign(_k=segment_key(df)).groupby("_k")["severity"].mean().to_dict()
        elif self.kind == "S2":
            X = prepare(df)[NUMERIC + CATEGORICAL]
            s = np.clip(y, 0.0, 1.0)
            pre = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                  ("sc", StandardScaler())]), NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                      sparse_output=False), CATEGORICAL)])
            self.pipeline = Pipeline([("pre", pre), ("clf", LogisticRegression(C=1.0, max_iter=2000))])
            Xd = pd.concat([X, X], ignore_index=True)
            yd = np.r_[np.ones(len(X)), np.zeros(len(X))]
            wd = np.r_[s, 1 - s]
            self.pipeline.fit(Xd, yd, clf__sample_weight=wd)
        elif self.kind != "S0":
            raise ValueError(self.kind)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "S0":
            return np.full(len(df), self.mean_)
        if self.kind == "S1":
            return segment_key(df).map(self.segments_).fillna(self.mean_).to_numpy(dtype="float64")
        return self.pipeline.predict_proba(prepare(df)[NUMERIC + CATEGORICAL])[:, 1]


def severity_metrics(y: np.ndarray, p: np.ndarray, weight: np.ndarray, n_boot: int = 1000,
                     seed: int = 20261013) -> dict:
    y, p, w = (np.asarray(a, dtype="float64") for a in (y, p, weight))
    rho = spearmanr(y, p).statistic if np.std(p) > 0 else float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    if np.std(p) > 0:
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            boots.append(spearmanr(y[i], p[i]).statistic)
    ci = (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))) if boots else (None, None)
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "mean_observed": float(np.mean(y)), "mean_predicted": float(np.mean(p)),
            "bias_mean": float(np.mean(p - y)),
            "upb_weighted_observed": float(np.sum(w * y) / np.sum(w)),
            "upb_weighted_predicted": float(np.sum(w * p) / np.sum(w)),
            "spearman": float(rho), "spearman_ci": ci}


def eltv_link(df: pd.DataFrame) -> dict:
    """Descriptive fractional-logit link: severity ~ a + b * ELTV_at_default (for house-price stress)."""
    d = df.dropna(subset=["eltv_at_default"])
    x = d["eltv_at_default"].to_numpy(dtype="float64") / 100.0
    s = np.clip(d["severity"].to_numpy(dtype="float64"), 0, 1)
    # quasi-likelihood fractional logit: each loan as a weighted loss row and a weighted no-loss row
    xx = np.r_[x, x]
    yy = np.r_[np.ones(len(x)), np.zeros(len(x))]
    ww = np.r_[s, 1 - s]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(xx.reshape(-1, 1), yy, sample_weight=ww)
    a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    return {"n": int(len(d)), "intercept": a, "slope_per_unit_eltv": b,
            "severity_at_eltv": {str(v): float(1 / (1 + np.exp(-(a + b * v / 100)))) for v in (60, 80, 100, 120)}}
