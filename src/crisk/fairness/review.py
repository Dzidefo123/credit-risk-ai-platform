"""Fairness and geography review (pre-registered: docs/fairness_review_plan_v1.md).

There are no protected attributes in the data, so this is a value + proxy-screen + segment-error review, not a
disparate-impact test. State demographics come from public Census aggregates; loan records are never linked to
individual-level external data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crisk.config import REPO_ROOT, DatasetConfig
from crisk.evaluation.metrics import binary_metrics

CENSUS_FILE = REPO_ROOT / "data" / "raw" / "census" / "sc-est2024-alldata6.csv"
CENSUS_YEAR_COL = "POPESTIMATE2022"
FIPS_TO_POSTAL = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
    "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
    "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY",
}
REGIONS = {"1": "Northeast", "2": "Midwest", "3": "South", "4": "West"}
MIN_STATE_TRAIN_LOANS = 500
MIN_SEGMENT_LOANS, MIN_SEGMENT_EVENTS = 1000, 10


def census_state_shares(path: Path = CENSUS_FILE) -> pd.DataFrame:
    """Per state: total population and shares (not NH-White-alone, Black alone, Hispanic), Census region."""
    df = pd.read_csv(path, dtype={"STATE": str, "SUMLEV": str, "REGION": str})
    df = df[(df.SUMLEV == "040") & (df.SEX == 0)]
    pop = CENSUS_YEAR_COL
    total = df[df.ORIGIN == 0].groupby("STATE")[pop].sum()
    nh_white = df[(df.ORIGIN == 1) & (df.RACE == 1)].groupby("STATE")[pop].sum()
    black = df[(df.ORIGIN == 0) & (df.RACE == 2)].groupby("STATE")[pop].sum()
    hisp = df[df.ORIGIN == 2].groupby("STATE")[pop].sum()
    region = df.groupby("STATE")["REGION"].first()
    out = pd.DataFrame({"population": total, "share_not_nh_white": 1 - nh_white / total,
                        "share_black": black / total, "share_hispanic": hisp / total, "region": region.map(REGIONS)})
    out.index = out.index.map(FIPS_TO_POSTAL)
    return out[out.index.notna()]


def paired_auc_delta(y: np.ndarray, p_with: np.ndarray, p_without: np.ndarray, n_boot: int = 1000,
                     seed: int = 20261013) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    base = roc_auc_score(y, p_with) - roc_auc_score(y, p_without)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:
            d.append(roc_auc_score(y[idx], p_with[idx]) - roc_auc_score(y[idx], p_without[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_auc_with_minus_without": float(base), "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(d)}


def state_effects(bundle, frame: pd.DataFrame) -> pd.Series:
    """Mean logit contribution of property_state per state (relative to the training mean)."""
    contrib = bundle.contributions(frame)
    return contrib["property_state"].groupby(frame["property_state"].to_numpy()).mean()


def _band(cs: pd.Series) -> pd.Series:
    bins = [-np.inf, 659, 699, 739, 779, np.inf]
    labels = ["<660", "660-699", "700-739", "740-779", "780+"]
    return pd.cut(cs, bins=bins, labels=labels).astype("object").where(cs.notna(), "missing")


def segment_table(frame: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray], census: pd.DataFrame) -> list:
    seg = pd.DataFrame({
        "region": frame["property_state"].map(census["region"]).fillna("Other/territory").to_numpy(),
        "state": frame["property_state"].to_numpy(),
        "credit_score_band": _band(frame["credit_score"]).to_numpy(),
        "occupancy": frame["occupancy_status"].to_numpy(), "purpose": frame["loan_purpose"].to_numpy(),
        "channel": frame["channel"].to_numpy(), "first_time_buyer": frame["first_time_homebuyer_flag"].to_numpy(),
    })
    top_states = seg["state"].value_counts().head(10).index
    rows = []
    for dim in seg.columns:
        values = top_states if dim == "state" else sorted(seg[dim].astype(str).unique())
        for v in values:
            m = (seg[dim].astype(str) == str(v)).to_numpy()
            if m.sum() < MIN_SEGMENT_LOANS or y[m].sum() < MIN_SEGMENT_EVENTS:
                continue
            row = {"dimension": dim, "segment": str(v), "n": int(m.sum()), "events": int(y[m].sum()),
                   "observed_rate": float(y[m].mean())}
            for name, p in preds.items():
                mm = binary_metrics(y[m], p[m])
                row[f"{name}_auc"] = mm.get("roc_auc")
                row[f"{name}_oe"] = mm["observed_expected_ratio"]
                row[f"{name}_mean_pd"] = mm["mean_pd"]
            rows.append(row)
    return rows


def run_review(cfg: DatasetConfig, with_state_id: str, without_state_id: str, store, holdout_cfg=None) -> dict:
    from scipy.stats import spearmanr

    from crisk.features.snapshots import snapshot_path
    from crisk.models.registry import load_bundle
    from crisk.models.train import target_label_columns
    from crisk.utils import utc_now, write_json

    census = census_state_shares()
    b_with, m_with = load_bundle(cfg, with_state_id)
    b_wo, m_wo = load_bundle(cfg, without_state_id)
    if m_with["target_id"] != m_wo["target_id"]:
        raise ValueError("Both models must share a target")
    col = target_label_columns(cfg)[m_with["target_id"]]
    snap = pd.read_parquet(snapshot_path(cfg, m_with["feature_set_id"]))
    train, cal = snap[snap.split == "train"], snap[snap.split == "calibration"]
    cal = cal[cal[col].notna()]
    y = cal[col].astype(int).to_numpy()

    # 1. value of property_state (uncalibrated, development split)
    value = paired_auc_delta(y, b_with.predict_raw(cal), b_wo.predict_raw(cal))
    value["auc_with"] = float(roc_auc_score(y, b_with.predict_raw(cal)))
    value["auc_without"] = float(roc_auc_score(y, b_wo.predict_raw(cal)))

    # 2. proxy screen across states
    effects = state_effects(b_with, cal)
    counts = train["property_state"].value_counts()
    eligible = [s for s in effects.index if counts.get(s, 0) >= MIN_STATE_TRAIN_LOANS and s in census.index]
    screen = {}
    for share in ("share_not_nh_white", "share_black", "share_hispanic"):
        rho, pval = spearmanr(effects[eligible], census.loc[eligible, share])
        screen[share] = {"spearman_rho": float(rho), "p_value": float(pval), "states": len(eligible)}
    max_abs_rho = max(abs(v["spearman_rho"]) for v in screen.values())
    state_table = (pd.DataFrame({"state_effect_logit": effects[eligible],
                                 "train_loans": counts.reindex(eligible).astype(int)})
                   .join(census[["share_not_nh_white", "share_black", "share_hispanic", "region"]])
                   .sort_values("state_effect_logit"))

    # 3. segments (calibrated PDs)
    preds = {"with_state": b_with.predict(cal), "without_state": b_wo.predict(cal)}
    segments = {"calibration_split": segment_table(cal, y, preds, census)}
    if holdout_cfg is not None:
        hs = pd.read_parquet(snapshot_path(holdout_cfg, m_with["feature_set_id"]))
        hs = hs[(hs.split == "test") & hs[target_label_columns(holdout_cfg)[m_with["target_id"]]].notna()]
        hy = hs[target_label_columns(holdout_cfg)[m_with["target_id"]]].astype(int).to_numpy()
        segments["holdout_diagnostic_previously_viewed"] = segment_table(
            hs, hy, {"with_state": b_with.predict(hs), "without_state": b_wo.predict(hs)}, census)

    # Decision (pre-registered)
    adds_value = value["delta_auc_with_minus_without"] >= 0.010 and value["ci_low"] > 0
    weak_proxy = max_abs_rho < 0.5
    keep = adds_value and weak_proxy
    decision = {"keep_property_state": keep, "adds_clear_value": adds_value, "weak_proxy_screen": weak_proxy,
                "max_abs_spearman_rho": max_abs_rho,
                "action": "keep property_state; add state-segment monitoring" if keep else
                "remove property_state: M1-nostate becomes champion candidate (validate, then promote)"}
    report = {"plan": "docs/fairness_review_plan_v1.md", "reviewed_at": utc_now(), "target_id": m_with["target_id"],
              "model_with_state": with_state_id, "model_without_state": without_state_id,
              "census_source": {"file": CENSUS_FILE.name, "column": CENSUS_YEAR_COL},
              "value_of_property_state": value, "proxy_screen": screen,
              "state_table": state_table.reset_index(names="state").to_dict(orient="records"),
              "segments": segments, "decision": decision}
    write_json(cfg.reports_dir / "fairness_v1" / f"review__{m_with['target_id']}.json", report)
    store.append_event(actor="validator", action="fairness_review", entity_type="model",
                       entity_id=f"{with_state_id},{without_state_id}", details=decision)
    return report
