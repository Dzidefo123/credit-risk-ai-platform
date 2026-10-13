"""Portfolio monitoring (Phase I): drift, data quality, delayed outcomes and de-duplicated alerts.

The monitored stream is a historical replay of an evaluation dataset's application cohorts, by first-payment month,
scored with the serving model. Outcome metrics use only cohorts whose 12-month window has matured (all of them in a
historical replay; the logic is the same in production, where recent cohorts are still immature).

Thresholds (configs/monitoring/thresholds_v1.yaml) follow common industry PSI conventions (< 0.10 stable,
0.10-0.25 watch, >= 0.25 alert). Drift is a signal to investigate, not an automatic failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crisk.config import load_yaml

THRESHOLDS = "configs/monitoring/thresholds_v1.yaml"
EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype="float64"), EPS, None)
    a = np.clip(np.asarray(actual, dtype="float64"), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_bins(reference: pd.Series, n: int = 10) -> np.ndarray:
    edges = np.unique(np.nanquantile(reference.to_numpy(dtype="float64"), np.linspace(0, 1, n + 1)[1:-1]))
    return edges


def shares_numeric(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    v = x.to_numpy(dtype="float64")
    idx = np.where(np.isnan(v), len(edges) + 1, np.searchsorted(edges, v, side="right"))
    return np.bincount(idx, minlength=len(edges) + 2).astype("float64")


def shares_categorical(x: pd.Series, levels: list[str]) -> np.ndarray:
    s = x.astype(str).where(x.astype(str).isin(levels), "__OTHER__")
    return s.value_counts().reindex([*levels, "__OTHER__"], fill_value=0).to_numpy(dtype="float64")


def monitor(reference: pd.DataFrame, stream: pd.DataFrame, feature_cols: list[str], numeric_cols: list[str],
            ref_auc: float, approved_use: str, thresholds: dict | None = None,
            score_edges: list[float] | None = None) -> dict:
    """``reference``/``stream`` need columns: month (YYYY-MM), pd, label (NaN if immature) and the features.

    Score drift is measured against ``score_edges`` (the champion's current decile cut-offs, i.e. deciles of the
    current rank reference population), so the expected share is 10% per decile by construction.
    """
    th = thresholds or load_yaml(THRESHOLDS)
    score_edges = np.asarray(score_edges if score_edges is not None
                             else np.quantile(reference["pd"], np.linspace(0.1, 0.9, 9)), dtype="float64")
    ref_score = np.array([0.1] * 10 + [0.0])  # ten deciles + empty missing-score bin
    feat_ref = {}
    for c in feature_cols:
        if c in numeric_cols:
            edges = numeric_bins(reference[c])
            feat_ref[c] = ("num", edges, shares_numeric(reference[c], edges))
        else:
            levels = sorted(reference[c].astype(str).value_counts().index[:20])
            feat_ref[c] = ("cat", levels, shares_categorical(reference[c], levels))
    ref_missing = {c: float(reference[c].isna().mean()) for c in numeric_cols}

    months = []
    for m, g in stream.groupby("month", sort=True):
        row = {"month": m, "volume": int(len(g)),
               "score_psi": psi(ref_score, shares_numeric(g["pd"], score_edges)),
               "share_top_reference_decile": float((g["pd"] > score_edges[-1]).mean()),
               "mean_pd": float(g["pd"].mean())}
        fp = {}
        for c, (kind, spec, ref) in feat_ref.items():
            cur = shares_numeric(g[c], spec) if kind == "num" else shares_categorical(g[c], spec)
            fp[c] = psi(ref, cur)
        row["feature_psi"] = fp
        row["missing_rate_shift"] = {c: float(g[c].isna().mean() - ref_missing[c]) for c in numeric_cols}
        lab = g["label"].dropna()
        row["matured_loans"] = int(len(lab))
        months.append(row)

    # delayed outcomes by quarter (matured only)
    s = stream.dropna(subset=["label"]).copy()
    s["quarter"] = pd.PeriodIndex(s["month"], freq="M").asfreq("Q").astype(str)
    quarters = []
    for q, g in s.groupby("quarter", sort=True):
        y = g["label"].to_numpy(dtype=int)
        quarters.append({"quarter": q, "loans": int(len(g)), "events": int(y.sum()),
                         "observed_rate": float(y.mean()), "mean_pd": float(g["pd"].mean()),
                         "observed_expected": float(y.mean() / g["pd"].mean()),
                         "roc_auc": float(roc_auc_score(y, g["pd"])) if 0 < y.sum() < len(y) else None})
    alerts = _alerts(months, quarters, th, ref_auc, approved_use)
    return {"thresholds": th, "reference": {"loans": int(len(reference)), "score_decile_edges": score_edges.tolist(),
                                            "roc_auc": ref_auc},
            "months": months, "quarters": quarters, "alerts": alerts}


def _alerts(months: list, quarters: list, th: dict, ref_auc: float, approved_use: str) -> list[dict]:
    raw = []  # (key, period, severity, metric value, message)
    for r in months:
        if r["score_psi"] >= th["psi_alert"]:
            raw.append(("score_drift", r["month"], "high", r["score_psi"], "Score distribution shifted vs reference"))
        elif r["score_psi"] >= th["psi_watch"]:
            raw.append(("score_drift", r["month"], "medium", r["score_psi"], "Score distribution moving"))
        for c, v in r["feature_psi"].items():
            if v >= th["psi_alert"]:
                raw.append((f"feature_drift:{c}", r["month"], "medium", v, f"Input '{c}' distribution shifted"))
        for c, v in r["missing_rate_shift"].items():
            if abs(v) >= th["missing_rate_shift_alert"]:
                raw.append((f"missing_rate:{c}", r["month"], "high", v, f"Missing rate of '{c}' changed"))
        if r["volume"] < th["min_monthly_volume"]:
            raw.append(("low_volume", r["month"], "low", r["volume"], "Unusually low scoring volume"))
    for q in quarters:
        if q["roc_auc"] is not None and q["roc_auc"] < ref_auc - th["auc_drop_alert"]:
            raw.append(("discrimination_drop", q["quarter"], "high", q["roc_auc"],
                        "Rank-ordering weaker than reference"))
        band = th["observed_expected_band"]
        if not band[0] <= q["observed_expected"] <= band[1]:
            sev = "high" if approved_use == "pd_level" else "info"
            raw.append(("calibration", q["quarter"], sev, q["observed_expected"],
                        "Observed/expected outside band" + ("" if sev == "high" else
                                                            " (informational: model approved for rank-ordering only)")))
    # de-duplicate: one alert per key, spanning first/last period, worst severity, peak value
    order = {"info": 0, "low": 1, "medium": 2, "high": 3}
    owners = {"score_drift": ("model owner", "RB-02 score drift: investigate, then decile refresh with reviewer"),
              "feature_drift": ("data owner", "RB-01 bad or shifted data: check source release and mapping"),
              "missing_rate": ("data owner", "RB-01 bad or shifted data: check ingestion quarantine"),
              "low_volume": ("operations", "RB-03 service or feed outage"),
              "discrimination_drop": ("model owner", "RB-04 performance deterioration: validation review"),
              "calibration": ("model owner", "RB-04 calibration: level overlay review (see F-001)")}
    alerts: dict[str, dict] = {}
    for key, period, sev, val, msg in raw:
        a = alerts.setdefault(key, {"alert_id": key, "severity": sev, "first_period": period, "last_period": period,
                                    "periods": 0, "peak_value": val, "message": msg})
        a["periods"] += 1
        a["last_period"] = period
        if order[sev] > order[a["severity"]]:
            a["severity"], a["message"] = sev, msg
        if abs(val) > abs(a["peak_value"]):
            a["peak_value"] = val
        a["owner"], a["action"] = owners[key.split(":")[0]]
    return sorted(alerts.values(), key=lambda a: (-order[a["severity"]], a["alert_id"]))


def run_monitoring(model_cfg, eval_cfg, model_id: str | None = None) -> dict:
    from crisk.features.snapshots import snapshot_path
    from crisk.models.registry import champion, load_bundle
    from crisk.models.train import target_label_columns
    from crisk.utils import utc_now, write_json

    ch = champion(model_cfg)
    model_id = model_id or ch["model_id"]
    bundle, meta = load_bundle(model_cfg, model_id)
    spec = bundle.feature_spec
    ref = pd.read_parquet(snapshot_path(model_cfg, meta["feature_set_id"])).query("split == 'calibration'").copy()
    stream = pd.read_parquet(snapshot_path(eval_cfg, meta["feature_set_id"])).query("split == 'test'").copy()
    col = target_label_columns(eval_cfg)[meta["target_id"]]
    for df in (ref, stream):
        df["pd"] = bundle.predict(df)
        df["month"] = pd.to_datetime(df["first_payment_date"]).dt.strftime("%Y-%m")
    stream["label"] = pd.to_numeric(stream[col], errors="coerce")
    ref_col = target_label_columns(model_cfg)[meta["target_id"]]
    ry = pd.to_numeric(ref[ref_col], errors="coerce")
    ref_auc = float(roc_auc_score(ry[ry.notna()].astype(int), ref.loc[ry.notna(), "pd"]))
    approved_use = (ch or {}).get("approved_use", "not_promoted") if ch and ch["model_id"] == model_id else \
        "not_promoted"
    edges = (ch or {}).get("risk_decile_cutoffs") if ch and ch["model_id"] == model_id else None
    result = monitor(ref, stream, spec.model_columns, spec.numeric_columns, ref_auc, approved_use,
                     score_edges=edges)
    computed = "model calibration split (computed)"
    result["reference"]["rank_reference"] = (ch or {}).get("risk_decile_reference") if edges else computed
    report = {"generated_at": utc_now(), "model_id": model_id, "approved_use": approved_use,
              "stream": f"historical replay of {eval_cfg.dataset_id} test cohorts", "eval_dataset": eval_cfg.dataset_id,
              "reference_population": f"{model_cfg.dataset_id} calibration split", **result}
    out = model_cfg.reports_dir / "monitoring" / "latest.json"
    write_json(out, report)
    write_json(model_cfg.reports_dir / "monitoring" / f"run_{report['generated_at'][:19].replace(':', '')}.json",
               report)
    return report
