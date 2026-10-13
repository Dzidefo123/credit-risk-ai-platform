"""Walk-forward PD recalibration on recent mature cohorts (evaluation plan v3).

The ranking model is fixed. For each first-payment cohort month T, the PD *level* is refreshed using
only cohorts whose outcomes would have been known when loans in cohort T were scored: FPD in
[T - (lag + window - 1), T - lag] months. Procedures:

- R0: no recalibration, ``p = p0``
- R1: intercept update, ``p = sigmoid(a_T + logit p0)``, with a_T fitted by offset logistic regression
- R2: Platt refit, ``p = sigmoid(a_T + b_T * logit p0)``

Guard: a window with fewer than ``min_events`` events carries forward the last fitted parameters
(or R0 if none exist yet).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crisk.evaluation.metrics import binary_metrics, fit_logistic_1d, logit, sigmoid

METHODS = ("R0", "R1", "R2")
BAND = (0.8, 1.25)


@dataclass(frozen=True)
class RecalSpec:
    lag_months: int = 20
    window_months: int = 12
    min_events: int = 20


DEFAULT_SPEC = RecalSpec()  # pre-registered: 20-month lag, 12-month window, >= 20 events

def window_for(t: pd.Period, spec: RecalSpec) -> tuple[pd.Period, pd.Period]:
    return t - (spec.lag_months + spec.window_months - 1), t - spec.lag_months


def walk_forward(months: pd.Series, p0: np.ndarray, y: np.ndarray, eval_months: list[pd.Period],
                 spec: RecalSpec = DEFAULT_SPEC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-row predictions for eval cohorts, per-month fitted parameters).

    ``months`` holds each loan's FPD as a monthly ``pd.Period``; ``y`` may be NaN for unlabelled loans
    (they are never used for fitting or scoring metrics).
    """
    months = pd.Series(months).reset_index(drop=True)
    lp = logit(np.asarray(p0, dtype="float64"))
    y = np.asarray(y, dtype="float64")
    labelled = ~np.isnan(y)
    rows, params = [], []
    last = {"R1": (0.0, 1.0), "R2": (0.0, 1.0)}
    for t in eval_months:
        lo, hi = window_for(t, spec)
        w = ((months >= lo) & (months <= hi)).to_numpy() & labelled
        events = int(y[w].sum()) if w.any() else 0
        fitted = events >= spec.min_events
        if fitted:
            a1, _ = fit_logistic_1d(lp[w], y[w], offset=lp[w], fit_slope=False)
            a2, b2 = fit_logistic_1d(lp[w], y[w])
            last = {"R1": (a1, 1.0), "R2": (a2, b2)}
        params.append({"month": str(t), "window_start": str(lo), "window_end": str(hi), "window_loans": int(w.sum()),
                       "window_events": events, "refitted": fitted,
                       "R1_a": last["R1"][0], "R2_a": last["R2"][0], "R2_b": last["R2"][1]})
        idx = np.where(((months == t).to_numpy()) & labelled)[0]
        if len(idx) == 0:
            continue
        x = lp[idx]
        rows.append(pd.DataFrame({
            "row": idx, "month": str(t), "y": y[idx],
            "R0": sigmoid(x), "R1": sigmoid(last["R1"][0] + x), "R2": sigmoid(last["R2"][0] + last["R2"][1] * x),
        }))
    preds = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["row", "month", "y", *METHODS])
    return preds, pd.DataFrame(params)


def summarise(preds: pd.DataFrame, periods: dict[str, tuple[str, str]], band: tuple[float, float] = BAND) -> dict:
    """Metrics overall, per FPD year and per labelled period, for each method."""
    out: dict = {"overall": {}, "by_year": {}, "by_period": {}, "in_band_share": {}}
    preds = preds.assign(year=preds["month"].str[:4])
    for m in METHODS:
        met = binary_metrics(preds.y.to_numpy(), preds[m].to_numpy())
        met.pop("reliability", None)
        out["overall"][m] = met
        years = {}
        for yr, g in preds.groupby("year"):
            ym = binary_metrics(g.y.to_numpy(), g[m].to_numpy())
            ym.pop("reliability", None)
            years[yr] = ym
        out["by_year"][m] = years
        oe = [v["observed_expected_ratio"] for v in years.values()]
        out["in_band_share"][m] = float(np.mean([band[0] <= r <= band[1] for r in oe])) if oe else None
        out["by_period"][m] = {}
        for name, (start, end) in periods.items():
            g = preds[(preds.month >= start) & (preds.month <= end)]
            if len(g):
                pm = binary_metrics(g.y.to_numpy(), g[m].to_numpy())
                pm.pop("reliability", None)
                out["by_period"][m][name] = pm
    return out


def decide(summary_by_target: dict[str, dict], r2_min_rel_gain: float = 0.01) -> dict:
    """Pre-registered decision rule (evaluation_plan_v3.md)."""
    def ll(t, m):
        return summary_by_target[t]["overall"][m]["log_loss"]

    def band(t, m):
        return summary_by_target[t]["in_band_share"][m]

    targets = list(summary_by_target)
    r1 = all(ll(t, "R1") < ll(t, "R0") and band(t, "R1") > band(t, "R0") for t in targets)
    choice, reasons = "R0", []
    if r1:
        choice = "R1"
        reasons.append("R1 lowers log loss and raises in-band share vs R0 for all targets")
        r2 = all((ll(t, "R1") - ll(t, "R2")) / ll(t, "R1") >= r2_min_rel_gain and band(t, "R2") >= band(t, "R1")
                 for t in targets)
        if r2:
            choice = "R2"
            reasons.append(f"R2 lowers log loss vs R1 by >= {r2_min_rel_gain:.0%} without reducing in-band share")
    else:
        reasons.append("R1 did not beat R0 on both criteria for every target")
    return {"adopted": choice, "reasons": reasons}


PERIODS_V3 = {"P1_covid_never_evaluated": ("2019-04", "2021-12"),
              "P2_overlaps_viewed_test": ("2022-01", "2023-06"),
              "P3_viewed_holdout": ("2023-07", "2025-03")}
EVAL_RANGE_V3 = ("2019-04", "2025-03")


def run_backtest(model_cfg, model_ids: list[str], eval_cfg, store, spec: RecalSpec = DEFAULT_SPEC,
                 actor: str = "validator") -> dict:
    """Walk-forward backtest for each model (one per target); applies the pre-registered decision rule."""
    from crisk.features.snapshots import snapshot_path
    from crisk.models.registry import load_bundle
    from crisk.models.train import target_label_columns
    from crisk.utils import utc_now, write_json

    eval_months = list(pd.period_range(*EVAL_RANGE_V3, freq="M"))
    per_target, params_out = {}, {}
    label_cols = target_label_columns(eval_cfg)
    for mid in model_ids:
        bundle, meta = load_bundle(model_cfg, mid)
        snap = pd.read_parquet(snapshot_path(eval_cfg, meta["feature_set_id"]))
        sources = set(snap.source_id)
        if len(sources) != 1:
            raise ValueError(f"Backtest data must come from a single source, got {sources}")
        y = pd.to_numeric(snap[label_cols[meta["target_id"]]], errors="coerce").to_numpy(dtype="float64")
        months = pd.to_datetime(snap.first_payment_date).dt.to_period("M")
        preds, params = walk_forward(months, bundle.predict(snap), y, eval_months, spec)
        per_target[meta["target_id"]] = summarise(preds, PERIODS_V3)
        per_target[meta["target_id"]]["model_id"] = mid
        params_out[meta["target_id"]] = params.to_dict(orient="records")
    decision = decide({t: s for t, s in per_target.items()})
    report = {"plan": "docs/evaluation_plan_v3.md", "evaluated_at": utc_now(), "eval_dataset": eval_cfg.dataset_id,
              "data_origin": eval_cfg.data_origin, "spec": spec.__dict__, "eval_range": EVAL_RANGE_V3,
              "periods": PERIODS_V3, "results": per_target, "parameters_by_month": params_out, "decision": decision}
    out = model_cfg.reports_dir / "recalibration_v3" / f"backtest__{eval_cfg.dataset_id}.json"
    write_json(out, report)
    store.append_event(actor=actor, action="recalibration_backtest", entity_type="model", entity_id=",".join(model_ids),
                       details={"eval_dataset": eval_cfg.dataset_id, "decision": decision["adopted"]})
    return report
