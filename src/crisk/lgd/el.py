"""12-month expected loss at origination with scenarios (illustrative; plan docs/lgd_el_plan_v1.md).

EL_i = PD_overlay(decile_i) x P(liquidation | default) x Severity_i x EAD_i
- PD_overlay: observed 12-month default rate per current champion decile on the decile-reference applications
  (the champion is approved for ranking only, so the level comes from a documented overlay).
- Severity_i: selected severity model (origination features).
- EAD_i: original UPB x mean EAD ratio at first 90+ (fixed-rate amortising loans stop amortising when delinquent).
Scenario stress: PD multiplier; P(liquidation) shift; house-price shock applied through the ELTV link
(orig LTV used as the proxy for LTV at default: delta logit = slope x LTV x (1/(1 - shock) - 1) / 100).
Not CECL/IFRS 9: single period, no lifetime PD, no discounting, no staging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = {
    "base": {"pd_multiplier": 1.0, "p_liq_shift": 0.0, "house_price_shock": 0.0},
    "adverse": {"pd_multiplier": 2.0, "p_liq_shift": 0.10, "house_price_shock": 0.10},
    "severe": {"pd_multiplier": 4.0, "p_liq_shift": 0.20, "house_price_shock": 0.20},
}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stressed_severity(sev: np.ndarray, ltv: np.ndarray, slope: float, shock: float) -> np.ndarray:
    if shock == 0:
        return sev
    delta = slope * np.nan_to_num(ltv, nan=np.nanmedian(ltv)) * (1 / (1 - shock) - 1) / 100.0
    return 1 / (1 + np.exp(-(_logit(sev) + delta)))


def expected_loss(frame: pd.DataFrame, overlay: dict[int, float], p_liq: float, ead_ratio: float, eltv_slope: float,
                  scenarios: dict = SCENARIOS) -> dict:
    """``frame`` needs: risk_decile, severity, orig_upb, orig_ltv. Returns portfolio and per-decile EL by scenario."""
    pd_base = frame["risk_decile"].map(overlay).to_numpy(dtype="float64")
    ead = frame["orig_upb"].to_numpy(dtype="float64") * ead_ratio
    out = {}
    for name, s in scenarios.items():
        pdv = np.clip(pd_base * s["pd_multiplier"], 0, 1)
        pl = min(max(p_liq + s["p_liq_shift"], 0.0), 1.0)
        sev = stressed_severity(frame["severity"].to_numpy(dtype="float64"),
                                frame["orig_ltv"].to_numpy(dtype="float64"), eltv_slope, s["house_price_shock"])
        el = pdv * pl * sev * ead
        by_dec = (pd.DataFrame({"d": frame["risk_decile"].to_numpy(), "el": el, "ead": ead})
                  .groupby("d").agg(el=("el", "sum"), ead=("ead", "sum"), loans=("el", "size")))
        out[name] = {"assumptions": {**s, "p_liquidation": pl}, "loans": int(len(frame)),
                     "exposure_usd": float(ead.sum()), "expected_loss_usd": float(el.sum()),
                     "el_rate_bps": float(1e4 * el.sum() / ead.sum()), "mean_pd": float(pdv.mean()),
                     "mean_severity": float(sev.mean()),
                     "by_decile": {int(d): {"loans": int(r.loans), "el_usd": float(r.el),
                                            "el_rate_bps": float(1e4 * r.el / r.ead)} for d, r in by_dec.iterrows()}}
    return out


def run_el(model_cfg, eval_cfg, lgd_cfg, lgd_id: str, overlay_window: tuple[str, str],
           portfolio_window: tuple[str, str]) -> dict:
    import duckdb

    from crisk.features.snapshots import snapshot_path
    from crisk.lgd.workflow import load_lgd, overlay_from_reference
    from crisk.models.registry import champion, load_bundle
    from crisk.utils import utc_now, write_json

    sev_model, lmeta = load_lgd(lgd_cfg, lgd_id)
    overlay = overlay_from_reference(model_cfg, eval_cfg, *overlay_window)
    ch = champion(model_cfg)
    bundle, meta = load_bundle(model_cfg, ch["model_id"])
    snap = pd.read_parquet(snapshot_path(eval_cfg, meta["feature_set_id"]))
    ym = pd.to_datetime(snap.first_payment_date).dt.strftime("%Y-%m")
    book = snap[(ym >= portfolio_window[0]) & (ym <= portfolio_window[1])].copy()
    book["risk_decile"] = np.searchsorted(np.asarray(ch["risk_decile_cutoffs"]), bundle.predict(book),
                                          side="right") + 1
    with duckdb.connect() as con:
        con.register("ids", book[["loan_id"]])
        src = (eval_cfg.processed_dir / "loan_origination.parquet").as_posix()
        raw = con.execute(f"SELECT o.* FROM read_parquet('{src}') o JOIN ids USING (loan_id)").df()
    book = book[["loan_id", "risk_decile"]].merge(raw, on="loan_id")
    book["severity"] = sev_model.predict(book)
    res = expected_loss(book, overlay["decile_default_rate"], lmeta["p_liquidation_given_default"],
                        lmeta["ead_ratio_mean"], lmeta["eltv_link"]["slope_per_unit_eltv"])
    report = {"generated_at": utc_now(), "plan": "docs/lgd_el_plan_v1.md",
              "label": "ILLUSTRATIVE 12-month expected loss at origination; not CECL/IFRS 9",
              "pd_model": ch["model_id"], "pd_overlay": overlay, "lgd_id": lgd_id,
              "p_liquidation_given_default": lmeta["p_liquidation_given_default"],
              "ead_ratio": lmeta["ead_ratio_mean"], "eltv_link": lmeta["eltv_link"],
              "portfolio": f"{eval_cfg.dataset_id} applications FPD {portfolio_window[0]}..{portfolio_window[1]}",
              "scenarios": res}
    write_json(model_cfg.reports_dir / "el_v1" / f"el__{lgd_id}.json", report)
    return report
