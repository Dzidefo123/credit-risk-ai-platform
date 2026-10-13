"""Historical policy simulation (observational, illustrative economics).

Applies policy variants to historical, *funded and acquired* loans with mature outcomes. This is not a causal
estimate of what would have happened under a different policy: declined applicants and behavioural responses are not
observed, and the population is already selected by past underwriting. Economics use explicit scenario assumptions
(configs/policies/simulation_assumptions_v1.yaml): EAD = original UPB, constant LGD, a 12-month net margin and per-loan
costs. These are not fitted models and not accounting ECL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crisk.config import load_yaml
from crisk.policies.engine import Policy

POLICY_FIELDS = ("credit_score", "orig_dti", "orig_ltv", "occupancy_status", "orig_upb")


def _decisions(policy: Policy, frame: pd.DataFrame, approved_use: str) -> pd.Series:
    out = []
    for rec in frame.to_dict(orient="records"):
        app = {k: (None if pd.isna(rec.get(k)) else rec.get(k)) for k in POLICY_FIELDS}
        score = {"status": "scored", "risk_decile": int(rec["risk_decile"]), "pd": float(rec["pd"])}
        out.append(policy.evaluate(app, score, approved_use)["decision"])
    return pd.Series(out, index=frame.index)


def economics(frame: pd.DataFrame, approved: np.ndarray, lgd: float, margin: float, a: dict,
              referred: np.ndarray | None = None) -> dict:
    sel = approved if referred is None else (approved | referred)
    upb, y = frame["orig_upb"].to_numpy(dtype="float64"), frame["label"].to_numpy(dtype="float64")
    revenue = float((upb[sel] * margin).sum())
    losses = float((upb[sel] * y[sel] * lgd).sum())
    costs = float(sel.sum() * a["operating_cost_per_loan_usd"]
                  + (referred.sum() if referred is not None else 0) * a["manual_review_cost_per_loan_usd"])
    return {"funded_loans": int(sel.sum()), "exposure_usd": float(upb[sel].sum()), "net_margin_usd": revenue,
            "credit_losses_usd": losses, "costs_usd": costs, "contribution_usd": revenue - losses - costs,
            "contribution_bps_of_exposure": 1e4 * (revenue - losses - costs) / max(float(upb[sel].sum()), 1.0)}


def simulate(frame: pd.DataFrame, policies: list[Policy], approved_use: str, assumptions: dict) -> dict:
    """``frame`` needs POLICY_FIELDS, label (0/1), pd and risk_decile."""
    a = assumptions
    results = {}
    y = frame["label"].to_numpy()
    for pol in policies:
        dec = _decisions(pol, frame, approved_use)
        app, ref, dcl = (dec == "approve").to_numpy(), (dec == "refer").to_numpy(), (dec == "decline").to_numpy()

        def rate(m):
            return float(y[m].mean()) if m.any() else None

        results[pol.version] = {
            "policy_sha256": pol.sha256, "n": len(frame),
            "approve_rate": float(app.mean()), "refer_rate": float(ref.mean()), "decline_rate": float(dcl.mean()),
            "default_rate_approved": rate(app), "default_rate_referred": rate(ref), "default_rate_declined": rate(dcl),
            "defaults_in_declined": int(y[dcl].sum()), "non_defaults_in_declined": int((1 - y[dcl]).sum()),
            "economics_base_referred_declined": economics(frame, app, a["lgd_base"], a["net_margin_base"], a),
            "economics_base_referred_approved": economics(frame, app, a["lgd_base"], a["net_margin_base"], a,
                                                          referred=ref),
            "sensitivity_contribution_usd": {
                f"lgd={lgd}|margin={mg}": economics(frame, app, lgd, mg, a)["contribution_usd"]
                for lgd in a["lgd_sensitivity"] for mg in a["net_margin_sensitivity"]},
        }
    return results


def run_simulation(model_cfg, model_id: str, eval_cfg, policy_paths: list[str],
                   assumptions_path: str = "configs/policies/simulation_assumptions_v1.yaml") -> dict:
    import duckdb

    from crisk.features.snapshots import snapshot_path
    from crisk.models.registry import champion, load_bundle, reference_deciles
    from crisk.models.train import target_label_columns
    from crisk.utils import utc_now, write_json

    bundle, meta = load_bundle(model_cfg, model_id)
    ch = champion(model_cfg)
    if ch and ch["model_id"] == model_id and ch.get("risk_decile_cutoffs"):
        cutoffs, approved_use = ch["risk_decile_cutoffs"], ch.get("approved_use", "demonstration_rank_ordering")
    else:  # not the champion: same reference population as promotion would use
        ref = pd.read_parquet(snapshot_path(model_cfg, meta["feature_set_id"])).query("split == 'calibration'")
        cutoffs, approved_use = reference_deciles(bundle, ref), "demonstration_rank_ordering"
    snap = pd.read_parquet(snapshot_path(eval_cfg, meta["feature_set_id"]))
    col = target_label_columns(eval_cfg)[meta["target_id"]]
    snap = snap[(snap.split == "test") & snap[col].notna()].copy()
    snap["label"] = snap[col].astype(int)
    snap["pd"] = bundle.predict(snap)
    snap["risk_decile"] = np.searchsorted(np.asarray(cutoffs), snap["pd"].to_numpy(), side="right") + 1
    with duckdb.connect() as con:
        con.register("ids", snap[["loan_id"]])
        src = (eval_cfg.processed_dir / "loan_origination.parquet").as_posix()
        cols = ", ".join("o." + f for f in POLICY_FIELDS)
        raw = con.execute(f"SELECT o.loan_id, {cols} FROM read_parquet('{src}') o JOIN ids USING (loan_id)").df()
    frame = snap[["loan_id", "label", "pd", "risk_decile", "first_payment_date"]].merge(raw, on="loan_id")
    assumptions = load_yaml(assumptions_path)
    policies = [Policy.load(p) for p in policy_paths]
    results = simulate(frame, policies, approved_use, assumptions)
    report = {"generated_at": utc_now(), "model_id": model_id, "target_id": meta["target_id"],
              "eval_dataset": eval_cfg.dataset_id, "population": "funded, Freddie Mac-acquired loans with mature "
              "12-month outcomes (test split of the evaluation dataset)", "n_loans": len(frame),
              "fpd_range": [str(frame.first_payment_date.min())[:10], str(frame.first_payment_date.max())[:10]],
              "approved_use": approved_use, "decile_cutoffs": cutoffs, "assumptions": assumptions,
              "results": results,
              "caveats": ["Observational: declined loans' outcomes are observed only because these loans were funded "
                          "historically; a real policy change alters who applies and is funded.",
                          "Economics are scenario arithmetic (constant LGD, EAD = original UPB, assumed margin and "
                          "costs), not fitted LGD/EAD models or accounting ECL.",
                          "FICTIONAL demonstration policies."]}
    write_json(model_cfg.reports_dir / "policy_v1" / f"simulation__{eval_cfg.dataset_id}__{model_id}.json", report)
    return report
