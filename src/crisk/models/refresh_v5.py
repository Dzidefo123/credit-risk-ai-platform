"""Plan v5 (docs/evaluation_plan_v5.md): compare the control C0 and the refit candidates C1/C2 on the same loans.

Two stages, both audited in the refresh dataset's audit store:

- ``selection``: development set (sample 2023, already-examined period). Applies the pre-registered rule: default
  to C2, and choose C1 only if C1's AUC exceeds C2's by >= 0.005 with a paired-bootstrap 95% CI lower bound > 0.
- ``confirmatory``: unseen Release 48 2025 vintage, run once for the selected candidate. Pass requires: AUC CI
  lower bound >= 0.70; paired delta AUC (candidate - C0) >= -0.01; observed/expected in [0.8, 1.25].

Every model is scored with its own feature set on the evaluation dataset's ``test`` split. Rows are aligned on
``loan_id``, so all models are compared on identical loans and labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crisk.audit.store import AuditStore
from crisk.config import DatasetConfig
from crisk.evaluation.metrics import binary_metrics, bootstrap_ci
from crisk.features.snapshots import snapshot_path
from crisk.models.registry import load_bundle
from crisk.models.train import target_label_columns
from crisk.utils import utc_now, write_json

SELECTION_MARGIN = 0.005
CONFIRM = {"auc_ci_low_min": 0.70, "delta_auc_vs_c0_min": -0.01, "oe_band": (0.8, 1.25)}
CREDIT_BANDS = [-np.inf, 660, 700, 740, 780, np.inf]
RATE_BANDS = [-np.inf, 4.0, 5.0, 6.0, 7.0, np.inf]


def paired_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 20261014) -> dict:
    """AUC(a) - AUC(b) with a loan-level paired bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)), "ci_low": float(lo), "ci_high": float(hi)}


def _band_oe(frame: pd.DataFrame, col: str, bins: list, p: np.ndarray) -> list[dict]:
    b = pd.cut(frame[col].astype("float64"), bins)
    out = []
    for k, g in pd.DataFrame({"b": b, "y": frame["label"].to_numpy(), "p": p}).groupby("b", observed=True):
        out.append({"band": str(k), "loans": int(len(g)), "events": int(g.y.sum()), "observed_rate": float(g.y.mean()),
                    "mean_pd": float(g.p.mean()), "observed_expected": float(g.y.mean() / g.p.mean())})
    return out


def score_models(models: dict[str, tuple[DatasetConfig, str]], data_cfg: DatasetConfig,
                 target_id: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return a frame with loan_id, label, credit_score, orig_interest_rate and one PD column per model."""
    frames, metas = {}, {}
    for name, (mcfg, model_id) in models.items():
        bundle, meta = load_bundle(mcfg, model_id)
        tid = target_id or meta["target_id"]
        snap = pd.read_parquet(snapshot_path(data_cfg, meta["feature_set_id"]))
        test = snap[snap.split == "test"]
        col = target_label_columns(data_cfg)[tid]
        test = test[test[col].notna()].copy()
        test["label"] = test[col].astype("int64")
        frames[name] = pd.DataFrame({"loan_id": test.loan_id.to_numpy(), "label": test.label.to_numpy(),
                                     "p": bundle.predict(test)}).set_index("loan_id")
        metas[name] = {"model_id": model_id, "feature_set_id": meta["feature_set_id"], "target_id": tid,
                       "training_dataset": mcfg.dataset_id}
        if "orig" not in frames:
            raw = pd.read_parquet(data_cfg.processed_dir / "loan_origination.parquet",
                                  columns=["loan_id", "credit_score", "orig_interest_rate"])
            frames["orig"] = raw.set_index("loan_id")
    names = list(models)
    base = frames[names[0]][["label"]]
    for n in names:
        if not frames[n]["label"].reindex(base.index).equals(base["label"]):
            raise RuntimeError(f"labels or loans differ for {n}: feature snapshots are not aligned")
    out = base.join([frames[n]["p"].rename(n) for n in names]).join(frames["orig"], how="left")
    return out.reset_index(), metas


def compare(models: dict[str, tuple[DatasetConfig, str]], data_cfg: DatasetConfig, stage: str, store: AuditStore,
            report_cfg: DatasetConfig, actor: str, target_id: str | None = None, candidate: str | None = None) -> dict:
    if stage not in ("selection", "confirmatory"):
        raise ValueError(stage)
    action = f"v5_{stage}_evaluated" + (f"__{target_id}" if target_id else "")
    if stage == "confirmatory" and target_id is None and store.events(action=action):
        raise RuntimeError("v5 confirmatory test already run (plan v5: once; the Release 49 repeat is a new action)")
    df, metas = score_models(models, data_cfg, target_id)
    y = df["label"].to_numpy()
    rep: dict = {"generated_at": utc_now(), "plan": "docs/evaluation_plan_v5.md", "stage": stage,
                 "evaluation_dataset": data_cfg.dataset_id, "target_id": target_id or "primary",
                 "fpd_note": "test split of the evaluation dataset", "loans": int(len(df)), "events": int(y.sum()),
                 "models": metas, "metrics": {}, "ci": {}, "credit_score_bands": {}, "note_rate_bands": {}}
    for n in models:
        p = df[n].to_numpy()
        rep["metrics"][n] = binary_metrics(y, p)
        rep["ci"][n] = bootstrap_ci(y, p, n_boot=500, seed=20261014)
        rep["credit_score_bands"][n] = _band_oe(df, "credit_score", CREDIT_BANDS, p)
        rep["note_rate_bands"][n] = _band_oe(df, "orig_interest_rate", RATE_BANDS, p)
    rep["paired"] = {f"{a}-{b}": paired_delta(y, df[a].to_numpy(), df[b].to_numpy())
                     for a, b in (("C1", "C0"), ("C2", "C0"), ("C1", "C2")) if a in models and b in models}
    if stage == "selection" and target_id is None:
        d = rep["paired"]["C1-C2"]
        chosen = "C1" if d["delta_auc"] >= SELECTION_MARGIN and d["ci_low"] > 0 else "C2"
        rep["selection"] = {"rule": "default C2; C1 only if AUC(C1)-AUC(C2) >= 0.005 and CI low > 0",
                            "chosen": chosen, "model_id": metas[chosen]["model_id"]}
    if stage == "confirmatory" and target_id is None:
        c = candidate or next(n for n in models if n != "C0")
        auc_low = rep["ci"][c]["roc_auc"]["low"]
        dv = rep["paired"][f"{c}-C0"]["delta_auc"]
        oe = rep["metrics"][c]["observed_expected_ratio"]
        lo, hi = CONFIRM["oe_band"]
        checks = {"discrimination": auc_low >= CONFIRM["auc_ci_low_min"],
                  "no_ranking_loss_vs_c0": dv >= CONFIRM["delta_auc_vs_c0_min"],
                  "level": lo <= oe <= hi}
        rep["confirmatory"] = {"candidate": c, "criteria": {k: (list(v) if isinstance(v, tuple) else v)
                                                            for k, v in CONFIRM.items()},
                               "values": {"auc_ci_low": auc_low, "delta_auc_vs_c0": dv, "observed_expected": oe},
                               "checks": checks, "pass": all(checks.values()),
                               "consequence": ("eligible for validation; pd_level possible if V07 passes"
                                               if all(checks.values()) else
                                               "rank-ordering only" if checks["discrimination"] and
                                               checks["no_ranking_loss_vs_c0"] else "C0 stays")}
    store.append_event(actor=actor, action=action, entity_type="v5_evaluation", entity_id=data_cfg.dataset_id,
                       details={"models": {n: m["model_id"] for n, m in metas.items()},
                                "selection": rep.get("selection"), "confirmatory": rep.get("confirmatory")})
    write_json(report_cfg.reports_dir / "v5" / f"{stage}__{data_cfg.dataset_id}__{target_id or 'primary'}.json", rep)
    return rep
