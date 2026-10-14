"""LGD/EAD development (pre-registered selection), locked test, and artefact registry.

Training data: full-volume 2015 vintage liquidations (freddie_mac_historical). Locked test: provider-sample vintages
2016-2021 (freddie_mac_sample). Plan: docs/lgd_el_plan_v1.md.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from crisk.audit.store import AuditStore
from crisk.config import DatasetConfig
from crisk.lgd.data import default_resolution, liquidations
from crisk.lgd.models import SeverityModel, eltv_link, severity_metrics
from crisk.utils import code_version, read_json, sha256_file, utc_now, write_json

ORDER = ("S0", "S1", "S2")
MIN_REL_GAIN = 0.03
TRAIN_VINTAGES = ("2015",)
TEST_VINTAGES = ("2016", "2017", "2018", "2019", "2020", "2021")


def develop(train_cfg: DatasetConfig, store: AuditStore) -> dict:
    liq = liquidations(train_cfg)
    liq = liq[liq.vintage.isin(TRAIN_VINTAGES)].copy()
    if not liq.reconciles.all():
        raise RuntimeError(f"{(~liq.reconciles).sum()} liquidations fail the loss reconciliation")
    dev_tr, dev_va = liq[liq.liquidation_year <= 2019], liq[liq.liquidation_year >= 2020]
    dev = {}
    for k in ORDER:
        m = SeverityModel(k).fit(dev_tr)
        dev[k] = severity_metrics(dev_va.severity, m.predict(dev_va), dev_va.removal_upb)
    selected, trail = "S0", []
    for k in ORDER[1:]:
        gain = (dev[selected]["mae"] - dev[k]["mae"]) / dev[selected]["mae"]
        trail.append({"candidate": k, "vs": selected, "relative_mae_gain": gain, "adopted": gain >= MIN_REL_GAIN})
        if gain >= MIN_REL_GAIN:
            selected = k
    final = SeverityModel(selected).fit(liq)
    res = default_resolution(train_cfg, TRAIN_VINTAGES)
    counts = res.resolution.value_counts().to_dict()
    p_liq = counts.get("liquidated_with_loss", 0) / len(res)
    ead_ratio = float(res.ead_ratio.clip(0, 1.5).mean())
    link = eltv_link(liq)
    lgd_id = f"lgd-{selected.lower()}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = train_cfg.artifacts_dir / "lgd" / lgd_id
    out.mkdir(parents=True)
    joblib.dump(final, out / "severity_model.joblib")
    meta = {"lgd_id": lgd_id, "created_at": utc_now(), "plan": "docs/lgd_el_plan_v1.md", "selected": selected,
            "selection_trail": trail, "dev_validation_metrics": dev,
            "dev_split": {"train": "liquidated <= 2019", "validation": "liquidated >= 2020",
                          "n_train": len(dev_tr), "n_validation": len(dev_va)},
            "final_training": {"vintages": TRAIN_VINTAGES, "n": len(liq),
                               "mean_severity": float(liq.severity.mean()),
                               "median_workout_months": float(liq.workout_months.median())},
            "p_liquidation_given_default": p_liq, "resolution_counts": counts, "n_defaults": len(res),
            "ead_ratio_mean": ead_ratio,
            "ead_ratio_quantiles": {str(k): float(v) for k, v in res.ead_ratio.quantile([.1, .5, .9]).items()},
            "eltv_link": link, "artifact_sha256": sha256_file(out / "severity_model.joblib"),
            "code_version": code_version(), "test_evaluated": False}
    write_json(out / "metadata.json", meta)
    store.append_event("modeller", "lgd_developed", "lgd", lgd_id, {"selected": selected, "p_liq": p_liq})
    return meta


def load_lgd(train_cfg: DatasetConfig, lgd_id: str) -> tuple[SeverityModel, dict]:
    d = train_cfg.artifacts_dir / "lgd" / lgd_id
    meta = read_json(d / "metadata.json")
    if sha256_file(d / "severity_model.joblib") != meta["artifact_sha256"]:
        raise RuntimeError("LGD artefact hash mismatch")
    return joblib.load(d / "severity_model.joblib"), meta


def test_once(train_cfg: DatasetConfig, test_cfg: DatasetConfig, lgd_id: str, store: AuditStore) -> dict:
    model, meta = load_lgd(train_cfg, lgd_id)
    prior = store.events(action="lgd_test_evaluated", entity_id=lgd_id)
    liq = liquidations(test_cfg)
    test = liq[liq.vintage.isin(TEST_VINTAGES)]
    base = np.full(len(test), meta["final_training"]["mean_severity"])
    res = {"lgd_id": lgd_id, "evaluated_at": utc_now(), "prior_evaluations": len(prior),
           "test_population": f"{test_cfg.dataset_id} vintages {TEST_VINTAGES}",
           "selected": severity_metrics(test.severity, model.predict(test), test.removal_upb),
           "baseline_S0": severity_metrics(test.severity, base, test.removal_upb),
           "by_disposition": {c: severity_metrics(g.severity, model.predict(g), g.removal_upb)
                              for c, g in test.groupby("zero_balance_code") if len(g) >= 20}}
    diag = liquidations(train_cfg)
    diag = diag[diag.vintage == "2022"]
    if len(diag):  # diagnostic only: early, fast resolutions of the 2022 vintage
        res["diagnostic_2022_full_volume"] = severity_metrics(diag.severity, model.predict(diag), diag.removal_upb)
    write_json(train_cfg.reports_dir / "lgd" / lgd_id / "test.json", res)
    store.append_event("validator", "lgd_test_evaluated", "lgd", lgd_id,
                       {"n": res["selected"]["n"], "mae": res["selected"]["mae"], "prior": len(prior)})
    return res


def overlay_from_reference(model_cfg: DatasetConfig, eval_cfg: DatasetConfig, fpd_from: str, fpd_to: str) -> dict:
    """PD level overlay: observed 12-month default rate per current champion decile on the reference applications."""
    from crisk.features.snapshots import snapshot_path
    from crisk.models.registry import champion, load_bundle
    from crisk.models.train import target_label_columns

    ch = champion(model_cfg)
    bundle, meta = load_bundle(model_cfg, ch["model_id"])
    snap = pd.read_parquet(snapshot_path(eval_cfg, meta["feature_set_id"]))
    ym = pd.to_datetime(snap.first_payment_date).dt.strftime("%Y-%m")
    col = target_label_columns(eval_cfg)[meta["target_id"]]
    ref = snap[(ym >= fpd_from) & (ym <= fpd_to) & snap[col].notna()]
    dec = np.searchsorted(np.asarray(ch["risk_decile_cutoffs"]), bundle.predict(ref), side="right") + 1
    rates = pd.Series(ref[col].astype(float).to_numpy()).groupby(dec).agg(["mean", "size", "sum"])
    return {"model_id": ch["model_id"], "reference": f"{eval_cfg.dataset_id} FPD {fpd_from}..{fpd_to}",
            "decile_default_rate": {int(k): float(v) for k, v in rates["mean"].items()},
            "decile_loans": {int(k): int(v) for k, v in rates["size"].items()},
            "decile_defaults": {int(k): int(v) for k, v in rates["sum"].items()}}
