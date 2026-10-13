"""Train the prevalence baseline and the regularised logistic PD model; evaluate on locked test."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from crisk.audit.store import AuditStore
from crisk.config import DatasetConfig, load_yaml
from crisk.evaluation.metrics import binary_metrics, bootstrap_ci
from crisk.features.origination import FeatureSpec
from crisk.features.snapshots import snapshot_path, split_manifest_path
from crisk.labels.default_horizon import TargetSpec
from crisk.models.bundle import ModelBundle
from crisk.models.registry import load_bundle, load_metadata, save_bundle
from crisk.utils import code_version, environment_info, read_json, sha256_file, utc_now, write_json

EVIDENCE_NOTE = {
    "SYNTHETIC": "SYNTHETIC fixture data: metrics test the pipeline only and are not evidence about credit risk.",
    "EMPIRICAL": "Empirical research result on Freddie Mac SFLLD; funded-loan population only.",
}


def _load_snapshots(cfg: DatasetConfig, feature_set_id: str) -> pd.DataFrame:
    return pd.read_parquet(snapshot_path(cfg, feature_set_id))


def target_label_columns(cfg: DatasetConfig) -> dict[str, str]:
    """target_id -> snapshot label column (primary target uses ``label``)."""
    cols = {TargetSpec.load(cfg.target_config).target_id: "label"}
    for path in cfg.sensitivity_target_configs:
        t = TargetSpec.load(path).target_id
        cols[t] = f"label__{t}"
    return cols


def resolve_target(cfg: DatasetConfig, target_id: str | None) -> tuple[TargetSpec, str]:
    specs = [TargetSpec.load(cfg.target_config)] + [TargetSpec.load(p) for p in cfg.sensitivity_target_configs]
    spec = specs[0] if target_id is None else next((t for t in specs if t.target_id == target_id), None)
    if spec is None:
        raise KeyError(f"Unknown target {target_id!r}; configured: {[t.target_id for t in specs]}")
    return spec, target_label_columns(cfg)[spec.target_id]


def _with_label(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rows labelled under the chosen target, with that target copied into ``label``."""
    out = df[df[col].notna()].copy()
    out["label"] = out[col].astype("int64")
    return out


def train_model(cfg: DatasetConfig, store: AuditStore, actor: str = "modeller",
                target_id: str | None = None, feature_config: str | None = None,
                model_config: str | None = None) -> dict:
    mcfg = load_yaml(model_config or cfg.model_config)
    fspec = FeatureSpec.load(feature_config or cfg.feature_config)
    tspec, label_col = resolve_target(cfg, target_id)
    snap = _with_label(_load_snapshots(cfg, fspec.feature_set_id), label_col)
    split_manifest = read_json(split_manifest_path(cfg, fspec.feature_set_id))
    macro = None
    if fspec.needs_macro:
        from crisk.macro.pmms import MarketRateTable

        macro = MarketRateTable.from_fred_csv()
        if split_manifest.get("macro", {}).get("source_sha256") != macro.source_sha256:
            raise RuntimeError("Market-rate files changed since the feature snapshot was built; rebuild features")
    train, cal = snap[snap.split == "train"], snap[snap.split == "calibration"]
    if train.label.nunique() < 2:
        raise ValueError("Training split needs both classes")
    if cal.empty or cal.label.nunique() < 2:
        raise ValueError("Calibration split needs both classes")

    seed = int(mcfg["seed"])
    np.random.seed(seed)
    min_freq = float(mcfg["preprocessing"]["categorical_min_frequency"])
    fit_info: dict = {}
    if mcfg["family"] == "logistic_regression":
        lg = mcfg["logistic"]
        bundle = ModelBundle.fit_logistic(
            train, train.label.to_numpy(), fspec, C=float(lg["C"]), max_iter=int(lg["max_iter"]),
            min_frequency=min_freq, seed=seed, class_weight=lg.get("class_weight"), macro=macro)
    elif mcfg["family"] == "lightgbm":
        lgc = mcfg["lightgbm"]
        # temporal early-stopping slice: the last N months of the training window
        fpd = pd.to_datetime(train.first_payment_date)
        es_from = (fpd.max() - pd.DateOffset(months=int(lgc["early_stopping_last_months"]) - 1)).strftime("%Y-%m")
        es_mask = (fpd.dt.strftime("%Y-%m") >= es_from).to_numpy()
        bundle, fit_info = ModelBundle.fit_lightgbm(
            train, train.label.to_numpy(), fspec, params=dict(lgc["params"]),
            monotone={k: int(v) for k, v in (lgc.get("monotone_constraints") or {}).items()},
            min_frequency=min_freq, seed=seed, early_stopping_mask=es_mask,
            early_stopping_rounds=int(lgc["early_stopping_rounds"]), macro=macro)
    else:
        raise ValueError(f"Unknown model family {mcfg['family']!r}")
    raw_cal_metrics = binary_metrics(cal.label.to_numpy(), bundle.predict_raw(cal))
    bundle.fit_platt(cal, cal.label.to_numpy())

    prevalence = float(train.label.mean())
    metrics = {
        "train": {"logistic_uncalibrated": binary_metrics(train.label.to_numpy(), bundle.predict_raw(train)),
                  "prevalence_baseline": binary_metrics(train.label.to_numpy(), np.full(len(train), prevalence))},
        "calibration": {
            "logistic_uncalibrated": raw_cal_metrics,
            # in-sample for the Platt step: reported for transparency, not as validation evidence
            "logistic_platt_in_sample": binary_metrics(cal.label.to_numpy(), bundle.predict(cal)),
            "prevalence_baseline": binary_metrics(cal.label.to_numpy(), np.full(len(cal), prevalence)),
        },
    }

    snap_file = snapshot_path(cfg, fspec.feature_set_id)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    tag = "" if label_col == "label" else "-sens"
    family_tag = {"logistic_regression": "logit", "lightgbm": "lgbm"}[mcfg["family"]]
    model_id = f"pd-{family_tag}-{fspec.feature_set_id}{tag}-{stamp}-{uuid.uuid4().hex[:6]}"
    metadata = {
        "created_at": utc_now(),
        "dataset_id": cfg.dataset_id,
        "data_origin": cfg.data_origin,
        "evidence_note": EVIDENCE_NOTE[cfg.data_origin],
        "status": "candidate",
        "model_family": bundle.model_family,
        "model_config": mcfg,
        "seed": seed,
        "feature_set_id": fspec.feature_set_id,
        "feature_columns": fspec.model_columns,
        "target_id": tspec.target_id,
        "label_column": label_col,
        "horizon_months": tspec.horizon_months,
        "split_id": split_manifest["split_id"],
        "split_manifest": split_manifest["splits"],
        "fit_info": fit_info,
        "macro": macro.metadata() if macro is not None else None,
        "training_data": {"feature_snapshot_sha256": sha256_file(snap_file),
                          "train_rows": len(train), "calibration_rows": len(cal),
                          "train_events": int(train.label.sum()), "calibration_events": int(cal.label.sum()),
                          "train_prevalence": prevalence},
        "calibration": {"method": "platt", "a": bundle.platt_a, "b": bundle.platt_b},
        "coefficients": bundle.coefficients(),
        "metrics": metrics,
        "test_set_evaluated": False,
        "code_version": code_version(),
        "environment": environment_info(),
    }
    metadata = save_bundle(cfg, bundle, model_id, metadata)
    store.append_event(actor=actor, action="model_trained", entity_type="model", entity_id=model_id,
                       details={"artifact_sha256": metadata["artifact_sha256"], "data_origin": cfg.data_origin,
                                "feature_snapshot_sha256": metadata["training_data"]["feature_snapshot_sha256"]})
    return metadata


def evaluate_on_test(cfg: DatasetConfig, model_id: str, store: AuditStore, actor: str = "validator") -> dict:
    """Locked test evaluation. Every access is written to the audit chain so repeated peeking is visible."""
    bundle, meta = load_bundle(cfg, model_id)
    snap = _load_snapshots(cfg, meta["feature_set_id"])
    if sha256_file(snapshot_path(cfg, meta["feature_set_id"])) != meta["training_data"]["feature_snapshot_sha256"]:
        raise RuntimeError("Feature snapshot changed since training; rebuild/retrain before test evaluation")
    all_test = snap[snap.split == "test"]
    label_cols = target_label_columns(cfg)
    test = _with_label(all_test, meta.get("label_column", "label"))
    prior = store.events(action="test_set_evaluated", entity_id=model_id)
    y = test.label.to_numpy()
    prevalence_train = meta["training_data"].get("train_prevalence", meta["split_manifest"]["train"]["prevalence"])
    result = {
        "model_id": model_id,
        "data_origin": meta["data_origin"],
        "evidence_note": EVIDENCE_NOTE[meta["data_origin"]],
        "evaluated_at": utc_now(),
        "split": "test",
        "target_id": meta["target_id"],
        "prior_test_evaluations": len(prior),
        "logistic_calibrated": binary_metrics(y, bundle.predict(test)),
        "logistic_uncalibrated": binary_metrics(y, bundle.predict_raw(test)),
        "prevalence_baseline": binary_metrics(y, np.full(len(test), prevalence_train)),
        "by_fpd_year": {},
    }
    test = test.assign(pd_cal=bundle.predict(test), year=pd.to_datetime(test.first_payment_date).dt.year)
    result["bootstrap_ci_logistic_calibrated"] = bootstrap_ci(y, test.pd_cal.to_numpy(), seed=int(meta["seed"]))
    for yr, g in test.groupby("year"):
        m = binary_metrics(g.label.to_numpy(), g.pd_cal.to_numpy())
        m.pop("reliability", None)
        result["by_fpd_year"][str(yr)] = m
    # Same model scored against the other configured target definitions (sensitivity analysis).
    result["against_other_targets"] = {}
    for t, col in label_cols.items():
        if t == meta["target_id"]:
            continue
        other = _with_label(all_test, col)
        m = binary_metrics(other.label.to_numpy(), bundle.predict(other))
        m.pop("reliability", None)
        result["against_other_targets"][t] = m
    out = cfg.reports_dir / model_id / "evaluation_test.json"
    write_json(out, result)
    store.append_event(actor=actor, action="test_set_evaluated", entity_type="model", entity_id=model_id,
                       details={"n": len(test), "roc_auc": result["logistic_calibrated"].get("roc_auc"),
                                "prior_evaluations": len(prior)})
    return result


def model_summary(cfg: DatasetConfig, model_id: str) -> dict:
    return load_metadata(cfg, model_id)
