"""Build ``feature_snapshots``: one row per labelled loan with features, label and split."""

from __future__ import annotations

import duckdb
import pandas as pd

from crisk.config import DatasetConfig
from crisk.features.origination import FeatureSpec, build_feature_frame
from crisk.features.splits import SplitSpec
from crisk.labels.default_horizon import TargetSpec, outcomes_path
from crisk.utils import sha256_file, sha256_text, utc_now, write_json

ID_COLUMNS = ["loan_id", "first_payment_date", "split", "label", "outcome", "data_origin"]


def build_feature_snapshots(cfg: DatasetConfig) -> dict:
    fspec = FeatureSpec.load(cfg.feature_config)
    tspec = TargetSpec.load(cfg.target_config)
    sspec = SplitSpec.load(cfg.split_config, tspec.horizon_months)
    pdir = cfg.processed_dir
    sens_ids = [TargetSpec.load(p).target_id for p in cfg.sensitivity_target_configs]
    sens_cols = "".join(f", s{i}.label AS label__{t}, s{i}.outcome AS outcome__{t}" for i, t in enumerate(sens_ids))
    sens_joins = "".join(
        f" LEFT JOIN read_parquet('{outcomes_path(cfg, t).as_posix()}') s{i} USING (loan_id)"
        for i, t in enumerate(sens_ids))
    con = duckdb.connect()
    try:
        df = con.execute(f"""
            SELECT o.*, y.outcome, y.label {sens_cols}
            FROM read_parquet('{(pdir / 'loan_origination.parquet').as_posix()}') o
            JOIN read_parquet('{outcomes_path(cfg).as_posix()}') y USING (loan_id) {sens_joins}
            WHERE y.label IS NOT NULL
            ORDER BY o.loan_id""").df()
    finally:
        con.close()

    feats = build_feature_frame(df, fspec)
    extra = [c for t in sens_ids for c in (f"label__{t}", f"outcome__{t}")]
    snap = pd.concat([df[["loan_id", "first_payment_date", "label", "outcome", "data_origin", "source_id", *extra]],
                      feats], axis=1)
    for t in sens_ids:
        snap[f"label__{t}"] = snap[f"label__{t}"].astype("Int64")  # nullable: excluded under that target
    snap["split"] = sspec.assign(snap["first_payment_date"], snap["source_id"])
    snap["label"] = snap["label"].astype("int64")
    snap["feature_set_id"] = fspec.feature_set_id
    snap["target_id"] = tspec.target_id
    snap["split_id"] = sspec.split_id
    out = pdir / "feature_snapshots.parquet"
    snap.to_parquet(out, index=False)

    manifest = {
        "dataset_id": cfg.dataset_id,
        "data_origin": cfg.data_origin,
        "built_at": utc_now(),
        "split_id": sspec.split_id,
        "feature_set_id": fspec.feature_set_id,
        "target_id": tspec.target_id,
        "date_column": sspec.date_column,
        "min_gap_months": sspec.min_gap_months,
        "ranges": sspec.ranges,
        "snapshot_file_sha256": sha256_file(out),
        "splits": {},
    }
    for name, g in snap.groupby("split"):
        manifest["splits"][name] = {
            "loans": int(len(g)),
            "defaults": int(g["label"].sum()),
            "prevalence": float(g["label"].mean()) if len(g) else None,
            "fpd_min": str(g["first_payment_date"].min()),
            "fpd_max": str(g["first_payment_date"].max()),
            "loan_ids_sha256": sha256_text("\n".join(sorted(g["loan_id"]))),
            "source_ids": {str(k): int(v) for k, v in g["source_id"].value_counts().items()},
            "sensitivity_targets": {
                t: {"labelled": int(g[f"label__{t}"].notna().sum()), "defaults": int(g[f"label__{t}"].sum())}
                for t in sens_ids},
        }
    manifest["missing_rate_by_feature"] = {c: float(feats[c].isna().mean()) for c in fspec.model_columns}
    write_json(pdir / "split_manifest.json", manifest)
    return manifest
