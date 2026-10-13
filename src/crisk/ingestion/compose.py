"""Compose a dataset from interim files already ingested for other datasets.

Used to combine full-volume vintages (train/test) with provider-sample vintages (calibration)
without re-parsing tens of GB. Only validated interim Parquet is reused; the per-file ingestion
manifests (checksums, counts) are copied alongside so lineage is preserved. Every component must
use the same contract version. Rows keep their own ``source_id`` / ``source_file`` columns.
"""

from __future__ import annotations

import fnmatch
import shutil

from crisk.config import DatasetConfig, load_dataset_config
from crisk.utils import read_json, utc_now, write_json


def compose_interim(cfg: DatasetConfig) -> dict:
    if not cfg.compose:
        raise ValueError(f"{cfg.dataset_id} has no 'compose' section")
    copied, skipped = [], []
    for comp in cfg.compose:
        src = load_dataset_config(comp["dataset"], root=cfg.root)
        if src.contract_version != cfg.contract_version:
            raise ValueError(f"{src.dataset_id} uses {src.contract_version}, expected {cfg.contract_version}")
        if src.data_origin != cfg.data_origin:
            raise ValueError("Cannot compose SYNTHETIC and EMPIRICAL data")
        for kind in ("origination", "performance"):
            for f in sorted((src.interim_dir / kind).glob("*.parquet")):
                if not any(fnmatch.fnmatch(f.stem, pat) for pat in comp["include"]):
                    continue
                manifest = src.interim_dir / "_ingestion" / f"{kind}__{f.stem}.json"
                if not manifest.exists():
                    raise FileNotFoundError(f"{manifest} missing: ingest {src.dataset_id} first")
                dst = cfg.interim_dir / kind / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and dst.stat().st_size == f.stat().st_size:
                    skipped.append(f.name)
                else:
                    shutil.copy2(f, dst)
                    copied.append(f.name)
                entry = {**read_json(manifest), "composed_from_dataset": src.dataset_id}
                write_json(cfg.interim_dir / "_ingestion" / f"{kind}__{f.stem}.json", entry)
    # remove files no longer selected so the composition is exactly what the config says
    selected = set(copied) | set(skipped)
    removed = []
    for kind in ("origination", "performance"):
        for f in (cfg.interim_dir / kind).glob("*.parquet"):
            if f.name not in selected:
                f.unlink()
                removed.append(f.name)
    result = {"dataset_id": cfg.dataset_id, "composed_at": utc_now(), "components": cfg.compose,
              "copied": copied, "skipped_unchanged": skipped, "removed": removed}
    write_json(cfg.interim_dir / "_ingestion" / "compose_manifest.json", result)
    return result


def is_composed(cfg: DatasetConfig) -> bool:
    return bool(cfg.compose)

