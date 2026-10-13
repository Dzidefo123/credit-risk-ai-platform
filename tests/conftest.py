"""Shared fixtures. All data here is SYNTHETIC and generated deterministically into temp dirs."""

from __future__ import annotations

from pathlib import Path

import pytest

from crisk.audit.store import AuditStore
from crisk.config import load_dataset_config
from crisk.features.snapshots import build_feature_snapshots
from crisk.fixtures.synthetic import generate
from crisk.ingestion.raw_files import ingest_dataset
from crisk.labels.default_horizon import build_labels
from crisk.quality.canonical import build_canonical


def make_workspace(tmp: Path, dataset_id: str, n_random: int):
    cfg = load_dataset_config(dataset_id).with_root(tmp, raw_dir=tmp / "raw")
    key = generate(cfg.raw_dir, n_random=n_random, contract_version=cfg.contract_version)
    return cfg, key


@pytest.fixture(scope="session")
def scenario_ws(tmp_path_factory):
    """Scenario loans only, processed through labels."""
    tmp = tmp_path_factory.mktemp("scenarios")
    cfg, key = make_workspace(tmp, "synthetic_scenarios", 0)
    ingest = ingest_dataset(cfg)
    canonical = build_canonical(cfg)
    labels = build_labels(cfg)
    return {"cfg": cfg, "key": key, "ingest": ingest, "canonical": canonical, "labels": labels}


@pytest.fixture(scope="session")
def trained_ws(tmp_path_factory):
    """Small simulated portfolio processed through training (SYNTHETIC)."""
    from crisk.models.train import train_model

    tmp = tmp_path_factory.mktemp("trained")
    cfg, key = make_workspace(tmp, "synthetic_fixture", 4000)
    ingest_dataset(cfg)
    build_canonical(cfg)
    build_labels(cfg)
    build_feature_snapshots(cfg)
    store = AuditStore(cfg.audit_db)
    meta = train_model(cfg, store)
    return {"cfg": cfg, "key": key, "store": store, "meta": meta}


@pytest.fixture(scope="session")
def promoted(trained_ws):
    """trained_ws with its model promoted as champion (demonstration use)."""
    from crisk.models.registry import promote

    cfg, store, meta = trained_ws["cfg"], trained_ws["store"], trained_ws["meta"]
    promote(cfg, meta["model_id"], "test-reviewer", "unit test", store)
    return trained_ws
