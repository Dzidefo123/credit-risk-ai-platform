"""Dataset configuration and workspace paths.

A *dataset* is one ingestion context (e.g. the synthetic fixture or a Freddie Mac
sample download). Every derived table, model and audit record is namespaced by its
``dataset_id`` and carries the dataset's ``data_origin`` (SYNTHETIC or EMPIRICAL) so
synthetic results can never be presented as empirical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

DataOrigin = Literal["SYNTHETIC", "EMPIRICAL"]


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    data_origin: DataOrigin
    source_id: str
    loan_id_profile: Literal["empirical", "synthetic"]
    raw_dir: Path
    origination_glob: str
    performance_glob: str
    performance_cutoff: str  # YYYY-MM: last reporting period the release is declared to cover
    max_structural_reject_rate: float
    target_config: str
    feature_config: str
    split_config: str
    model_config: str
    contract_version: str
    sensitivity_target_configs: tuple[str, ...] = ()
    compose: tuple[dict, ...] = ()  # build interim data from other datasets' validated files
    root: Path = REPO_ROOT

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim" / self.dataset_id

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed" / self.dataset_id

    @property
    def quarantine_dir(self) -> Path:
        return self.interim_dir / "_quarantine"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts" / self.dataset_id

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def audit_db(self) -> Path:
        return self.artifacts_dir / "audit.sqlite"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports" / self.dataset_id

    def with_root(self, root: Path, raw_dir: Path | None = None) -> DatasetConfig:
        """Relocate all workspace outputs (used by tests to write into a temp dir)."""
        d = dict(self.__dict__)
        d["root"] = root
        if raw_dir is not None:
            d["raw_dir"] = raw_dir
        return DatasetConfig(**d)


def load_dataset_config(dataset_id: str, root: Path | None = None) -> DatasetConfig:
    raw = load_yaml(CONFIG_DIR / "datasets" / f"{dataset_id}.yaml")
    base = root or REPO_ROOT
    raw_dir = Path(raw["raw_dir"])
    cfg = DatasetConfig(
        dataset_id=raw["dataset_id"],
        data_origin=raw["data_origin"],
        source_id=raw["source_id"],
        loan_id_profile=raw["loan_id_profile"],
        raw_dir=raw_dir if raw_dir.is_absolute() else base / raw_dir,
        origination_glob=raw["file_globs"]["origination"],
        performance_glob=raw["file_globs"]["performance"],
        performance_cutoff=str(raw["performance_cutoff"]),
        max_structural_reject_rate=float(raw.get("max_structural_reject_rate", 0.01)),
        target_config=raw["target_config"],
        feature_config=raw["feature_config"],
        split_config=raw["split_config"],
        model_config=raw["model_config"],
        contract_version=raw["contract_version"],
        sensitivity_target_configs=tuple(raw.get("sensitivity_target_configs", [])),
        compose=tuple(raw.get("compose", [])),
        root=base,
    )
    if cfg.data_origin not in ("SYNTHETIC", "EMPIRICAL"):
        raise ValueError(f"data_origin must be SYNTHETIC or EMPIRICAL, got {cfg.data_origin!r}")
    return cfg
