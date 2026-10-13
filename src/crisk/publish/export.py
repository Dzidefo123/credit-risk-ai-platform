"""Publication gate: export aggregate results that are safe to share.

Freddie Mac SFLLD terms allow noncommercial research results to be made public only if they
"cannot be used to derive or recreate any part of the Single-Family Loan-Level Dataset". This gate:

1. Copies only aggregate JSON reports (evaluations, label/cohort summaries, split counts, data-quality
   check counts) and selected model metadata (configuration, coefficients/importances, metrics,
   code version). It never copies Parquet, model artefacts, audit stores or batch scores.
2. Suppresses any aggregate row covering fewer than ``MIN_LOANS`` loans, and replaces event counts
   below ``MIN_EVENTS`` with a "<MIN_EVENTS" marker.
3. Fails if any output contains a string shaped like a Freddie Mac Loan Sequence Number.

Passing the gate does not replace human review: read the export before publishing it.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from crisk.config import REPO_ROOT, DatasetConfig
from crisk.utils import code_version, sha256_file, utc_now, write_json

MIN_LOANS = 1000
MIN_EVENTS = 10
COUNT_KEYS = ("n", "loans", "labelled", "train_rows", "calibration_rows")
EVENT_KEYS = ("events", "defaults", "train_events", "calibration_events")
# Fields from which a masked event count could be recomputed (rate x n); masked together with it.
RATE_KEYS = ("observed_rate", "prevalence", "default_rate_labelled", "observed_expected_ratio", "pr_auc")
LOAN_ID_RE = re.compile(r"[FAX][0-9]{2}Q[1-4][0-9]{7}")
# Model metadata keys safe to publish (no file system paths, no environment details beyond versions).
MODEL_KEYS = ("model_id", "created_at", "data_origin", "evidence_note", "model_family", "model_config", "seed",
              "feature_set_id", "feature_columns", "target_id", "horizon_months", "split_id", "calibration",
              "coefficients", "metrics", "fit_info", "macro", "code_version", "artifact_sha256")


class PublicationGateError(RuntimeError):
    pass


def _suppress(obj: Any, stats: dict) -> Any:
    if isinstance(obj, dict):
        size = next((obj[k] for k in COUNT_KEYS if isinstance(obj.get(k), (int, float))), None)
        if size is not None and size < MIN_LOANS:
            stats["suppressed_rows"] += 1
            keep = {k: v for k, v in obj.items() if k in ("fpd_year", "fpd_quarter", "bin", "split")}
            return {**keep, "suppressed": f"fewer than {MIN_LOANS} loans"}
        out = {}
        small = any(k in EVENT_KEYS and isinstance(v, (int, float)) and v < MIN_EVENTS for k, v in obj.items())
        for k, v in obj.items():
            if k in EVENT_KEYS and isinstance(v, (int, float)) and v < MIN_EVENTS:
                stats["masked_event_counts"] += 1
                out[k] = f"<{MIN_EVENTS}"
            elif small and k in RATE_KEYS and isinstance(v, (int, float)):
                stats["masked_rates"] = stats.get("masked_rates", 0) + 1
                out[k] = "masked (small event count)"
            elif k == "loan_ids_sha256":
                out[k] = "removed"
            else:
                out[k] = _suppress(v, stats)
        return out
    if isinstance(obj, list):
        return [_suppress(v, stats) for v in obj]
    return obj


def export_results(cfg: DatasetConfig, model_ids: list[str], out_root: Path | None = None,
                   extra_datasets: list[DatasetConfig] | None = None) -> dict:
    if cfg.data_origin != "EMPIRICAL":
        raise PublicationGateError("Only EMPIRICAL datasets are exported; synthetic results are system tests")
    stamp = utc_now()[:19].replace(":", "").replace("-", "")
    out = (out_root or REPO_ROOT / "publish") / cfg.dataset_id / stamp
    out.mkdir(parents=True, exist_ok=False)
    stats = {"suppressed_rows": 0, "masked_event_counts": 0}
    sources: list[tuple[str, Path]] = []
    pdir = cfg.processed_dir
    for name in ("canonical_build_report.json", "label_report.json"):
        sources.append((name, pdir / name))
    sources += [(p.name, p) for p in sorted(pdir.glob("label_report__*.json"))]
    sources += [(p.name, p) for p in sorted(pdir.glob("split_manifest__*.json"))]
    for extra in extra_datasets or []:  # e.g. an evaluation-only holdout dataset
        if extra.data_origin != "EMPIRICAL":
            raise PublicationGateError("Only EMPIRICAL datasets are exported")
        xd = extra.processed_dir
        for f in ["canonical_build_report.json", "label_report.json", *[q.name for q in xd.glob("label_report__*.json")],
                  *[q.name for q in xd.glob("split_manifest__*.json")]]:
            sources.append((f"{extra.dataset_id}/{f}", xd / f))
    for mid in model_ids:
        meta_path = cfg.models_dir / mid / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        write_json(out / "models" / f"{mid}.json", _suppress({k: meta.get(k) for k in MODEL_KEYS}, stats))
        for rep in sorted((cfg.reports_dir / mid).glob("*.json")) if (cfg.reports_dir / mid).exists() else []:
            sources.append((f"evaluations/{mid}__{rep.name}", rep))
    for rel, src in sources:
        if not src.exists():
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        write_json(out / rel, _suppress(data, stats))
    docs = REPO_ROOT / "docs"
    if (docs / "results").exists():
        shutil.copytree(docs / "results", out / "docs" / "results")
    for f in ("evaluation_plan_v1.md", "evaluation_plan_v2.md", "target_specification.md", "feature_dictionary.md"):
        if (docs / f).exists():
            (out / "docs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(docs / f, out / "docs" / f)

    # Gate: no loan-identifier-shaped strings anywhere in the export.
    files = sorted(p for p in out.rglob("*") if p.is_file())
    for p in files:
        if LOAN_ID_RE.search(p.read_text(encoding="utf-8", errors="ignore")):
            shutil.rmtree(out)
            raise PublicationGateError(f"{p.name} contains a loan-identifier-shaped string; export removed")
    manifest = {
        "exported_at": utc_now(), "dataset_id": cfg.dataset_id, "data_origin": cfg.data_origin,
        "rules": {"min_loans_per_row": MIN_LOANS, "min_events_shown": MIN_EVENTS,
                  "loan_id_scan": LOAN_ID_RE.pattern, "copied": "aggregate JSON and docs only"},
        "stats": stats, "code_version": code_version(),
        "terms_note": ("Freddie Mac SFLLD: noncommercial research results only; must not allow the dataset to be "
                       "recreated; attribute Freddie Mac. Human review required before publication."),
        "files": {str(p.relative_to(out)).replace("\\", "/"): sha256_file(p) for p in files},
    }
    write_json(out / "export_manifest.json", manifest)
    return {"out_dir": str(out), **{k: manifest[k] for k in ("stats", "rules")}, "n_files": len(files)}
