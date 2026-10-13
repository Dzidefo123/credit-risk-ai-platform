"""Command-line entry point: ``uv run crisk <command> --dataset <id>``."""

from __future__ import annotations

import argparse
import json
import sys

from crisk.audit.store import AuditStore
from crisk.config import DatasetConfig, load_dataset_config
from crisk.utils import utc_now, write_json


def _p(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_fixtures(cfg: DatasetConfig, a) -> dict:
    from crisk.fixtures.synthetic import generate

    if cfg.data_origin != "SYNTHETIC":
        raise SystemExit("fixtures can only be generated for SYNTHETIC datasets")
    key = generate(cfg.raw_dir, n_random=a.n_random, seed=a.seed, contract_version=cfg.contract_version)
    return {"raw_dir": str(cfg.raw_dir), "files": key["files"], "n_random": a.n_random, "seed": a.seed}


def cmd_unpack(cfg, a) -> dict:
    from crisk.ingestion.archives import unpack_archives

    m = unpack_archives(cfg)
    return {"out_dir": m["out_dir"], "archives": len(m["archives"]),
            "members": [(e["member"], e["status"], e["bytes"]) for e in m["members"]]}


def cmd_ingest(cfg, a) -> dict:
    from crisk.ingestion.raw_files import ingest_dataset

    if cfg.compose:
        from crisk.ingestion.compose import compose_interim

        return compose_interim(cfg)

    return {"files": [r.__dict__ for r in ingest_dataset(cfg)]}


def cmd_canonical(cfg, a) -> dict:
    from crisk.quality.canonical import build_canonical

    return build_canonical(cfg)


def cmd_labels(cfg, a) -> dict:
    from crisk.labels.default_horizon import build_labels

    r = build_labels(cfg)
    return {k: r[k] for k in ("target_id", "cohort_flow", "labelled_loans", "defaults", "prevalence_labelled")}


def cmd_features(cfg, a) -> dict:
    from crisk.features.snapshots import build_feature_snapshots

    m = build_feature_snapshots(cfg, feature_config=a.feature_config)
    return {k: m[k] for k in ("split_id", "feature_set_id", "splits")}


def cmd_train(cfg, a) -> dict:
    from crisk.models.train import train_model

    meta = train_model(cfg, AuditStore(cfg.audit_db), target_id=a.target_id, feature_config=a.feature_config,
                       model_config=a.model_config)
    cal = meta["metrics"]["calibration"]["logistic_uncalibrated"]
    return {"model_id": meta["model_id"], "target_id": meta["target_id"], "data_origin": meta["data_origin"],
            "evidence_note": meta["evidence_note"], "training_data": meta["training_data"],
            "calibration_split_uncalibrated": {k: cal.get(k) for k in ("n", "events", "roc_auc", "brier",
                                                                        "calibration_intercept", "calibration_slope")}}


def cmd_evaluate(cfg, a) -> dict:
    from crisk.models.train import evaluate_on_test

    r = evaluate_on_test(cfg, a.model_id, AuditStore(cfg.audit_db))
    m = r["logistic_calibrated"]
    return {"model_id": a.model_id, "data_origin": r["data_origin"], "evidence_note": r["evidence_note"],
            "prior_test_evaluations": r["prior_test_evaluations"],
            "test": {k: m.get(k) for k in ("n", "events", "prevalence", "roc_auc", "gini", "ks", "pr_auc", "brier",
                                           "log_loss", "calibration_intercept", "calibration_slope")}}


def cmd_promote(cfg, a) -> dict:
    from crisk.models.registry import promote

    return promote(cfg, a.model_id, a.reviewer, a.rationale, AuditStore(cfg.audit_db))


def cmd_rollback(cfg, a) -> dict:
    from crisk.models.registry import rollback

    return rollback(cfg, a.reviewer, a.rationale, AuditStore(cfg.audit_db))


def cmd_score_batch(cfg, a) -> dict:
    from crisk.scoring.service import batch_score_split

    df = batch_score_split(cfg, AuditStore(cfg.audit_db), a.split, a.limit, a.model_id)
    out = cfg.reports_dir / f"batch_scores_{a.split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return {"scored": int((df.status == "scored").sum()), "unavailable": int((df.status != "scored").sum()),
            "output": str(out), "first_score_id": df.score_id.iloc[0] if len(df) else None}


def cmd_replay(cfg, a) -> dict:
    from crisk.scoring.service import replay

    return replay(cfg, AuditStore(cfg.audit_db), a.score_id)


def cmd_export_results(cfg, a) -> dict:
    from crisk.publish.export import export_results

    if not a.model_id:
        raise SystemExit("--model-id is required (comma-separated for several models)")
    return export_results(cfg, [m.strip() for m in a.model_id.split(",") if m.strip()])


def cmd_verify_audit(cfg, a) -> dict:
    ok, msg = AuditStore(cfg.audit_db).verify_chain()
    return {"chain_ok": ok, "detail": msg}


def cmd_serve(cfg, a) -> dict:
    import uvicorn

    from crisk.scoring.api import create_app

    uvicorn.run(create_app(cfg, a.model_id), host="127.0.0.1", port=a.port)
    return {}


def cmd_run_slice(cfg, a) -> dict:
    """Fixture -> ingest -> validate -> labels -> features -> train -> test eval -> promote -> score -> replay."""
    summary: dict = {"dataset_id": cfg.dataset_id, "data_origin": cfg.data_origin, "started_at": utc_now()}
    if cfg.data_origin == "SYNTHETIC":
        summary["fixtures"] = cmd_fixtures(cfg, a)
    summary["ingest"] = cmd_ingest(cfg, a)
    summary["canonical"] = cmd_canonical(cfg, a)["checks"]
    summary["labels"] = cmd_labels(cfg, a)
    summary["features"] = cmd_features(cfg, a)
    summary["train"] = cmd_train(cfg, a)
    model_id = summary["train"]["model_id"]
    a.model_id = model_id
    summary["test_evaluation"] = cmd_evaluate(cfg, a)
    a.reviewer = a.reviewer or "demo-operator"
    a.rationale = a.rationale or f"Vertical-slice demonstration on {cfg.data_origin} data; not a validation approval."
    summary["promotion"] = cmd_promote(cfg, a)
    a.split, a.limit = "test", a.limit or 200
    summary["batch_scoring"] = cmd_score_batch(cfg, a)
    a.score_id = summary["batch_scoring"]["first_score_id"]
    summary["replay"] = cmd_replay(cfg, a)
    summary["audit_chain"] = cmd_verify_audit(cfg, a)
    summary["finished_at"] = utc_now()
    write_json(cfg.reports_dir / "slice_summary.json", summary)
    return summary


COMMANDS = {
    "fixtures": cmd_fixtures, "unpack": cmd_unpack, "ingest": cmd_ingest, "build-canonical": cmd_canonical,
    "build-labels": cmd_labels, "build-features": cmd_features, "train": cmd_train, "evaluate-test": cmd_evaluate,
    "promote": cmd_promote, "rollback": cmd_rollback, "score-batch": cmd_score_batch, "replay": cmd_replay,
    "verify-audit": cmd_verify_audit, "export-results": cmd_export_results, "serve": cmd_serve,
    "run-slice": cmd_run_slice,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="crisk", description=__doc__)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--dataset", required=True, help="dataset id (configs/datasets/<id>.yaml)")
    ap.add_argument("--n-random", type=int, default=20000, help="fixtures: number of random synthetic loans")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--model-id")
    ap.add_argument("--target-id", help="train: sensitivity target id (default: primary target)")
    ap.add_argument("--feature-config", help="build-features/train: feature set YAML (default: dataset config)")
    ap.add_argument("--model-config", help="train: model YAML (default: dataset config)")
    ap.add_argument("--reviewer")
    ap.add_argument("--rationale")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--score-id")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.dataset)
    _p(COMMANDS[a.command](cfg, a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
