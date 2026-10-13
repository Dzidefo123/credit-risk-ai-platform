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


def cmd_evaluate_holdout(cfg, a) -> dict:
    from crisk.models.train import evaluate_on_test

    if not a.holdout_dataset or not a.model_id:
        raise SystemExit("--model-id and --holdout-dataset are required")
    r = evaluate_on_test(cfg, a.model_id, AuditStore(cfg.audit_db), holdout_cfg=load_dataset_config(a.holdout_dataset))
    m = r["logistic_calibrated"]
    return {"model_id": a.model_id, "evaluation_dataset": r["evaluation_dataset"], "fpd_range": r["fpd_range"],
            "target_id": r["target_id"], "prior_holdout_evaluations": r["prior_test_evaluations"],
            "holdout": {k: m.get(k) for k in ("n", "events", "prevalence", "mean_pd", "observed_expected_ratio",
                                              "roc_auc", "gini", "ks", "pr_auc", "brier", "calibration_intercept",
                                              "calibration_slope")},
            "bootstrap_ci": r["bootstrap_ci_logistic_calibrated"]}


def cmd_backtest_recalibration(cfg, a) -> dict:
    from crisk.models.recalibration import run_backtest

    if not a.model_id or not a.eval_dataset:
        raise SystemExit("--model-id (comma-separated, one per target) and --eval-dataset are required")
    r = run_backtest(cfg, [m.strip() for m in a.model_id.split(",")], load_dataset_config(a.eval_dataset),
                     AuditStore(cfg.audit_db))
    return {"decision": r["decision"], "in_band_share": {t: s["in_band_share"] for t, s in r["results"].items()},
            "overall_log_loss": {t: {m: s["overall"][m]["log_loss"] for m in ("R0", "R1", "R2")}
                                 for t, s in r["results"].items()}}


def cmd_backtest_ttc(cfg, a) -> dict:
    from crisk.models.recalibration import run_ttc_backtest

    if not a.model_id or not a.eval_dataset:
        raise SystemExit("--model-id (comma-separated, one per target) and --eval-dataset are required")
    r = run_ttc_backtest(cfg, [m.strip() for m in a.model_id.split(",")], load_dataset_config(a.eval_dataset),
                         AuditStore(cfg.audit_db))
    return {"evidence": r["evidence"], "decision": r["decision"],
            "rmsle_oe": {t: s["rmsle_oe"] for t, s in r["results"].items()},
            "log_loss": {t: {m: s["overall"][m]["log_loss"] for m in ("L0", "L1", "L2")}
                         for t, s in r["results"].items()}}


def cmd_promote(cfg, a) -> dict:
    import pandas as pd

    from crisk.features.snapshots import snapshot_path
    from crisk.models.registry import load_bundle, promote, reference_deciles

    if not a.model_id or not a.reviewer or not a.rationale:
        raise SystemExit("--model-id, --reviewer and --rationale are required")
    bundle, meta = load_bundle(cfg, a.model_id)
    extra = {}
    snap_file = snapshot_path(cfg, meta["feature_set_id"])
    if snap_file.exists():  # risk-decile cut-offs on the out-of-time calibration population
        ref = pd.read_parquet(snap_file).query("split == 'calibration'")
        if len(ref):
            extra = {"risk_decile_cutoffs": reference_deciles(bundle, ref),
                     "risk_decile_reference": f"{cfg.dataset_id} calibration split ({len(ref)} loans)"}
    if cfg.data_origin == "EMPIRICAL":  # validation gate (release_criteria_v1)
        from crisk.validation.runner import latest_validation

        val = latest_validation(cfg, a.model_id)
        if val is None or a.approved_use not in val["approved_for"]:
            raise SystemExit(f"Promotion refused: no validation report approving '{a.approved_use}' for "
                             f"{a.model_id}. Run `crisk validate` first.")
        extra["validation_id"] = val["validation_id"]
    return promote(cfg, a.model_id, a.reviewer, a.rationale, AuditStore(cfg.audit_db),
                   approved_use=a.approved_use, extra=extra)


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
    extra = [load_dataset_config(a.holdout_dataset)] if a.holdout_dataset else None
    return export_results(cfg, [m.strip() for m in a.model_id.split(",") if m.strip()], extra_datasets=extra)


def cmd_create_api_key(cfg, a) -> dict:
    from crisk.security.auth import KeyStore, key_store_path

    if not a.name or not a.role:
        raise SystemExit("--name and --role are required")
    key, entry = KeyStore(key_store_path(cfg)).create(a.name, a.role, AuditStore(cfg.audit_db))
    return {"api_key": key, "warning": "Shown once. Store it in a secret manager or env var; it cannot be recovered.",
            **entry}


def cmd_revoke_api_key(cfg, a) -> dict:
    from crisk.security.auth import KeyStore, key_store_path

    return KeyStore(key_store_path(cfg)).revoke(a.key_id, AuditStore(cfg.audit_db))


def cmd_list_api_keys(cfg, a) -> dict:
    from crisk.security.auth import KeyStore, key_store_path

    return {"keys": KeyStore(key_store_path(cfg)).list()}


def cmd_validate(cfg, a) -> dict:
    from crisk.validation.runner import validate_model

    if not a.model_id or not a.validator:
        raise SystemExit("--model-id and --validator are required")
    r = validate_model(cfg, a.model_id, a.validator)
    return {k: r[k] for k in ("validation_id", "approved_for", "summary", "open_findings_by_severity",
                              "segregation_of_duties")} | {
        "failed_checks": [c["id"] + " " + c["title"] for c in r["checks"] if c["status"] == "fail"],
        "report": str(cfg.reports_dir / "validation" / a.model_id / "validation_report.md")}


def cmd_fairness_review(cfg, a) -> dict:
    from crisk.fairness.review import run_review

    ids = [m.strip() for m in (a.model_id or "").split(",") if m.strip()]
    if len(ids) != 2:
        raise SystemExit("--model-id must be '<with_state_model>,<without_state_model>'")
    holdout = load_dataset_config(a.holdout_dataset) if a.holdout_dataset else None
    r = run_review(cfg, ids[0], ids[1], AuditStore(cfg.audit_db), holdout_cfg=holdout)
    return {"target_id": r["target_id"], "value": r["value_of_property_state"], "proxy_screen": r["proxy_screen"],
            "decision": r["decision"]}


def cmd_policy_simulate(cfg, a) -> dict:
    from crisk.policies.simulate import run_simulation

    if not a.model_id or not a.eval_dataset:
        raise SystemExit("--model-id and --eval-dataset are required")
    pols = a.policies.split(",") if a.policies else ["configs/policies/demo_policy_v1.yaml",
                                                     "configs/policies/demo_policy_v1_tight.yaml",
                                                     "configs/policies/demo_policy_v1_loose.yaml"]
    ref = (a.decile_from, a.decile_to) if a.decile_from and a.decile_to else None
    r = run_simulation(cfg, a.model_id, load_dataset_config(a.eval_dataset), pols, fpd_from=a.fpd_from,
                       decile_reference=ref)
    return {"n_loans": r["n_loans"], "approved_use": r["approved_use"],
            "results": {v: {k: x[k] for k in ("approve_rate", "refer_rate", "decline_rate", "default_rate_approved",
                                               "default_rate_declined")} | {
                "contribution_usd": x["economics_base_referred_declined"]["contribution_usd"]}
                for v, x in r["results"].items()}}


def cmd_monitor(cfg, a) -> dict:
    from crisk.monitoring.monitor import run_monitoring

    if not a.eval_dataset:
        raise SystemExit("--eval-dataset is required (the stream to monitor)")
    r = run_monitoring(cfg, load_dataset_config(a.eval_dataset), a.model_id)
    return {"model_id": r["model_id"], "months": len(r["months"]), "quarters": len(r["quarters"]),
            "score_psi_by_month": {m["month"]: round(m["score_psi"], 3) for m in r["months"]},
            "alerts": [{k: x[k] for k in ("alert_id", "severity", "first_period", "last_period", "periods", "owner")}
                       for x in r["alerts"]]}


def cmd_refresh_deciles(cfg, a) -> dict:
    import pandas as pd

    from crisk.features.snapshots import snapshot_path
    from crisk.models.registry import champion, load_bundle, reference_deciles, refresh_deciles

    if not (a.eval_dataset and a.fpd_from and a.fpd_to and a.reviewer and a.rationale):
        raise SystemExit("--eval-dataset, --fpd-from, --fpd-to, --reviewer and --rationale are required")
    ch = champion(cfg)
    bundle, meta = load_bundle(cfg, ch["model_id"])
    ev = load_dataset_config(a.eval_dataset)
    snap = pd.read_parquet(snapshot_path(ev, meta["feature_set_id"]))
    ym = pd.to_datetime(snap["first_payment_date"]).dt.strftime("%Y-%m")
    ref = snap[(ym >= a.fpd_from) & (ym <= a.fpd_to)]  # applications only: outcomes are not used
    cut = reference_deciles(bundle, ref)
    return refresh_deciles(cfg, cut, f"{ev.dataset_id} applications FPD {a.fpd_from}..{a.fpd_to} ({len(ref)} loans)",
                           a.reviewer, a.rationale, AuditStore(cfg.audit_db))


def cmd_verify_audit(cfg, a) -> dict:
    ok, msg = AuditStore(cfg.audit_db).verify_chain()
    return {"chain_ok": ok, "detail": msg}


def cmd_serve(cfg, a) -> dict:
    import uvicorn

    from crisk.scoring.api import create_app

    uvicorn.run(create_app(cfg, a.model_id), host=a.host, port=a.port)
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
    "fixtures": cmd_fixtures,
    "unpack": cmd_unpack,
    "ingest": cmd_ingest,
    "build-canonical": cmd_canonical,
    "build-labels": cmd_labels,
    "build-features": cmd_features,
    "train": cmd_train,
    "evaluate-test": cmd_evaluate,
    "evaluate-holdout": cmd_evaluate_holdout,
    "backtest-recalibration": cmd_backtest_recalibration,
    "backtest-ttc": cmd_backtest_ttc,
    "promote": cmd_promote,
    "rollback": cmd_rollback,
    "score-batch": cmd_score_batch,
    "replay": cmd_replay,
    "verify-audit": cmd_verify_audit,
    "monitor": cmd_monitor,
    "refresh-deciles": cmd_refresh_deciles,
    "policy-simulate": cmd_policy_simulate,
    "fairness-review": cmd_fairness_review,
    "validate": cmd_validate,
    "create-api-key": cmd_create_api_key,
    "revoke-api-key": cmd_revoke_api_key,
    "list-api-keys": cmd_list_api_keys,
    "export-results": cmd_export_results,
    "serve": cmd_serve,
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
    ap.add_argument("--holdout-dataset", help="evaluate-holdout: evaluation-only dataset id")
    ap.add_argument("--fpd-from", help="policy-simulate/refresh-deciles: first FPD month (YYYY-MM)")
    ap.add_argument("--fpd-to", help="refresh-deciles: last FPD month of the reference applications")
    ap.add_argument("--decile-from", help="policy-simulate scenario: decile reference window start")
    ap.add_argument("--decile-to", help="policy-simulate scenario: decile reference window end")
    ap.add_argument("--policies", help="policy-simulate: comma-separated policy YAML paths")
    ap.add_argument("--validator", help="validate: name of the person performing validation")
    ap.add_argument("--name", help="create-api-key: key holder name")
    ap.add_argument("--role", help="create-api-key: modeller | validator | analyst | administrator | auditor")
    ap.add_argument("--key-id", help="revoke-api-key: key id")
    ap.add_argument("--approved-use", default="demonstration_rank_ordering",
                    help="promote: demonstration_rank_ordering (default) or pd_level")
    ap.add_argument("--eval-dataset", help="backtest-recalibration: dataset providing the walk-forward cohorts")
    ap.add_argument("--reviewer")
    ap.add_argument("--rationale")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--score-id")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="serve: bind address (0.0.0.0 only inside a container)")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.dataset)
    _p(COMMANDS[a.command](cfg, a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
