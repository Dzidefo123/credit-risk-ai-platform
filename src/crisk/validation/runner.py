"""Independent validation workflow (Phase D).

Runs separately from model fitting, against release criteria fixed in configs/validation/release_criteria_v1.yaml.
It produces a JSON and Markdown report with:
  - 14 reproducible checks (V01-V14), each with status pass / fail / not_applicable and its evidence;
  - a findings register: the standing findings plus one finding per failed check;
  - the outcome: the list of uses the model is approved for (demonstration_rank_ordering, pd_level).
The run is written to the audit chain. A prototype validation report is not an institutional approval.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from crisk.audit.store import AuditStore
from crisk.config import DatasetConfig, load_yaml
from crisk.utils import code_version, sha256_file, utc_now, write_json

CRITERIA = "configs/validation/release_criteria_v1.yaml"
FINDINGS = "configs/validation/findings_register.yaml"
LEAKAGE_FIELDS = ("servicer_name", "mi_cancellation_indicator", "seller_name", "loan_id", "pre_relief_refi_loan_id")
CHECK_TITLES = {
    "V01": "Artefact integrity (SHA-256)", "V02": "Code lineage (clean commit or audited reproduction)",
    "V03": "Training snapshot unchanged since training", "V04": "Leakage guard on feature inputs",
    "V05": "Out-of-time evaluation evidence exists", "V06": "Discrimination: AUC CI lower bound",
    "V07": "PD-level calibration (O/E and slope bands)", "V08": "Test/holdout access discipline",
    "V09": "Probability bounds on extreme inputs", "V10": "Economic direction of key inputs",
    "V11": "Missing-input handling", "V12": "Invalid-input rejection (API schema)",
    "V13": "Offline/online parity", "V14": "Score replay from audit record",
}


def _check(cid: str, passed: bool | None, evidence: dict, severity: str = "high") -> dict:
    status = "not_applicable" if passed is None else ("pass" if passed else "fail")
    return {"id": cid, "title": CHECK_TITLES[cid], "status": status, "severity": severity, "evidence": evidence}


def _evidence_ids(store: AuditStore, model_id: str) -> list[str]:
    """The model plus any model it was audited as a bit-identical reproduction of."""
    ids = [model_id]
    for e in store.events(action="model_reproduced", entity_id=model_id):
        src = e["details"].get("reproduces_model_id")
        if src and src not in ids:
            ids.append(src)
    return ids


def _latest_evaluation(cfg: DatasetConfig, ids: list[str]) -> tuple[str | None, dict | None]:
    """Prefer a holdout evaluation over the test evaluation (the most recent out-of-time evidence)."""
    for pattern in ("evaluation_holdout__*.json", "evaluation_test.json"):
        for mid in ids:
            files = sorted((cfg.reports_dir / mid).glob(pattern)) if (cfg.reports_dir / mid).exists() else []
            if files:
                return str(files[-1].relative_to(cfg.root)), json.loads(files[-1].read_text(encoding="utf-8"))
    return None, None


def _median_applicant(snap: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        s = snap[c].dropna()
        if s.empty:
            out[c] = None
        elif pd.api.types.is_numeric_dtype(s):
            med = float(s.median())
            # integer-valued fields stay integers so schema checks fail only for the intended reason
            out[c] = int(round(med)) if np.all(np.mod(s.to_numpy(dtype="float64"), 1) == 0) else med
        else:
            out[c] = str(s.mode().iloc[0])
    return out


def validate_model(cfg: DatasetConfig, model_id: str, validator: str, store: AuditStore | None = None) -> dict:
    from crisk.features.origination import FeatureSpec
    from crisk.features.snapshots import snapshot_path
    from crisk.models.registry import ModelIntegrityError, champion, load_bundle, load_metadata
    from crisk.scoring.schemas import REQUIRED_FOR_SCORING, application_model
    from crisk.scoring.service import ScoringService, replay

    if not validator.strip():
        raise ValueError("A named validator is required")
    crit = load_yaml(CRITERIA)
    store = store or AuditStore(cfg.audit_db)
    checks: list[dict] = []

    # V01 integrity
    try:
        bundle, meta = load_bundle(cfg, model_id)
        checks.append(_check("V01", True, {"artifact_sha256": meta["artifact_sha256"]}, "critical"))
    except ModelIntegrityError as e:
        checks.append(_check("V01", False, {"error": str(e)}, "critical"))
        meta = load_metadata(cfg, model_id)
        bundle = None
    ids = _evidence_ids(store, model_id)

    # V02 lineage
    cv = meta.get("code_version", {})
    reproduced = len(ids) > 1
    clean = cv.get("git_commit") not in (None, "uncommitted") and not cv.get("git_dirty", True)
    checks.append(_check("V02", clean, {"code_version": cv, "audited_reproduction_of": ids[1:] or None,
                                        "note": "evaluation evidence from reproduced model applies" if reproduced
                                        else None}))

    # V03 training snapshot
    snap_file = snapshot_path(cfg, meta["feature_set_id"])
    same = snap_file.exists() and sha256_file(snap_file) == meta["training_data"]["feature_snapshot_sha256"]
    checks.append(_check("V03", same, {"snapshot": str(snap_file.relative_to(cfg.root)) if snap_file.exists()
                                       else None}))

    # V04 leakage guard
    try:
        spec = bundle.feature_spec if bundle is not None else FeatureSpec.load(cfg.feature_config)
        spec.check_availability()
        leaked = [c for c in meta["feature_columns"] if c in LEAKAGE_FIELDS]
        checks.append(_check("V04", not leaked, {"raw_inputs": spec.raw_inputs, "leaked_fields": leaked}, "critical"))
    except ValueError as e:
        checks.append(_check("V04", False, {"error": str(e)}, "critical"))

    # V05/V06/V07 evaluation evidence
    eval_path, ev = _latest_evaluation(cfg, ids)
    checks.append(_check("V05", ev is not None, {"evaluation": eval_path}))
    if ev is not None:
        m, ci = ev["logistic_calibrated"], ev.get("bootstrap_ci_logistic_calibrated", {})
        lo = ci.get("roc_auc", {}).get("low")
        checks.append(_check("V06", lo is not None and lo > crit["discrimination"]["auc_ci_lower_gt"],
                             {"roc_auc": m.get("roc_auc"), "ci_low": lo, "evaluation": eval_path}))
        band, sband = crit["pd_level"]["observed_expected_band"], crit["pd_level"]["calibration_slope_band"]
        oe, slope = m.get("observed_expected_ratio"), m.get("calibration_slope")
        checks.append(_check("V07", oe is not None and band[0] <= oe <= band[1] and sband[0] <= slope <= sband[1],
                             {"observed_expected": oe, "band": band, "calibration_slope": slope, "slope_band": sband},
                             "high"))
    else:
        checks += [_check("V06", False, {"reason": "no evaluation"}), _check("V07", False, {"reason": "no evaluation"})]

    # V08 test-access discipline
    counts = {}
    for mid in ids:
        for action in ("test_set_evaluated", "holdout_evaluated"):
            counts[f"{mid}:{action}"] = len(store.events(action=action, entity_id=mid))
    mx = crit["test_access"]["max_evaluations_per_split"]
    checks.append(_check("V08", all(v <= mx for v in counts.values()), {"evaluations": counts, "max": mx}))

    # Robustness checks need a working bundle and the reference population
    snap = pd.read_parquet(snap_file) if snap_file.exists() else None
    ref = snap[snap.split == "calibration"] if snap is not None else None
    with tempfile.TemporaryDirectory() as tmp:
        tmp_store = AuditStore(Path(tmp) / "validation_audit.sqlite")  # robustness scoring never touches prod audit
        svc = ScoringService(cfg, tmp_store, model_id=model_id) if bundle is not None else None
        raw_inputs = bundle.feature_spec.raw_inputs if bundle is not None else []
        median = _median_applicant(_canonical_reference(cfg, ref), raw_inputs) if ref is not None and len(ref) else {}

        # V09 bounds
        if svc is not None and median:
            from crisk.contracts.freddie_mac import CONTRACT_SETS, DEFAULT_CONTRACT_VERSION

            orig = CONTRACT_SETS[DEFAULT_CONTRACT_VERSION]["origination"]
            extremes = []
            for bound in ("min_value", "max_value"):
                app = dict(median)
                for c in raw_inputs:
                    v = getattr(orig.by_name(c), bound, None) if c in orig.names else None
                    if v is not None and orig.by_name(c).dtype in ("int", "float") and c != "orig_upb":
                        app[c] = v
                extremes.append(app)
            extremes.append({c: (median[c] if c in REQUIRED_FOR_SCORING else None) for c in raw_inputs})
            res = svc.compute(extremes)
            ok = all((r.status == "scored" and np.isfinite(r.pd) and 0 < r.pd < 1) or r.status == "unavailable"
                     for r in res)
            checks.append(_check("V09", ok, {"pds": [r.pd for r in res], "statuses": [r.status for r in res]}))
        else:
            checks.append(_check("V09", False, {"reason": "model or reference population unavailable"}))

        # V10 economic direction
        if svc is not None and median:
            details, ok = {}, True
            for feat, rule in crit["monotonic_expectations"].items():
                if feat not in raw_inputs or median.get(feat) is None:
                    continue
                hi = dict(median, **{feat: median[feat] + rule["step"]})
                base_pd, hi_pd = (r.pd for r in svc.compute([median, hi]))
                good = (hi_pd - base_pd) * rule["direction"] > 0
                ok &= good
                details[feat] = {"base_pd": base_pd, "pd_after_step": hi_pd, "step": rule["step"],
                                 "expected_direction": rule["direction"], "ok": bool(good)}
            checks.append(_check("V10", ok, details))
        else:
            checks.append(_check("V10", False, {"reason": "model or reference population unavailable"}))

        # V11 missing inputs
        if svc is not None and median:
            details, ok = {}, True
            for c in raw_inputs:
                r = svc.compute([dict(median, **{c: None})])[0]
                expect_unavailable = c in REQUIRED_FOR_SCORING
                good = (r.status == "unavailable") if expect_unavailable else (
                    r.status == "scored" and any(c in f for f in r.dq_flags))
                ok &= good
                details[c] = {"status": r.status, "flags": r.dq_flags, "ok": bool(good)}
            checks.append(_check("V11", ok, details))
        else:
            checks.append(_check("V11", False, {"reason": "model or reference population unavailable"}))

        # V12 invalid input rejection by the API schema
        if bundle is not None and median:
            Application = application_model(bundle.feature_spec)
            cases = {"credit_score_out_of_range": dict(median, credit_score=1200),
                     "unknown_occupancy_code": dict(median, occupancy_status="Z"),
                     "post_origination_field": dict(median, servicer_name="X"),
                     "provider_sentinel": dict(median, credit_score=9999)}
            results = {}
            for name, app in cases.items():
                try:
                    Application(**{k: v for k, v in app.items() if k in raw_inputs or k == "servicer_name"})
                    results[name] = "accepted"
                except Exception:  # noqa: BLE001 - pydantic ValidationError
                    results[name] = "rejected"
            checks.append(_check("V12", all(v == "rejected" for v in results.values()), results))
        else:
            checks.append(_check("V12", False, {"reason": "model unavailable"}))

        # V13 offline/online parity
        if svc is not None and ref is not None and len(ref):
            n = min(int(crit["robustness"]["parity_sample_loans"]), len(ref))
            sample = ref.sort_values("loan_id").head(n)
            canon = _canonical_reference(cfg, sample).set_index("loan_id").loc[sample.loan_id]
            apps = [{c: _py(canon.at[i, c]) for c in raw_inputs} for i in canon.index]
            online = np.array([r.pd for r in svc.compute(apps)], dtype="float64")
            offline = bundle.predict(sample)
            diff = float(np.max(np.abs(online - offline)))
            checks.append(_check("V13", diff <= float(crit["robustness"]["parity_tolerance"]),
                                 {"loans": n, "max_abs_diff": diff}, "critical"))

            # V14 replay
            r = svc.score(apps[0], requested_by=f"validator:{validator}")
            rep = replay(cfg, tmp_store, r["score_id"], actor=f"validator:{validator}")
            checks.append(_check("V14", bool(rep["match"]), {"abs_diff": rep.get("abs_diff"),
                                                             "artifact_hash_match": rep.get("artifact_hash_match")},
                                 "critical"))
        else:
            checks += [_check("V13", False, {"reason": "unavailable"}, "critical"),
                       _check("V14", False, {"reason": "unavailable"}, "critical")]

    # Outcome
    by_id = {c["id"]: c for c in checks}
    approved = [use for use, rule in crit["uses"].items()
                if all(by_id[cid]["status"] == "pass" for cid in rule["critical_checks"])]
    reg = load_yaml(FINDINGS)["findings"]
    findings = [dict(f) for f in reg]
    for c in checks:
        if c["status"] == "fail":
            findings.append({"id": f"AUTO-{c['id']}", "title": f"Check failed: {c['title']}",
                             "severity": c["severity"], "status": "open", "owner": "modeller",
                             "evidence": "validation report", "detail": json.dumps(c["evidence"], default=str)[:400]})
    ch = champion(cfg)
    sod = None
    if ch and ch.get("model_id") == model_id:
        sod = "met" if ch.get("reviewer", "").split(" ")[0].lower() != validator.split(" ")[0].lower() else \
            "NOT met: the promotion reviewer and the validator are the same person"
        if "demonstration_rank_ordering" in approved or "pd_level" in approved:
            for f in findings:
                if f["id"] == "F-006":
                    f["status"] = "closed"
                    f["detail"] += f" Closed by validation on {utc_now()[:10]}."
    report = {
        "validation_id": f"val-{model_id}-{utc_now()[:19].replace(':', '').replace('-', '')}",
        "model_id": model_id, "dataset_id": cfg.dataset_id, "data_origin": meta["data_origin"],
        "validator": validator, "validated_at": utc_now(), "criteria": crit["criteria_id"],
        "validator_code_version": code_version(), "evidence_models": ids, "checks": checks,
        "summary": {s: sum(c["status"] == s for c in checks) for s in ("pass", "fail", "not_applicable")},
        "approved_for": approved, "findings": findings,
        "open_findings_by_severity": {s: sum(f["status"] == "open" and f["severity"] == s for f in findings)
                                      for s in ("critical", "high", "medium", "low")},
        "segregation_of_duties": sod,
        "disclaimer": "Prototype validation report. Not an institutional independent model validation approval.",
    }
    out_dir = cfg.reports_dir / "validation" / model_id
    write_json(out_dir / "validation_report.json", report)
    (out_dir / "validation_report.md").write_text(_markdown(report), encoding="utf-8", newline="\n")
    store.append_event(actor=f"{validator} (validator)", action="model_validated", entity_type="model",
                       entity_id=model_id, details={"approved_for": approved, "summary": report["summary"],
                                                    "validation_id": report["validation_id"]})
    return report


def latest_validation(cfg: DatasetConfig, model_id: str) -> dict | None:
    p = cfg.reports_dir / "validation" / model_id / "validation_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _py(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    return v


def _canonical_reference(cfg: DatasetConfig, ref: pd.DataFrame) -> pd.DataFrame:
    """Canonical origination records for the reference loans (raw inputs as the API would receive them)."""
    import duckdb

    ids = pd.DataFrame({"loan_id": ref["loan_id"].astype(str)})
    with duckdb.connect() as con:
        con.register("ids", ids)
        src = (cfg.processed_dir / "loan_origination.parquet").as_posix()
        return con.execute(f"SELECT o.* FROM read_parquet('{src}') o JOIN ids USING (loan_id)").df()


def _markdown(r: dict) -> str:
    lines = [f"# Validation report: `{r['model_id']}`", "",
             f"_{r['disclaimer']}_", "",
             f"- Dataset: `{r['dataset_id']}` ({r['data_origin']}) · criteria `{r['criteria']}`",
             f"- Validator: **{r['validator']}** · {r['validated_at']}",
             f"- Evidence models: {', '.join('`' + i + '`' for i in r['evidence_models'])}",
             f"- **Approved for: {', '.join(r['approved_for']) or 'nothing'}**",
             f"- Segregation of duties: {r['segregation_of_duties'] or 'n/a (model is not the champion)'}", "",
             "## Checks", "", "| ID | Check | Status | Severity |", "|---|---|---|---|"]
    lines += [f"| {c['id']} | {c['title']} | **{c['status']}** | {c['severity']} |" for c in r["checks"]]
    lines += ["", "## Findings", "", "| ID | Title | Severity | Status | Owner |", "|---|---|---|---|---|"]
    lines += [f"| {f['id']} | {f['title']} | {f['severity']} | {f['status']} | {f['owner']} |" for f in r["findings"]]
    lines += ["", "Check evidence is in `validation_report.json`."]
    return "\n".join(lines) + "\n"
