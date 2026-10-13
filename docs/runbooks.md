# Incident runbooks (prototype)

Each monitoring alert names a runbook. Severity: high means act within 1 business day, medium within 1 week, low or
info means review at the next monthly monitoring meeting. Every action that changes the model, the deciles or the
policy needs a named reviewer, and is audited.

## RB-01 Bad or shifted input data (`feature_drift:*`, `missing_rate:*`)
1. Check the ingestion manifests and quarantine for the affected release (`data/interim/<ds>/_ingestion`, `_quarantine`).
2. Compare against the provider release notes for layout or enumeration changes (Release 47 changed several).
3. If a source change is confirmed, add a contract version (never edit one in place), re-ingest, and re-run monitoring.
4. If the population genuinely shifted (e.g. note rates rising with the market), record it. Check RB-02 and the model's
   validity range (finding F-005).

## RB-02 Score drift (`score_drift`)
1. Confirm the drift is genuine: volume normal, no RB-01 data issue.
2. Rank-based policies need current cut-offs. Refresh the deciles on recent **applications**, which needs no outcomes:
   `crisk refresh-deciles --eval-dataset <ds> --fpd-from <YYYY-MM> --fpd-to <YYYY-MM> --reviewer <name> --rationale <text>`.
3. Re-run `crisk policy-simulate` (expect referrals to return to about the rule's design share) and `crisk monitor`.
4. If drift comes with a discrimination drop (RB-04), escalate to model review instead of refreshing.

## RB-03 Service unavailable or low volume (`low_volume`, `/ready` = 503)
1. `GET /ready`: check `load_error` (missing champion, failed artefact hash check) and audit-store health.
2. A failed artefact hash means the file changed: do **not** bypass it. Restore from the registry or roll back
   (`crisk rollback --reviewer ... --rationale ...`).
3. While unavailable, the API returns `model_unavailable` (no score is produced) and applications go to manual
   underwriting.

## RB-04 Performance or calibration deterioration (`discrimination_drop`, `calibration`)
1. Discrimination drop (high): open a validation review (`crisk validate`), then compare segments and inputs.
2. Calibration outside band: for `demonstration_rank_ordering` models this is informational (finding F-001). For
   `pd_level` models it is high: suspend PD-based rules (the policy engine already blocks them for non-pd_level
   models) and review the level overlay.
3. Any retraining produces a **candidate**. It needs a validation report and a named reviewer before promotion.

## RB-05 Suspicious document inputs (Phase G, not yet built)
Placeholder: documents are untrusted content. Quarantine them, and never let an unvalidated extraction become a feature.

## RB-06 AI copilot citation failure (Phase H, not yet built)
Placeholder: withdraw the answer, log the failing question in the evaluation set, and fall back to plain retrieval.
