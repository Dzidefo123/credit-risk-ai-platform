# Validation report: `pd-logit-origination_v1-20261013T011316-b6249e`

_Prototype validation report. Not an institutional independent model validation approval._

- Dataset: `freddie_mac_combined` (EMPIRICAL) · criteria `release_criteria_v1`
- Validator: **Dzidefo123** · 2026-10-13T11:32:41.716119Z
- Evidence models: `pd-logit-origination_v1-20261013T011316-b6249e`, `pd-logit-origination_v1-20261012T234214-1ba6d5`
- **Approved for: demonstration_rank_ordering**
- Segregation of duties: NOT met: the promotion reviewer and the validator are the same person

## Checks

| ID | Check | Status | Severity |
|---|---|---|---|
| V01 | Artefact integrity (SHA-256) | **pass** | critical |
| V02 | Code lineage (clean commit or audited reproduction) | **pass** | high |
| V03 | Training snapshot unchanged since training | **pass** | high |
| V04 | Leakage guard on feature inputs | **pass** | critical |
| V05 | Out-of-time evaluation evidence exists | **pass** | high |
| V06 | Discrimination: AUC CI lower bound | **pass** | high |
| V07 | PD-level calibration (O/E and slope bands) | **fail** | high |
| V08 | Test/holdout access discipline | **pass** | high |
| V09 | Probability bounds on extreme inputs | **pass** | high |
| V10 | Economic direction of key inputs | **pass** | high |
| V11 | Missing-input handling | **pass** | high |
| V12 | Invalid-input rejection (API schema) | **pass** | high |
| V13 | Offline/online parity | **pass** | critical |
| V14 | Score replay from audit record | **pass** | critical |

## Findings

| ID | Title | Severity | Status | Owner |
|---|---|---|---|---|
| F-001 | Absolute PD level unstable across vintages | high | open | modeller |
| F-002 | property_state proxy / fair-lending review not performed | medium | open | validator |
| F-003 | Funded-loan selection; no reject inference | medium | accepted | modeller |
| F-004 | Reason codes are model attributions, not adverse-action reasons | medium | accepted | administrator |
| F-005 | Note-rate extrapolation outside training range | medium | open | modeller |
| F-006 | Champion promoted before the validation workflow existed | low | closed | validator |
| AUTO-V07 | Check failed: PD-level calibration (O/E and slope bands) | high | open | modeller |

Check evidence is in `validation_report.json`.
