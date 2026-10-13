# Feature dictionary: `origination_v1`

The source of truth is [configs/features/origination_v1.yaml](../configs/features/origination_v1.yaml), validated
against field availability in [src/crisk/contracts/freddie_mac.py](../src/crisk/contracts/freddie_mac.py).
`FeatureSpec.check_availability()` refuses any input whose availability is post-origination, performance or
identifier.

## Model inputs

| Feature | Type | Source field (guide position) | Availability | Missing handling | Notes |
|---|---|---|---|---|---|
| credit_score | numeric | CREDIT SCORE (1) | origination | 9999 → NULL; median + indicator | 300–850 |
| orig_dti | numeric | ORIGINAL DTI (10) | origination | 999 → NULL; median + indicator | > 65% and all HARP loans are masked to 999 |
| orig_ltv | numeric | ORIGINAL LTV (12) | origination | 999 → NULL | Valid range changed in 2018Q2 |
| orig_cltv | numeric | ORIGINAL CLTV (9) | origination | 999 → NULL | |
| orig_interest_rate | numeric | ORIGINAL INTEREST RATE (13) | origination | **required** | A rate spread vs market needs macro data (Phase A3) |
| orig_loan_term | numeric | ORIGINAL LOAN TERM (22) | origination | **required** | Months |
| mi_pct | numeric | MI % (6) | acquisition (accepted) | 999 → NULL | 0 = no MI |
| num_units | numeric | NUMBER OF UNITS (7) | origination | 99 → NULL | |
| log_orig_upb | derived | ln(ORIGINAL UPB (11)) | origination | **required** input | UPB is rounded to $1,000 |
| multiple_borrowers | derived | NUMBER OF BORROWERS (23) ≥ 2 | origination | 99 → NULL | Harmonises the 2018Q2 coding change |
| cltv_minus_ltv | derived | CLTV − LTV | origination | NULL if either is missing | Secondary-financing proxy |
| occupancy_status | categorical | (8) | origination | 9 → `__MISSING__` | P/I/S |
| channel | categorical | (14) | origination | 9 → `__MISSING__` | R/B/C/T |
| property_type | categorical | (18) | origination | 99 → `__MISSING__` | |
| loan_purpose | categorical | (21) | origination | 9 → `__MISSING__` | |
| first_time_homebuyer_flag | categorical | (3) | origination | 9 → `__MISSING__` | |
| property_state | categorical | (17) | origination | `__MISSING__` | **Proxy review required** (A-013) |

Learned transformations (median, scaling, category pooling at min frequency 0.5%) are fitted on the
**train split only**. Unseen categories at scoring time are pooled as infrequent and flagged
(`unseen_category:…`).

## Excluded fields (with reason)

| Field | Reason |
|---|---|
| servicer_name | Reported as of the **last** performance period: leakage |
| mi_cancellation_indicator | Post-purchase event: leakage |
| seller_name | Masked by a quarter-level rule computed after the quarter; institution identifier |
| loan_id, pre_relief_refi_loan_id | Identifiers |
| first_payment_date, maturity_date | Used for cohorting and splits, or redundant |
| postal_code, msa_code | High-cardinality geography; deferred pending proxy review |
| amortization_type, ppm_flag, io_indicator, super_conforming_flag | Constant or near-constant in the Standard population (verify empirically) |
| special_eligibility_program, relief_refi_indicator, property_valuation_method | Coding changes or time-dependent missingness; deferred to segment analysis |
| All performance-file fields | Post-origination outcomes. Reserved for labels and the later behavioural model |
