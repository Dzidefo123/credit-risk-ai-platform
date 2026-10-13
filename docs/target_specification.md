# Target specification: `default_90dpd_12m_v1`

**Research definition. Not a regulatory or accounting default definition.**
Config: [configs/target/default_90dpd_12m_v1.yaml](../configs/target/default_90dpd_12m_v1.yaml).
Implementation: [src/crisk/labels/default_horizon.py](../src/crisk/labels/default_horizon.py).

## Question answered

For a funded, Freddie Mac-acquired fixed-rate mortgage, what is the probability that it becomes seriously
delinquent or has a credit-related termination within its first 12 reporting months after the first payment
date? The probability is estimated from information disclosed at origination.

## Definitions

| Item | Definition |
|---|---|
| Unit of observation | One loan (Loan Sequence Number). Borrower identities are not disclosed, so borrower-level overlap cannot be checked (A-010). |
| Population | Loans in the origination file with `amortization_type = FRM`, at least one performance record and a first record by m ≤ 6. |
| Origin (time zero) | The **first payment date (FPD) month**. The note and decision dates are not disclosed; they normally precede the FPD by 1–2 months. FPD is a documented proxy, **not** the decision date. |
| Score time (intended use) | At or before origination. Features are the origination-file values (reconstructed origination state). The dataset gives no evidence of when each value became available in real time (A-003). |
| Month index | m = months(reporting_period − FPD) + 1, so m = 1 is the FPD month. It is computed from dates because the provider's Loan Age resets after modification. |
| Horizon | m ∈ [1, 12] |
| Default event | The first m in the horizon with: delinquency status ≥ 3 (90+ days past due, MBA method), **or** status `RA` (REO acquisition), **or** zero balance code 02 (third-party sale), 03 (short sale/charge-off) or 09 (REO disposition). |
| Censoring removals | Zero balance codes 15 (whole-loan sale), 16 (RPL securitisation) and 96 (defect). If one occurs before any default, the outcome is unobservable and the loan is excluded. |
| Prepayment | Zero balance code 01 in the window before any default gives label **0** (`non_default_prepaid`). This is a competing-risk choice (D-007); a censored treatment is planned as a sensitivity analysis. |
| Cure | Not relevant to the event: the first 90+ status counts even if the loan later cures. |
| Modification | Not an event in itself. Flagged (`modified_in_window`); default is still detected from status. |
| Missing reporting | More than 2 missing months in [max(first record, 1), end of window or payoff] with no observed default gives `excluded_incomplete_reporting`. A default observed before or within a gap still counts. |
| Maturity | **Cohort-level**: the loan is labelled only if FPD + 11 months ≤ the performance cutoff. Immature loans are excluded even when they have already defaulted, which avoids biasing prevalence upward. |
| Units | Balances are USD. Rates are percentages. Dates are month-level (`YYYYMM`) in the source. |

## Outcome classes

`default` (1), `non_default` (0), `non_default_prepaid` (0), and these exclusions (NULL):
`excluded_ineligible`, `excluded_no_performance`, `excluded_immature`, `excluded_late_acquisition`,
`excluded_censored_removal`, `excluded_incomplete_reporting`.

The order of evaluation is documented in the module docstring. All 25 scenario loans in
[tests/fixtures/sflld_synthetic_scenarios/answer_key.json](../tests/fixtures/sflld_synthetic_scenarios/answer_key.json)
are verified by `tests/test_labels.py`.

## Known limitations

- **Reporting lag**: delinquency for non-performing loans reflects the prior month's default reporting cycle.
  The event month is therefore approximate (±1 month).
- **COVID forbearance** (2020–2021): forborne loans can be reported as delinquent. They are kept as defaults
  under this definition, and `forbearance_in_window` and `disaster_flag_in_window` are stored so analyses can
  run with and without them (open question Q-3).
- **Selection**: only loans that were funded *and* acquired by Freddie Mac are observed. No conclusion
  about rejected applicants follows (see Phase B and E notes in the phase plan).
- **Left truncation**: performance starts at acquisition. Loans first observed after m = 6 are excluded,
  and the number excluded is reported.

## Sensitivity target: `default_90dpd_12m_exdisaster_v1` (D-017)

This target is identical to the primary target with one change. A 90+/RA status reported while the loan is
flagged "delinquency due to disaster" (`Y`) or is in a forbearance plan (`F`) is **not** a default event.

- If the loan reaches an unflagged 90+ status, or a credit termination (02/03/09), within the window, it is a
  default.
- If its only in-window 90+ months are flagged, the outcome is `excluded_disaster_forbearance`. It is not
  labelled 0, because a credit-driven outcome cannot be inferred for it.

Measured on the Freddie Mac sample (2015–2022 vintages), the primary 12-month rate jumps to about 1.8% for 2019–2020
first-payment cohorts (COVID forbearance) and about 0.42% for 2017 (hurricanes Harvey and Irma). The sensitivity
rate stays at about 0.1–0.25% until it rises to about 0.34–0.47% for 2022–23. See
`data/processed/freddie_mac_sample/label_report__default_90dpd_12m_exdisaster_v1.json` (local only).

## Distinction from conditional delinquency

This target is an **origination PD**. It does not estimate the probability of becoming delinquent
*conditional on current status*; that behavioural model is a separate, later target.
