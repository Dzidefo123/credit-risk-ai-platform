# Data handling and security (prototype baseline)

This is a design baseline for a local research prototype. It is **not** a compliance certification.

## Data rules

- Raw Freddie Mac files stay unmodified under `data/raw/` and are never committed (`.gitignore`). The same
  applies to interim and processed data, model artefacts, audit stores and reports, all of which can carry
  loan-level information.
- Under the SFLLD terms (effective November 2025), the dataset and Derived Products may not be distributed to
  third parties without a separate agreement. Noncommercial research results may be published only if they
  cannot recreate the data. Linking records to identify individuals is prohibited, and this project makes no
  attempt to do so.
- The dataset is masked: month-level dates, 3-digit postal codes, rounded balances and no borrower identity.
  Even so, loan-level outputs are treated as restricted.
- Synthetic fixtures use fictional `X…` loan IDs and are labelled SYNTHETIC in every output.
- Credentials (Clarity login; later the FRED API key and LLM provider keys) stay outside the repository, in
  environment variables or an OS secret store. `.env*` files are ignored.

## Current controls

| Control | Status |
|---|---|
| Append-only audit tables (triggers) and hash-chain tamper evidence | implemented, tested |
| Model artefact SHA-256 verification before use | implemented, tested |
| Promotion and rollback need a named reviewer and rationale, and are audited | implemented, tested |
| Locked test-set access is audited | implemented, tested |
| Payload-free error logging in the API | implemented |
| Authentication and authorisation (roles: modeller, validator, analyst, administrator, auditor) | **not implemented**: do not expose the API beyond localhost |
| Encryption at rest, retention and deletion schedule, WORM audit storage | **design only** (production) |
| Threat model | to be written in Phase F/D |

## Retention (proposed)

Scoring events keep their input snapshot so scores can be replayed. Production would set a retention period
per jurisdiction and purpose, protect snapshots (encryption or tokenised references), and separate the
auditor's read access from the service's write credentials.
