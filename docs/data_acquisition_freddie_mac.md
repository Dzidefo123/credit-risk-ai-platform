# Freddie Mac SFLLD: acquisition and import

Status (2026-10-12): **acquired**. Sample and full-volume files for vintages 2015 and 2022, Release 47 (July 2026).
Release 47 changed the file layout (31 origination / 35 performance fields) and the sample performance file name
(`sample_perf_YYYY.txt`). The full-volume archives contain `orig_YYYYQn.txt` / `perf_YYYYQn.txt`; run
`uv run crisk unpack --dataset freddie_mac_historical` to extract them (archives stay untouched).
Facts below were checked against official pages on 2026-10-12. Re-check them when you download.

## What to obtain first

The **Standard Dataset sample files**: a provider-drawn simple random sample of 50,000 loans per full vintage
year, in the same layout as the full dataset. Start with vintages **2015–2022**, which gives about 400k loans.
Each year comes as one zip:

| Archive | Contents | Type |
|---|---|---|
| `sample_YYYY.zip` | `sample_orig_YYYY.txt` | Origination (32 fields, `|`-delimited, no header) |
| | `sample_svcg_YYYY.txt` | Monthly performance (32 fields, `|`-delimited, no header) |

Also download the **release notes**, the **General User Guide** (the contract is coded against the January 2026
version) and **`file_layout.xlsx`**. The release notes give the Origination Cutoff Date and the Performance Cutoff
Date. The dataset page states that performance is disclosed through 2026-03-31; confirm this in the release notes.

## Steps (manual: do not script the login)

1. Read the terms: <https://capitalmarkets.freddiemac.com/crt/docs/pdfs/fre_terms_conditions_sflld.pdf>.
   They allow internal, research and academic use. Third parties may not receive the data or derived products.
   Linking records to identify individuals is prohibited.
2. Register or sign in to Clarity Data Intelligence: <https://capitalmarkets.freddiemac.com/clarity>.
3. Open the SFLLD Data Download page, which the dataset page links as <https://claritydownload.fmapps.freddiemac.com/CRT/>.
4. Download `sample_2015.zip` … `sample_2022.zip` and the release notes.
5. Unzip **without modifying** the files into:
   ```
   data/raw/freddie_mac/sample/sample_orig_2015.txt
   data/raw/freddie_mac/sample/sample_svcg_2015.txt
   ...
   ```
   Keep the zips too if you like (`data/raw/freddie_mac/archives/`). Both locations are git-excluded.
6. Set `performance_cutoff` in [configs/datasets/freddie_mac_sample.yaml](../configs/datasets/freddie_mac_sample.yaml)
   to the release's Performance Cutoff Date (`YYYY-MM`). Record the release identifier and download date in
   [configs/sources/source_manifest.yaml](../configs/sources/source_manifest.yaml).
7. Import and validate:
   ```
   uv run crisk ingest          --dataset freddie_mac_sample
   uv run crisk build-canonical --dataset freddie_mac_sample
   uv run crisk build-labels    --dataset freddie_mac_sample
   ```
   Ingestion records a SHA-256 checksum, line counts and rejects for each file in
   `data/interim/freddie_mac_sample/_ingestion/`. If more than 0.1% of a file's lines have the wrong field
   count, ingestion **refuses the file**, because that usually means a layout version change. In that case
   compare against the guide and add a new contract version rather than editing the existing one.
8. Before training, review `data/processed/freddie_mac_sample/label_report.json`: event counts by vintage,
   maturity, and forbearance-affected defaults. Then confirm the provisional split in
   [configs/splits/freddie_mac_temporal_v1.yaml](../configs/splits/freddie_mac_temporal_v1.yaml) (question Q-3).

## Memory and size notes

Sample performance files are expected (not yet measured) to be up to several hundred MB per vintage year,
because older vintages carry 100+ monthly records per surviving loan. DuckDB streams them, so a 16 GB
laptop is expected to be sufficient. This has not yet been measured on real files; record it in PROJECT_STATUS
when done. The full quarterly files (`historical_data_YYYYQn.zip`) are much larger. Move to them only after
the sample baseline is reviewed.

## Known data characteristics to handle (from the guide)

- Performance starts at **acquisition** by Freddie Mac, not origination, so early months can be missing.
  The target requires a first record by month 6 (assumption A-006).
- **Reporting gaps** and **irregular delinquency progressions** occur. The target tolerates ≤ 2 missing months
  in the window.
- Delinquency for non-performing loans is reported with a **one-month lag**.
- Before 2019-05, accounting cycles ran from the 16th to the 15th.
- `SERVICER NAME` is reported as of the **last** period, so it is excluded as leakage.
  `MI CANCELLATION INDICATOR` describes post-purchase events, so it is excluded too.
- LTV/CLTV valid ranges and number-of-borrowers coding changed in **2018Q2**.
- Original UPB is rounded to $1,000. Current UPB is rounded for loan age ≤ 6. Dates are month-level only.
  Postal codes are masked to 3 digits.
- 2020–2021 delinquency includes COVID **forbearance**. The label is kept as reported, with flags
  (`forbearance_in_window`, `disaster_flag_in_window`).
