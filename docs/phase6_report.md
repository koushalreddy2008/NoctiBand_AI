# Phase 6 Report: Dataset v2 Ingestion & Feature Engineering

**Status: NO model metrics on real data computed yet.** Real data plumbing, inspection, adapter conversion, coverage gating, and participant-level split verification completed.

## 1. Measured Inspection & Outlier Summary
- **Verification**: 125 files verified against `SHA256SUMS.txt` (124 signal files, `LICENSE.txt`). 0 bad.
- **Outliers**: `9618981` (380.9 MB, 100% night cov, recorded 3 days post-study); `5383425` & `8258170` (12 MB, 100% night cov, recorded at 10 Hz $f_s$); `7749105` (17.8 MB, motion halted at $t=4024.5$ s, 13.6% motion cov, 22.3% HR cov; flagged & excluded).
- **Steps Decision**: Excluded. 8/31 subjects (25.8%) have 0 in-night steps; proprietary algorithmically derived watch pedometer metric violates device-agnostic headband requirement.

## 2. Coverage, Gating & Splitting (70/15/15, Seed 42)
- **Coverage Guard ($\ge 0.95$)**: 26 subjects passed; 5 flagged (`1066528`, `1449548`, `2598705`, `7749105`, `8530312`).
- **`real_data_gate`**: Refused on uncurated 31-subject cohort (`7749105` valid motion 13.6% < 50%); PASSED on curated 26-subject cohort (`n_subjects=26`, `n_epochs=22,385`, `n_wake=1,836`).
- **Splits**: Train (18 subj, 15,146 ep, 1,306 Wake); Val (4 subj, 3,332 ep, 210 Wake); Test (4 subj, 3,907 ep, 320 Wake). **All splits have $\ge 30$ Wake epochs.**

## 3. Test & Linter Verification Outputs
```
$ python -m pytest -q -rs
............................s.............s..........                    [100%]
SKIPPED [1] tests\test_features.py:437: truncated or missing download
SKIPPED [1] tests\test_loader.py:315: truncated or missing download
51 passed, 2 skipped in 10.31s
```
```
$ python -m ruff check .
All checks passed!
```
