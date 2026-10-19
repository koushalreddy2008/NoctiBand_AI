# Phase 4 Report: Feature Engineering & Dataset Splitting

> PIPELINE CHECK ONLY: results are not scientifically meaningful

## 1. Built Components
- `configs/features.yaml`: Centered context windows (+-2, +-5), thresholds, `hours_since_start: false` (circadian bias documented).
- `src/noctiband/features/`: `build_epoch_table`, `build_dataset`, `make_split`, allowlists (`FEATURE_COLUMNS`, `LABEL_COLUMNS`, `METADATA_COLUMNS`).
- Leakage Guard & Minimum Subjects: `make_split` strictly requires >= 3 distinct subjects (raises `ValueError` otherwise; night-level splits prohibited).
- Motion & HR Validity: Zero-count epochs emit NaNs + `motion_missing = True`; counts < 500 emit `motion_partial = True`.

## 2. Test & Lint Output
- Lint: `python -m ruff check .` -> All checks passed!
- Tests: `python -m pytest -q -rs` -> 28 passed in 2.39s (100%).
- Data Tests: `pytest -m data -rs` -> 2 passed, 26 deselected in 1.63s.

## 3. Dataset Coverage & Truncated Tails
Total: n_subjects=1, n_epochs=1928, n_wake=71 (Night 1: n_subjects=1, n_wake=36, n_epochs=935; Night 3: n_subjects=1, n_wake=35, n_epochs=993).
Local motion missing: Night 1 = 99.25% (928/935), Night 3 = 98.69% (980/993); HR missing: Night 1 = 0.11% (1/935), Night 3 = 0.20% (2/993) [n_subjects=1, n_wake=71].

| Night | Epoch | HR Count | HR Mean | Motion Count | Motion Mag | HR Miss | Motion Miss | Motion Part | Label |
|:-----:|:-----:|:--------:|:-------:|:------------:|:----------:|:-------:|:-----------:|:-----------:|:-----:|
| 1 | 0 | 6 | 64.33 | 1495 | 1.005 | False | False | False | 0 (Wake) |
| 1 | 1 | 3 | 69.33 | 1496 | 1.017 | False | False | False | 0 (Wake) |
| 1 | 2 | 3 | 74.00 | 1495 | 1.049 | False | False | False | 0 (Wake) |
| 1 | 932 | 5 | 65.20 | 0 | NaN | False | True | False | 0 (Wake) |
| 1 | 933 | 1 | 68.00 | 0 | NaN | False | True | False | 0 (Wake) |
| 1 | 934 | 0 | NaN | 0 | NaN | True | True | False | 0 (Wake) |
| 3 | 0 | 4 | 66.50 | 1496 | 1.009 | False | False | False | 0 (Wake) |
| 3 | 1 | 2 | 73.50 | 1496 | 1.045 | False | False | False | 0 (Wake) |
| 3 | 2 | 5 | 76.20 | 1496 | 1.062 | False | False | False | 0 (Wake) |
| 3 | 990 | 1 | 89.00 | 0 | NaN | False | True | False | 4 (REM) |
| 3 | 991 | 0 | NaN | 0 | NaN | True | True | False | 4 (REM) |
| 3 | 992 | 0 | NaN | 0 | NaN | True | True | False | 1 (N1) |

## 4. Key Decisions & Open Questions
- Decisions: Normalization uses per-night HR median/IQR only. Centered context (+-2, +-5) is valid offline. `hours_since_start` excluded.
- Open Questions: When full benchmark motion files are downloaded, verify full 50 Hz motion across all epochs 0..990/933.
