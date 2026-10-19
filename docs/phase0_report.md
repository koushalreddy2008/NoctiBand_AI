# Phase 0 Preflight Report

> PIPELINE CHECK ONLY: results are not scientifically meaningful (n_subjects = 1 < 10)

## 1. Part 3 Artifact Verification & Test Results
- Confirmed all 7 Part 3 deliverables exist: `docs/audit_epoching.md`, `scripts/diagnose_motion.py`, `docs/motion_findings.md`, `src/noctiband/io/loader.py`, `src/noctiband/preprocessing/epoching.py`, `tests/test_loader.py`, `tests/test_epoching.py`, and `pyproject.toml` (`data` marker).
- `python -m pytest -q -rs`: 19 passed in 5.91s (0 skipped, 0 failed).
- `python -m pytest -m data -rs`: 1 passed, 18 deselected in 1.52s (`test_load_night_integration_benchmark`).
- `python -m ruff check .`: All checks passed.
- Git state: Branch `phase4-A` based on `part3-loader-A` (HEAD `a1cfa6c`).

## 2. Dataset Counts
- `data/raw/<subject>/<night>/` contains **1 subject** (`Bidslab00`) and **2 nights** (`1` and `3`) with all 3 files (`hr.csv`, `motion.csv`, `labels.mat`). Total epochs: Night 3 = 993, Night 1 = 935.

## 3. Summary of Audit and Motion Findings (<= 15 lines)
1. `audit_epoching.md`: 0-based indexing `floor((t - recStart)/30)` and `reindex(range(n_epochs))` are valid and reused in `assign_epochs` and `epoch_coverage`.
2. Previous pipeline had hard-coded flat paths, lacked timezone/DST guards, header autodetection, duplicate/NaN drop logging, and alignment guards.
3. `motion_findings.md`: `hr.csv` has no header (0.2 Hz sampling); `motion.csv` has header `Timestamp,x,y,z` (50 Hz sampling, resting magnitude $\approx 1.002$ g in $g$ units). Both are strictly sorted with zero duplicate timestamps and zero internal gaps > 60 s.
4. "Incomplete" verdict: Caused by a prior local aborted download (`motion.csv.partial`, 4.16 MB, ~18 min).
5. On benchmark data, incomplete means **truncated end relative to the label window**: motion and HR end ~87 s before window end on Night 3 (epochs 991–992 zero readings, epoch 990 partial), and ~45 s before on Night 1 (epoch 934 zero readings, epoch 933 partial).

## 4. Decisions & Open Questions
- Decision: Use leave-one-night/leave-one-subject-out folds for Phase 4 split logic given single-subject availability.
- Open question: For epochs with zero motion/HR at window boundaries (e.g. 991–992), emit NaN with missing flag; confirm downstream imputation strategy.
