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
3. `motion_findings.md`: `hr.csv` has no header (0.2 Hz sampling); `motion.csv` has header `Timestamp,x,y,z` (50 Hz sampling). Both are strictly sorted with zero duplicate timestamps.
4. **CORRECTION (Data Integrity Audit)**: The previous verdict ("benign 1-2 epoch truncated tail") was **WRONG**. Local `motion.csv` files are truncated downloads saved under the real filename (Night 1: 1.07 MB, Night 3: 1.64 MB vs ~100 MB official release), covering only ~5–8 minutes (< 1.5% coverage) rather than the full overnight window.
5. All reported motion-feature behaviour and downstream motion metrics are invalid until data is re-downloaded with SHA-256 verification.

## 4. Decisions & Open Questions
- Decision: Enforce coverage guards in `load_night` (< 50% raises ValueError) to prevent truncated data from contaminating pipelines.
- Open question: Re-download full 50 Hz motion data via `download_bidsleep.py` once network access is established.
