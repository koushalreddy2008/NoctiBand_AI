# Phase 5 Report: Modeling & Evaluation Plumbing

**Status: NO real-data results exist.** Local data (`data/raw/`) is truncated and missing; all real-data evaluations are blocked under the zero-hallucination data policy. All work in Phase 5 is purely plumbing, verified exclusively using synthetic programmatic fixtures.

## 1. Components Built
- **Canonical Schema (`noctiband.data_schema`)**: Enforces standard column names (`subject_id`, `night_id`, `epoch_idx`, `label`), standardizes aliases, validates missingness/types, and provides `real_data_gate` rejecting cohorts with <10 subjects or <50% valid motion per night.
- **Model Registry & Baselines (`noctiband.models`)**: Decoupled registry supporting runtime registration (`register_estimator`), built-in classifiers (`majority`, `motion_threshold`, `logistic_regression`, `random_forest`). `MotionThresholdBaseline` optimizes motion decision boundaries exclusively on training folds.
- **Evaluation Harness (`noctiband.evaluation`)**: `GroupKFold` cross-validation grouped strictly by subject; `Pipeline` fitting median imputer + standard scaler only on training data; drops NaN/Unknown labels; computes fold/pooled macro-F1, wake-F1, sleep-F1, balanced accuracy, Cohen's kappa, Wake PR-AUC, and confusion matrix.
- **Documentation**: `docs/dataset_adapter.md` (guide for integrating alternative wearable datasets) and `docs/handoff_B_phase5.md` (instructions for Person B).

## 2. Verification Outputs
```
$ python -m pytest -q -rs
....................s.............s..........                            [100%]
=========================== short test summary info ===========================
SKIPPED [1] tests\test_features.py:437: truncated or missing download
SKIPPED [1] tests\test_loader.py:315: truncated or missing download
43 passed, 2 skipped in 16.07s
```
```
$ python -m ruff check .
All checks passed!
```

## 3. Next Steps for Person B
1. Re-download full BIDSleep v1.0.0 via `scripts/download_bidsleep.py` when network bandwidth is restored; verify SHA-256 hashes against `SHA256SUMS.txt`.
2. Register additional architectures (e.g. SVM, LightGBM, temporal models) via `register_estimator`.
3. Run `evaluate_cv` across candidate models once `real_data_gate` clears full cohort.
