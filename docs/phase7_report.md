# Phase 7 Report: Signal Conditioning & Cross-Validation Metrics

## 1. Fixes & Verification Summary
- **Deduplication & Monotonicity**: Adapter drops 33,526 duplicate HR timestamps and sorts non-monotonic rows (6 HR, 293 motion) prior to epoching; tested in `test_deduplication_and_sorting_in_adapter`.
- **Sampling-Rate Invariance**: Motion count/activity count scale with $f_s$; converted to invariant per-sample fractions (`motion_activity_fraction`, `motion_fraction_still`). Verified on 10/33/50 Hz synthetic signals (max diff < 0.04).
- **`MODEL_FEATURES`**: Explicitly defined; excludes all QC/sample counts (`hr_count`, `motion_count`, `hr_coverage`, `motion_coverage`, etc.). Recalibrated `epochs_since_last_high_activity` (cap 120) has 95.5% coverage but 53.1% ceiling saturation $\rightarrow$ dropped from `MODEL_FEATURES`.
- **Integrity**: Real-data test `test_real_data_v2_integration_gated` ran and passed. Rolling features confirmed within-night only. All model features on 26 curated subjects are $\ge 99.6\%$ non-null.

## 2. Cross-Validation Metrics (22 Train+Val Subjects; 4 Test Untouched)
*Caveats: 22 subjects, 1 night each, ~8.2% Wake prevalence, Apple Watch wrist data rather than target headband.*

| Model | Binary Acc | Binary Macro-F1 | Bal Acc | Wake F1 | Sleep F1 | Kappa | Wake PR-AUC | 3-Class Macro-F1 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Majority** | **0.9166** | 0.4783 | 0.5000 | 0.0000 | 0.9565 | 0.0000 | 0.0834 | 0.2891 |
| **Motion Threshold** | 0.6700 | 0.4582 | 0.4875 | 0.1194 | 0.7969 | -0.0117 | 0.1698 | 0.3694 |
| **Logistic Regression** | 0.8517 | **0.6645** | **0.7500** | **0.4138** | 0.9151 | 0.3401 | **0.5151** | **0.5563** |
| **Random Forest** | **0.9297** | **0.6718** | 0.6250 | 0.3808 | **0.9628** | **0.3521** | 0.4634 | 0.5196 |

- **Verdict**: LR and RF decisively beat the motion-threshold baseline across all metrics (+20.6 to +21.4 Macro-F1).
- **Sampling Rate Outliers**: In the per-subject table, the 10 Hz subjects (`5383425`: BalAcc=0.644; `8258170`: BalAcc=0.624) and 33 Hz subject (`9618981`: BalAcc=0.684) lie squarely in the center of the 50 Hz cohort distribution (range 0.499–0.816). They are **not outliers**.

## 3. Verification Outputs
```
$ python -m pytest -q -rs
............................s.............s..............                [100%]
SKIPPED [1] tests\test_features.py:437: truncated or missing download
SKIPPED [1] tests\test_loader.py:315: truncated or missing download
55 passed, 2 skipped in 9.30s
$ python -m ruff check .
All checks passed!
```
