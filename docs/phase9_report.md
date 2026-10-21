# Phase 9 Report: Audit Gaps, Ablations & Nested Threshold Tuning

## 1. Audit Explanations
- **Fold-Means vs Pooled Metrics**: Phase 7 reported arithmetic fold-means in summary rows (`0.5056` Acc / `0.3694` Macro-F1). Corrected pooled metrics match the confusion matrix identically: Majority (Acc=0.7022, Macro-F1=0.2750); Motion Threshold (Acc=0.5020, Macro-F1=0.2762, BalAcc=0.3387, Kappa=0.0120).
- **Per-Subject Discrepancy**: Phase 7 quoted the 3-class balanced accuracy range (0.499–0.816, median ~0.62); Phase 8 reported the binary range (0.480–0.887, median ~0.737).
- **Motion Threshold Dip**: Uses `motion_mag_mean` (ROC-AUC 0.56). In Fold 5 (`1818471`, `2638030`, `3997827`, `9961348`), training-fold tuning chose threshold $\theta=-0.04$, over-flagging Wake (61.7% flag rate) and yielding val BalAcc 0.3642.

## 2. Ablations & Paired Bootstrap (22 Subjects, 18,187 Epochs)
*Caveats: 22 subjects, 1 night each, ~8.2% Wake prevalence, Apple Watch data.*

| Model / Feature Set | Pooled ROC-AUC | Pooled PR-AUC | Wake F1 | Bal Acc | Macro-F1 | Median Subj ROC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Best Single (`enmo_std_w11`)** | 0.8508 | 0.4474 | 0.3394 | **0.7807** | 0.5829 | 0.8794 |
| **LR (Motion-only, 14 feats)** | **0.8203** | **0.5204** | **0.4207** | 0.7540 | **0.6688** | **0.8846** |
| **LR (HR-only, 16 feats)** | 0.6972 | 0.2181 | 0.2614 | 0.6474 | 0.5580 | 0.7447 |
| **LR (Motion+HR, 30 feats)** | 0.8194 | 0.5151 | 0.4138 | 0.7500 | 0.6645 | 0.8796 |
| **RF (Motion+HR, 30 feats)** | 0.8221 | 0.4634 | 0.3808 | 0.6250 | 0.6718 | 0.8795 |

- **Paired Bootstrap [LR(30) - Best Single]**: Delta PR-AUC = **+0.0679** (95% CI: [+0.0340, +0.1032], significant); Delta ROC-AUC = -0.0308 (95% CI: [-0.0606, -0.0015]).
- **Paired Bootstrap [LR(30) - LR(Motion-only)]**: Delta PR-AUC = **-0.0051** (95% CI: [-0.0196, +0.0116], not significant). Motion features dominate wake prediction.

## 3. Nested Threshold Tuning & Rate Bias Checks
- **Nested Tuning (Inner 4-fold GroupKFold)**: Optimizing threshold on training folds yields Wake F1 = **0.4798** (+6.6 pts) and Macro-F1 = **0.7190** for LR (chosen thresholds 0.74–0.90); Wake F1 = **0.4353** (+5.4 pts) for RF (thresholds 0.24–0.39).
- **Rate Bias (Non-50Hz Subjects)**: Training on 19 50Hz vs all 21 subjects shows **no evidence of rate bias** on LR for `5383425` (Delta PR=-0.013), `8258170` (Delta PR=-0.009), or `9618981` (Delta PR=+0.003).

## 4. Verification Outputs
```
$ python -m pytest -q -rs
............................s.............s................              [100%]
SKIPPED [1] tests\test_features.py:437: truncated or missing download
SKIPPED [1] tests\test_loader.py:315: truncated or missing download
59 passed, 2 skipped in 37.11s
$ python -m ruff check .
All checks passed!
```
