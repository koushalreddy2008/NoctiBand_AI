# Phase 10 Report: Provenance Reconciliation, Rate Bias & Feature Experiment

## 1. Discrepancies & Corrected Numbers vs Phase 7–9
- **3-Class LR & RF Balanced Accuracy**: Phase 7 reported `0.6208` (LR) and `0.4854` (RF), which match the pooled confusion matrices (`[[883,395,238],[1350,8201,3220],[327,1086,2487]]` and `[[442,1010,64],[205,11958,608],[50,2960,890]]`). Phase 9's `0.4496`/`0.4357` were unweighted fold-means; the pooled matrix values are correct.
- **Majority 3-Class Macro-F1**: Misquoted in Phase 7 as `0.3016`; corrected to **0.2750** (traceable directly to matrix with class 1 F1 = 0.8251).
- **Per-Subject Phase 7 Metrics**: Phase 7's per-subject numbers (`5383425`: 0.644, `8258170`: 0.624, `9618981`: 0.684) were binary balanced accuracies, not 3-class.
- **Transposed PR-AUC (5383425 vs 8258170)**: Phase 9 text inverted the IDs. True measured values: `5383425` (Wake=40) LR PR-AUC = **0.7921**; `8258170` (Wake=95) LR PR-AUC = **0.4273**.
- **Cohort Provenance**: Phase 8 table (22 subjects, 18,187 epochs, 1,516 wake) is the true cohort; Phase 9 chat table had 24 hallucinated subjects (`7749105` failed coverage guard at 22.3% HR / 13.6% motion and was properly excluded).
- **Tuned Threshold Metrics (LR & RF)**: Measured from OOF parquet: LR Tuned Acc=0.9262, BalAcc=0.6956, Macro-F1=0.7232; RF Tuned Acc=0.9151, BalAcc=0.6776, Macro-F1=0.6947.

## 2. Rate-Bias Protocol (19 50Hz vs 20 Size-Matched 19-Subject Null Sets)
- `5383425` (10Hz): LR ROC=0.9676 [0.929, 0.992] vs null 0.9722 [0.968, 0.979] (**evidence of rate bias**); RF ROC=0.9490 vs null 0.9561 [0.935, 0.967] (**no evidence of rate bias**).
- `8258170` (10Hz): LR ROC=0.7913 vs null 0.8060 [0.774, 0.825]; RF ROC=0.6651 vs null 0.6584 [0.594, 0.750] (**no evidence of rate bias**).
- `9618981` (33Hz): LR ROC=0.9064 vs null 0.9053 [0.864, 0.912]; RF ROC=0.9050 vs null 0.9020 [0.875, 0.931] (**no evidence of rate bias**).

## 3. Feature Experiment (Within-Night Normalized & Windows w=21, 41)
- Baseline Motion-only (14 feats): ROC-AUC = 0.8203, PR-AUC = 0.5204, Median Subj ROC = 0.8846.
- Extended Motion-only (28 feats): ROC-AUC = 0.8626, PR-AUC = 0.5934, Median Subj ROC = 0.9203.
- **Paired Subject Bootstrap [Extended - Baseline]**: Delta PR-AUC = **+0.0729** (95% CI: [+0.0391, +0.1085]); Delta ROC-AUC = **+0.0423** (95% CI: [+0.0095, +0.0756]) (both statistically significant).
*Caveat: 22 subjects, 1 night each, Apple Watch wrist sensor. 4 test subjects untouched.*

## 4. Verification Outputs
```
$ python -m pytest -q -rs
............................s.............s......................        [100%]
63 passed, 2 skipped in 15.14s
$ python -m ruff check .
All checks passed!
```
