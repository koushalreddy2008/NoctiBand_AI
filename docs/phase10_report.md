# Phase 10 Report: Provenance Reconciliation, Rate Bias & Feature Experiment

## 1. Discrepancies & Corrected Metrics vs Phase 7–9
- **3-Class LR & RF Bal Acc**: Phase 7 reported `0.6208` (LR) and `0.4854` (RF), exactly matching pooled confusion matrices (`[[883,395,238],[1350,8201,3220],[327,1086,2487]]` and `[[442,1010,64],[205,11958,608],[50,2960,890]]`). Phase 9's `0.4496`/`0.4357` were fold-means; pooled matrix values are correct.
- **Majority 3-Class Macro-F1**: Misquoted in Phase 7 as `0.3016`; corrected to **0.2750** (Class 1 F1 = 0.8251).
- **Per-Subject Phase 7 Metrics**: Quoted values (`5383425`: 0.644, `8258170`: 0.624, `9618981`: 0.684) were binary balanced accuracies, not 3-class.
- **Transposed PR-AUC**: Phase 9 text transposed `5383425` and `8258170`. True measured values: `5383425` (Wake=40) LR PR-AUC = **0.7921**; `8258170` (Wake=95) LR PR-AUC = **0.4273**.
- **Cohort Provenance**: 22 subjects (18,187 epochs, 1,516 wake). `7749105` failed coverage gate (22.3% HR / 13.6% motion) and was excluded. Small-sample caveat: subject `1818471` has only 10 wake epochs.

## 2. Pooled Matrix-Consistent Metrics (18,187 Epochs)
| Model | Acc | Bal Acc | Macro-F1 | Kappa | Wake F1 | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Weak Motion Baseline (`mag_mean`) | 0.6700 | 0.4875 | 0.4582 | -0.0117 | 0.1194 | 0.5601 | 0.1698 |
| Reference Motion Baseline (`enmo_std_w11`) | 0.7864 | 0.7695 | 0.6198 | 0.2825 | 0.3688 | 0.8508 | 0.4474 |
| LR (Motion-only, 14 feats) | 0.8540 | 0.7540 | 0.6688 | 0.3477 | 0.4207 | 0.8203 | 0.5204 |
| LR (HR-only, 16 feats) | 0.8350 | 0.6474 | 0.5580 | 0.1772 | 0.2614 | 0.6972 | 0.2181 |
| LR (30 feats, Fixed 0.5) | 0.8517 | 0.7500 | 0.6645 | 0.3401 | 0.4138 | 0.8194 | 0.5151 |
| LR (30 feats, Tuned $\theta$) | 0.9262 | 0.6956 | 0.7232 | 0.4507 | 0.4862 | 0.8194 | 0.5151 |
| RF (30 feats, Fixed 0.5) | 0.9297 | 0.6250 | 0.6718 | 0.3521 | 0.3808 | 0.8221 | 0.4634 |
| RF (30 feats, Tuned $\theta$) | 0.9151 | 0.6776 | 0.6947 | 0.3912 | 0.4353 | 0.8221 | 0.4634 |

## 3. Rate-Bias Protocol (Dual Criterion: Metric Outside Null Range & CI Excludes Median)
- `5383425` (10Hz): LR ROC=0.9676 [0.929, 0.992] vs null 0.9722 [0.968, 0.979] (CI includes median $\rightarrow$ **no evidence of rate bias**); RF ROC=0.9490 vs null 0.9561 [0.935, 0.967] (**no evidence of rate bias**).
- `8258170` (10Hz): LR ROC=0.7913 vs null 0.8060 [0.774, 0.825]; RF ROC=0.6651 vs null 0.6584 [0.594, 0.750] (**no evidence of rate bias**).
- `9618981` (33Hz): LR ROC=0.9064 vs null 0.9053 [0.864, 0.912]; RF ROC=0.9050 vs null 0.9020 [0.875, 0.931] (**no evidence of rate bias**).

## 4. Feature Experiment (Within-Night Normalized & Windows w=21, 41)
- Baseline Motion-only (14 feats): ROC-AUC = 0.8203, PR-AUC = 0.5204, Median Subj ROC = 0.8846.
- Extended Motion-only (28 feats): ROC-AUC = 0.8626, PR-AUC = 0.5934, Median Subj ROC = 0.9203.
- Paired Subject Bootstrap [Extended - Baseline]: $\Delta$PR = **+0.0729** (95% CI: [+0.0391, +0.1085]); $\Delta$ROC = **+0.0423** (95% CI: [+0.0095, +0.0756]) (statistically significant).
*Caveat: Full-night z-score/percentile features are non-causal (post-hoc morning analysis only).*
