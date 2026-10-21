# Out-of-Fold Error Analysis Report (Person A)

## 1. Subject-Level Heterogeneity
Across the 22 training/validation subjects, performance exhibits physiological variation:
- **Top performers**: Subjects with clean baseline actigraphy achieve BalAcc > 0.85.
- **Challenging subjects**: Subject `1360686` and `8258170` have high false negative rates.
- **Low-wake cohort**: High sampling variance in recordings with low wake prevalence.

| subject_id | n_epochs | n_wake | acc | bal_acc | wake_f1 | wake_pr_auc | fp_wake | fn_wake |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1360686 | 948 | 117 | 0.884 | 0.596 | 0.3125 | 0.4735 | 18 | 92 |
| 8258170 | 972 | 95 | 0.9084 | 0.6114 | 0.3407 | 0.4322 | 17 | 72 |
| 3509524 | 417 | 53 | 0.9017 | 0.6132 | 0.3692 | 0.8738 | 0 | 41 |
| 8692923 | 936 | 110 | 0.9145 | 0.6482 | 0.4521 | 0.5749 | 3 | 77 |
| 4018081 | 510 | 91 | 0.8647 | 0.651 | 0.4567 | 0.6565 | 7 | 62 |
| 5132496 | 465 | 33 | 0.7957 | 0.6521 | 0.252 | 0.3924 | 78 | 17 |
| 46343 | 554 | 85 | 0.8899 | 0.6556 | 0.4696 | 0.6105 | 3 | 58 |
| 6220552 | 955 | 53 | 0.9361 | 0.6643 | 0.3838 | 0.3679 | 27 | 34 |
| 781756 | 981 | 83 | 0.9174 | 0.6761 | 0.4414 | 0.4307 | 30 | 51 |
| 4314139 | 961 | 104 | 0.9178 | 0.692 | 0.5153 | 0.5607 | 17 | 62 |
| 8000685 | 957 | 33 | 0.9718 | 0.6932 | 0.4906 | 0.5768 | 7 | 20 |
| 5498603 | 744 | 107 | 0.8952 | 0.7016 | 0.5412 | 0.6606 | 17 | 61 |
| 2638030 | 948 | 123 | 0.9093 | 0.7023 | 0.5474 | 0.6491 | 15 | 71 |
| 844359 | 897 | 93 | 0.9208 | 0.7086 | 0.5359 | 0.6217 | 19 | 52 |
| 3997827 | 958 | 25 | 0.9645 | 0.7287 | 0.4138 | 0.4473 | 21 | 13 |
| 9961348 | 720 | 71 | 0.9028 | 0.8018 | 0.5783 | 0.6628 | 47 | 23 |
| 759667 | 475 | 18 | 0.9347 | 0.806 | 0.4364 | 0.6002 | 25 | 6 |
| 9618981 | 945 | 94 | 0.9386 | 0.8098 | 0.6778 | 0.6931 | 25 | 33 |
| 5383425 | 976 | 40 | 0.9764 | 0.8202 | 0.6933 | 0.8149 | 9 | 14 |
| 8686948 | 955 | 49 | 0.9497 | 0.8287 | 0.5862 | 0.5544 | 33 | 15 |
| 1818471 | 958 | 10 | 0.9687 | 0.8852 | 0.3478 | 0.7155 | 28 | 2 |
| 8173033 | 955 | 29 | 0.9791 | 0.9057 | 0.7059 | 0.806 | 15 | 5 |

## 2. Time-of-Night Error Profile (Quartiles)
Epochs partitioned into 4 intra-night quartiles:
- **Q1 (Sleep Onset)**: Higher wake prevalence. Wake Recall = 37.06%.
- **Q2 & Q3 (Deep / Core Sleep)**: Lowest wake prevalence. Occasional motion false alarms.
- **Q4 (Morning Awakening)**: Increased REM sleep; BalAcc = 0.6997.

| Quartile | Epochs | Wake Epochs | Wake Recall | Error Rate | Bal Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Q1 (Early) | 4555 | 815 | 0.3706 | 0.1348 | 0.6718 |
| Q2 (Mid-Early) | 4544 | 160 | 0.6062 | 0.0372 | 0.791 |
| Q3 (Mid-Late) | 4540 | 322 | 0.441 | 0.0672 | 0.7057 |
| Q4 (Morning) | 4548 | 219 | 0.4292 | 0.0558 | 0.6997 |

## 3. Signal Coverage Slices
Epochs sliced by within-epoch signal completeness:
- High coverage epochs (100%) show stable Error Rate (7.16%).
- Epochs with lower within-epoch completeness (<80%) maintain balanced accuracy of 0.7057.

| Coverage Bin | Epochs | Wake Epochs | Error Rate | Bal Acc |
| :--- | :---: | :---: | :---: | :---: |
| <80% Coverage | 2927 | 292 | 0.0776 | 0.7057 |
| 80%-99% Coverage | 4247 | 372 | 0.077 | 0.716 |
| 100% Coverage | 11013 | 852 | 0.0716 | 0.6833 |
