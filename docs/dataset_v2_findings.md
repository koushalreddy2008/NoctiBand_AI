# Dataset v2 Findings: Apple Watch / PSG Sleep Dataset

## 1. Overview
- **Dataset Source**: PhysioNet / Apple Watch Sleep Study (ODC-By license).
- **Directory**: `data/raw_v2/`
- **Total Files**: 125 files verified against `SHA256SUMS.txt` (124 signal files + `LICENSE.txt`). 0 bad.
- **Subject Count**: 31 subjects, exactly 1 night per subject.
- **Files per Subject**: 4 files per subject (`heart_rate/<id>_heartrate.txt`, `labels/<id>_labeled_sleep.txt`, `motion/<id>_acceleration.txt`, `steps/<id>_steps.txt`).

---

## 2. Signal Characteristics (Measured Facts)

| Signal | Separator | Columns | Units | Sampling Rate ($f_s$) | Total Rows | Total NaNs | Duplicates | Non-Monotonic |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Heart Rate** | Comma (`,`) | `timestamp, heart_rate` | s, bpm | Non-uniform (~0.05–0.2 Hz) | 254,425 | 0 | 33,526 | 6 |
| **Labels** | Whitespace (`\s+`) | `timestamp, stage` | s, integer code | Fixed 30 s epochs (1/30 Hz) | 27,211 | 0 | 0 | 0 |
| **Motion** | Whitespace (`\s+`) | `timestamp, x, y, z` | s, $g$ | ~50 Hz (28 subj), ~10 Hz (2 subj), ~33 Hz (1 subj) | 51,819,151 | 0 | 0 | 293 |
| **Steps** | Comma (`,`) | `timestamp, steps` | s, step count | 10-minute bins (~1/600 Hz) | 33,459 | 0 | 0 | 0 |

### Time Reference & Alignment
- **Labels**: Timestamps start at `0` and step strictly by `+30` s ($0, 30, 60, \dots$). The duration of the labeled night is $N_{\text{epochs}} \times 30$ seconds.
- **Heart Rate & Motion**: Timestamps are floating-point seconds relative to the start of the polysomnography (PSG) recording ($t = 0$). Negative timestamps represent pre-study recording while the device was worn prior to lights out (up to 7 days prior). Positive timestamps during $[0, N_{\text{epochs}} \times 30]$ align directly with the sleep scoring window.

---

## 3. Label Distribution & Canonical Schema Mapping

The raw label codes in `*_labeled_sleep.txt` follow standard PSG sleep staging annotations:

| Raw Code | Stage Name | Count | Fraction of Scored | Canonical Code (`label`) | Canonical Binary (`label_binary`) | Canonical 3-Class (`label_3class`) |
|:---|:---|:---|:---|:---|:---|:---|
| **`0`** | Wake | 2,429 | 9.07% | `0` (Wake) | `0` (Wake) | `0` (Wake) |
| **`1`** | N1 (NREM1) | 1,821 | 6.80% | `1` (N1) | `1` (Sleep) | `1` (NREM) |
| **`2`** | N2 (NREM2) | 12,954 | 48.38% | `2` (N2) | `1` (Sleep) | `1` (NREM) |
| **`3`** | N3 (NREM3) | 3,329 | 12.43% | `3` (N3) | `1` (Sleep) | `1` (NREM) |
| **`4`** | N4 (Slow Wave) | 356 | 1.33% | `3` (Merged N3 under AASM) | `1` (Sleep) | `1` (NREM) |
| **`5`** | REM | 5,884 | 21.98% | `4` (REM) | `1` (Sleep) | `2` (REM) |
| **`-1`** | Unscored / Unknown | 438 | — | `NaN` (`label_unknown=True`) | `NaN` | `NaN` |
| **Total** | | **27,211** | | | | |

- Total scored epochs: 26,773.
- REM percentage of sleep: $5884 / 24344 = 24.17\%$ (physiologically consistent with healthy adult sleep architecture).

---

## 4. Size Outliers Analysis

| Subject ID | Motion File Size | Motion Rows | Time Range ($t_{\min}, t_{\max}$) | Median $\Delta t$ ($f_s$) | Labeled Epochs | Night Duration | Labeled Motion Cov | Labeled HR Cov | Root Cause / Status |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **`9618981`** | 380.9 MB | 8,980,334 | -26,379.0 s to 248,054.9 s | 0.0299 s (~33.4 Hz) | 945 | 7.88 h | **1.0000** | **1.0000** | Device kept recording for ~3 days post-study. Night coverage is 100% complete. |
| **`5383425`** | 12.6 MB | 298,000 | -23.9 s to 29,762.0 s | 0.0999 s (~10.0 Hz) | 978 | 8.15 h | **1.0000** | **1.0000** | Configured to record at 10 Hz nominal rate instead of 50 Hz. Night coverage is 100%. |
| **`8258170`** | 12.5 MB | 295,000 | -10.4 s to 29,925.7 s | 0.0999 s (~10.0 Hz) | 972 | 8.10 h | **1.0000** | **1.0000** | Configured to record at 10 Hz nominal rate instead of 50 Hz. Night coverage is 100%. |
| **`7749105`** | 17.8 MB | 422,000 | -220,620.4 s to 29,495.6 s | 0.0199 s (~50.2 Hz) | 960 | 8.00 h | **0.1365** | **0.2229** | Severe data truncation: motion stopped at $t=4024.5$ s (~1.1 h in). **Flagged & Excluded**. |

---

## 5. Subject Coverage Audit (Labeled Night Threshold: $\ge 0.95$)

Epoch coverage was measured across the labeled night $[0, N_{\text{epochs}} \times 30]$ for each subject.

- **Passed ($\ge 0.95$ HR & Motion)**: 26 subjects (`1360686`, `1455390`, `1818471`, `2638030`, `3509524`, `3997827`, `4018081`, `4314139`, `4426783`, `46343`, `5132496`, `5383425`, `5498603`, `5797046`, `6220552`, `759667`, `781756`, `8000685`, `8173033`, `8258170`, `844359`, `8686948`, `8692923`, `9106476`, `9618981`, `9961348`).
- **Flagged (< 0.95 HR or Motion)**: 5 subjects:
  1. `1066528`: HR coverage = 91.91%, Motion coverage = 86.97%
  2. `1449548`: HR coverage = 100.0%, Motion coverage = 92.75%
  3. `2598705`: HR coverage = 96.95%, Motion coverage = 89.32%
  4. `7749105`: HR coverage = 22.29%, Motion coverage = 13.65% (severe mid-night drop)
  5. `8530312`: HR coverage = 100.0%, Motion coverage = 93.79%

---

## 6. Decision on Steps Feature

**Decision: Steps will NOT be used as a feature.**
- **Missingness**: 8 of 31 subjects (25.8%) have zero step entries during the entire labeled night (`1066528`, `4426783`, `5132496`, `759667`, `781756`, `8258170`, `9618981`, `9961348`).
- **Data Policy & Device-Agnostic Requirement**: Steps are an algorithmically derived Apple Watch proprietary metric, not a raw measured biosignal. The target deployment is a custom headband equipped solely with raw accelerometer and PPG (HR). Using proprietary watch pedometer values introduces device dependence and feature leakage.
