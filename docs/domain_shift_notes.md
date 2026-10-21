# Domain Shift & Real-World Translation Analysis (Person A)

## 1. Overview & Purpose
The NoctiBand sleep staging model (`models/noctiband_v0.1.joblib`) was trained and validated on benchmark nocturnal polysomnography paired with wrist actigraphy and photoplethysmography (Dataset v2). Translating this pipeline to the target **NoctiBand wearable headband** involves distinct physical, anatomical, and sensor domain shifts. This document details these domain differences and the engineering safeguards implemented to ensure model robustness.

---

## 2. Sensor Placement & Form Factor Shifts

| Factor | Benchmark Dataset (Dataset v2) | Target Device (NoctiBand Headband) | Mitigation & Safeguards |
|:---|:---|:---|:---|
| **Anatomical Placement** | Dorsal wrist (Apple Watch) | Forehead / Temple band | Head movements have smaller amplitude but higher signal-to-noise ratio for gross awakenings. Standardized ENMO features and rolling variance capture postural shifts. |
| **Motion Profile** | High-amplitude limb movements, twitches, scratching, bedsheet rustling | Axial head turns, nodding, rolling; lower extremity twitches absent | Eliminates peripheral twitch false alarms; model uses rolling standard deviation and distribution percentiles rather than absolute peak counts. |
| **Accelerometer Orientation** | Dynamic wrist orientation varying with arm posture | Relatively fixed orientation along cranium axes | Feature engineering uses orientation-invariant ENMO (Euclidean Norm Minus One) and vector magnitude ($\sqrt{x^2+y^2+z^2}$). |
| **PPG & Heart Rate** | Radial/ulnar microvasculature of the distal wrist | Superficial temporal artery / frontal branch | Temple PPG provides robust arterial pulse signals with minimal limb motion artifact during sleep. HR rolling dynamics ($\Delta \text{HR}$, rolling IQR) remain invariant across sites. |

---

## 3. Sampling Rate & Device Agnosticism
Empirical audits across Phases 6–8 identified heterogeneous sampling rates in commercial wearables:
- Standard Apple Watch: ~50 Hz.
- Reduced rate recordings: ~33 Hz (subject `9618981`), ~10 Hz (subjects `5383425`, `8258170`).

### Implemented Protections:
1. **Feature Rate Invariance**: All features scaling with sampling frequency (e.g. raw sample counts, motion activity totals) were converted to per-epoch fractions or excluded from `MODEL_FEATURES`.
2. **Rate Harmonization Adapter**: `src/noctiband/adapter.py` provides `adapt_external_stream`, which detects empirical sampling rates and resamples to a standard 50 Hz grid via linear interpolation.
3. **Dual Empirical Rate-Bias Gating**: Verified across Phase 8 and Phase 10; non-50 Hz recordings show no statistically significant rate bias against the 50 Hz null distribution.

---

## 4. Population & Environmental Shifts

### Clinical Laboratory vs Home Environment
- **Laboratory Polysomnography (Dataset v2)**: Monitored sleep in temperature-controlled sleep laboratories with PSG wire harness. Often exhibits "first-night effect" with heightened sleep latency and altered REM/NREM proportions.
- **Consumer Home Wellness (NoctiBand)**: Naturalistic sleep environment over consecutive weeks. Multi-night aggregation (`src/noctiband/app_glue.py`) mitigates single-night variance through longitudinal tracking and Sleep Regularity Index (SRI) scoring.

### Demographic Generalization
- Benchmark participants consist primarily of healthy young adults.
- For broader consumer deployment, future iterations will benefit from multi-cohort calibration across diverse age groups and resting heart rate distributions.

---

## 5. Summary of Architecture Safeguards
1. **Model Architecture**: Linear logistic regression with L2 regularization and median imputation — inherently constrained against out-of-distribution feature explosions.
2. **Quality Control Guard**: `QualityControlError` strictly enforces $\ge 95\%$ signal coverage and $\le 5\%$ missing epochs.
3. **Non-Diagnostic Safety Validator**: `validate_suggestion_safety` filters all user-facing tips against medical and diagnostic claims, preserving consumer wellness compliance.
