# NoctiBand Model Card: v0.1

## Model Overview
- **Name**: NoctiBand Sleep Stager v0.1
- **Architecture**: `SimpleImputer(median)` -> `StandardScaler` -> `LogisticRegression(C=1.0)`.
- **Target**: Binary 30-second epoch sleep staging (`0: Wake`, `1: Sleep`).
- **Post-Processing**: Threshold $\theta = 0.85$, median smoothing ($k=1$).
- **Feature Space**: 30 device-agnostic features (14 accelerometer, 16 heart rate dynamics).

## Training Cohort
- **Dataset**: Apple Watch / Polysomnography Benchmark (Dataset v2).
- **Participants**: 22 curated subjects with $\ge 95\%$ signal coverage.
- **Sample Size**: 18,187 epochs (1,516 Wake).
- **Cohort Checksum**: `e130cce7859e386a`.

## Test Split Performance (Held-Out Sequestered Test Split)
Evaluated strictly once on the 4 sequestered test participants (`9106476, 1455390, 5797046, 4426783`):

| Metric | Measured Value |
|:---|:---:|
| **Test Epochs ($N$)** | 3,835 |
| **Test Wake Epochs** | 320 (8.34%) |
| **Accuracy** | **0.9301** |
| **Balanced Accuracy** | **0.6466** |
| **Macro-F1** | **0.6926** |
| **Wake-F1** | **0.4224** |
| **Cohen's Kappa** | **0.3909** |
| **Wake ROC-AUC** | **0.7709** |
| **Wake PR-AUC** | **0.4826** |

### Confusion Matrix (Test Split)
```
                Predicted Wake (0)    Predicted Sleep (1)
True Wake (0)              98                   222
True Sleep (1)             46                  3469
```

## Intended Use & Boundaries
- Intended for nocturnal sleep wellness tracking and sleep architecture estimation.
- Non-diagnostic consumer wellness tool; not a medical device.
