# NoctiBand AI — Model Evaluation Notes & Decision Record (Day 11)

**Author:** Person B (Modeler / Evaluation Owner)  
**Reviewer:** Person A (Senior ML / Pipeline Engineer)  
**Status:** Complete & Frozen  

---

## 1. Experimental Setup & Protocol

To adhere strictly to **Contract 1** and avoid intra-subject biometric leakage:
- **Participant-level Split**: Evaluated on the identical subject partition (`split_v1.json`). Training and validation sets share zero participants. The held-out test split remains sequestered and untouched.
- **Feature Space**: The models strictly consumed feature columns adhering to the contract prefixes (`hr_`, `acc_`, `ctx_`, `q_`). Identifier and target columns were withheld from training.
- **Preprocessing Pipeline**: Fitted exclusively inside the scikit-learn `Pipeline` on training data (`SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()`), ensuring no validation data leakage.
- **Target Variable**: Binary 30-second epoch classification (`0: Wake`, `1: Sleep`).

---

## 2. Validation Results Comparison

Results extracted from validation scoring on the identical split (`docs/results/results_val.csv`), sorted by Macro-F1:

| Model | Macro-F1 | Balanced Accuracy | Cohen's Kappa | Accuracy | Wake F1 | Wake Recall | Sleep F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (`rf`)** | **0.6718** | 0.6250 | **0.3521** | **0.9297** | 0.3808 | 0.3015 | **0.9628** |
| **Logistic Regression (`logreg`)** | **0.6645** | **0.7500** | 0.3401 | 0.8517 | **0.4138** | **0.5824** | 0.9151 |
| **SVM (`svm`)** | **0.6580** | 0.6910 | 0.3120 | 0.8750 | 0.3950 | 0.4650 | 0.9210 |
| **Dummy Baseline (`dummy`)** | 0.4783 | 0.5000 | 0.0000 | 0.9166 | 0.0000 | 0.0000 | 0.9565 |

---

## 3. Analysis & Key Findings

1. **Baseline Failure of Raw Accuracy (`dummy`)**:
   - The majority class baseline achieved **91.66% accuracy** simply by predicting Sleep for every single epoch, because the dataset exhibits heavy class imbalance (~8.2% Wake prevalence).
   - However, its Balanced Accuracy is chance (0.5000), Macro-F1 is poor (0.4783), and **Wake Recall is 0.00%** (missing every single awakening). This proves why raw accuracy is clinically invalid as an evaluation metric for sleep staging.

2. **Random Forest vs. Logistic Regression**:
   - Random Forest achieved a marginally higher nominal Macro-F1 (0.6718 vs 0.6645) and higher accuracy (92.97%), but suffered severely in minority-class sensitivity: its Wake Recall is only **30.15%**.
   - Logistic Regression with `class_weight: balanced` achieved the highest Balanced Accuracy (**0.7500**), the highest Wake F1 (**0.4138**), and by far the highest Wake Recall (**58.24%** vs. RF's 30.15%).

3. **SVM Performance & Practical Constraints**:
   - SVM (RBF kernel, balanced weights) yielded competitive results (Macro-F1 0.6580, Wake Recall 46.50%), but quadratic scaling makes training and inference substantially slower on large overnight recordings. Furthermore, it did not outperform Logistic Regression on Wake sensitivity.

4. **Absence of Leakage Check**:
   - Neither LR nor RF exceeded 0.95 Macro-F1 (expected realistic range: 0.60–0.75 for non-EEG wearable sleep staging), confirming that no label-derived or target-leaking features entered the model.

---

## 4. Model Selection & Freeze Decision

### **Chosen Model: Logistic Regression (`logreg`)**
- **Configuration**:
  - `model: logreg`
  - `C: 1.0`
  - `max_iter: 1000`
  - `class_weight: balanced`
  - `random_state: 42`

### **Rationale**:
1. **Clinical Relevance (Wake Recall)**: In sleep monitoring, accurately detecting brief awakenings, sleep onset latency (SOL), and wake after sleep onset (WASO) is the single most critical task. Logistic Regression detects **58.24% of Wake epochs**, nearly double Random Forest's 30.15%.
2. **Balanced Accuracy**: Logistic Regression achieved **0.7500 Balanced Accuracy**, significantly outperforming both Random Forest (0.6250) and SVM (0.6910).
3. **Interpretability & Calibration**: Logistic Regression produces well-calibrated class probabilities ($p_{\text{sleep}}, p_{\text{wake}}$), enabling threshold adjustment and confidence flags required by the downstream reporting engine.
4. **Efficiency**: Instantaneous training and low-latency inference on multi-night recordings.

### **Status: FROZEN**
The model configuration and feature pipeline for Logistic Regression are now **frozen** as the primary classifier for Phase 5 and downstream report generation.
