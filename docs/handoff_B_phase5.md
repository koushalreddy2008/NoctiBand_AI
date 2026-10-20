# Phase 5 Handoff: Estimator Registry & Model Integration

To: Person B (Modeler / Reporting Owner)
From: Person A (Senior ML / Pipeline Engineer)

---

## 1. Estimator Registry Integration

All model evaluation is orchestrated through the decoupled estimator registry in `src/noctiband/models/registry.py`. You do not need to modify the cross-validation harness or pipeline to introduce new models (such as SVM or LightGBM).

### How to Register a New Model (e.g. SVM):
```python
from sklearn.svm import SVC
from noctiband.models import register_estimator

# Register a factory function returning a configured scikit-learn estimator
register_estimator(
    "svm",
    lambda: SVC(
        C=1.0,
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
)
```

### Pre-Registered Estimators:
- `"majority"`: Majority class constant baseline.
- `"motion_threshold"`: Rule-based motion threshold baseline fit on training folds.
- `"logistic_regression"`: Balanced L2-penalized logistic regression.
- `"random_forest"`: Balanced subsample random forest (100 trees).

---

## 2. Evaluation Harness API

The cross-validation harness in `src/noctiband/evaluation/harness.py` handles imputer and scaler fitting strictly on training folds, subject-grouped splitting via `GroupKFold`, and complete metric computation:

```python
from noctiband.evaluation import evaluate_cv

result = evaluate_cv(
    df=epoch_table,
    estimator="svm",              # or pass estimator instance
    target_col="label_binary",    # or "label_3class"
    n_splits=5,
    is_synthetic=True,            # Set False for validated real benchmark cohorts
)

print("Pooled Macro-F1:", result.pooled_metrics["macro_f1"])
print("Pooled Wake-F1:", result.pooled_metrics["wake_f1"])
print("Per-Subject Table:\n", result.per_subject_table)
```
- Unknown epochs (code 5) and NaN targets are automatically dropped prior to training and scoring.
- Pipeline structure: `SimpleImputer(median) -> StandardScaler() -> Estimator`.
- Real-data gate: Automatically enforces `n_subjects >= 10` and `valid-motion >= 50%` per night when `is_synthetic=False`.
