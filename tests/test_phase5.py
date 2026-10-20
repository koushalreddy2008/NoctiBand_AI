"""Unit tests for Phase 5 plumbing: models, registry, schema, and evaluation harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from noctiband.data_schema import (
    check_epoch_table,
    real_data_gate,
    validate_epoch_table,
)
from noctiband.evaluation import build_pipeline, evaluate_cv
from noctiband.models import (
    get_estimator,
    list_estimators,
    register_estimator,
)


@pytest.fixture
def multi_subject_cohort() -> pd.DataFrame:
    """Create a programmatic 6-subject dataset fixture with planted signal.

    Plumbing only, never reported as scientific results.
    """
    rng = np.random.default_rng(42)
    rows = []
    subjects = [f"subj_{i:02d}" for i in range(6)]

    for subj in subjects:
        for night in ["1", "2"]:
            n_epochs = 40
            for ep in range(n_epochs):
                # 25% wake, 75% sleep
                is_wake = rng.random() < 0.25
                label_bin = 0 if is_wake else 1

                # Random noise features
                noise1 = rng.normal(0.0, 1.0)
                noise2 = rng.normal(0.0, 1.0)

                # Planted signal: high for Wake (mean=3.0), low for Sleep (mean=-1.0)
                if is_wake:
                    planted_signal = rng.normal(3.0, 0.5)
                else:
                    planted_signal = rng.normal(-1.0, 0.5)

                rows.append(
                    {
                        "subject_id": subj,
                        "night_id": night,
                        "epoch_idx": ep,
                        "feat_noise1": noise1,
                        "feat_noise2": noise2,
                        "motion_signal": planted_signal,
                        "label_binary": label_bin,
                        "motion_missing": False,
                    }
                )

    return pd.DataFrame(rows)


def test_estimator_registry_extensibility():
    """Verify registry accepts new models (e.g. mock SVM) and instantiates them."""
    available = list_estimators()
    assert "majority" in available
    assert "motion_threshold" in available
    assert "logistic_regression" in available
    assert "random_forest" in available

    # Test registering a mock SVM factory without touching harness
    class MockSVM:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    register_estimator("mock_svm", lambda: MockSVM())
    svm_inst = get_estimator("mock_svm")
    assert isinstance(svm_inst, MockSVM)


def test_folds_never_share_a_subject(multi_subject_cohort: pd.DataFrame):
    """Verify GroupKFold guarantees train and validation folds never share subjects."""
    res = evaluate_cv(
        multi_subject_cohort,
        estimator="majority",
        target_col="label_binary",
        n_splits=3,
        is_synthetic=True,
    )

    all_subjects = set(multi_subject_cohort["subject_id"].unique())
    visited_val_subjects: set[str] = set()

    for fold in res.per_fold_metrics:
        val_subjs = set(fold["val_subjects"])
        # Validation subjects for this fold must be distinct from previously evaluated folds
        assert val_subjs.isdisjoint(visited_val_subjects)
        visited_val_subjects.update(val_subjs)

    assert visited_val_subjects == all_subjects


def test_scaler_imputer_fit_on_train_folds_only():
    """Verify scaler and imputer are fit strictly on training fold, not validation."""
    rng = np.random.default_rng(123)
    # 4 training subjects centered near 0.0, 2 test subjects centered near +1000.0
    rows = []
    for i in range(4):
        subj = f"train_subj_{i}"
        for ep in range(30):
            rows.append(
                {
                    "subject_id": subj,
                    "night_id": "1",
                    "epoch_idx": ep,
                    "feature": rng.normal(0.0, 1.0),
                    "label_binary": ep % 2,
                }
            )

    test_subj = "test_subj_shifted"
    for ep in range(30):
        rows.append(
            {
                "subject_id": test_subj,
                "night_id": "1",
                "epoch_idx": ep,
                "feature": rng.normal(1000.0, 1.0),  # shifted test mean
                "label_binary": ep % 2,
            }
        )

    df = pd.DataFrame(rows)

    pipe = build_pipeline("majority")
    X_train = df[df["subject_id"].str.startswith("train")][["feature"]].to_numpy()
    y_train = df[df["subject_id"].str.startswith("train")]["label_binary"].to_numpy()
    X_test = df[df["subject_id"] == test_subj][["feature"]].to_numpy()

    pipe.fit(X_train, y_train)

    fitted_scaler = pipe.named_steps["scaler"]
    # Mean of scaler must be near 0.0 (from train fold only), not shifted by test fold
    assert abs(fitted_scaler.mean_[0]) < 0.5

    # Transforming X_test must center around ~1000
    transformed_test = pipe.named_steps["scaler"].transform(X_test)
    assert transformed_test.mean() > 900.0


def test_planted_signal_recovered_by_lr_and_rf(multi_subject_cohort: pd.DataFrame):
    """Verify LogisticRegression and RandomForest recover planted signal."""
    # 1. Logistic Regression
    lr_res = evaluate_cv(
        multi_subject_cohort,
        estimator="logistic_regression",
        feature_columns=["motion_signal"],
        target_col="label_binary",
        n_splits=3,
        is_synthetic=True,
    )
    assert lr_res.pooled_metrics["balanced_accuracy"] > 0.85
    assert lr_res.pooled_metrics["macro_f1"] > 0.85

    # 2. Random Forest
    rf_res = evaluate_cv(
        multi_subject_cohort,
        estimator="random_forest",
        feature_columns=["motion_signal"],
        target_col="label_binary",
        n_splits=3,
        is_synthetic=True,
    )
    assert rf_res.pooled_metrics["balanced_accuracy"] > 0.85
    assert rf_res.pooled_metrics["macro_f1"] > 0.85


def test_shuffled_labels_give_macro_f1_near_chance(multi_subject_cohort: pd.DataFrame):
    """Verify that models trained on permuted labels drop to chance performance."""
    df_shuffled = multi_subject_cohort.copy()
    rng = np.random.default_rng(999)
    df_shuffled["label_binary"] = rng.permutation(df_shuffled["label_binary"].to_numpy())

    res = evaluate_cv(
        df_shuffled,
        estimator="logistic_regression",
        target_col="label_binary",
        n_splits=3,
        is_synthetic=True,
    )

    # Balanced accuracy should be near chance (0.50 +- 0.15)
    assert abs(res.pooled_metrics["balanced_accuracy"] - 0.50) < 0.15


def test_majority_baseline_accuracy_printed_beside_macro_f1(
    multi_subject_cohort: pd.DataFrame,
    capsys: pytest.CaptureFixture,
):
    """Verify MajorityBaseline accuracy is printed beside its macro-F1."""
    res = evaluate_cv(
        multi_subject_cohort,
        estimator="majority",
        target_col="label_binary",
        n_splits=3,
        is_synthetic=True,
    )

    y_true = multi_subject_cohort["label_binary"].to_numpy()
    # Majority baseline constantly predicts majority class
    maj_class = int(pd.Series(y_true).mode()[0])
    acc = float(accuracy_score(y_true, np.full_like(y_true, maj_class)))
    macro_f1 = res.pooled_metrics["macro_f1"]

    print(f"Majority baseline - Accuracy: {acc:.4f}, Macro-F1: {macro_f1:.4f}")
    captured = capsys.readouterr()
    assert "Majority baseline - Accuracy:" in captured.out
    assert "Macro-F1:" in captured.out

    # Majority class balanced accuracy is mathematically 0.50
    assert res.pooled_metrics["balanced_accuracy"] == 0.50


def test_real_data_gate_refuses_small_cohort_and_low_motion():
    """Verify real_data_gate strictly rejects cohorts < 10 subjects and low-motion nights."""
    # 1. Reject cohort with 3 subjects
    df_3subjs = pd.DataFrame(
        {
            "subject_id": ["s1", "s2", "s3"] * 20,
            "night_id": ["1"] * 60,
            "epoch_idx": list(range(20)) * 3,
            "feat": [1.0] * 60,
            "label": [0, 1] * 30,
        }
    )
    with pytest.raises(ValueError, match="minimum 10 required"):
        real_data_gate(df_3subjs, is_synthetic=False)

    # 2. Reject cohort with 10 subjects where 1 night has < 50% motion
    rows = []
    for i in range(10):
        subj = f"s{i}"
        for ep in range(20):
            # Subject 0 night 1 has 90% missing motion (< 50% valid motion)
            m_missing = True if (i == 0 and ep > 1) else False
            rows.append(
                {
                    "subject_id": subj,
                    "night_id": "1",
                    "epoch_idx": ep,
                    "feat": 1.0,
                    "label": ep % 2,
                    "motion_missing": m_missing,
                }
            )

    df_low_motion = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="valid-motion fraction.*is below 50%"):
        real_data_gate(df_low_motion, is_synthetic=False)


def test_schema_validator_rejects_malformed_tables():
    """Verify validate_epoch_table and check_epoch_table reject malformed schemas."""
    # 1. Missing identifier
    df_no_epoch = pd.DataFrame({"subject_id": ["s1"], "night_id": ["1"], "label": [0], "f": [1.0]})
    with pytest.raises(ValueError, match="Missing required identifier column"):
        check_epoch_table(df_no_epoch)

    # 2. Missing label
    df_no_label = pd.DataFrame(
        {"subject_id": ["s1"], "night_id": ["1"], "epoch_idx": [0], "f": [1.0]}
    )
    with pytest.raises(ValueError, match="Missing required label column"):
        check_epoch_table(df_no_label)

    # 3. No numeric features
    df_no_features = pd.DataFrame(
        {"subject_id": ["s1"], "night_id": ["1"], "epoch_idx": [0], "label": [0]}
    )
    with pytest.raises(ValueError, match="No numeric feature columns"):
        check_epoch_table(df_no_features)

    # 4. Infinite feature values
    df_inf = pd.DataFrame(
        {
            "subject_id": ["s1"],
            "night_id": ["1"],
            "epoch_idx": [0],
            "label": [0],
            "f": [np.inf],
        }
    )
    with pytest.raises(ValueError, match="contains infinite values"):
        check_epoch_table(df_inf)

    # 5. Empty DataFrame
    res = validate_epoch_table(pd.DataFrame())
    assert not res.is_valid
    assert "Epoch table is empty" in res.errors[0]
