"""Tests for XGBoost Sleep Stager v0.2 model, registry, and inference."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from noctiband.features.epoch_features import MODEL_FEATURES
from noctiband.inference.predict import load_inference_model, predict_night
from noctiband.models.manifest import ModelManifest
from noctiband.models.registry import build_model, get_estimator, list_estimators
from noctiband.models.xgboost_model import (
    build_xgboost_pipeline,
    cross_validate_xgboost,
    evaluate_predictions,
    save_xgboost_artifact,
    train_xgboost,
    tune_wake_threshold,
)
from noctiband.types import NightData, NightPrediction

ROOT = Path(__file__).resolve().parent.parent
XGB_JOBLIB_PATH = ROOT / "models" / "noctiband_xgb_v0.2.joblib"
XGB_META_PATH = ROOT / "models" / "noctiband_xgb_v0.2_meta.json"


@pytest.fixture
def synthetic_features_df() -> pd.DataFrame:
    """Create a minimal synthetic feature DataFrame with 3 subjects."""
    rng = np.random.default_rng(42)
    n_rows = 60
    data = {feat: rng.normal(size=n_rows) for feat in MODEL_FEATURES}
    data["subject_id"] = ["sub_1"] * 20 + ["sub_2"] * 20 + ["sub_3"] * 20
    # Ensure both classes 0 and 1 are reliably present in each subject partition
    labels = []
    for _ in range(3):
        labels.extend([0] * 4 + [1] * 16)
    data["label_binary"] = labels
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_8h_night() -> NightData:
    """Create a synthetic 8-hour night with 50 Hz motion and 1 Hz HR."""
    n_epochs = 960  # 8 hours = 480 min = 960 epochs of 30s
    total_seconds = n_epochs * 30

    t_hr = np.arange(0, total_seconds, 1.0)
    hr_vals = 60.0 + 5.0 * np.sin(2 * np.pi * t_hr / 3600.0)
    hr_df = pd.DataFrame({"timestamp": t_hr, "hr": hr_vals})

    t_mot = np.arange(0, total_seconds, 0.02)
    x = 0.05 * np.sin(2 * np.pi * t_mot / 10.0)
    y = 0.05 * np.cos(2 * np.pi * t_mot / 10.0)
    z = 0.98 + 0.02 * np.sin(2 * np.pi * t_mot / 5.0)
    mot_df = pd.DataFrame({"timestamp": t_mot, "x": x, "y": y, "z": z})

    return NightData(
        subject="test_xgb_subject",
        night="1",
        hr=hr_df,
        motion=mot_df,
        labels=None,
        rec_start=0.0,
    )


def test_registry_integration():
    """Verify xgboost and xgb are registered and accessible via get_estimator."""
    estimators = list_estimators()
    assert "xgboost" in estimators
    assert "xgb" in estimators

    est = get_estimator("xgboost")
    assert isinstance(est, XGBClassifier)
    assert est.n_estimators == 100
    assert est.max_depth == 4


def test_build_model_xgboost():
    """Verify build_model builds valid pipeline for xgboost."""
    pipe = build_model("xgboost", {"max_depth": 3, "n_estimators": 50})
    assert "impute" in pipe.named_steps
    assert "scale" in pipe.named_steps
    assert "clf" in pipe.named_steps
    assert isinstance(pipe.named_steps["clf"], XGBClassifier)
    assert pipe.named_steps["clf"].max_depth == 3
    assert pipe.named_steps["clf"].n_estimators == 50


def test_pipeline_fit_and_evaluate(synthetic_features_df):
    """Verify XGBoost pipeline fits, predicts, and generates evaluation metrics."""
    pipe = build_xgboost_pipeline({"n_estimators": 10, "max_depth": 2})
    X = synthetic_features_df[MODEL_FEATURES].to_numpy()
    y = synthetic_features_df["label_binary"].to_numpy()

    pipe.fit(X, y)
    probs = pipe.predict_proba(X)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = pipe.predict(X)
    metrics = evaluate_predictions(y, preds, y_prob=probs[:, 0])

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "cohen_kappa" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert "Wake" in metrics["per_class"]
    assert "Sleep" in metrics["per_class"]
    assert "wake_roc_auc" in metrics
    assert "wake_pr_auc" in metrics
    assert "confusion_matrix" in metrics


def test_tune_wake_threshold():
    """Verify optimal threshold search logic."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    # Higher wake probability for wake epochs
    p_wake = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3, 0.05, 0.15, 0.2, 0.1])
    best_th, best_f1 = tune_wake_threshold(y_true, p_wake)

    assert 0.10 <= best_th <= 0.90
    assert best_f1 > 0.80  # Separation is strong


def test_cross_validation_participant_leakage_free(synthetic_features_df):
    """Verify GroupKFold cross-validation groups strictly by subject."""
    cv_res = cross_validate_xgboost(
        synthetic_features_df,
        features=MODEL_FEATURES,
        group_col="subject_id",
        target_col="label_binary",
        params={"n_estimators": 5, "max_depth": 2},
        n_splits=3,
        tune_threshold=True,
    )

    assert cv_res["n_splits"] == 3
    assert cv_res["n_subjects"] == 3
    assert len(cv_res["per_fold_metrics"]) == 3
    assert "pooled_metrics" in cv_res
    assert len(cv_res["oof_preds"]) == len(synthetic_features_df)


def test_train_and_save_artifact(synthetic_features_df, tmp_path):
    """Verify model training and serialization to joblib and json manifest."""
    joblib_out = tmp_path / "test_xgb.joblib"
    meta_out = tmp_path / "test_xgb_meta.json"

    pipe, opt_th, trn_metrics = train_xgboost(
        synthetic_features_df,
        features=MODEL_FEATURES,
        params={"n_estimators": 5, "max_depth": 2},
    )

    assert opt_th is not None
    assert "accuracy" in trn_metrics

    save_xgboost_artifact(
        pipeline=pipe,
        features=MODEL_FEATURES,
        optimal_threshold=opt_th,
        smoothing_window=1,
        output_joblib_path=joblib_out,
        output_meta_path=meta_out,
    )

    assert joblib_out.exists()
    assert meta_out.exists()

    # Load and validate manifest
    manifest = ModelManifest.from_file(meta_out)
    manifest.validate()
    assert manifest.version == "v0.2"
    assert manifest.n_features == len(MODEL_FEATURES)


def test_frozen_v0_2_artifacts_exist_and_valid():
    """Verify that models/noctiband_xgb_v0.2.joblib and its manifest exist and pass checks."""
    assert XGB_JOBLIB_PATH.exists(), f"Missing frozen model: {XGB_JOBLIB_PATH}"
    assert XGB_META_PATH.exists(), f"Missing manifest: {XGB_META_PATH}"

    manifest = ModelManifest.from_file(XGB_META_PATH)
    manifest.validate()

    assert manifest.model_name == "NoctiBand Sleep Stager XGBoost"
    assert manifest.version == "v0.2"
    assert manifest.n_features == 30
    assert 0.0 <= manifest.optimal_wake_threshold <= 1.0

    artifact = load_inference_model(XGB_JOBLIB_PATH)
    assert "pipeline" in artifact
    assert "features" in artifact
    assert "optimal_threshold" in artifact


def test_predict_night_with_xgboost_model(synthetic_8h_night):
    """Verify that predict_night seamlessly runs inference using noctiband_xgb_v0.2.joblib."""
    pred = predict_night(
        synthetic_8h_night,
        model_path=XGB_JOBLIB_PATH,
        allow_partial=True,
    )

    assert isinstance(pred, NightPrediction)
    assert len(pred.epoch_labels) == 960
    assert len(pred.probs) == 960
    assert len(pred.coverage) == 960
    assert np.all(np.isin(pred.epoch_labels, [0, 1]))
    assert pred.qc_summary["n_epochs"] == 960
