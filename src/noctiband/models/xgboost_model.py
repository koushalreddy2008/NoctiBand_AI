"""XGBoost Sleep Staging Classifier & Training Pipeline (v0.2).

Provides an end-to-end training, cross-validation, and model serialization
engine using XGBClassifier for 30-second epoch sleep staging from actigraphy
and autonomic features. Adheres strictly to participant-level splits with zero
data leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from noctiband.features.epoch_features import MODEL_FEATURES
from noctiband.models.manifest import ModelManifest
from noctiband.models.smoothing import TemporalSmoother, smooth_labels

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COHORT_PATH = ROOT / "data" / "processed" / "cohort_v2_clean.pkl"
DEFAULT_SPLIT_PATH = ROOT / "configs" / "split.json"
DEFAULT_RAW_V2_DIR = ROOT / "data" / "raw_v2"
DEFAULT_OUTPUT_JOBLIB = ROOT / "models" / "noctiband_xgb_v0.2.joblib"
DEFAULT_OUTPUT_META = ROOT / "models" / "noctiband_xgb_v0.2_meta.json"

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def build_xgboost_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """Build a leak-free sklearn Pipeline containing imputer, scaler, and XGBClassifier.

    Parameters
    ----------
    params : dict, optional
        Keyword arguments to pass to the XGBClassifier constructor.

    Returns
    -------
    Pipeline
        Composed sklearn Pipeline.
    """
    from xgboost import XGBClassifier

    merged_params = dict(DEFAULT_XGB_PARAMS)
    if params:
        merged_params.update(params)

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(**merged_params)),
    ])


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculate comprehensive sleep staging classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0 = Wake, 1 = Sleep).
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : np.ndarray, optional
        Predicted wake probabilities (prob[:, 0]).

    Returns
    -------
    dict[str, Any]
        Dictionary containing accuracy, balanced_accuracy, cohen_kappa,
        macro_f1, per-class metrics, PR-AUC, ROC-AUC, and confusion matrix.
    """
    classes = [0, 1]
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=classes))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    wake_prec = float(precision_score(y_true == 0, y_pred == 0, zero_division=0))
    wake_rec = float(recall_score(y_true == 0, y_pred == 0, zero_division=0))
    wake_f1 = float(f1_score(y_true == 0, y_pred == 0, zero_division=0))
    wake_support = int((y_true == 0).sum())

    sleep_prec = float(precision_score(y_true == 1, y_pred == 1, zero_division=0))
    sleep_rec = float(recall_score(y_true == 1, y_pred == 1, zero_division=0))
    sleep_f1 = float(f1_score(y_true == 1, y_pred == 1, zero_division=0))
    sleep_support = int((y_true == 1).sum())

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    metrics: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "cohen_kappa": round(kappa, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            "Wake": {
                "precision": round(wake_prec, 4),
                "recall": round(wake_rec, 4),
                "f1": round(wake_f1, 4),
                "support": wake_support,
            },
            "Sleep": {
                "precision": round(sleep_prec, 4),
                "recall": round(sleep_rec, 4),
                "f1": round(sleep_f1, 4),
                "support": sleep_support,
            },
        },
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        try:
            metrics["wake_roc_auc"] = round(float(roc_auc_score(y_true == 0, y_prob)), 4)
            metrics["wake_pr_auc"] = round(float(average_precision_score(y_true == 0, y_prob)), 4)
        except Exception:
            metrics["wake_roc_auc"] = float("nan")
            metrics["wake_pr_auc"] = float("nan")

    return metrics


def tune_wake_threshold(
    y_true: np.ndarray,
    p_wake: np.ndarray,
    min_th: float = 0.10,
    max_th: float = 0.90,
    n_steps: int = 81,
) -> tuple[float, float]:
    """Find the optimal wake decision threshold maximizing Wake F1.

    Parameters
    ----------
    y_true : np.ndarray
        Binary true labels (0 = Wake, 1 = Sleep).
    p_wake : np.ndarray
        Model output probabilities for Wake (class 0).
    min_th : float, default=0.10
        Lower bound of threshold search range.
    max_th : float, default=0.90
        Upper bound of threshold search range.
    n_steps : int, default=81
        Number of threshold candidates to evaluate.

    Returns
    -------
    tuple[float, float]
        (optimal_threshold, best_wake_f1)
    """
    candidates = np.linspace(min_th, max_th, n_steps)
    best_th = 0.50
    best_f1 = -1.0

    for th in candidates:
        pred = np.where(p_wake >= th, 0, 1)
        wf1 = float(f1_score(y_true == 0, pred == 0, zero_division=0))
        if wf1 > best_f1:
            best_f1 = wf1
            best_th = float(th)

    return round(best_th, 4), round(best_f1, 4)


def cross_validate_xgboost(
    df: pd.DataFrame,
    features: list[str] | None = None,
    group_col: str = "subject_id",
    target_col: str = "label_binary",
    params: dict[str, Any] | None = None,
    n_splits: int = 5,
    tune_threshold: bool = True,
) -> dict[str, Any]:
    """Perform participant-level GroupKFold cross-validation with zero data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Training cohort DataFrame containing features, subject IDs, and targets.
    features : list[str], optional
        List of feature column names (defaults to MODEL_FEATURES).
    group_col : str, default="subject_id"
        Column used for grouping to prevent intra-participant leakage.
    target_col : str, default="label_binary"
        Target binary column (0 = Wake, 1 = Sleep).
    params : dict, optional
        Hyperparameters for XGBClassifier.
    n_splits : int, default=5
        Number of cross-validation folds.
    tune_threshold : bool, default=True
        Whether to tune the wake probability threshold on each fold's train partition.

    Returns
    -------
    dict[str, Any]
        Aggregated cross-validation metrics, fold details, and out-of-fold predictions.
    """
    features = features or MODEL_FEATURES
    clean_df = df[df[target_col].notna()].copy()
    clean_df[target_col] = clean_df[target_col].astype(int)

    X = clean_df[features].to_numpy()
    y = clean_df[target_col].to_numpy()
    groups = clean_df[group_col].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(clean_df), dtype=float)
    oof_preds = np.zeros(len(clean_df), dtype=int)
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (trn_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        pipe = build_xgboost_pipeline(params)
        pipe.fit(X_trn, y_trn)

        # Probabilities for class 0 (Wake)
        p_wake_trn = pipe.predict_proba(X_trn)[:, 0]
        p_wake_val = pipe.predict_proba(X_val)[:, 0]

        if tune_threshold:
            opt_th, _ = tune_wake_threshold(y_trn, p_wake_trn)
        else:
            opt_th = 0.50

        preds_val = np.where(p_wake_val >= opt_th, 0, 1)

        oof_probs[val_idx] = p_wake_val
        oof_preds[val_idx] = preds_val

        f_metrics = evaluate_predictions(y_val, preds_val, y_prob=p_wake_val)
        f_metrics["fold"] = fold_idx + 1
        f_metrics["optimal_threshold"] = opt_th
        f_metrics["n_val_epochs"] = len(val_idx)
        f_metrics["val_subjects"] = sorted(np.unique(groups[val_idx]).tolist())
        fold_metrics.append(f_metrics)

    # Compute overall pooled OOF metrics
    pooled_metrics = evaluate_predictions(y, oof_preds, y_prob=oof_probs)

    return {
        "n_splits": n_splits,
        "n_subjects": int(len(np.unique(groups))),
        "n_epochs": len(clean_df),
        "pooled_metrics": pooled_metrics,
        "per_fold_metrics": fold_metrics,
        "oof_probs_wake": oof_probs,
        "oof_preds": oof_preds,
    }


def train_xgboost(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target_col: str = "label_binary",
    params: dict[str, Any] | None = None,
    tune_threshold: bool = True,
    smoothing_window: int = 1,
) -> tuple[Pipeline, float, dict[str, Any]]:
    """Train XGBoost pipeline on a complete dataset and calibrate decision threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Cohort DataFrame with features and binary labels.
    features : list[str], optional
        Feature columns to use. Defaults to MODEL_FEATURES.
    target_col : str, default="label_binary"
        Target binary column.
    params : dict, optional
        Hyperparameters for XGBClassifier.
    tune_threshold : bool, default=True
        Whether to calibrate wake decision threshold on training cohort.
    smoothing_window : int, default=1
        Temporal smoothing window size.

    Returns
    -------
    tuple[Pipeline, float, dict[str, Any]]
        (fitted_pipeline, optimal_wake_threshold, train_metrics)
    """
    features = features or MODEL_FEATURES
    clean_df = df[df[target_col].notna()].copy()
    clean_df[target_col] = clean_df[target_col].astype(int)

    X_train = clean_df[features].to_numpy()
    y_train = clean_df[target_col].to_numpy()

    pipeline = build_xgboost_pipeline(params)
    pipeline.fit(X_train, y_train)

    p_wake_train = pipeline.predict_proba(X_train)[:, 0]

    if tune_threshold:
        optimal_th, _ = tune_wake_threshold(y_train, p_wake_train)
    else:
        optimal_th = 0.50

    raw_preds = np.where(p_wake_train >= optimal_th, 0, 1)
    if smoothing_window > 1:
        preds = smooth_labels(raw_preds, window_size=smoothing_window)
    else:
        preds = raw_preds

    train_metrics = evaluate_predictions(y_train, preds, y_prob=p_wake_train)

    return pipeline, optimal_th, train_metrics


def save_xgboost_artifact(
    pipeline: Pipeline,
    features: list[str],
    optimal_threshold: float,
    smoothing_window: int = 1,
    output_joblib_path: Path | str = DEFAULT_OUTPUT_JOBLIB,
    output_meta_path: Path | str = DEFAULT_OUTPUT_META,
    training_cohort_meta: dict[str, Any] | None = None,
    held_out_test_meta: dict[str, Any] | None = None,
    cv_metrics: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Serialize the trained XGBoost model and generate versioned ModelManifest metadata.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    features : list[str]
        Exact feature column list expected by the model.
    optimal_threshold : float
        Calibrated decision threshold for Wake detection.
    smoothing_window : int, default=1
        Temporal smoothing window size (odd integer).
    output_joblib_path : Path or str
        Destination path for serialized .joblib payload.
    output_meta_path : Path or str
        Destination path for JSON manifest metadata.
    training_cohort_meta : dict, optional
        Metadata describing the training cohort.
    held_out_test_meta : dict, optional
        Evaluation results on sequestered test split.
    cv_metrics : dict, optional
        Pooled cross-validation metrics.

    Returns
    -------
    tuple[Path, Path]
        (joblib_path, meta_path)
    """
    joblib_path = Path(output_joblib_path)
    meta_path = Path(output_meta_path)

    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save .joblib artifact adhering to predict_night contract
    payload = {
        "pipeline": pipeline,
        "features": features,
        "optimal_threshold": float(optimal_threshold),
        "smoothing_window": int(smoothing_window),
        "smoother": TemporalSmoother(window_size=smoothing_window, threshold=optimal_threshold),
    }
    joblib.dump(payload, joblib_path)
    logger.info(f"Saved XGBoost model artifact to {joblib_path}")

    # 2. Build and validate ModelManifest
    manifest_dict: dict[str, Any] = {
        "model_name": "NoctiBand Sleep Stager XGBoost",
        "version": "v0.2",
        "architecture": (
            "SimpleImputer(strategy='median') -> StandardScaler() -> "
            "XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08)"
        ),
        "features": list(features),
        "n_features": len(features),
        "target_mapping": {"0": "Wake", "1": "Sleep"},
        "optimal_wake_threshold": float(optimal_threshold),
        "smoothing_window_size": int(smoothing_window),
        "min_sampling_rate_hz": 10.0,
        "max_sampling_rate_hz": 100.0,
        "training_cohort": training_cohort_meta or {},
    }

    if held_out_test_meta:
        manifest_dict["held_out_test_split"] = held_out_test_meta
    if cv_metrics:
        manifest_dict["cv_metrics"] = cv_metrics

    # Validate schema
    manifest = ModelManifest.from_dict(manifest_dict)
    manifest.validate()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)
    logger.info(f"Saved ModelManifest metadata to {meta_path}")

    return joblib_path, meta_path


def train_and_benchmark(
    cohort_path: Path | str = DEFAULT_COHORT_PATH,
    split_path: Path | str = DEFAULT_SPLIT_PATH,
    raw_v2_dir: Path | str = DEFAULT_RAW_V2_DIR,
    output_joblib_path: Path | str = DEFAULT_OUTPUT_JOBLIB,
    output_meta_path: Path | str = DEFAULT_OUTPUT_META,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute end-to-end XGBoost cross-validation, training, test evaluation, and artifact freeze.

    Parameters
    ----------
    cohort_path : Path or str
        Path to training cohort pickle or CSV.
    split_path : Path or str
        Path to split.json metadata.
    raw_v2_dir : Path or str
        Path to raw_v2 directory containing sequestered test subject recordings.
    output_joblib_path : Path or str
        Path to save frozen .joblib model.
    output_meta_path : Path or str
        Path to save manifest .json.
    params : dict, optional
        Custom hyperparameters for XGBoost.

    Returns
    -------
    dict[str, Any]
        Complete benchmark summary including CV and sequestered test metrics.
    """
    cohort_path = Path(cohort_path)
    if cohort_path.suffix == ".pkl":
        df_cohort = pd.read_pickle(cohort_path)
    else:
        df_cohort = pd.read_csv(cohort_path)

    features = MODEL_FEATURES

    # Compute cohort hash for provenance
    sub_str = "".join(sorted(df_cohort["subject_id"].astype(str).unique()))
    h = hashlib.sha256()
    h.update(sub_str.encode("utf-8"))
    h.update(str(len(df_cohort)).encode("utf-8"))
    cohort_hash = h.hexdigest()[:16]

    cohort_meta = {
        "n_subjects": int(df_cohort["subject_id"].nunique()),
        "n_epochs": len(df_cohort),
        "n_wake": int((df_cohort["label_binary"] == 0).sum()),
        "cohort_hash": cohort_hash,
        "train_val_subjects": sorted(df_cohort["subject_id"].astype(str).unique().tolist()),
    }

    # 1. 5-Fold Cross-Validation across subjects (zero participant leakage)
    cv_res = cross_validate_xgboost(
        df_cohort,
        features=features,
        group_col="subject_id",
        target_col="label_binary",
        params=params,
        n_splits=5,
        tune_threshold=True,
    )

    # 2. Fit Final Model on all 22 subjects & tune threshold
    pipeline, opt_th, train_metrics = train_xgboost(
        df_cohort,
        features=features,
        target_col="label_binary",
        params=params,
        tune_threshold=True,
    )

    # 3. Evaluate once on Sequestered Test Split if available
    split_path = Path(split_path)
    raw_v2_dir = Path(raw_v2_dir)
    test_metrics: dict[str, Any] | None = None

    if split_path.exists() and raw_v2_dir.exists():
        from noctiband.io.adapter_v2 import build_subject_features_v2

        with open(split_path, encoding="utf-8") as f:
            split_info = json.load(f)
        test_subs = split_info.get("test_subjects", [])

        test_dfs = []
        for sid in test_subs:
            try:
                sub_df = build_subject_features_v2(base_dir=raw_v2_dir, subject_id=sid)
                test_dfs.append(sub_df)
            except Exception as e:
                logger.warning(f"Could not load test subject {sid}: {e}")

        if test_dfs:
            df_test = pd.concat(test_dfs, ignore_index=True)
            df_test = df_test[df_test["label_binary"].notna()].copy()
            df_test["label_binary"] = df_test["label_binary"].astype(int)

            X_test = df_test[features].to_numpy()
            y_test = df_test["label_binary"].to_numpy()

            p_wake_test = pipeline.predict_proba(X_test)[:, 0]
            preds_test = np.where(p_wake_test >= opt_th, 0, 1)

            t_metrics = evaluate_predictions(y_test, preds_test, y_prob=p_wake_test)
            test_metrics = {
                "test_subjects": test_subs,
                "n_epochs": len(df_test),
                "n_wake": int((y_test == 0).sum()),
                "metrics": t_metrics,
            }

    # 4. Save Artifacts
    save_xgboost_artifact(
        pipeline=pipeline,
        features=features,
        optimal_threshold=opt_th,
        smoothing_window=1,
        output_joblib_path=output_joblib_path,
        output_meta_path=output_meta_path,
        training_cohort_meta=cohort_meta,
        held_out_test_meta=test_metrics,
        cv_metrics=cv_res["pooled_metrics"],
    )

    return {
        "cohort_meta": cohort_meta,
        "cv_pooled_metrics": cv_res["pooled_metrics"],
        "train_metrics": train_metrics,
        "optimal_threshold": opt_th,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    import sys

    print(f"Running XGBoost benchmark in {sys.executable}")
    bench = train_and_benchmark()
    print("XGBoost Training and Evaluation Benchmark Complete.")
    print(f"Optimal Threshold: {bench['optimal_threshold']}")
    print(f"CV Pooled Metrics: {bench['cv_pooled_metrics']}")
    if bench["test_metrics"]:
        print(f"Sequestered Test Metrics: {bench['test_metrics']['metrics']}")
