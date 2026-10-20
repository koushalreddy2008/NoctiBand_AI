"""Evaluation harness for sleep stage estimation models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from noctiband.data_schema import (
    CANONICAL_SUBJECT_COL,
    check_epoch_table,
    real_data_gate,
    standardize_columns,
)
from noctiband.models.registry import get_estimator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation report across folds, pooled metrics, and subjects."""

    estimator_name: str
    target_col: str
    n_subjects: int
    n_epochs: int
    n_wake: int
    per_fold_metrics: list[dict[str, Any]]
    pooled_metrics: dict[str, Any]
    per_subject_table: pd.DataFrame


def build_pipeline(estimator: str | BaseEstimator) -> Pipeline:
    """Create a standardized leak-free pipeline with imputer, scaler, and model."""
    if isinstance(estimator, str):
        est_obj = get_estimator(estimator)
    elif isinstance(estimator, BaseEstimator):
        est_obj = estimator
    else:
        raise TypeError(f"Expected estimator name or BaseEstimator, got {type(estimator)}")

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", est_obj),
        ]
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute standard sleep staging metrics."""
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # Wake F1 (Wake is class 0)
    wake_f1 = float(
        f1_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]
        if 0 in classes
        else np.nan
    )

    # Sleep F1 (Sleep is class 1 for binary classification)
    sleep_f1 = float(
        f1_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0]
        if 1 in classes
        else np.nan
    )

    # Wake PR-AUC
    wake_pr_auc = np.nan
    if y_prob is not None and 0 in classes:
        try:
            wake_idx = int(np.where(classes == 0)[0][0])
            if wake_idx < y_prob.shape[1]:
                wake_true_binary = (y_true == 0).astype(int)
                if len(np.unique(wake_true_binary)) > 1:
                    wake_pr_auc = float(
                        average_precision_score(wake_true_binary, y_prob[:, wake_idx])
                    )
        except Exception as exc:
            logger.debug("Could not compute Wake PR-AUC: %s", exc)

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    return {
        "macro_f1": round(macro_f1, 4),
        "wake_f1": round(wake_f1, 4) if pd.notna(wake_f1) else np.nan,
        "sleep_f1": round(sleep_f1, 4) if pd.notna(sleep_f1) else np.nan,
        "balanced_accuracy": round(bal_acc, 4),
        "cohen_kappa": round(kappa, 4),
        "wake_pr_auc": round(wake_pr_auc, 4) if pd.notna(wake_pr_auc) else np.nan,
        "confusion_matrix": cm,
    }


def evaluate_cv(
    df: pd.DataFrame,
    estimator: str | BaseEstimator = "majority",
    feature_columns: list[str] | None = None,
    target_col: str = "label_binary",
    n_splits: int = 5,
    is_synthetic: bool = False,
) -> EvaluationResult:
    """Run participant-level GroupKFold cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing subject identifiers, features, and targets.
    estimator : str or BaseEstimator
        Model to evaluate (e.g. 'majority', 'logistic_regression', 'random_forest').
    feature_columns : list[str] | None
        Explicit list of feature columns. If None, automatically detected.
    target_col : str, default='label_binary'
        Target label column to classify (e.g. 'label_binary' or 'label_3class').
    n_splits : int, default=5
        Number of cross-validation folds.
    is_synthetic : bool, default=False
        Whether the input data is programmatic fixture data (bypassing real-data gate).

    Returns
    -------
    EvaluationResult
        Evaluation metrics across folds, pooled metrics, and per-subject breakdown.
    """
    # 1. Standardize schema and validate real-data gate
    norm_df = standardize_columns(df)
    gate_meta = real_data_gate(norm_df, is_synthetic=is_synthetic)
    schema_res = check_epoch_table(norm_df, feature_columns=feature_columns)
    active_feats = schema_res.feature_columns

    est_name = estimator if isinstance(estimator, str) else estimator.__class__.__name__

    # 2. Drop NaN-label rows (Unknown code 5 or unannotated)
    valid_mask = norm_df[target_col].notna()
    if "label_unknown" in norm_df.columns:
        valid_mask &= ~norm_df["label_unknown"].astype(bool)

    clean_df = norm_df.loc[valid_mask].copy().reset_index(drop=True)
    if clean_df.empty:
        raise ValueError(f"No valid labeled epochs remain for target '{target_col}'.")

    X = clean_df[active_feats].to_numpy(dtype=float)
    y = clean_df[target_col].to_numpy(dtype=int)
    groups = clean_df[CANONICAL_SUBJECT_COL].to_numpy()

    unique_subjects = np.unique(groups)
    effective_splits = min(n_splits, len(unique_subjects))
    if effective_splits < 2:
        raise ValueError(
            f"Need at least 2 distinct subjects for GroupKFold, found {len(unique_subjects)}."
        )

    gkf = GroupKFold(n_splits=effective_splits)

    per_fold_metrics: list[dict[str, Any]] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_prob_all: list[np.ndarray] = []
    pred_subject_ids: list[str] = []

    classes = np.unique(y)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        pipe = build_pipeline(estimator)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Pipeline fits imputer + scaler + estimator on training fold ONLY
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)

        prob_val: np.ndarray | None = None
        if hasattr(pipe, "predict_proba"):
            try:
                prob_val = pipe.predict_proba(X_val)
            except Exception:
                prob_val = None

        fold_metric = compute_metrics(y_val, y_pred, y_prob=prob_val, classes=classes)
        fold_metric["fold"] = fold_idx
        fold_metric["val_subjects"] = list(np.unique(groups[val_idx]))
        per_fold_metrics.append(fold_metric)

        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)
        pred_subject_ids.extend(groups[val_idx])
        if prob_val is not None:
            y_prob_all.append(prob_val)

    pooled_probs = np.vstack(y_prob_all) if len(y_prob_all) == effective_splits else None
    pooled_metrics = compute_metrics(
        np.array(y_true_all),
        np.array(y_pred_all),
        y_prob=pooled_probs,
        classes=classes,
    )

    # Per-subject breakdown table
    res_df = pd.DataFrame(
        {
            "subject": pred_subject_ids,
            "y_true": y_true_all,
            "y_pred": y_pred_all,
        }
    )

    subj_rows = []
    for subj_id, grp in res_df.groupby("subject"):
        sub_y_true = grp["y_true"].to_numpy()
        sub_y_pred = grp["y_pred"].to_numpy()
        subj_rows.append(
            {
                "subject": subj_id,
                "n_epochs": len(grp),
                "n_wake": int((sub_y_true == 0).sum()),
                "balanced_accuracy": round(
                    float(balanced_accuracy_score(sub_y_true, sub_y_pred)), 4
                ),
                "macro_f1": round(
                    float(f1_score(sub_y_true, sub_y_pred, average="macro", zero_division=0)),
                    4,
                ),
            }
        )
    per_subject_table = pd.DataFrame(subj_rows)

    return EvaluationResult(
        estimator_name=est_name,
        target_col=target_col,
        n_subjects=gate_meta["n_subjects"],
        n_epochs=gate_meta["n_epochs"],
        n_wake=gate_meta["n_wake"],
        per_fold_metrics=per_fold_metrics,
        pooled_metrics=pooled_metrics,
        per_subject_table=per_subject_table,
    )
