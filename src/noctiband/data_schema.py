"""Dataset-agnostic epoch table schema definition and validation guards.

Defines the standard canonical contract for epoch-level features and real-data
integrity gates to prevent unvalidated or under-powered datasets from entering
modeling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column names required for downstream model training
CANONICAL_SUBJECT_COL = "subject_id"
CANONICAL_NIGHT_COL = "night_id"
CANONICAL_EPOCH_COL = "epoch_idx"
CANONICAL_LABEL_COL = "label"

# Permissible alias mapping for input normalization
COLUMN_ALIASES: dict[str, str] = {
    "subject": CANONICAL_SUBJECT_COL,
    "night": CANONICAL_NIGHT_COL,
    "epoch": CANONICAL_EPOCH_COL,
}


@dataclass(frozen=True)
class SchemaValidationResult:
    """Outcome of epoch table validation."""

    is_valid: bool
    errors: list[str]
    n_subjects: int
    n_epochs: int
    n_wake: int
    feature_columns: list[str]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common alias column names to the canonical schema names."""
    rename_map = {col: COLUMN_ALIASES[col] for col in df.columns if col in COLUMN_ALIASES}
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def validate_epoch_table(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> SchemaValidationResult:
    """Validate that an epoch table strictly conforms to the canonical contract.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    feature_columns : list[str] | None, optional
        Explicit feature list. If None, all numeric columns except identifiers
        and labels are treated as features.

    Returns
    -------
    SchemaValidationResult
        Validation status, error details, and summary counts.
    """
    errors: list[str] = []

    if df.empty:
        errors.append("Epoch table is empty.")
        return SchemaValidationResult(
            is_valid=False,
            errors=errors,
            n_subjects=0,
            n_epochs=0,
            n_wake=0,
            feature_columns=[],
        )

    norm_df = standardize_columns(df)

    # 1. Identifier checks
    for col in [CANONICAL_SUBJECT_COL, CANONICAL_NIGHT_COL, CANONICAL_EPOCH_COL]:
        if col not in norm_df.columns:
            errors.append(f"Missing required identifier column: '{col}' (or alias).")

    # 2. Label column check
    if CANONICAL_LABEL_COL not in norm_df.columns and "label_binary" not in norm_df.columns:
        errors.append("Missing required label column: must contain 'label' or 'label_binary'.")

    # 3. Determine feature columns
    non_feature_cols = {
        CANONICAL_SUBJECT_COL,
        CANONICAL_NIGHT_COL,
        CANONICAL_EPOCH_COL,
        CANONICAL_LABEL_COL,
        "label_binary",
        "label_3class",
        "label_unknown",
        "epoch_timestamp",
        "subject",
        "night",
        "epoch",
    }

    if feature_columns is not None:
        missing_feats = [f for f in feature_columns if f not in norm_df.columns]
        if missing_feats:
            errors.append(f"Specified feature columns missing from table: {missing_feats}")
        active_features = [f for f in feature_columns if f in norm_df.columns]
    else:
        active_features = [
            c
            for c in norm_df.columns
            if c not in non_feature_cols
            and pd.api.types.is_numeric_dtype(norm_df[c])
            and not pd.api.types.is_bool_dtype(norm_df[c])
        ]

    if not active_features:
        errors.append("No numeric feature columns found in epoch table.")

    # 4. Check for infinite values in features
    for feat in active_features:
        if np.isinf(norm_df[feat].to_numpy(dtype=float, na_value=0.0)).any():
            errors.append(f"Feature column '{feat}' contains infinite values.")

    n_subjects = norm_df[CANONICAL_SUBJECT_COL].nunique() if CANONICAL_SUBJECT_COL in norm_df else 0
    n_epochs = len(norm_df)

    # Compute Wake count (label 0)
    lbl_col = CANONICAL_LABEL_COL if CANONICAL_LABEL_COL in norm_df else "label_binary"
    if lbl_col in norm_df:
        n_wake = int((norm_df[lbl_col] == 0).sum())
    else:
        n_wake = 0

    return SchemaValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        n_subjects=n_subjects,
        n_epochs=n_epochs,
        n_wake=n_wake,
        feature_columns=active_features,
    )


def check_epoch_table(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> SchemaValidationResult:
    """Validate table schema, raising ValueError if malformed."""
    res = validate_epoch_table(df, feature_columns=feature_columns)
    if not res.is_valid:
        raise ValueError(
            "Epoch table validation failed with the following errors:\n"
            + "\n".join(f"  - {e}" for e in res.errors)
        )
    return res


def real_data_gate(df: pd.DataFrame, is_synthetic: bool = False) -> dict[str, Any]:
    """Strict gating check before allowing evaluation on real datasets.

    Enforces that real benchmark data must contain >= 10 distinct subjects
    and >= 50% valid motion per night to avoid reporting misleading or
    underpowered clinical metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset table.
    is_synthetic : bool, default=False
        If True, bypass the real-data gate (for unit tests / mock plumbing only).

    Returns
    -------
    dict[str, Any]
        Cohort summary metadata: n_subjects, n_epochs, n_wake.

    Raises
    ------
    ValueError
        If real data constraints are violated.
    """
    norm_df = standardize_columns(df)
    validation = check_epoch_table(norm_df)

    metadata = {
        "n_subjects": validation.n_subjects,
        "n_epochs": validation.n_epochs,
        "n_wake": validation.n_wake,
    }

    if is_synthetic:
        logger.info("Real data gate bypassed for synthetic test fixture: %s", metadata)
        return metadata

    # 1. Subject sample size gate
    if validation.n_subjects < 10:
        raise ValueError(
            f"Real-data gate REJECTED dataset: found only {validation.n_subjects} distinct "
            "subject(s) (minimum 10 required for scientific evaluation). Intra-subject biometrics "
            "and small cohort size prevent valid generalization."
        )

    # 2. Per-night valid-motion fraction gate
    if "motion_missing" in norm_df.columns:
        night_grouped = norm_df.groupby([CANONICAL_SUBJECT_COL, CANONICAL_NIGHT_COL])
        for (subj, night), grp in night_grouped:
            valid_frac = float((~grp["motion_missing"]).mean())
            if valid_frac < 0.5:
                raise ValueError(
                    f"Real-data gate REJECTED night {subj}/{night}: valid-motion fraction "
                    f"{valid_frac:.1%} is below 50% ({int((~grp['motion_missing']).sum())}/"
                    f"{len(grp)} epochs). Corrupted or truncated recordings cannot enter modeling."
                )

    return metadata
