"""End-to-end night prediction and inference engine (Person A)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from noctiband.features.epoch_features import MODEL_FEATURES, build_epoch_table
from noctiband.models.smoothing import smooth_labels
from noctiband.types import NightData, NightPrediction, QualityControlError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "noctiband_v0.1.joblib"

_MODEL_CACHE: dict[str, Any] = {}


def load_inference_model(model_path: Path | str | None = None) -> dict[str, Any]:
    """Load and cache frozen model artifact and metadata."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    path_key = str(path.resolve())

    if path_key in _MODEL_CACHE:
        return _MODEL_CACHE[path_key]

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run Stage 2 model freeze first."
        )

    artifact = joblib.load(path)
    _MODEL_CACHE[path_key] = artifact
    return artifact


def predict_night(
    night: NightData | Path | str,
    model_path: Path | str | None = None,
    allow_partial: bool = False,
    min_coverage: float = 0.95,
    max_missing_fraction: float = 0.05,
) -> NightPrediction:
    """Predict sleep stages for an entire night of wearable sensor data.

    Parameters
    ----------
    night : NightData or Path or str
        Standard container with raw HR and accelerometer streams, or directory path.
    model_path : Path or str, optional
        Path to serialized model artifact (.joblib). Defaults to frozen v0.1 model.
    allow_partial : bool, default=False
        If False, raises QualityControlError if coverage < min_coverage or missing > max.
        If True, returns predictions with qc_summary['passed_qc'] = False.
    min_coverage : float, default=0.95
        Minimum allowable mean epoch coverage fraction.
    max_missing_fraction : float, default=0.05
        Maximum allowable fraction of epochs with missing motion.

    Returns
    -------
    NightPrediction
        Container with rec_start, epoch_labels, probs, coverage, epoch_timestamps, qc_summary.
    """
    # 1. Resolve input to NightData
    if isinstance(night, (str, Path)):
        from noctiband.io.loader import load_night

        night_data = load_night(night)
    elif isinstance(night, NightData):
        night_data = night
    else:
        raise TypeError(f"Expected NightData or Path, got {type(night)}")

    # 2. Extract Canonical Epoch Features
    epoch_table = build_epoch_table(night_data)
    n_epochs = len(epoch_table)

    if n_epochs == 0:
        raise ValueError("Night data produced 0 epochs.")

    # 3. Quality Control Audit
    hr_cov = float(epoch_table["hr_coverage"].mean()) if "hr_coverage" in epoch_table else 1.0
    mot_cov = (
        float(epoch_table["motion_coverage"].mean()) if "motion_coverage" in epoch_table else 1.0
    )
    mean_cov = min(hr_cov, mot_cov)

    missing_motion_frac = (
        float(epoch_table["motion_missing"].mean()) if "motion_missing" in epoch_table else 0.0
    )
    n_missing_epochs = (
        int(epoch_table["motion_missing"].sum()) if "motion_missing" in epoch_table else 0
    )

    qc_passed = (mean_cov >= min_coverage) and (missing_motion_frac <= max_missing_fraction)
    warnings: list[str] = []

    if mean_cov < min_coverage:
        warnings.append(
            f"Mean epoch coverage {mean_cov:.1%} is below minimum threshold {min_coverage:.1%}."
        )
    if missing_motion_frac > max_missing_fraction:
        warnings.append(
            f"Missing motion fraction {missing_motion_frac:.1%} exceeds maximum allowable "
            f"{max_missing_fraction:.1%}."
        )

    if not qc_passed and not allow_partial:
        raise QualityControlError(
            f"Night recording failed quality control guard: {'; '.join(warnings)}"
        )

    qc_summary: dict[str, Any] = {
        "passed_qc": qc_passed,
        "hr_coverage": round(hr_cov, 4),
        "motion_coverage": round(mot_cov, 4),
        "mean_coverage": round(mean_cov, 4),
        "missing_motion_fraction": round(missing_motion_frac, 4),
        "n_epochs": n_epochs,
        "n_missing_epochs": n_missing_epochs,
        "warnings": warnings,
    }

    # 4. Load Model Artifact
    model_dict = load_inference_model(model_path)
    pipeline = model_dict["pipeline"]
    features = model_dict.get("features", MODEL_FEATURES)
    threshold = model_dict.get("optimal_threshold", 0.50)
    window_size = model_dict.get("smoothing_window", 1)

    # 5. Extract Feature Matrix and Run Inference
    X = epoch_table[features].to_numpy()
    probs = pipeline.predict_proba(X)
    p_wake = probs[:, 0]
    p_sleep = probs[:, 1]

    raw_labels = np.where(p_wake >= threshold, 0, 1)

    # 6. Apply Temporal Post-Processing
    if window_size > 1:
        smoothed_labels = smooth_labels(raw_labels, window_size=window_size)
    else:
        smoothed_labels = raw_labels

    # 7. Formulate Epoch Timestamps and Coverage Array
    rec_start = float(night_data.rec_start)
    epoch_timestamps = rec_start + np.arange(n_epochs) * 30.0

    if "hr_coverage" in epoch_table and "motion_coverage" in epoch_table:
        cov_array = np.minimum(
            epoch_table["hr_coverage"].to_numpy(),
            epoch_table["motion_coverage"].to_numpy(),
        )
    else:
        cov_array = np.ones(n_epochs, dtype=float)

    return NightPrediction(
        rec_start=rec_start,
        epoch_labels=smoothed_labels,
        probs=p_sleep,
        coverage=cov_array,
        epoch_timestamps=epoch_timestamps,
        qc_summary=qc_summary,
    )
