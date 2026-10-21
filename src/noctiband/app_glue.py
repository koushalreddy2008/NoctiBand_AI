"""Streamlit backend glue and multi-night aggregation (Person A).

Provides upload validation, cached model inference wrappers, and multi-night
aggregation for sleep regularity and trend detection.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from noctiband.inference.predict import load_inference_model
from noctiband.report_bridge import compute_sleep_metrics
from noctiband.types import NightData, NightPrediction

# Conditional Streamlit caching
try:
    import streamlit as st  # type: ignore

    def _cache_resource_decorator(func):
        return st.cache_resource(func)

except ImportError:

    def _cache_resource_decorator(func):
        # Fallback identity decorator when streamlit is not installed
        return func


@_cache_resource_decorator
def get_cached_model(model_path: Path | str | None = None) -> dict[str, Any]:
    """Cached loader for frozen sleep staging pipeline."""
    return load_inference_model(model_path)


def _read_dataframe_from_upload(file_obj: Any) -> pd.DataFrame:
    """Read a CSV or whitespace-delimited file object or buffer into DataFrame."""
    if hasattr(file_obj, "read"):
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        buffer = io.StringIO(content)
        # Rewind if file_obj supports seek
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    elif isinstance(file_obj, (str, Path)):
        with open(file_obj, encoding="utf-8", errors="replace") as f:
            buffer = io.StringIO(f.read())
    else:
        raise TypeError(f"Unsupported file object type: {type(file_obj)}")

    # Detect delimiter: comma vs whitespace
    sample = buffer.getvalue()[:2048]
    if not sample.strip():
        raise ValueError("Uploaded file is empty.")

    sep = r"\s+" if ("," not in sample.splitlines()[0] and "\t" in sample or " " in sample) else ","

    buffer.seek(0)
    # Check if header is present
    first_line = sample.splitlines()[0]
    has_header = any(col in first_line.lower() for col in ["time", "hr", "bpm", "x", "y", "z"])

    header = 0 if has_header else None
    df = pd.read_csv(buffer, sep=sep, header=header, engine="python")
    return df


def load_and_validate_upload(
    files: dict[str, Any] | list[Any],
    subject_id: str = "uploaded_user",
    night_id: str = "night_1",
) -> NightData:
    """Validate uploaded files and construct a standardized NightData container.

    Parameters
    ----------
    files : dict or list
        Either a dict {'hr': file_obj, 'motion': file_obj} or list of file objects
        with a name attribute containing 'hr'/'heartrate' and 'motion'/'accel'.
    subject_id : str, default='uploaded_user'
        Identifier for subject.
    night_id : str, default='night_1'
        Identifier for night.

    Returns
    -------
    NightData
        Standardized night data container.
    """
    hr_file = None
    mot_file = None

    if isinstance(files, dict):
        hr_file = files.get("hr") or files.get("heart_rate")
        mot_file = files.get("motion") or files.get("accel") or files.get("acceleration")
    elif isinstance(files, list):
        for f in files:
            name = getattr(f, "name", "").lower()
            if "hr" in name or "heart" in name:
                hr_file = f
            elif "mot" in name or "accel" in name:
                mot_file = f

    if hr_file is None:
        raise ValueError("Missing heart rate file. Upload must contain HR data.")
    if mot_file is None:
        raise ValueError("Missing accelerometer motion file. Upload must contain motion data.")

    # Parse HR DataFrame
    raw_hr = _read_dataframe_from_upload(hr_file)
    if raw_hr.shape[1] < 2:
        raise ValueError(
            f"HR file must have at least 2 columns (timestamp, hr), found {raw_hr.shape[1]}"
        )

    # Standardize column names
    raw_hr = raw_hr.iloc[:, :2].copy()
    raw_hr.columns = ["timestamp", "hr"]
    raw_hr["timestamp"] = pd.to_numeric(raw_hr["timestamp"], errors="coerce")
    raw_hr["hr"] = pd.to_numeric(raw_hr["hr"], errors="coerce")
    hr_clean = raw_hr.dropna().sort_values("timestamp").reset_index(drop=True)

    if len(hr_clean) == 0:
        raise ValueError("HR file contains no valid numeric rows.")

    # Parse Motion DataFrame
    raw_mot = _read_dataframe_from_upload(mot_file)
    if raw_mot.shape[1] < 4:
        raise ValueError(
            f"Motion file must have at least 4 columns (timestamp, x, y, z), "
            f"found {raw_mot.shape[1]}"
        )

    raw_mot = raw_mot.iloc[:, :4].copy()
    raw_mot.columns = ["timestamp", "x", "y", "z"]
    for col in ["timestamp", "x", "y", "z"]:
        raw_mot[col] = pd.to_numeric(raw_mot[col], errors="coerce")
    mot_clean = raw_mot.dropna().sort_values("timestamp").reset_index(drop=True)

    if len(mot_clean) == 0:
        raise ValueError("Motion file contains no valid numeric rows.")

    # Determine aligned rec_start
    rec_start = float(min(hr_clean["timestamp"].iloc[0], mot_clean["timestamp"].iloc[0]))

    return NightData(
        subject=subject_id,
        night=night_id,
        hr=hr_clean,
        motion=mot_clean,
        labels=None,
        rec_start=rec_start,
    )


def aggregate_multi_night(
    night_predictions: list[NightPrediction | dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate longitudinal metrics across multiple nights.

    Parameters
    ----------
    night_predictions : list of NightPrediction or dicts
        List of night predictions across sequential nights.

    Returns
    -------
    dict[str, Any]
        Aggregated statistics including means, standard deviations,
        sleep regularity index, and trend slopes.
    """
    if not night_predictions:
        return {
            "n_nights": 0,
            "mean_tst_min": 0.0,
            "std_tst_min": 0.0,
            "mean_sleep_efficiency": 0.0,
            "std_sleep_efficiency": 0.0,
            "mean_waso_min": 0.0,
            "std_waso_min": 0.0,
            "mean_sol_min": 0.0,
            "std_sol_min": 0.0,
            "sleep_regularity_index": 0.0,
            "efficiency_trend_slope": 0.0,
            "nights": [],
        }

    per_night_metrics: list[dict[str, Any]] = []
    tst_list: list[float] = []
    se_list: list[float] = []
    waso_list: list[float] = []
    sol_list: list[float] = []
    hypnograms: list[np.ndarray] = []

    for i, npred in enumerate(night_predictions):
        if isinstance(npred, NightPrediction):
            labels = npred.epoch_labels
            rec_start = npred.rec_start
        elif isinstance(npred, dict):
            labels = np.asarray(
                npred.get("epoch_labels", npred.get("hypnogram", [])),
                dtype=int,
            )
            rec_start = npred.get("rec_start", 0.0)
        else:
            raise TypeError(f"Unexpected prediction type in multi-night: {type(npred)}")

        m = compute_sleep_metrics(labels)
        m["night_index"] = i + 1
        m["rec_start"] = rec_start
        per_night_metrics.append(m)

        tst_list.append(m["tst_min"])
        se_list.append(m["sleep_efficiency"])
        waso_list.append(m["waso_min"])
        sol_list.append(m["sol_min"])
        hypnograms.append(labels)

    n_nights = len(per_night_metrics)

    # 1. Longitudinal Summary Statistics
    mean_tst = float(np.mean(tst_list))
    std_tst = float(np.std(tst_list, ddof=1)) if n_nights > 1 else 0.0

    mean_se = float(np.mean(se_list))
    std_se = float(np.std(se_list, ddof=1)) if n_nights > 1 else 0.0

    mean_waso = float(np.mean(waso_list))
    std_waso = float(np.std(waso_list, ddof=1)) if n_nights > 1 else 0.0

    mean_sol = float(np.mean(sol_list))
    std_sol = float(np.std(sol_list, ddof=1)) if n_nights > 1 else 0.0

    # 2. Trend Detection (Linear slope of sleep efficiency over nights)
    if n_nights > 1:
        x = np.arange(n_nights, dtype=float)
        slope, _ = np.polyfit(x, np.array(se_list, dtype=float), deg=1)
        trend_slope = float(slope)
    else:
        trend_slope = 0.0

    # 3. Sleep Regularity Index (SRI)
    # Measures the probability of being in the same state (sleep vs wake) across consecutive nights
    sri_scores: list[float] = []
    for i in range(n_nights - 1):
        h1 = hypnograms[i]
        h2 = hypnograms[i + 1]
        min_len = min(len(h1), len(h2))
        if min_len > 0:
            agreement = float((h1[:min_len] == h2[:min_len]).mean())
            # Scale agreement from [-1, 1] to [0, 100]
            sri_scores.append(round(agreement * 100.0, 2))

    sri_val = round(float(np.mean(sri_scores)), 2) if sri_scores else 100.0

    return {
        "n_nights": n_nights,
        "mean_tst_min": round(mean_tst, 2),
        "std_tst_min": round(std_tst, 2),
        "mean_sleep_efficiency": round(mean_se, 2),
        "std_sleep_efficiency": round(std_se, 2),
        "mean_waso_min": round(mean_waso, 2),
        "std_waso_min": round(std_waso, 2),
        "mean_sol_min": round(mean_sol, 2),
        "std_sol_min": round(std_sol, 2),
        "sleep_regularity_index": sri_val,
        "efficiency_trend_slope": round(trend_slope, 4),
        "nights": per_night_metrics,
    }
