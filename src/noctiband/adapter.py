"""External wearable data stream adapter and signal normalization (Person A).

Normalizes third-party smartwatch and smartband data into the canonical NightData
representation by handling unit conversions (m/s^2 vs g), timestamp variations
(Unix seconds, milliseconds, ISO 8601), and sampling rate harmonization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from noctiband.types import NightData

logger = logging.getLogger(__name__)

GRAVITY_MS2 = 9.80665


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse timestamps from ISO strings, Unix ms, or Unix seconds into float seconds."""
    s = series.copy()
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        # Attempt ISO 8601 datetime parsing
        try:
            parsed = pd.to_datetime(s, format="mixed", utc=True)
            return (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        except Exception:
            s = pd.to_numeric(s, errors="coerce")

    s = pd.to_numeric(s, errors="coerce")

    # Detect Unix milliseconds (e.g. 1.6e12 vs 1.6e9)
    sample_val = float(s.dropna().iloc[0]) if len(s.dropna()) > 0 else 0.0
    if sample_val > 1e11:
        s = s / 1000.0

    return s


def _normalize_acceleration_units(
    df: pd.DataFrame,
    target_unit: str = "g",
) -> pd.DataFrame:
    """Detect and convert acceleration from m/s^2 to g (or vice-versa)."""
    out = df.copy()
    cols = [c for c in ["x", "y", "z"] if c in out.columns]

    if not cols:
        return out

    # Compute magnitude
    mag = np.sqrt(np.sum(out[cols].to_numpy() ** 2, axis=1))
    median_mag = float(np.median(mag))

    # Static gravity reference: ~1.0 g vs ~9.81 m/s^2
    is_ms2 = median_mag > 4.5  # Typical human resting posture: ~9.8 m/s^2

    if is_ms2 and target_unit.lower() == "g":
        for c in cols:
            out[c] = out[c] / GRAVITY_MS2
        logger.info("Converted acceleration from m/s^2 to g (median mag was %.2f)", median_mag)
    elif (not is_ms2) and target_unit.lower() in ["ms2", "m/s2", "m/s^2"]:
        for c in cols:
            out[c] = out[c] * GRAVITY_MS2
        logger.info("Converted acceleration from g to m/s^2 (median mag was %.2f)", median_mag)

    return out


def _resample_motion(
    df: pd.DataFrame,
    target_hz: float = 50.0,
) -> pd.DataFrame:
    """Resample irregularly sampled motion to a regular target sampling rate."""
    if len(df) < 2 or target_hz <= 0:
        return df

    t = df["timestamp"].to_numpy()
    t_start = float(t[0])
    t_end = float(t[-1])
    dt = 1.0 / float(target_hz)

    # If already close to target_hz (within 5%), retain original
    observed_hz = (len(t) - 1) / max(1e-3, t_end - t_start)
    if np.isclose(observed_hz, target_hz, rtol=0.05):
        return df

    new_t = np.arange(t_start, t_end, dt)
    if len(new_t) == 0:
        return df

    resampled = {"timestamp": new_t}
    for c in ["x", "y", "z"]:
        if c in df.columns:
            resampled[c] = np.interp(new_t, t, df[c].to_numpy())

    return pd.DataFrame(resampled)


def adapt_external_stream(
    hr_data: pd.DataFrame | dict[str, Any] | str,
    motion_data: pd.DataFrame | dict[str, Any] | str,
    target_hz: float = 50.0,
    target_unit: str = "g",
    subject_id: str = "external_subject",
    night_id: str = "night_1",
) -> NightData:
    """Adapt raw external smartwatch/smartband streams into canonical NightData.

    Parameters
    ----------
    hr_data : pd.DataFrame or dict or file path
        Heart rate stream with columns (or keys) timestamp and hr.
    motion_data : pd.DataFrame or dict or file path
        Accelerometer stream with columns timestamp, x, y, (and optional z).
    target_hz : float, default=50.0
        Harmonized motion target frequency in Hz.
    target_unit : str, default='g'
        Target accelerometer unit ('g' or 'm/s2').
    subject_id : str, default='external_subject'
        Subject identifier.
    night_id : str, default='night_1'
        Night identifier.

    Returns
    -------
    NightData
        Standardized night data container compatible with predict_night.
    """
    # 1. Process Heart Rate
    if not isinstance(hr_data, pd.DataFrame):
        hr_df = pd.DataFrame(hr_data)
    else:
        hr_df = hr_data.copy()

    # Column mapping
    hr_cols = {c.lower(): c for c in hr_df.columns}
    t_col_hr = hr_cols.get("timestamp") or hr_cols.get("time") or list(hr_df.columns)[0]
    val_col_hr = (
        hr_cols.get("hr")
        or hr_cols.get("bpm")
        or hr_cols.get("pulse")
        or list(hr_df.columns)[1]
    )

    hr_clean = pd.DataFrame({
        "timestamp": _parse_timestamps(hr_df[t_col_hr]),
        "hr": pd.to_numeric(hr_df[val_col_hr], errors="coerce"),
    }).dropna().sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    if len(hr_clean) == 0:
        raise ValueError("Heart rate stream contains no valid numeric records after cleaning.")

    # 2. Process Motion
    if not isinstance(motion_data, pd.DataFrame):
        mot_df = pd.DataFrame(motion_data)
    else:
        mot_df = motion_data.copy()

    mot_cols = {c.lower(): c for c in mot_df.columns}
    t_col_mot = mot_cols.get("timestamp") or mot_cols.get("time") or list(mot_df.columns)[0]

    parsed_t = _parse_timestamps(mot_df[t_col_mot])

    x_col = mot_cols.get("x") or list(mot_df.columns)[1]
    y_col = mot_cols.get("y") or list(mot_df.columns)[2]
    has_z = ("z" in mot_cols) or (len(mot_df.columns) >= 4)
    z_col = mot_cols.get("z") if "z" in mot_cols else (list(mot_df.columns)[3] if has_z else None)

    mot_dict = {
        "timestamp": parsed_t,
        "x": pd.to_numeric(mot_df[x_col], errors="coerce"),
        "y": pd.to_numeric(mot_df[y_col], errors="coerce"),
    }
    if z_col is not None:
        mot_dict["z"] = pd.to_numeric(mot_df[z_col], errors="coerce")
    else:
        # Synthesize missing z assuming ~1g static gravity at rest
        mag_xy = np.sqrt(mot_dict["x"] ** 2 + mot_dict["y"] ** 2)
        mot_dict["z"] = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, mag_xy ** 2)))

    mot_clean = pd.DataFrame(mot_dict).dropna().sort_values("timestamp").drop_duplicates(
        subset=["timestamp"]
    ).reset_index(drop=True)

    if len(mot_clean) == 0:
        raise ValueError("Motion stream contains no valid numeric records after cleaning.")

    # Normalize acceleration units
    mot_clean = _normalize_acceleration_units(mot_clean, target_unit=target_unit)

    # Harmonize sampling rate
    if target_hz > 0:
        mot_clean = _resample_motion(mot_clean, target_hz=target_hz)

    # Align recording start
    rec_start = float(min(hr_clean["timestamp"].iloc[0], mot_clean["timestamp"].iloc[0]))

    return NightData(
        subject=subject_id,
        night=night_id,
        hr=hr_clean,
        motion=mot_clean,
        labels=None,
        rec_start=rec_start,
    )
