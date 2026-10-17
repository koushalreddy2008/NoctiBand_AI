"""
Timestamp synchronization and alignment utilities for NoctiBand AI.

Implements the documented BIDSleep alignment rule:
    epoch_0_based = floor((timestamp - rec_start) / epoch_duration)
and provides tools for auditing signal boundaries, missingness, and windowing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def timestamp_to_epoch(
    timestamp: float | np.ndarray | pd.Series,
    rec_start_unix: float,
    epoch_duration_sec: float = 30.0,
) -> int | np.ndarray | pd.Series:
    """Map Unix timestamp(s) to 0-based epoch index.

    The first epoch [rec_start, rec_start + epoch_duration) has epoch index 0.
    Timestamps prior to rec_start will receive negative epoch indices.

    Args:
        timestamp: Single timestamp or array/Series of timestamps in Unix seconds.
        rec_start_unix: Recording start time in Unix seconds.
        epoch_duration_sec: Duration of one epoch in seconds (default: 30.0).

    Returns:
        Integer epoch index or array/Series of integer epoch indices.
    """
    if isinstance(timestamp, (float, int)):
        return int(np.floor((timestamp - rec_start_unix) / epoch_duration_sec))
    elif isinstance(timestamp, pd.Series):
        return np.floor((timestamp - rec_start_unix) / epoch_duration_sec).astype(int)
    else:
        # np.ndarray or list-like
        arr = np.asarray(timestamp, dtype=float)
        return np.floor((arr - rec_start_unix) / epoch_duration_sec).astype(int)


def epoch_to_timestamps(
    epoch_idx: int,
    rec_start_unix: float,
    epoch_duration_sec: float = 30.0,
) -> tuple[float, float]:
    """Return the start and end Unix timestamps for a given 0-based epoch index.

    Interval is half-open: [start_unix, end_unix).

    Args:
        epoch_idx: 0-based epoch index.
        rec_start_unix: Recording start time in Unix seconds.
        epoch_duration_sec: Duration of one epoch in seconds (default: 30.0).

    Returns:
        Tuple of (start_unix, end_unix).
    """
    start_unix = rec_start_unix + (epoch_idx * epoch_duration_sec)
    end_unix = start_unix + epoch_duration_sec
    return start_unix, end_unix


def epoch_to_datetimes(
    epoch_idx: int,
    rec_start: pd.Timestamp,
    epoch_duration_sec: float = 30.0,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the start and end timezone-aware pd.Timestamps for a given epoch.

    Args:
        epoch_idx: 0-based epoch index.
        rec_start: Recording start pd.Timestamp.
        epoch_duration_sec: Duration of one epoch in seconds (default: 30.0).

    Returns:
        Tuple of (start_timestamp, end_timestamp).
    """
    offset_start = pd.Timedelta(seconds=epoch_idx * epoch_duration_sec)
    offset_end = offset_start + pd.Timedelta(seconds=epoch_duration_sec)
    return rec_start + offset_start, rec_start + offset_end


def audit_signal_alignment(
    df: pd.DataFrame,
    rec_start_unix: float,
    total_epochs: int,
    time_col: str = "timestamp",
    epoch_duration_sec: float = 30.0,
) -> dict[str, Any]:
    """Audit the alignment of a signal dataframe relative to recording boundaries.

    Identifies out-of-boundary readings (before recStart or after recEnd)
    and lists covered versus missing epochs.

    Args:
        df: Dataframe with a timestamp column.
        rec_start_unix: Recording start in Unix seconds.
        total_epochs: Number of expected labeled epochs.
        time_col: Name of timestamp column.
        epoch_duration_sec: Epoch length in seconds (default: 30.0).

    Returns:
        Dictionary containing audit metrics and missing epoch indices.
    """
    if df.empty or time_col not in df.columns:
        return {
            "total_samples": 0,
            "samples_before_start": 0,
            "samples_inside_window": 0,
            "samples_after_window": 0,
            "epochs_covered": 0,
            "epochs_missing": total_epochs,
            "missing_epoch_indices": list(range(total_epochs)),
            "coverage_ratio": 0.0,
            "min_timestamp": None,
            "max_timestamp": None,
        }

    timestamps = df[time_col].values
    epochs = timestamp_to_epoch(timestamps, rec_start_unix, epoch_duration_sec)

    before_mask = epochs < 0
    after_mask = epochs >= total_epochs
    inside_mask = (~before_mask) & (~after_mask)

    inside_epochs = set(epochs[inside_mask])
    expected_epochs = set(range(total_epochs))
    missing_epochs: list[int] = sorted(expected_epochs - inside_epochs)

    return {
        "total_samples": len(df),
        "samples_before_start": int(np.sum(before_mask)),
        "samples_inside_window": int(np.sum(inside_mask)),
        "samples_after_window": int(np.sum(after_mask)),
        "epochs_covered": len(inside_epochs),
        "epochs_missing": len(missing_epochs),
        "missing_epoch_indices": missing_epochs,
        "coverage_ratio": float(len(inside_epochs) / total_epochs) if total_epochs > 0 else 0.0,
        "min_timestamp": float(np.min(timestamps)),
        "max_timestamp": float(np.max(timestamps)),
    }


def filter_and_align_signal(
    df: pd.DataFrame,
    rec_start_unix: float,
    total_epochs: int | None = None,
    time_col: str = "timestamp",
    epoch_duration_sec: float = 30.0,
    drop_outside: bool = True,
) -> pd.DataFrame:
    """Assign epoch indices to a signal dataframe and optionally drop out-of-bounds rows.

    Args:
        df: Input dataframe containing timestamp column.
        rec_start_unix: Recording start in Unix seconds.
        total_epochs: Optional total number of valid epochs [0, total_epochs - 1].
        time_col: Name of timestamp column.
        epoch_duration_sec: Epoch length in seconds (default: 30.0).
        drop_outside: If True and total_epochs is specified, drops rows with epoch < 0
                      or epoch >= total_epochs.

    Returns:
        Dataframe with added 'epoch' column, sorted by timestamp.
    """
    if df.empty:
        df_out = df.copy()
        df_out["epoch"] = pd.Series(dtype=int)
        return df_out

    df_out = df.copy()
    df_out["epoch"] = timestamp_to_epoch(
        df_out[time_col], rec_start_unix, epoch_duration_sec=epoch_duration_sec
    )

    if drop_outside and total_epochs is not None:
        mask = (df_out["epoch"] >= 0) & (df_out["epoch"] < total_epochs)
        df_out = df_out[mask]

    return df_out.sort_values(time_col).reset_index(drop=True)
