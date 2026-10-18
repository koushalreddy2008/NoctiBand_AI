"""Diagnosis CLI script for wearable heart rate and motion data.

Usage:
    python scripts/diagnose_motion.py data/raw/Bidslab00/3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

logger = logging.getLogger(__name__)


def is_header_present(first_cell: str) -> bool:
    """Return True if first cell cannot be parsed as a float."""
    try:
        float(first_cell.strip())
        return False
    except ValueError:
        return True


def parse_rec_start(rec_start_raw: Any) -> tuple[float, str]:
    """Extract recStart string from labels.mat and convert to America/New_York unix seconds."""
    if isinstance(rec_start_raw, np.ndarray):
        rec_start_str = str(rec_start_raw.ravel()[0])
    else:
        rec_start_str = str(rec_start_raw)
    rec_start_str = rec_start_str.strip()

    try:
        ts = pd.Timestamp(rec_start_str)
        localized = ts.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise")
    except Exception as exc:
        raise ValueError(
            f"Ambiguous or non-existent local time '{rec_start_str}' in America/New_York: {exc}"
        ) from exc

    return float(localized.timestamp()), rec_start_str


def diagnose_csv(
    file_path: Path,
    signal_type: str,
    rec_start: float | None = None,
    n_epochs: int | None = None,
) -> dict[str, Any]:
    """Diagnose a single CSV file (hr.csv or motion.csv)."""
    if not file_path.exists():
        return {"exists": False, "error": f"File not found: {file_path}"}

    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if not first_line.strip():
        return {"exists": True, "row_count": 0, "error": "Empty file"}

    first_cell = first_line.split(",")[0]
    header_present = is_header_present(first_cell)

    if header_present:
        df = pd.read_csv(file_path, header=0)
        raw_cols = df.columns.tolist()
        norm_map = {col: col.strip().lower() for col in raw_cols}
        df = df.rename(columns=norm_map)
    else:
        if signal_type == "hr":
            names = ["timestamp", "hr"]
        else:
            names = ["timestamp", "x", "y", "z"]
        df = pd.read_csv(file_path, header=None, names=names)
        raw_cols = names

    for c in ["timestamp", "hr", "x", "y", "z"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    row_count = len(df)
    nan_count = {col: int(df[col].isna().sum()) for col in df.columns}
    dtypes = {col: str(df[col].dtype) for col in df.columns}

    if row_count == 0 or df["timestamp"].dropna().empty:
        return {
            "exists": True,
            "header_present": header_present,
            "columns": raw_cols,
            "dtypes": dtypes,
            "nan_count": nan_count,
            "row_count": 0,
        }

    ts_series = df["timestamp"].dropna()
    t_min = float(ts_series.min())
    t_max = float(ts_series.max())
    time_span_s = t_max - t_min
    time_span_hours = time_span_s / 3600.0

    diffs = ts_series.diff().dropna()
    if not diffs.empty:
        p05 = float(diffs.quantile(0.05))
        p50 = float(diffs.quantile(0.50))
        p95 = float(diffs.quantile(0.95))
        implied_rate_hz = float(1.0 / p50) if p50 > 0 else 0.0
    else:
        p05, p50, p95, implied_rate_hz = 0.0, 0.0, 0.0, 0.0

    is_sorted = bool(ts_series.is_monotonic_increasing)
    duplicate_count = int(ts_series.duplicated().sum())

    gaps = diffs[diffs > 60.0]
    gaps_count = int(len(gaps))
    longest_gap_s = float(gaps.max()) if gaps_count > 0 else 0.0
    first_five_gaps = []
    if gaps_count > 0:
        for idx in gaps.index[:5]:
            start_ts = float(ts_series.loc[idx - 1])
            first_five_gaps.append(
                {
                    "start_timestamp": start_ts,
                    "start_iso_utc": pd.to_datetime(start_ts, unit="s", utc=True).isoformat(),
                    "gap_duration_s": float(diffs.loc[idx]),
                }
            )

    unit_sanity = "seconds" if 1e9 <= t_min <= 2e9 else "milliseconds"
    first_iso_utc = pd.to_datetime(t_min, unit="s", utc=True).isoformat()
    last_iso_utc = pd.to_datetime(t_max, unit="s", utc=True).isoformat()
    first_iso_et = (
        pd.to_datetime(t_min, unit="s", utc=True)
        .tz_convert("America/New_York")
        .isoformat()
    )
    last_iso_et = (
        pd.to_datetime(t_max, unit="s", utc=True)
        .tz_convert("America/New_York")
        .isoformat()
    )

    result: dict[str, Any] = {
        "exists": True,
        "header_present": header_present,
        "columns": raw_cols,
        "dtypes": dtypes,
        "nan_count": nan_count,
        "row_count": row_count,
        "time_span_hours": round(time_span_hours, 4),
        "sample_interval_percentiles_s": {
            "p05": round(p05, 6),
            "median": round(p50, 6),
            "p95": round(p95, 6),
        },
        "implied_sampling_rate_hz": round(implied_rate_hz, 4),
        "is_sorted": is_sorted,
        "duplicate_timestamps_count": duplicate_count,
        "gaps_gt_60s": {
            "count": gaps_count,
            "longest_s": round(longest_gap_s, 3),
            "first_five": first_five_gaps,
        },
        "timestamp_unit_sanity": unit_sanity,
        "first_timestamp": t_min,
        "last_timestamp": t_max,
        "first_timestamp_readable": {"utc": first_iso_utc, "america_new_york": first_iso_et},
        "last_timestamp_readable": {"utc": last_iso_utc, "america_new_york": last_iso_et},
    }

    if rec_start is not None and n_epochs is not None and n_epochs > 0:
        window_start = rec_start
        window_end = rec_start + n_epochs * 30.0
        inside_mask = (ts_series >= window_start) & (ts_series < window_end)
        inside_ts = ts_series[inside_mask]

        epochs = np.floor((inside_ts - window_start) / 30.0).astype(int)
        counts_per_epoch = epochs.value_counts().reindex(range(n_epochs), fill_value=0)

        epochs_zero = int((counts_per_epoch == 0).sum())
        epochs_with_data = n_epochs - epochs_zero
        zero_epoch_indices = counts_per_epoch[counts_per_epoch == 0].index.tolist()

        result["label_window_coverage"] = {
            "rec_start": rec_start,
            "n_epochs": n_epochs,
            "window_duration_hours": round((n_epochs * 30.0) / 3600.0, 4),
            "samples_inside_window": int(inside_mask.sum()),
            "samples_before_window": int((ts_series < window_start).sum()),
            "samples_after_window": int((ts_series >= window_end).sum()),
            "fraction_epochs_with_data": round(epochs_with_data / n_epochs, 4),
            "samples_per_epoch": {
                "min": int(counts_per_epoch.min()),
                "median": float(counts_per_epoch.median()),
                "max": int(counts_per_epoch.max()),
            },
            "epochs_zero_samples_count": epochs_zero,
            "zero_sample_epochs": zero_epoch_indices,
        }

    if signal_type == "motion" and {"x", "y", "z"}.issubset(df.columns):
        clean_xyz = df[["x", "y", "z"]].dropna()
        if not clean_xyz.empty:
            x = clean_xyz["x"].to_numpy(dtype=float)
            y = clean_xyz["y"].to_numpy(dtype=float)
            z = clean_xyz["z"].to_numpy(dtype=float)
            mag = np.sqrt(x**2 + y**2 + z**2)
            mag_median = float(np.median(mag))
            result["accelerometer_magnitude_stats"] = {
                "min": round(float(np.min(mag)), 4),
                "mean": round(float(np.mean(mag)), 4),
                "median": round(mag_median, 4),
                "max": round(float(np.max(mag)), 4),
                "std": round(float(np.std(mag)), 4),
                "resting_level_near_1g": bool(abs(mag_median - 1.0) < 0.2),
            }
            result["x_range"] = [round(float(np.min(x)), 4), round(float(np.max(x)), 4)]
            result["y_range"] = [round(float(np.min(y)), 4), round(float(np.max(y)), 4)]
            result["z_range"] = [round(float(np.min(z)), 4), round(float(np.max(z)), 4)]

    return result


def diagnose(night_dir: str | Path) -> dict[str, Any]:
    """Diagnose hr.csv and motion.csv against labels.mat in night_dir."""
    night_path = Path(night_dir).resolve()
    if not night_path.exists():
        raise FileNotFoundError(f"Night directory not found: {night_path}")

    labels_file = night_path / "labels.mat"
    rec_start: float | None = None
    rec_start_str: str | None = None
    n_epochs: int | None = None
    expert_labels_summary: dict[str, Any] | None = None

    if labels_file.exists():
        mat = loadmat(str(labels_file))
        if "recStart" in mat:
            rec_start, rec_start_str = parse_rec_start(mat["recStart"])
        if "expert_label" in mat:
            labels_arr = mat["expert_label"].ravel()
            n_epochs = len(labels_arr)
            series = pd.Series(labels_arr)
            expert_labels_summary = {
                "total_epochs": n_epochs,
                "stage_counts": {int(k): int(v) for k, v in series.value_counts().sort_index().items()},
            }

    hr_report = diagnose_csv(
        night_path / "hr.csv", signal_type="hr", rec_start=rec_start, n_epochs=n_epochs
    )
    motion_report = diagnose_csv(
        night_path / "motion.csv", signal_type="motion", rec_start=rec_start, n_epochs=n_epochs
    )

    summary: dict[str, Any] = {
        "night_dir": str(night_path),
        "labels": {
            "exists": labels_file.exists(),
            "rec_start_text": rec_start_str,
            "rec_start_unix": rec_start,
            "expert_labels": expert_labels_summary,
        },
        "hr": hr_report,
        "motion": motion_report,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose hr.csv and motion.csv in a night directory.")
    parser.add_argument("night_dir", type=str, help="Path to raw night directory (e.g., data/raw/Bidslab00/3)")
    args = parser.parse_args()

    diagnosis = diagnose(args.night_dir)
    print(json.dumps(diagnosis, indent=2, default=str))


if __name__ == "__main__":
    main()
