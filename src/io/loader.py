"""
Data loading utilities for NoctiBand AI.

Provides modular loaders for wearable heart rate, 3-axis accelerometry,
multimodal physiological signals (BP, SpO2), and sleep stage ground-truth labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.io.contracts import NightRecording

logger = logging.getLogger(__name__)


def load_hr(file_path: str | Path) -> pd.DataFrame:
    """Load heart-rate time series from CSV.

    Expected formats:
    - Headerless 2-column CSV: [timestamp, heart_rate]
    - Headered CSV with 'timestamp' and 'heart_rate' (or 'hr') columns.

    Args:
        file_path: Path to the HR CSV file.

    Returns:
        pd.DataFrame with columns ['timestamp', 'heart_rate'] sorted by timestamp.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is empty or cannot be parsed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Heart-rate file not found: {path}")

    # Inspect first line to determine header presence
    first_line = path.open("r", encoding="utf-8").readline().strip()
    if not first_line:
        raise ValueError(f"Heart-rate file is empty: {path}")

    parts = [p.strip().lower() for p in first_line.split(",")]
    has_header = any(keyword in parts for keyword in ["time", "timestamp", "hr", "heart_rate"])

    if has_header:
        df = pd.read_csv(path)
        rename_dict = {}
        for col in df.columns:
            cl = str(col).strip().lower()
            if cl in ["timestamp", "time", "t"]:
                rename_dict[col] = "timestamp"
            elif cl in ["heart_rate", "hr", "heartrate", "bpm"]:
                rename_dict[col] = "heart_rate"
        df = df.rename(columns=rename_dict)
    else:
        df = pd.read_csv(path, header=None, names=["timestamp", "heart_rate"])

    if "timestamp" not in df.columns or "heart_rate" not in df.columns:
        raise ValueError(f"Could not identify 'timestamp' and 'heart_rate' in {path}")

    # Validate types and values
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    df = df.dropna(subset=["timestamp", "heart_rate"])

    # Basic physiological validation: HR between 20 and 260 BPM
    df = df[(df["heart_rate"] >= 20) & (df["heart_rate"] <= 260)]

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "heart_rate"]]


def load_motion(file_path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load 3-axis accelerometry time series from CSV.

    Expected columns: timestamp (or Timestamp), x, y, z.

    Args:
        file_path: Path to the motion CSV file.
        max_rows: Optional row limit for fast inspection.

    Returns:
        pd.DataFrame with columns ['timestamp', 'x', 'y', 'z'] sorted by timestamp.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Motion file not found: {path}")

    df = pd.read_csv(path, nrows=max_rows)
    if df.empty:
        raise ValueError(f"Motion file is empty: {path}")

    rename_dict = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl in ["timestamp", "time", "t"]:
            rename_dict[col] = "timestamp"
        elif cl in ["x", "acc_x", "accx"]:
            rename_dict[col] = "x"
        elif cl in ["y", "acc_y", "accy"]:
            rename_dict[col] = "y"
        elif cl in ["z", "acc_z", "accz"]:
            rename_dict[col] = "z"
    df = df.rename(columns=rename_dict)

    required = ["timestamp", "x", "y", "z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Motion file {path} missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[required]


def load_labels(
    file_path: str | Path,
    label_key: str = "expert_label",
    timezone: str = "America/New_York",
) -> tuple[np.ndarray, pd.Timestamp, np.ndarray | None]:
    """Load sleep-stage ground truth and recording start timestamp from MAT file.

    Args:
        file_path: Path to the labels.mat file.
        label_key: Primary key for sleep stages in MAT dictionary (default 'expert_label').
        timezone: Timezone string for recStart localization (default 'America/New_York').

    Returns:
        Tuple of (labels_array, rec_start_timestamp, dreem_labels_or_None).

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If recStart or specified label key is missing.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {path}")

    mat = loadmat(str(path))

    if label_key not in mat:
        available_keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Label key '{label_key}' not in {path}. Available keys: {available_keys}")

    if "recStart" not in mat:
        raise KeyError(f"'recStart' not found in {path}")

    # Parse primary labels
    labels = mat[label_key].ravel().astype(int)

    # Parse recStart string safely across MAT serialization variants
    rec_val = mat["recStart"]
    while isinstance(rec_val, (np.ndarray, list)):
        if len(rec_val) == 0:
            break
        rec_val = rec_val[0]
    rec_start_str = str(rec_val).strip()

    try:
        rec_start_dt = pd.Timestamp(rec_start_str)
        if rec_start_dt.tzinfo is None:
            rec_start_dt = rec_start_dt.tz_localize(timezone)
    except Exception as exc:
        raise ValueError(f"Failed to parse recStart string '{rec_start_str}': {exc}") from exc

    # Parse optional dreem_label if present
    dreem_labels: np.ndarray | None = None
    if "dreem_label" in mat:
        dreem_labels = mat["dreem_label"].ravel().astype(int)

    return labels, rec_start_dt, dreem_labels


def load_bp(file_path: str | Path) -> pd.DataFrame:
    """Load continuous blood pressure recordings from CSV.

    Expected columns: timestamp, systolic, diastolic, map, pulse_pressure.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Blood pressure file not found: {path}")

    df = pd.read_csv(path)
    rename_dict = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl in ["timestamp", "time"]:
            rename_dict[col] = "timestamp"
        elif cl in ["systolic", "sbp"]:
            rename_dict[col] = "systolic"
        elif cl in ["diastolic", "dbp"]:
            rename_dict[col] = "diastolic"
        elif cl in ["map", "mean_arterial_pressure"]:
            rename_dict[col] = "map"
        elif cl in ["pulse_pressure", "pp"]:
            rename_dict[col] = "pulse_pressure"
    df = df.rename(columns=rename_dict)

    req = ["timestamp", "systolic", "diastolic", "map", "pulse_pressure"]
    existing = [c for c in req if c in df.columns]
    for c in existing:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[existing]


def load_spo2(file_path: str | Path) -> pd.DataFrame:
    """Load continuous pulse oximetry (SpO2) recordings from CSV.

    Expected columns: timestamp, spo2.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"SpO2 file not found: {path}")

    df = pd.read_csv(path)
    rename_dict = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl in ["timestamp", "time"]:
            rename_dict[col] = "timestamp"
        elif cl in ["spo2", "sat", "o2"]:
            rename_dict[col] = "spo2"
    df = df.rename(columns=rename_dict)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["spo2"] = pd.to_numeric(df["spo2"], errors="coerce")
    df = df.dropna(subset=["timestamp", "spo2"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "spo2"]]


def load_night(
    data_dir: str | Path,
    subject_id: str = "Bidslab00",
    night_id: str = "3",
    timezone: str = "America/New_York",
    epoch_duration_sec: float = 30.0,
    hr_filename: str = "hr.csv",
    motion_filename: str = "motion.csv",
    labels_filename: str = "labels.mat",
    bp_filename: str | None = "blood_pressure.csv",
    spo2_filename: str | None = "spo2.csv",
) -> NightRecording:
    """Load a complete overnight recording session with all available channels.

    Args:
        data_dir: Directory containing raw data files.
        subject_id: Identifier for participant.
        night_id: Identifier for recording session/night.
        timezone: Timezone for timestamp localization.
        epoch_duration_sec: Default epoch length in seconds (30.0).
        hr_filename: Filename for HR CSV.
        motion_filename: Filename for motion CSV.
        labels_filename: Filename for labels MAT file.
        bp_filename: Optional filename for continuous BP CSV.
        spo2_filename: Optional filename for SpO2 CSV.

    Returns:
        NightRecording instance populated with all available channels and metadata.
    """
    root = Path(data_dir)
    metadata: dict[str, Any] = {
        "data_dir": str(root.resolve()),
        "timezone": timezone,
    }

    # 1. Load labels if present
    labels_path = root / labels_filename
    labels: np.ndarray | None = None
    dreem_labels: np.ndarray | None = None
    rec_start: pd.Timestamp | None = None

    if labels_path.is_file():
        try:
            labels, rec_start, dreem_labels = load_labels(
                labels_path, label_key="expert_label", timezone=timezone
            )
            metadata["labels_file"] = str(labels_path)
            metadata["num_labeled_epochs"] = len(labels)
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            logger.warning(f"Could not load labels from {labels_path}: {exc}")

    # 2. Load HR
    hr_path = root / hr_filename
    hr_df: pd.DataFrame | None = None
    if hr_path.is_file():
        try:
            hr_df = load_hr(hr_path)
            metadata["hr_file"] = str(hr_path)
            metadata["hr_readings"] = len(hr_df)
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            logger.warning(f"Could not load HR from {hr_path}: {exc}")

    # 3. Load Motion
    motion_path = root / motion_filename
    motion_df: pd.DataFrame | None = None
    if motion_path.is_file():
        try:
            motion_df = load_motion(motion_path)
            metadata["motion_file"] = str(motion_path)
            metadata["motion_readings"] = len(motion_df)
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            logger.warning(f"Could not load motion from {motion_path}: {exc}")

    # 4. Load BP (optional)
    bp_df: pd.DataFrame | None = None
    if bp_filename:
        bp_path = root / bp_filename
        if not bp_path.is_file():
            # Check alternative bp.csv
            alt_bp = root / "bp.csv"
            if alt_bp.is_file():
                bp_path = alt_bp
        if bp_path.is_file():
            try:
                bp_df = load_bp(bp_path)
                metadata["bp_file"] = str(bp_path)
                metadata["bp_readings"] = len(bp_df)
            except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
                logger.warning(f"Could not load BP from {bp_path}: {exc}")

    # 5. Load SpO2 (optional)
    spo2_df: pd.DataFrame | None = None
    if spo2_filename:
        spo2_path = root / spo2_filename
        if spo2_path.is_file():
            try:
                spo2_df = load_spo2(spo2_path)
                metadata["spo2_file"] = str(spo2_path)
                metadata["spo2_readings"] = len(spo2_df)
            except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
                logger.warning(f"Could not load SpO2 from {spo2_path}: {exc}")

    # Fallback for rec_start if labels are absent
    if rec_start is None:
        earliest_ts = None
        for df in [hr_df, motion_df, bp_df, spo2_df]:
            if df is not None and not df.empty:
                min_t = float(df["timestamp"].min())
                earliest_ts = min_t if earliest_ts is None else min(earliest_ts, min_t)
        if earliest_ts is not None:
            rec_start = pd.Timestamp(earliest_ts, unit="s", tz=timezone)
        else:
            rec_start = pd.Timestamp.now(tz=timezone)

    return NightRecording(
        subject_id=subject_id,
        night_id=night_id,
        rec_start=rec_start,
        epoch_duration_sec=epoch_duration_sec,
        hr=hr_df,
        motion=motion_df,
        bp=bp_df,
        spo2=spo2_df,
        labels=labels,
        dreem_labels=dreem_labels,
        metadata=metadata,
    )
