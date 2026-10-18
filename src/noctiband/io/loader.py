"""Data loader for multi-modal overnight wearable recordings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from noctiband.types import NightData

logger = logging.getLogger(__name__)


def is_header_present(first_cell: str) -> bool:
    """Return True if first cell cannot be parsed as a float."""
    try:
        float(first_cell.strip())
        return False
    except ValueError:
        return True


def parse_rec_start(rec_start_raw: Any) -> float:
    """Parse recStart string into unix seconds localized as America/New_York.

    Raises ValueError if the local time is ambiguous or non-existent (DST transition).
    """
    if isinstance(rec_start_raw, np.ndarray):
        rec_start_str = str(rec_start_raw.ravel()[0])
    else:
        rec_start_str = str(rec_start_raw)
    rec_start_str = rec_start_str.strip()

    try:
        ts = pd.Timestamp(rec_start_str)
        localized = ts.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise")
        return float(localized.timestamp())
    except Exception as exc:
        raise ValueError(
            f"Ambiguous or non-existent local time '{rec_start_str}' in America/New_York: {exc}"
        ) from exc


def load_night(
    night_dir: str | Path,
    subject: str | None = None,
    night: str | None = None,
    rec_start: float | None = None,
) -> NightData:
    """Load wearable HR, motion, and expert sleep labels for a single night.

    Parameters
    ----------
    night_dir : str | Path
        Directory containing hr.csv, motion.csv, and optionally labels.mat.
    subject : str | None, optional
        Subject identifier. Defaults to parent folder name of night_dir.
    night : str | None, optional
        Night identifier. Defaults to night_dir folder name.
    rec_start : float | None, optional
        Unix timestamp (seconds) at start of epoch 0. Required if labels.mat is missing.

    Returns
    -------
    NightData
        Dataclass containing subject, night, hr, motion, labels, and rec_start.
    """
    night_path = Path(night_dir).resolve()
    if not night_path.exists():
        raise FileNotFoundError(f"Night directory not found: {night_path}")

    night_id = night if night is not None else night_path.name
    subject_id = subject if subject is not None else night_path.parent.name

    labels_file = night_path / "labels.mat"
    labels: np.ndarray | None = None
    rec_start_val: float | None = None

    if labels_file.exists():
        mat = loadmat(str(labels_file))
        if "expert_label" in mat:
            labels = mat["expert_label"].ravel().astype(int)
        if "recStart" in mat:
            rec_start_val = parse_rec_start(mat["recStart"])
        elif rec_start is not None:
            rec_start_val = float(rec_start)
        else:
            raise ValueError("recStart not found in labels.mat and rec_start was not provided.")
    else:
        if rec_start is None:
            raise ValueError("labels.mat is missing and no rec_start was provided.")
        rec_start_val = float(rec_start)

    # Ingest hr.csv
    hr_file = night_path / "hr.csv"
    if not hr_file.exists():
        raise FileNotFoundError(f"hr.csv not found in {night_path}")

    df_hr = pd.read_csv(hr_file, header=None)
    if df_hr.shape[1] < 2:
        raise ValueError(f"hr.csv must contain at least 2 columns, found {df_hr.shape[1]}")
    if df_hr.shape[1] > 2:
        extra_hr_cols = df_hr.columns[2:].tolist()
        logger.warning("Ignoring extra columns in %s: %s", hr_file, extra_hr_cols)
        df_hr = df_hr.iloc[:, :2]

    df_hr.columns = ["timestamp", "hr"]
    df_hr["timestamp"] = pd.to_numeric(df_hr["timestamp"], errors="coerce")
    df_hr["hr"] = pd.to_numeric(df_hr["hr"], errors="coerce")

    hr_nans = int(df_hr.isna().any(axis=1).sum())
    if hr_nans > 0:
        df_hr = df_hr.dropna(subset=["timestamp", "hr"]).copy()
        logger.info("Dropped %d NaN rows from hr.csv in %s", hr_nans, night_path)

    hr_dups = int(df_hr.duplicated().sum())
    if hr_dups > 0:
        df_hr = df_hr.drop_duplicates().copy()
        logger.info("Dropped %d duplicate rows from hr.csv in %s", hr_dups, night_path)

    df_hr = df_hr.sort_values(by="timestamp").reset_index(drop=True)

    # Ingest motion.csv
    motion_file = night_path / "motion.csv"
    if not motion_file.exists():
        raise FileNotFoundError(f"motion.csv not found in {night_path}")

    with open(motion_file, encoding="utf-8") as f:
        first_line = f.readline()
    if not first_line.strip():
        raise ValueError(f"motion.csv in {night_path} is empty")

    first_cell = first_line.split(",")[0].strip()
    header_present = is_header_present(first_cell)

    if header_present:
        df_motion = pd.read_csv(motion_file, header=0)
        norm_cols = [c.strip().lower() for c in df_motion.columns]
        df_motion.columns = norm_cols
        required_cols = ["timestamp", "x", "y", "z"]
        missing_cols = [c for c in required_cols if c not in norm_cols]
        if missing_cols:
            raise ValueError(f"motion.csv missing required columns: {missing_cols}")
        extra_cols = [c for c in norm_cols if c not in required_cols]
        if extra_cols:
            logger.warning("Ignoring extra columns in motion.csv: %s", extra_cols)
        df_motion = df_motion[required_cols].copy()
    else:
        df_motion = pd.read_csv(motion_file, header=None)
        if df_motion.shape[1] < 4:
            raise ValueError(
                "motion.csv without header must contain at least 4 columns, "
                f"found {df_motion.shape[1]}"
            )
        if df_motion.shape[1] > 4:
            extra_cols_no_hdr = df_motion.columns[4:].tolist()
            logger.warning("Ignoring extra columns in motion.csv: %s", extra_cols_no_hdr)
            df_motion = df_motion.iloc[:, :4]
        df_motion.columns = ["timestamp", "x", "y", "z"]

    for col in ["timestamp", "x", "y", "z"]:
        df_motion[col] = pd.to_numeric(df_motion[col], errors="coerce")

    motion_nans = int(df_motion.isna().any(axis=1).sum())
    if motion_nans > 0:
        df_motion = df_motion.dropna(subset=["timestamp", "x", "y", "z"]).copy()
        logger.info("Dropped %d NaN rows from motion.csv in %s", motion_nans, night_path)

    motion_dups = int(df_motion.duplicated().sum())
    if motion_dups > 0:
        df_motion = df_motion.drop_duplicates().copy()
        logger.info("Dropped %d duplicate rows from motion.csv in %s", motion_dups, night_path)

    df_motion = df_motion.sort_values(by="timestamp").reset_index(drop=True)

    # Alignment sanity check
    for sig_name, df_sig in [("HR", df_hr), ("motion", df_motion)]:
        if not df_sig.empty:
            first_ts = float(df_sig["timestamp"].iloc[0])
            diff = abs(first_ts - rec_start_val)
            if diff > 3600.0:
                raise ValueError(
                    f"First {sig_name} timestamp {first_ts} differs from rec_start {rec_start_val} "
                    f"by {diff:.1f}s (> 3600s; likely timezone/unit bug)."
                )
            if diff > 300.0:
                logger.warning(
                    "First %s timestamp %s differs from rec_start %s by %.1fs (> 300s).",
                    sig_name,
                    first_ts,
                    rec_start_val,
                    diff,
                )

    return NightData(
        subject=subject_id,
        night=night_id,
        hr=df_hr,
        motion=df_motion,
        labels=labels,
        rec_start=rec_start_val,
    )
