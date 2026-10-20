"""Adapter for Dataset v2 (Apple Watch / Polysomnography Benchmark).

Converts raw text exports (heart_rate, labels, motion, steps) to the
canonical epoch feature schema defined in docs/dataset_adapter.md.

Design & Data Policy:
1. Steps are excluded from modeling features: 8/31 subjects have zero in-night
   steps, and steps are an algorithmically derived proprietary watch metric
   rather than a measured raw biosignal, violating device-agnostic headband
   specifications.
2. Signal coverage is strictly measured across the labeled night window [0, N*30).
   Subjects failing the >=0.95 coverage threshold are flagged and excluded by default.
3. Label coding:
   Raw 0 -> 0 (Wake)
   Raw 1 -> 1 (N1)
   Raw 2 -> 2 (N2)
   Raw 3 -> 3 (N3)
   Raw 4 -> 3 (N4 merged into N3 under AASM guidelines)
   Raw 5 -> 4 (REM)
   Raw -1 -> 5 (Unscored/Unknown; mapped to NaN target and label_unknown=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from noctiband.data_schema import standardize_columns, validate_epoch_table
from noctiband.features.epoch_features import build_epoch_table
from noctiband.types import EPOCH_S, NightData

logger = logging.getLogger(__name__)

# Raw to canonical internal stage code mapping
# Internal coding before schema conversion: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, 5=Unknown
RAW_LABEL_MAP: dict[int, int] = {
    0: 0,   # Wake
    1: 1,   # N1
    2: 2,   # N2
    3: 3,   # N3
    4: 3,   # N4 merged into slow-wave N3
    5: 4,   # REM
    -1: 5,  # Unscored / unknown
}


def compute_epoch_coverage_fraction(timestamps: np.ndarray, n_epochs: int) -> float:
    """Compute fraction of 30-second epochs having at least 1 reading."""
    if n_epochs <= 0 or len(timestamps) == 0:
        return 0.0

    sorted_ts = np.sort(timestamps)
    epochs_present = 0
    for i in range(n_epochs):
        t0 = i * float(EPOCH_S)
        t1 = (i + 1) * float(EPOCH_S)
        idx0 = np.searchsorted(sorted_ts, t0, side="left")
        idx1 = np.searchsorted(sorted_ts, t1, side="left")
        if idx1 > idx0:
            epochs_present += 1

    return float(epochs_present / n_epochs)


def load_subject_v2(
    base_dir: Path | str,
    subject_id: str | int,
    allow_partial: bool = False,
    min_coverage: float = 0.95,
) -> NightData:
    """Load and validate raw data files for a single Dataset v2 subject.

    Parameters
    ----------
    base_dir : Path or str
        Root directory containing 'heart_rate', 'labels', 'motion' subfolders.
    subject_id : str or int
        Subject identifier.
    allow_partial : bool, default=False
        If True, allow loading subjects with coverage below `min_coverage`.
    min_coverage : float, default=0.95
        Minimum required epoch coverage fraction for both HR and motion.

    Returns
    -------
    NightData
        Standard container with subject, night='1', hr, motion, labels, and rec_start=0.0.
    """
    base_path = Path(base_dir)
    sub_str = str(subject_id)

    lbl_file = base_path / "labels" / f"{sub_str}_labeled_sleep.txt"
    hr_file = base_path / "heart_rate" / f"{sub_str}_heartrate.txt"
    mot_file = base_path / "motion" / f"{sub_str}_acceleration.txt"

    if not lbl_file.exists():
        raise FileNotFoundError(f"Missing label file: {lbl_file}")
    if not hr_file.exists():
        raise FileNotFoundError(f"Missing HR file: {hr_file}")
    if not mot_file.exists():
        raise FileNotFoundError(f"Missing motion file: {mot_file}")

    # 1. Labels
    lbl_df = pd.read_csv(lbl_file, sep=r"\s+", header=None, names=["timestamp", "raw_stage"])
    n_epochs = len(lbl_df)
    if n_epochs == 0:
        raise ValueError(f"Label file {lbl_file} is empty.")

    # Map labels to internal coding
    mapped_labels = lbl_df["raw_stage"].map(RAW_LABEL_MAP).to_numpy(dtype=int)

    # 2. Heart rate
    hr_df = pd.read_csv(hr_file, header=None, names=["timestamp", "hr"])
    hr_df = hr_df.sort_values("timestamp").reset_index(drop=True)

    # 3. Motion
    mot_df = pd.read_csv(mot_file, sep=r"\s+", header=None, names=["timestamp", "x", "y", "z"])
    mot_df = mot_df.sort_values("timestamp").reset_index(drop=True)

    # 4. Coverage checks across labeled night window [0, n_epochs * 30)
    hr_cov = compute_epoch_coverage_fraction(hr_df["timestamp"].to_numpy(), n_epochs)
    mot_cov = compute_epoch_coverage_fraction(mot_df["timestamp"].to_numpy(), n_epochs)

    logger.info(
        "Subject %s coverage: HR=%.2f%%, Motion=%.2f%% (threshold=%.1f%%)",
        sub_str,
        hr_cov * 100,
        mot_cov * 100,
        min_coverage * 100,
    )

    if (hr_cov < min_coverage) or (mot_cov < min_coverage):
        msg = (
            f"Subject {sub_str} failed coverage guard: HR coverage {hr_cov:.1%}, "
            f"motion coverage {mot_cov:.1%} (min required: {min_coverage:.1%})."
        )
        if not allow_partial:
            raise ValueError(msg)
        logger.warning("%s Proceeding because allow_partial=True.", msg)

    return NightData(
        subject=sub_str,
        night="1",
        hr=hr_df,
        motion=mot_df,
        labels=mapped_labels,
        rec_start=0.0,
    )


def build_subject_features_v2(
    base_dir: Path | str,
    subject_id: str | int,
    config: dict[str, Any] | str | Path | None = None,
    allow_partial: bool = False,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Load subject, extract epoch features, and format to canonical schema."""
    night = load_subject_v2(
        base_dir=base_dir,
        subject_id=subject_id,
        allow_partial=allow_partial,
        min_coverage=min_coverage,
    )
    raw_table = build_epoch_table(night, config=config)

    # Standardize column naming to canonical schema
    canonical_df = standardize_columns(raw_table)
    return canonical_df


def build_dataset_v2(
    base_dir: Path | str = "data/raw_v2",
    output_path: Path | str | None = None,
    config: dict[str, Any] | str | Path | None = None,
    min_coverage: float = 0.95,
    allow_partial: bool = False,
    skip_flagged: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all subjects in Dataset v2 into a canonical epoch feature table.

    Parameters
    ----------
    base_dir : Path or str
        Root dataset directory (e.g. 'data/raw_v2').
    output_path : Path or str, optional
        Destination CSV file path for the combined table.
    config : dict or Path, optional
        Feature configuration override.
    min_coverage : float, default=0.95
        Coverage threshold for HR and motion.
    allow_partial : bool, default=False
        Whether to allow processing below threshold when not skipping.
    skip_flagged : bool, default=True
        Whether to gracefully exclude flagged subjects (< min_coverage)
        rather than aborting with an exception.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Combined canonical epoch feature DataFrame.
        - Summary DataFrame reporting per-subject metrics and inclusion status.
    """
    base_path = Path(base_dir)
    lbl_dir = base_path / "labels"
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {lbl_dir}")

    label_files = sorted(list(lbl_dir.glob("*_labeled_sleep.txt")))
    subjects = [f.name.split("_")[0] for f in label_files]

    tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for sub in subjects:
        # Check coverage without throwing to populate audit summary
        try:
            night = load_subject_v2(
                base_dir=base_path,
                subject_id=sub,
                allow_partial=True,
                min_coverage=min_coverage,
            )
            n_epochs = len(night.labels) if night.labels is not None else 0
            n_wake = int((night.labels == 0).sum()) if night.labels is not None else 0
            hr_cov = compute_epoch_coverage_fraction(night.hr["timestamp"].to_numpy(), n_epochs)
            mot_cov = compute_epoch_coverage_fraction(
                night.motion["timestamp"].to_numpy(), n_epochs
            )

            epoch_tbl = build_epoch_table(night, config=config)
            canonical_tbl = standardize_columns(epoch_tbl)

            # Valid motion fraction inside the table
            valid_m_count = int((~canonical_tbl["motion_missing"]).sum())
            valid_m_fraction = float(valid_m_count / n_epochs) if n_epochs > 0 else 0.0

            is_below = (hr_cov < min_coverage) or (mot_cov < min_coverage)
            should_include = not is_below or allow_partial or not skip_flagged

            status = "included" if should_include else "flagged_excluded"

            summary_rows.append(
                {
                    "subject_id": sub,
                    "epochs": n_epochs,
                    "wake_epochs": n_wake,
                    "hr_coverage": hr_cov,
                    "motion_coverage": mot_cov,
                    "valid_motion_fraction": valid_m_fraction,
                    "status": status,
                }
            )

            if should_include:
                tables.append(canonical_tbl)
            elif not skip_flagged and not allow_partial:
                raise ValueError(
                    f"Subject {sub} coverage below {min_coverage:.1%}: "
                    f"HR={hr_cov:.1%}, Motion={mot_cov:.1%}"
                )

        except Exception as exc:
            if not skip_flagged and not allow_partial:
                raise
            logger.warning("Failed or skipped subject %s: %s", sub, exc)
            summary_rows.append(
                {
                    "subject_id": sub,
                    "epochs": 0,
                    "wake_epochs": 0,
                    "hr_coverage": 0.0,
                    "motion_coverage": 0.0,
                    "valid_motion_fraction": 0.0,
                    "status": f"error: {exc}",
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    if not tables:
        combined_df = pd.DataFrame()
    else:
        combined_df = pd.concat(tables, ignore_index=True)
        validate_epoch_table(combined_df)

    if output_path is not None and not combined_df.empty:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(out, index=False)
        logger.info("Saved Dataset v2 epoch features to %s (%d rows)", out, len(combined_df))

    return combined_df, summary_df
