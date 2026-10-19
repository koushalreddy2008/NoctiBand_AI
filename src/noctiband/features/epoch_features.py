"""30-second epoch feature extraction and participant-level dataset splitting.

Offline nightly processing:
In offline overnight sleep staging, the full recording is available post-acquisition.
Centered rolling windows incorporating +-2 and +-5 epochs (+-60 s and +-150 s) of future
context are biologically standard (matching human expert EEG review, which inspects
surrounding epochs for stage transitions) and valid without real-time causality constraints.

Strict leakage guards:
Splitting is performed exclusively at the subject level. At least 3 distinct subjects
are strictly required to form valid train, validation, and test sets. Fallback to
night-level, epoch-level, or leave-one-night-out splitting is strictly prohibited to prevent
intra-subject biometric correlation leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from noctiband.io.loader import load_night
from noctiband.preprocessing.epoching import assign_epochs
from noctiband.types import EPOCH_S, NightData

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/features.yaml")

METADATA_COLUMNS: list[str] = [
    "subject",
    "night",
    "epoch",
    "epoch_timestamp",
]

LABEL_COLUMNS: list[str] = [
    "label",
    "label_binary",
    "label_3class",
]

BASE_FEATURE_COLUMNS: list[str] = [
    # HR base features
    "hr_count",
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_range",
    "hr_last_minus_first",
    "hr_mean_norm",
    "hr_delta",
    "hr_coverage",
    "hr_missing",
    "hr_partial",
    # Motion base features
    "motion_count",
    "motion_mag_mean",
    "motion_mag_std",
    "motion_mag_max",
    "motion_enmo_mean",
    "motion_activity_count",
    "motion_fraction_still",
    "motion_coverage",
    "motion_missing",
    "motion_partial",
    "partial_motion",
    "epochs_since_last_high_activity",
    # Centered rolling context features (w=5)
    "hr_mean_roll_mean_w5",
    "hr_mean_roll_std_w5",
    "hr_mean_norm_roll_mean_w5",
    "hr_mean_norm_roll_std_w5",
    "motion_mag_mean_roll_mean_w5",
    "motion_mag_mean_roll_std_w5",
    "motion_enmo_mean_roll_mean_w5",
    "motion_enmo_mean_roll_std_w5",
    # Centered rolling context features (w=11)
    "hr_mean_roll_mean_w11",
    "hr_mean_roll_std_w11",
    "hr_mean_norm_roll_mean_w11",
    "hr_mean_norm_roll_std_w11",
    "motion_mag_mean_roll_mean_w11",
    "motion_mag_mean_roll_std_w11",
    "motion_enmo_mean_roll_mean_w11",
    "motion_enmo_mean_roll_std_w11",
]

# Standard exported FEATURE_COLUMNS allowlist
FEATURE_COLUMNS: list[str] = list(BASE_FEATURE_COLUMNS)


def load_feature_config(config: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Load feature configuration from dict, file path, or default YAML."""
    if isinstance(config, dict):
        return config

    cfg_path = Path(config) if config is not None else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                return loaded

    # Fallback default configuration
    return {
        "epoch_duration_s": EPOCH_S,
        "hr": {
            "delta": True,
            "robust_norm": True,
            "min_samples_per_epoch": 2,
            "expected_samples_per_epoch": 6,
        },
        "motion": {
            "activity_threshold": 0.05,
            "still_threshold": 0.02,
            "high_activity_threshold": 0.15,
            "min_samples_per_epoch": 500,
            "expected_samples_per_epoch": 1500,
        },
        "context": {
            "windows": [2, 5],
            "features": ["hr_mean", "hr_mean_norm", "motion_mag_mean", "motion_enmo_mean"],
        },
        "time": {
            "include_hours_since_start": False,
        },
    }


def build_epoch_table(
    night: NightData,
    config: dict[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Extract 30-second epoch features for a single night recording.

    Parameters
    ----------
    night : NightData
        Container with subject, night, hr, motion, labels, and rec_start.
    config : dict or Path or None, optional
        Configuration dict or path to YAML config file.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed 0..n_epochs-1 containing METADATA_COLUMNS,
        FEATURE_COLUMNS, and LABEL_COLUMNS.
    """
    cfg = load_feature_config(config)
    hr_cfg = cfg.get("hr", {})
    motion_cfg = cfg.get("motion", {})
    context_cfg = cfg.get("context", {})
    time_cfg = cfg.get("time", {})

    expected_hr_samples = float(hr_cfg.get("expected_samples_per_epoch", 6))
    min_hr_samples = int(hr_cfg.get("min_samples_per_epoch", 2))

    expected_motion_samples = float(motion_cfg.get("expected_samples_per_epoch", 1500))
    min_motion_samples = int(motion_cfg.get("min_samples_per_epoch", 500))
    activity_threshold = float(motion_cfg.get("activity_threshold", 0.05))
    still_threshold = float(motion_cfg.get("still_threshold", 0.02))
    high_activity_threshold = float(motion_cfg.get("high_activity_threshold", 0.15))

    # Determine total epochs in recording
    if night.labels is not None:
        n_epochs = len(night.labels)
    else:
        max_hr_t = (
            float(night.hr["timestamp"].max()) if not night.hr.empty else night.rec_start
        )
        max_m_t = (
            float(night.motion["timestamp"].max()) if not night.motion.empty else night.rec_start
        )
        n_epochs = max(1, int(np.ceil((max(max_hr_t, max_m_t) - night.rec_start) / float(EPOCH_S))))

    epochs = np.arange(n_epochs, dtype=int)
    epoch_timestamps = night.rec_start + epochs * float(EPOCH_S)

    # 1. Metadata
    df = pd.DataFrame(
        {
            "subject": night.subject,
            "night": night.night,
            "epoch": epochs,
            "epoch_timestamp": epoch_timestamps,
        }
    )

    # 2. Heart Rate Features
    hr_assigned = assign_epochs(night.hr, night.rec_start, n_epochs)
    if not hr_assigned.empty:
        hr_grp = hr_assigned.groupby("epoch")["hr"]
        hr_count_s = hr_grp.count()
        hr_mean_s = hr_grp.mean()
        hr_std_s = hr_grp.std()
        hr_min_s = hr_grp.min()
        hr_max_s = hr_grp.max()
        hr_first_s = hr_grp.first()
        hr_last_s = hr_grp.last()

        df["hr_count"] = hr_count_s.reindex(epochs, fill_value=0).astype(int).to_numpy()
        df["hr_mean"] = hr_mean_s.reindex(epochs).to_numpy()
        df["hr_std"] = hr_std_s.reindex(epochs).to_numpy()
        df["hr_min"] = hr_min_s.reindex(epochs).to_numpy()
        df["hr_max"] = hr_max_s.reindex(epochs).to_numpy()
        hr_first = hr_first_s.reindex(epochs).to_numpy()
        hr_last = hr_last_s.reindex(epochs).to_numpy()

        df["hr_range"] = df["hr_max"] - df["hr_min"]
        df["hr_last_minus_first"] = hr_last - hr_first
    else:
        df["hr_count"] = 0
        df["hr_mean"] = np.nan
        df["hr_std"] = np.nan
        df["hr_min"] = np.nan
        df["hr_max"] = np.nan
        df["hr_range"] = np.nan
        df["hr_last_minus_first"] = np.nan

    # Robust normalization across this night only
    if not night.hr.empty:
        night_hr_series = night.hr["hr"].dropna()
        if not night_hr_series.empty:
            med_hr = float(night_hr_series.median())
            q25 = float(night_hr_series.quantile(0.25))
            q75 = float(night_hr_series.quantile(0.75))
            iqr_hr = q75 - q25
            if iqr_hr > 0:
                df["hr_mean_norm"] = (df["hr_mean"] - med_hr) / iqr_hr
            else:
                df["hr_mean_norm"] = df["hr_mean"] - med_hr
        else:
            df["hr_mean_norm"] = np.nan
    else:
        df["hr_mean_norm"] = np.nan

    # HR delta against previous epoch
    df["hr_delta"] = df["hr_mean"].diff()

    # HR quality and missing flags
    df["hr_coverage"] = (df["hr_count"] / expected_hr_samples).clip(0.0, 1.0)
    df["hr_missing"] = df["hr_count"] == 0
    df["hr_partial"] = (df["hr_count"] > 0) & (df["hr_count"] < min_hr_samples)

    # 3. Motion Features
    motion_assigned = assign_epochs(night.motion, night.rec_start, n_epochs)
    if not motion_assigned.empty:
        x = motion_assigned["x"].to_numpy(dtype=float)
        y = motion_assigned["y"].to_numpy(dtype=float)
        z = motion_assigned["z"].to_numpy(dtype=float)
        mag = np.sqrt(x * x + y * y + z * z)
        enmo = np.clip(mag - 1.0, 0.0, None)

        motion_proc = motion_assigned.copy()
        motion_proc["mag"] = mag
        motion_proc["enmo"] = enmo
        motion_proc["is_active"] = (enmo > activity_threshold).astype(int)
        motion_proc["is_still"] = (enmo < still_threshold).astype(int)

        m_grp = motion_proc.groupby("epoch")
        m_count_s = m_grp.size()
        m_mag_mean_s = m_grp["mag"].mean()
        m_mag_std_s = m_grp["mag"].std()
        m_mag_max_s = m_grp["mag"].max()
        m_enmo_mean_s = m_grp["enmo"].mean()
        m_act_count_s = m_grp["is_active"].sum()
        m_still_frac_s = m_grp["is_still"].mean()

        df["motion_count"] = m_count_s.reindex(epochs, fill_value=0).astype(int).to_numpy()
        df["motion_mag_mean"] = m_mag_mean_s.reindex(epochs).to_numpy()
        df["motion_mag_std"] = m_mag_std_s.reindex(epochs).to_numpy()
        df["motion_mag_max"] = m_mag_max_s.reindex(epochs).to_numpy()
        df["motion_enmo_mean"] = m_enmo_mean_s.reindex(epochs).to_numpy()
        df["motion_activity_count"] = m_act_count_s.reindex(epochs).to_numpy()
        df["motion_fraction_still"] = m_still_frac_s.reindex(epochs).to_numpy()
    else:
        df["motion_count"] = 0
        df["motion_mag_mean"] = np.nan
        df["motion_mag_std"] = np.nan
        df["motion_mag_max"] = np.nan
        df["motion_enmo_mean"] = np.nan
        df["motion_activity_count"] = np.nan
        df["motion_fraction_still"] = np.nan

    # Zero-reading validity guard (Amendment 2: Emit NaN for epochs without valid motion)
    zero_motion_mask = df["motion_count"] == 0
    df.loc[zero_motion_mask, "motion_mag_mean"] = np.nan
    df.loc[zero_motion_mask, "motion_mag_std"] = np.nan
    df.loc[zero_motion_mask, "motion_mag_max"] = np.nan
    df.loc[zero_motion_mask, "motion_enmo_mean"] = np.nan
    df.loc[zero_motion_mask, "motion_activity_count"] = np.nan
    df.loc[zero_motion_mask, "motion_fraction_still"] = np.nan

    # Motion quality and missing flags
    df["motion_coverage"] = (df["motion_count"] / expected_motion_samples).clip(0.0, 1.0)
    df["motion_missing"] = zero_motion_mask
    df["motion_partial"] = (df["motion_count"] > 0) & (df["motion_count"] < min_motion_samples)
    df["partial_motion"] = df["motion_partial"].copy()

    # Distance since last high activity epoch
    epochs_since_last_high = np.full(n_epochs, np.nan, dtype=float)
    last_high_idx: int | None = None
    for i in range(n_epochs):
        if df["motion_missing"].iloc[i]:
            epochs_since_last_high[i] = np.nan
            continue

        enmo_val = df["motion_enmo_mean"].iloc[i]
        if pd.notna(enmo_val) and enmo_val >= high_activity_threshold:
            last_high_idx = i
            epochs_since_last_high[i] = 0.0
        elif last_high_idx is not None:
            epochs_since_last_high[i] = float(i - last_high_idx)
        else:
            epochs_since_last_high[i] = np.nan

    df["epochs_since_last_high_activity"] = epochs_since_last_high

    # 4. Centered Rolling Context Features (Offline future context valid)
    radius_windows = context_cfg.get("windows", [2, 5])
    roll_target_feats = context_cfg.get(
        "features", ["hr_mean", "hr_mean_norm", "motion_mag_mean", "motion_enmo_mean"]
    )

    for radius in radius_windows:
        w_size = 2 * int(radius) + 1
        for feat in roll_target_feats:
            if feat in df.columns:
                roll_obj = df[feat].rolling(window=w_size, center=True, min_periods=1)
                df[f"{feat}_roll_mean_w{w_size}"] = roll_obj.mean()
                df[f"{feat}_roll_std_w{w_size}"] = roll_obj.std()

    # 5. Optional Time Feature
    if time_cfg.get("include_hours_since_start", False):
        df["hours_since_start"] = epochs * float(EPOCH_S) / 3600.0

    # 6. Labels (if present)
    if night.labels is not None:
        df["label"] = night.labels.astype(int)
        df["label_binary"] = (night.labels > 0).astype(int)
        # 0 = Wake (0), 1 = NREM (1, 2, 3), 2 = REM (4)
        df["label_3class"] = np.where(
            night.labels == 0,
            0,
            np.where(night.labels == 4, 2, 1),
        ).astype(int)
    else:
        df["label"] = np.nan
        df["label_binary"] = np.nan
        df["label_3class"] = np.nan

    # Arrange columns deterministically
    present_meta = [c for c in METADATA_COLUMNS if c in df.columns]
    present_labels = [c for c in LABEL_COLUMNS if c in df.columns]
    known_feats = [c for c in BASE_FEATURE_COLUMNS if c in df.columns]
    extra_feats = [
        c
        for c in df.columns
        if c not in present_meta and c not in present_labels and c not in known_feats
    ]

    return df[present_meta + known_feats + extra_feats + present_labels]


def build_dataset(
    raw_root: str | Path,
    out_path: str | Path | None = None,
    config: dict[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Scan raw data directory, load all nights, extract features, and concatenate.

    Parameters
    ----------
    raw_root : str or Path
        Root directory containing `<subject>/<night>/` structure.
    out_path : str or Path or None, optional
        If provided, save the combined dataset to CSV.
    config : dict or Path or None, optional
        Feature configuration override.

    Returns
    -------
    pd.DataFrame
        Concatenated epoch-level dataset across all valid subjects and nights.
    """
    raw_path = Path(raw_root)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data root directory not found: {raw_root}")

    tables: list[pd.DataFrame] = []
    # Deterministic alphabetical ordering
    for subj_dir in sorted(raw_path.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("."):
            continue
        for night_dir in sorted(subj_dir.iterdir()):
            if not night_dir.is_dir() or night_dir.name.startswith("."):
                continue

            try:
                night_data = load_night(
                    night_dir,
                    subject=subj_dir.name,
                    night=night_dir.name,
                )
                tbl = build_epoch_table(night_data, config=config)
                tables.append(tbl)
                logger.info(
                    "Extracted %d epochs from %s/%s",
                    len(tbl),
                    subj_dir.name,
                    night_dir.name,
                )
            except Exception as exc:
                logger.warning("Skipping night at %s: %s", night_dir, exc)

    if not tables:
        return pd.DataFrame(columns=METADATA_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS)

    combined_df = pd.concat(tables, ignore_index=True)

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(out, index=False)
        logger.info("Saved combined epoch dataset to %s (%d rows)", out, len(combined_df))

    return combined_df


def make_split(
    table: pd.DataFrame,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    out_path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Partition dataset by subject into train, validation, and test splits.

    Strict leakage guard:
    Splitting is performed exclusively at the subject level. At least 3 distinct subjects
    are strictly required to form valid train, validation, and test sets. Fallback to
    night-level, epoch-level, or leave-one-night-out splitting is prohibited to prevent
    intra-subject biometric correlation leakage.

    Parameters
    ----------
    table : pd.DataFrame
        Dataset table containing a 'subject' column.
    seed : int, default=42
        Random seed for deterministic subject shuffling.
    ratios : tuple of float, default=(0.7, 0.15, 0.15)
        Proportions for (train, val, test). Must sum to 1.0.
    out_path : str or Path, optional
        If provided, save the split dictionary as YAML.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys 'train', 'val', 'test' mapping to lists of subject identifiers.

    Raises
    ------
    ValueError
        If fewer than 3 distinct subjects are present in `table`.
    """
    if "subject" not in table.columns:
        raise ValueError(
            "DataFrame must contain a 'subject' column to perform subject-level split."
        )

    subjects = sorted(table["subject"].dropna().unique())
    n_subjects = len(subjects)

    # Amendment 1: strictly require >= 3 distinct subjects; no fallback
    if n_subjects < 3:
        raise ValueError(
            f"make_split requires at least 3 distinct subjects to partition into train, "
            f"validation, and test splits without data leakage, but found only {n_subjects} "
            f"distinct subject(s): {subjects}. Fallback to night-level or epoch-level splitting "
            f"is strictly prohibited to prevent intra-subject biometric leakage."
        )

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    rng = np.random.default_rng(seed)
    shuffled_subjects = [str(s) for s in rng.permutation(subjects)]

    # Guarantee at least 1 subject per split when n_subjects >= 3
    n_val = max(1, int(round(n_subjects * ratios[1])))
    n_test = max(1, int(round(n_subjects * ratios[2])))
    if n_val + n_test >= n_subjects:
        n_val = 1
        n_test = 1
    n_train = n_subjects - n_val - n_test

    train_subjects = shuffled_subjects[:n_train]
    val_subjects = shuffled_subjects[n_train : n_train + n_val]
    test_subjects = shuffled_subjects[n_train + n_val :]

    split_dict = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(split_dict, f, sort_keys=False)
        logger.info("Saved subject split to %s: %s", out, split_dict)

    return split_dict
