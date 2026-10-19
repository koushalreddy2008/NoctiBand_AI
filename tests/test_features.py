"""Unit tests for epoch-level feature extraction and participant splitting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from noctiband.features import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    build_dataset,
    build_epoch_table,
    make_split,
)
from noctiband.types import EPOCH_S, NightData


@pytest.fixture
def synthetic_night() -> NightData:
    """Create a minimal synthetic NightData instance for fast unit tests."""
    rec_start = 1000.0
    labels = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int)

    # Generate HR samples (e.g. 5 samples per epoch for epochs 0..8, none for 9)
    hr_records = []
    for ep in range(9):
        base_t = rec_start + ep * EPOCH_S
        for i in range(5):
            hr_records.append({"timestamp": base_t + i * 5.0, "hr": 60.0 + ep + i})
    hr_df = pd.DataFrame(hr_records)

    # Generate motion samples:
    # Epoch 0: 600 samples (full)
    # Epoch 1: 50 samples (partial, < 500)
    # Epochs 2..9: 0 samples (missing)
    motion_records = []
    # Epoch 0: 600 samples
    for i in range(600):
        motion_records.append(
            {
                "timestamp": rec_start + i * 0.05,
                "x": 0.0,
                "y": 0.0,
                "z": 1.0,
            }
        )
    # Epoch 1: 50 samples
    for i in range(50):
        motion_records.append(
            {
                "timestamp": rec_start + EPOCH_S + i * 0.5,
                "x": 0.0,
                "y": 0.0,
                "z": 1.0,
            }
        )
    motion_df = pd.DataFrame(motion_records)

    return NightData(
        subject="test_subj_01",
        night="1",
        hr=hr_df,
        motion=motion_df,
        labels=labels,
        rec_start=rec_start,
    )


def test_feature_and_label_columns_disjoint():
    """Verify FEATURE_COLUMNS, LABEL_COLUMNS, and METADATA_COLUMNS are mutually disjoint."""
    feat_set = set(FEATURE_COLUMNS)
    label_set = set(LABEL_COLUMNS)
    meta_set = set(METADATA_COLUMNS)

    assert feat_set.isdisjoint(label_set), f"Overlap: {feat_set & label_set}"
    assert feat_set.isdisjoint(meta_set), f"Overlap: {feat_set & meta_set}"
    assert label_set.isdisjoint(meta_set), f"Overlap: {label_set & meta_set}"


def test_epoch_counts_match_labels(synthetic_night: NightData):
    """Verify build_epoch_table produces exactly len(labels) rows."""
    tbl = build_epoch_table(synthetic_night)
    assert len(tbl) == len(synthetic_night.labels)
    assert list(tbl["epoch"]) == list(range(len(synthetic_night.labels)))


def test_features_invariant_to_label_shuffling(synthetic_night: NightData):
    """Verify that features are completely invariant to ground truth label shuffling."""
    tbl_original = build_epoch_table(synthetic_night)

    # Shuffle the labels
    shuffled_labels = np.random.default_rng(123).permutation(synthetic_night.labels)
    night_shuffled = replace(synthetic_night, labels=shuffled_labels)
    tbl_shuffled = build_epoch_table(night_shuffled)

    # Features must match identically
    pd.testing.assert_frame_equal(
        tbl_original[FEATURE_COLUMNS],
        tbl_shuffled[FEATURE_COLUMNS],
        check_exact=True,
    )

    # But labels should differ
    assert not np.array_equal(tbl_original["label"], tbl_shuffled["label"])


def test_per_night_normalization_isolation(synthetic_night: NightData):
    """Verify robust HR normalization uses only the focal night's HR readings."""
    tbl_night_a = build_epoch_table(synthetic_night)

    # Modify HR for another night
    night_b = replace(
        synthetic_night,
        night="2",
        hr=synthetic_night.hr.copy().assign(hr=lambda df: df["hr"] + 50.0),
    )
    tbl_night_b = build_epoch_table(night_b)

    # Night A features must be entirely unaffected by Night B's HR scale
    tbl_night_a_repeat = build_epoch_table(synthetic_night)
    pd.testing.assert_frame_equal(
        tbl_night_a[FEATURE_COLUMNS],
        tbl_night_a_repeat[FEATURE_COLUMNS],
        check_exact=True,
    )

    # Night B normalized HR reflects its own internal distribution
    assert not np.array_equal(
        tbl_night_a["hr_mean"].dropna(), tbl_night_b["hr_mean"].dropna()
    )


def test_missing_and_partial_motion_flags(synthetic_night: NightData):
    """Verify zero-count epochs emit NaNs + motion_missing, and partial counts flag correctly."""
    tbl = build_epoch_table(synthetic_night)

    # Epoch 0: 600 samples (>= min_samples_per_epoch 500)
    assert tbl.loc[0, "motion_count"] == 600
    assert not tbl.loc[0, "motion_missing"]
    assert not tbl.loc[0, "motion_partial"]
    assert not tbl.loc[0, "partial_motion"]
    assert pd.notna(tbl.loc[0, "motion_mag_mean"])
    assert pd.notna(tbl.loc[0, "motion_enmo_mean"])

    # Epoch 1: 50 samples (< min_samples_per_epoch 500)
    assert tbl.loc[1, "motion_count"] == 50
    assert not tbl.loc[1, "motion_missing"]
    assert tbl.loc[1, "motion_partial"]
    assert tbl.loc[1, "partial_motion"]
    assert pd.notna(tbl.loc[1, "motion_mag_mean"])

    # Epoch 2: 0 samples (zero motion)
    assert tbl.loc[2, "motion_count"] == 0
    assert tbl.loc[2, "motion_missing"]
    assert not tbl.loc[2, "motion_partial"]
    assert not tbl.loc[2, "partial_motion"]
    # Amendment 2: zero-motion epochs must emit NaN for motion metrics
    assert pd.isna(tbl.loc[2, "motion_mag_mean"])
    assert pd.isna(tbl.loc[2, "motion_mag_std"])
    assert pd.isna(tbl.loc[2, "motion_mag_max"])
    assert pd.isna(tbl.loc[2, "motion_enmo_mean"])
    assert pd.isna(tbl.loc[2, "motion_activity_count"])
    assert pd.isna(tbl.loc[2, "motion_fraction_still"])


def test_missing_and_partial_hr_flags(synthetic_night: NightData):
    """Verify HR missing and partial flags behavior."""
    tbl = build_epoch_table(synthetic_night)

    # Epoch 0: 5 HR readings
    assert tbl.loc[0, "hr_count"] == 5
    assert not tbl.loc[0, "hr_missing"]
    assert not tbl.loc[0, "hr_partial"]
    assert pd.notna(tbl.loc[0, "hr_mean"])

    # Epoch 9: 0 HR readings
    assert tbl.loc[9, "hr_count"] == 0
    assert tbl.loc[9, "hr_missing"]
    assert not tbl.loc[9, "hr_partial"]
    assert pd.isna(tbl.loc[9, "hr_mean"])


def test_make_split_under_3_subjects_raises_value_error():
    """Verify make_split strictly raises ValueError when distinct subjects < 3."""
    # 1 subject
    df1 = pd.DataFrame({"subject": ["s1"] * 20})
    with pytest.raises(ValueError, match="requires at least 3 distinct subjects"):
        make_split(df1)

    # 2 subjects
    df2 = pd.DataFrame({"subject": ["s1"] * 10 + ["s2"] * 10})
    with pytest.raises(ValueError, match="requires at least 3 distinct subjects"):
        make_split(df2)


def test_make_split_valid_3_and_more_subjects(tmp_path: Path):
    """Verify make_split cleanly partitions >= 3 subjects into disjoint sets."""
    # Exactly 3 subjects
    df3 = pd.DataFrame({"subject": ["s1", "s2", "s3"] * 10})
    split3 = make_split(df3, seed=42)
    assert len(split3["train"]) == 1
    assert len(split3["val"]) == 1
    assert len(split3["test"]) == 1
    all_s3 = set(split3["train"]) | set(split3["val"]) | set(split3["test"])
    assert all_s3 == {"s1", "s2", "s3"}

    # 6 subjects with output path
    df6 = pd.DataFrame({"subject": [f"s{i}" for i in range(6)] * 5})
    out_yaml = tmp_path / "split.yaml"
    split6 = make_split(df6, seed=42, ratios=(0.6, 0.2, 0.2), out_path=out_yaml)

    train_set = set(split6["train"])
    val_set = set(split6["val"])
    test_set = set(split6["test"])

    # Disjointness check
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == set(df6["subject"])

    # Verify written YAML
    assert out_yaml.exists()
    with open(out_yaml, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded == split6


@pytest.mark.data
def test_build_dataset_on_raw_data():
    """Verify build_dataset runs on real raw benchmark data."""
    raw_root = Path("data/raw")
    if not raw_root.exists():
        pytest.skip("data/raw not present")

    df = build_dataset(raw_root)
    # Night 3: 993 epochs, Night 1: 935 epochs -> 1928 epochs total
    assert len(df) == 1928
    assert set(df["subject"].unique()) == {"Bidslab00"}
    assert set(df["night"].unique()) == {"1", "3"}
    assert list(df.columns) == METADATA_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS
