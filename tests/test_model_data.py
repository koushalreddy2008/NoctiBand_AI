"""
Tests for model data helpers and split verification.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from noctiband.models.data import (
    check_split,
    feature_columns,
    get_subset,
    load_split,
    load_table,
)


def test_feature_columns_filtering():
    df = pd.DataFrame(
        {
            "subject_id": ["sub1"],
            "night_id": ["n1"],
            "epoch_idx": [0],
            "hr_mean": [65.0],
            "hr_std": [2.5],
            "acc_mag_mean": [0.05],
            "ctx_hour_of_night": [1.0],
            "q_is_missing": [0],
            "sleep_stage": [1],
            "unknown_col": [99],
        }
    )
    cols = feature_columns(df)
    assert cols == ["hr_mean", "hr_std", "acc_mag_mean", "ctx_hour_of_night", "q_is_missing"]
    assert "subject_id" not in cols
    assert "sleep_stage" not in cols
    assert "unknown_col" not in cols


def test_check_split_valid():
    split = {
        "train": ["sub1", "sub2", "sub3"],
        "val": ["sub4"],
        "test": ["sub5"],
    }
    # Should pass without assertion error
    check_split(split)


def test_check_split_train_val_overlap():
    split = {
        "train": ["sub1", "sub2"],
        "val": ["sub2", "sub3"],
        "test": ["sub4"],
    }
    with pytest.raises(AssertionError, match="train and val share subjects"):
        check_split(split)


def test_check_split_train_test_overlap():
    split = {
        "train": ["sub1", "sub2"],
        "val": ["sub3"],
        "test": ["sub1", "sub4"],
    }
    with pytest.raises(AssertionError, match="train and test share subjects"):
        check_split(split)


def test_check_split_val_test_overlap():
    split = {
        "train": ["sub1"],
        "val": ["sub2", "sub3"],
        "test": ["sub3", "sub4"],
    }
    with pytest.raises(AssertionError, match="val and test share subjects"):
        check_split(split)


def test_get_subset_filtering_and_sorting():
    df = pd.DataFrame(
        {
            "subject_id": ["sub2", "sub1", "sub1", "sub3"],
            "night_id": ["n1", "n1", "n1", "n1"],
            "epoch_idx": [0, 1, 0, 0],
            "hr_mean": [60.0, 62.0, 61.0, 59.0],
        }
    )
    split = {"train": ["sub1", "sub2"], "val": ["sub3"], "test": []}

    subset = get_subset(df, split, "train")

    assert len(subset) == 3
    assert set(subset["subject_id"]) == {"sub1", "sub2"}
    # Verify sorting by KEYS (subject_id, night_id, epoch_idx)
    assert list(subset["subject_id"]) == ["sub1", "sub1", "sub2"]
    assert list(subset["epoch_idx"]) == [0, 1, 0]


def test_load_split_and_table_with_temp_files(tmp_path: Path):
    split_file = tmp_path / "test_split.json"
    data_file = tmp_path / "test_data.csv"

    split_data = {"train": ["sub1"], "val": ["sub2"], "test": ["sub3"]}
    split_file.write_text(json.dumps(split_data))

    df_data = pd.DataFrame(
        {
            "subject_id": ["sub1"],
            "night_id": ["n1"],
            "epoch_idx": [0],
            "hr_mean": [70.0],
        }
    )
    df_data.to_csv(data_file, index=False)

    loaded_split = load_split(split_file)
    assert loaded_split == split_data

    loaded_table = load_table(data_file)
    assert len(loaded_table) == 1
    assert loaded_table["hr_mean"].iloc[0] == 70.0
