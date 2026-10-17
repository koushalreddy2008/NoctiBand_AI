"""
Tests for raw signal loaders, MAT label parsing, and NightRecording assembly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from src.io.loader import (
    load_bp,
    load_hr,
    load_labels,
    load_motion,
    load_night,
    load_spo2,
)


def test_load_hr_headerless(tmp_path: Path):
    file = tmp_path / "hr.csv"
    file.write_text("1000.0,65.0\n1005.0,66.5\n1010.0,64.0\n", encoding="utf-8")

    df = load_hr(file)
    assert list(df.columns) == ["timestamp", "heart_rate"]
    assert len(df) == 3
    assert df["heart_rate"].iloc[0] == 65.0


def test_load_hr_headered(tmp_path: Path):
    file = tmp_path / "hr_header.csv"
    file.write_text("Time,HR\n1000.0,72.0\n1005.0,73.0\n", encoding="utf-8")

    df = load_hr(file)
    assert list(df.columns) == ["timestamp", "heart_rate"]
    assert len(df) == 2
    assert df["heart_rate"].iloc[0] == 72.0


def test_load_hr_filters_implausible(tmp_path: Path):
    file = tmp_path / "hr_bad.csv"
    # 5.0 BPM (too low) and 350.0 BPM (too high) should be filtered out
    file.write_text("1000.0,5.0\n1005.0,70.0\n1010.0,350.0\n", encoding="utf-8")

    df = load_hr(file)
    assert len(df) == 1
    assert df["heart_rate"].iloc[0] == 70.0


def test_load_hr_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_hr(tmp_path / "nonexistent.csv")

    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_hr(empty_file)


def test_load_motion(tmp_path: Path):
    file = tmp_path / "motion.csv"
    file.write_text("Timestamp,x,y,z\n1000.0,0.1,-0.2,0.98\n1001.0,0.11,-0.19,0.97\n", encoding="utf-8")

    df = load_motion(file)
    assert list(df.columns) == ["timestamp", "x", "y", "z"]
    assert len(df) == 2
    assert df["z"].iloc[0] == 0.98


def test_load_motion_missing_columns(tmp_path: Path):
    file = tmp_path / "bad_motion.csv"
    file.write_text("Timestamp,x\n1000.0,0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_motion(file)


def test_load_labels(tmp_path: Path):
    file = tmp_path / "labels.mat"
    mat_dict = {
        "expert_label": np.array([[0, 1, 2, 3, 4]]),
        "dreem_label": np.array([[0, 1, 2, 2, 4]]),
        "recStart": np.array(["2021-12-04 23:40:29"], dtype=object),
    }
    savemat(str(file), mat_dict)

    labels, rec_start, dreem = load_labels(file)
    assert list(labels) == [0, 1, 2, 3, 4]
    assert isinstance(rec_start, pd.Timestamp)
    assert rec_start.year == 2021
    assert dreem is not None and list(dreem) == [0, 1, 2, 2, 4]


def test_load_bp_and_spo2(tmp_path: Path):
    bp_file = tmp_path / "blood_pressure.csv"
    bp_file.write_text("timestamp,systolic,diastolic,map,pulse_pressure\n1000.0,120.0,80.0,93.3,40.0\n", encoding="utf-8")

    bp_df = load_bp(bp_file)
    assert len(bp_df) == 1
    assert bp_df["systolic"].iloc[0] == 120.0

    spo2_file = tmp_path / "spo2.csv"
    spo2_file.write_text("timestamp,spo2\n1000.0,98.5\n", encoding="utf-8")

    spo2_df = load_spo2(spo2_file)
    assert len(spo2_df) == 1
    assert spo2_df["spo2"].iloc[0] == 98.5


def test_load_night_synthetic(tmp_path: Path):
    (tmp_path / "hr.csv").write_text("1000.0,60.0\n1010.0,62.0\n", encoding="utf-8")
    (tmp_path / "motion.csv").write_text("Timestamp,x,y,z\n1000.0,0.0,0.0,1.0\n", encoding="utf-8")
    mat_dict = {
        "expert_label": np.array([[0, 2]]),
        "recStart": np.array(["2021-12-04 23:40:29"], dtype=object),
    }
    savemat(str(tmp_path / "labels.mat"), mat_dict)

    rec = load_night(tmp_path, subject_id="sub_syn", night_id="night_01")
    assert rec.has_hr is True
    assert rec.has_motion is True
    assert rec.has_labels is True
    assert rec.total_epochs == 2
