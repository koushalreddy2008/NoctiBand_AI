"""Tests for noctiband.io.loader."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from noctiband.io.loader import is_header_present, load_night, parse_rec_start
from noctiband.types import NightData


def create_synthetic_night(
    night_dir: Path,
    rec_start_str: str = "2021-12-04 23:40:29",
    expert_labels: list[int] | None = None,
    hr_rows: list[list[float]] | None = None,
    motion_header: bool = True,
    motion_rows: list[list[float]] | None = None,
    extra_hr_col: bool = False,
    extra_motion_col: bool = False,
) -> float:
    """Helper to create synthetic raw night fixtures in tmp_path."""
    night_dir.mkdir(parents=True, exist_ok=True)

    ts = pd.Timestamp(rec_start_str).tz_localize("America/New_York")
    rec_start_unix = float(ts.timestamp())

    if expert_labels is None:
        expert_labels = [0, 1, 2, 3, 4]
    savemat(
        str(night_dir / "labels.mat"),
        {
            "recStart": np.array([rec_start_str], dtype="<U19"),
            "expert_label": np.array(expert_labels, dtype=np.int32).reshape(-1, 1),
        },
    )

    if hr_rows is None:
        hr_rows = [
            [rec_start_unix, 65.0],
            [rec_start_unix + 70.0, 66.0],
            [rec_start_unix + 145.0, 67.0],
        ]
    with open(night_dir / "hr.csv", "w", encoding="utf-8") as f:
        for r in hr_rows:
            if extra_hr_col:
                f.write(f"{r[0]},{r[1]},999.0\n")
            else:
                f.write(f"{r[0]},{r[1]}\n")

    if motion_rows is None:
        motion_rows = [
            [rec_start_unix, 0.0, 0.0, 1.0],
            [rec_start_unix + 70.0, 0.01, -0.01, 0.99],
            [rec_start_unix + 148.0, 0.02, 0.00, 1.01],
        ]
    with open(night_dir / "motion.csv", "w", encoding="utf-8") as f:
        if motion_header:
            if extra_motion_col:
                f.write("Timestamp,x,y,z,extra\n")
            else:
                f.write("Timestamp,x,y,z\n")
        for r in motion_rows:
            if extra_motion_col:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},888.0\n")
            else:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")

    return rec_start_unix


def test_is_header_present():
    assert is_header_present("Timestamp") is True
    assert is_header_present("time") is True
    assert is_header_present("1638679128.54") is False
    assert is_header_present(" 123.45 ") is False


def test_load_night_basic(tmp_path: Path):
    subject_dir = tmp_path / "Subj01"
    night_dir = subject_dir / "Night01"
    rec_start = create_synthetic_night(night_dir)

    data = load_night(night_dir)
    assert isinstance(data, NightData)
    assert data.subject == "Subj01"
    assert data.night == "Night01"
    assert data.rec_start == rec_start
    assert data.labels is not None
    assert np.array_equal(data.labels, np.array([0, 1, 2, 3, 4]))
    assert list(data.hr.columns) == ["timestamp", "hr"]
    assert list(data.motion.columns) == ["timestamp", "x", "y", "z"]
    assert len(data.hr) == 3
    assert len(data.motion) == 3


def test_load_night_custom_subject_and_night(tmp_path: Path):
    night_dir = tmp_path / "somedir"
    create_synthetic_night(night_dir)

    data = load_night(night_dir, subject="MySubject", night="MyNight")
    assert data.subject == "MySubject"
    assert data.night == "MyNight"


def test_load_night_header_variations(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    # Motion with no header
    night_dir_no_hdr = tmp_path / "no_hdr"
    create_synthetic_night(night_dir_no_hdr, motion_header=False)
    data1 = load_night(night_dir_no_hdr)
    assert list(data1.motion.columns) == ["timestamp", "x", "y", "z"]

    # Motion and HR with extra columns
    night_dir_extra = tmp_path / "extra_cols"
    create_synthetic_night(
        night_dir_extra, motion_header=True, extra_hr_col=True, extra_motion_col=True
    )
    with caplog.at_level(logging.WARNING):
        data2 = load_night(night_dir_extra)
    assert list(data2.hr.columns) == ["timestamp", "hr"]
    assert list(data2.motion.columns) == ["timestamp", "x", "y", "z"]
    assert any("Ignoring extra columns" in r.message for r in caplog.records)


def test_load_night_unsorted_and_duplicates(tmp_path: Path):
    night_dir = tmp_path / "unsorted_dup"
    rec_start = 1638679229.0
    hr_rows = [
        [rec_start + 145.0, 70.0],
        [rec_start + 70.0, 65.0],
        [rec_start + 70.0, 65.0],  # duplicate
        [rec_start, 60.0],
    ]
    motion_rows = [
        [rec_start + 148.0, 0.1, 0.1, 0.9],
        [rec_start + 70.0, 0.0, 0.0, 1.0],
        [rec_start + 70.0, 0.0, 0.0, 1.0],  # duplicate
        [rec_start, 0.0, 0.1, 1.0],
    ]
    create_synthetic_night(
        night_dir, hr_rows=hr_rows, motion_rows=motion_rows, motion_header=True
    )

    data = load_night(night_dir)
    # Deduplication and sorting check
    assert len(data.hr) == 3
    assert data.hr["timestamp"].is_monotonic_increasing
    assert len(data.motion) == 3
    assert data.motion["timestamp"].is_monotonic_increasing


def test_load_night_drops_nans(tmp_path: Path):
    night_dir = tmp_path / "nans"
    rec_start = 1638679229.0
    hr_rows = [
        [rec_start, 60.0],
        [rec_start + 70.0, float("nan")],
        [rec_start + 145.0, 62.0],
    ]
    motion_rows = [
        [rec_start, 0.0, 0.0, 1.0],
        [rec_start + 70.0, float("nan"), 0.0, 1.0],
        [rec_start + 148.0, 0.0, 0.0, 1.0],
    ]
    create_synthetic_night(night_dir, hr_rows=hr_rows, motion_rows=motion_rows)

    data = load_night(night_dir)
    assert len(data.hr) == 2
    assert len(data.motion) == 2
    assert not data.hr.isna().any().any()
    assert not data.motion.isna().any().any()


def test_timezone_dst_handling():
    # Standard time
    ts_std = parse_rec_start("2021-12-04 23:40:29")
    assert isinstance(ts_std, float)

    # Daylight saving time
    ts_dst = parse_rec_start("2021-07-04 23:40:29")
    assert isinstance(ts_dst, float)

    # Non-existent spring forward time (e.g. 2021-03-14 02:30:00 does not exist in ET)
    with pytest.raises(ValueError, match="Ambiguous or non-existent"):
        parse_rec_start("2021-03-14 02:30:00")

    # Ambiguous fall back time (e.g. 2021-11-07 01:30:00 occurs twice in ET)
    with pytest.raises(ValueError, match="Ambiguous or non-existent"):
        parse_rec_start("2021-11-07 01:30:00")


def test_load_night_alignment_guard(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    night_dir_fail = tmp_path / "align_fail"
    rec_start = 1638679229.0
    # First HR reading 4000s away (> 3600s)
    hr_rows_far = [[rec_start + 4000.0, 60.0]]
    motion_rows = [[rec_start, 0.0, 0.0, 1.0]]
    create_synthetic_night(night_dir_fail, hr_rows=hr_rows_far, motion_rows=motion_rows)

    with pytest.raises(ValueError, match="likely timezone/unit bug"):
        load_night(night_dir_fail)

    # First HR reading 400s away (> 300s, <= 3600s): should warn but succeed with allow_partial=True
    night_dir_warn = tmp_path / "align_warn"
    hr_rows_warn = [[rec_start + 400.0, 60.0]]
    create_synthetic_night(night_dir_warn, hr_rows=hr_rows_warn, motion_rows=motion_rows)

    with caplog.at_level(logging.WARNING):
        data = load_night(night_dir_warn, allow_partial=True)
    assert len(data.hr) == 1
    assert any("> 300s" in r.message for r in caplog.records)


def test_load_night_coverage_guard_full(tmp_path: Path):
    """Full coverage (> 95% of window) loads cleanly without warnings."""
    night_dir = tmp_path / "full_cov"
    create_synthetic_night(night_dir)
    data = load_night(night_dir)
    assert len(data.hr) > 0
    assert len(data.motion) > 0


def test_load_night_coverage_guard_partial_10_percent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """10% coverage raises ValueError unless allow_partial=True."""
    night_dir = tmp_path / "partial_10"
    rec_start = 1638679229.0
    # 5 epochs = 150s window; 10% is 15s
    hr_rows_10pct = [[rec_start, 60.0], [rec_start + 15.0, 62.0]]
    motion_rows = [[rec_start, 0.0, 0.0, 1.0], [rec_start + 148.0, 0.0, 0.0, 1.0]]
    create_synthetic_night(night_dir, hr_rows=hr_rows_10pct, motion_rows=motion_rows)

    with pytest.raises(ValueError, match="HR covered fraction 10.0%.*is below 50%"):
        load_night(night_dir, allow_partial=False)

    with caplog.at_level(logging.WARNING):
        data = load_night(night_dir, allow_partial=True)
    assert len(data.hr) == 2
    assert any("below 50%" in r.message for r in caplog.records)


def test_load_night_coverage_guard_early_stop(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Early stop at 40% of window raises ValueError unless allow_partial=True."""
    night_dir = tmp_path / "early_stop"
    rec_start = 1638679229.0
    # 5 epochs = 150s window; 40% is 60s
    motion_rows_40pct = [[rec_start, 0.0, 0.0, 1.0], [rec_start + 60.0, 0.0, 0.0, 1.0]]
    hr_rows = [[rec_start, 60.0], [rec_start + 145.0, 62.0]]
    create_synthetic_night(night_dir, hr_rows=hr_rows, motion_rows=motion_rows_40pct)

    with pytest.raises(ValueError, match="motion covered fraction 40.0%.*is below 50%"):
        load_night(night_dir, allow_partial=False)

    with caplog.at_level(logging.WARNING):
        data = load_night(night_dir, allow_partial=True)
    assert len(data.motion) == 2
    assert any("below 50%" in r.message for r in caplog.records)


def test_load_night_missing_labels_mat(tmp_path: Path):
    night_dir = tmp_path / "no_labels"
    create_synthetic_night(night_dir)
    (night_dir / "labels.mat").unlink()

    # When rec_start is omitted, must raise ValueError
    with pytest.raises(ValueError, match="labels.mat is missing and no rec_start was provided"):
        load_night(night_dir)

    # When rec_start is provided, should succeed with labels=None
    data = load_night(night_dir, rec_start=1638679229.0)
    assert data.labels is None
    assert data.rec_start == 1638679229.0


def test_load_night_missing_csvs(tmp_path: Path):
    night_dir = tmp_path / "missing_csvs"
    create_synthetic_night(night_dir)

    (night_dir / "hr.csv").unlink()
    with pytest.raises(FileNotFoundError, match="hr.csv not found"):
        load_night(night_dir)

    create_synthetic_night(night_dir)
    (night_dir / "motion.csv").unlink()
    with pytest.raises(FileNotFoundError, match="motion.csv not found"):
        load_night(night_dir)


EXPECTED_NIGHT3_MOTION_SHA256 = (
    "7afec69f8cbb1bb41f5c45cf2c2e7c0332a468d510eb6495e082ca529bb2f817"
)


def _compute_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


@pytest.mark.data
def test_load_night_integration_benchmark():
    night_dir = Path("data/raw/Bidslab00/3")
    motion_file = night_dir / "motion.csv"
    if not motion_file.exists() or _compute_sha256(motion_file) != EXPECTED_NIGHT3_MOTION_SHA256:
        pytest.skip("truncated or missing download")

    data = load_night(night_dir)
    assert data.subject == "Bidslab00"
    assert data.night == "3"
    assert data.labels is not None
    assert len(data.labels) == 993

    counts = pd.Series(data.labels).value_counts().to_dict()
    assert counts.get(0, 0) == 35   # Wake
    assert counts.get(1, 0) == 65   # N1
    assert counts.get(2, 0) == 451  # N2
    assert counts.get(3, 0) == 203  # N3
    assert counts.get(4, 0) == 239  # REM

    # Motion covered fraction >= 0.95
    n_epochs = len(data.labels)
    window_start = data.rec_start
    window_end = data.rec_start + n_epochs * 30.0
    m_first = max(float(data.motion["timestamp"].iloc[0]), window_start)
    m_last = min(float(data.motion["timestamp"].iloc[-1]), window_end)
    m_cov = (m_last - m_first) / (n_epochs * 30.0)
    assert m_cov >= 0.95
