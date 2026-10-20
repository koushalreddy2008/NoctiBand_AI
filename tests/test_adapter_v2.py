"""Unit and integration tests for Dataset v2 adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from noctiband.data_schema import check_epoch_table, validate_epoch_table
from noctiband.io.adapter_v2 import (
    RAW_LABEL_MAP,
    build_dataset_v2,
    build_subject_features_v2,
    compute_epoch_coverage_fraction,
    load_subject_v2,
)


@pytest.fixture
def mock_raw_v2(tmp_path: Path) -> Path:
    """Create a small programmatic fixture mimicking Dataset v2 layout."""
    base = tmp_path / "raw_v2_mock"
    (base / "heart_rate").mkdir(parents=True)
    (base / "labels").mkdir(parents=True)
    (base / "motion").mkdir(parents=True)
    (base / "steps").mkdir(parents=True)

    # Subject 101: 5 epochs (150 s), full coverage
    # Labels: 0 (Wake), 1 (N1), 2 (N2), 4 (N4), -1 (unscored)
    labels_content = "0 0\n30 1\n60 2\n90 4\n120 -1\n"
    (base / "labels" / "101_labeled_sleep.txt").write_text(labels_content, encoding="utf-8")

    # HR: 1 sample every 5s from t=-20 to t=160
    hr_lines = [f"{t},{70 + (t % 10)}" for t in range(-20, 165, 5)]
    (base / "heart_rate" / "101_heartrate.txt").write_text("\n".join(hr_lines), encoding="utf-8")

    # Motion: 50 Hz from t=-5 to t=155
    mot_lines = []
    for t_step in range(-250, 7750):
        t = t_step * 0.02
        mot_lines.append(f"{t:.4f} 0.01 0.02 -0.99")
    (base / "motion" / "101_acceleration.txt").write_text("\n".join(mot_lines), encoding="utf-8")

    # Steps: 10-minute bins
    steps_content = "-600,0\n0,12\n"
    (base / "steps" / "101_steps.txt").write_text(steps_content, encoding="utf-8")

    # Subject 102: Partial motion coverage (stops at t=40s out of 150s = ~26.7%)
    (base / "labels" / "102_labeled_sleep.txt").write_text(labels_content, encoding="utf-8")
    (base / "heart_rate" / "102_heartrate.txt").write_text("\n".join(hr_lines), encoding="utf-8")
    partial_mot_lines = [f"{t * 0.02:.4f} 0.01 0.02 -0.99" for t in range(0, 2000)]
    (base / "motion" / "102_acceleration.txt").write_text(
        "\n".join(partial_mot_lines), encoding="utf-8"
    )
    (base / "steps" / "102_steps.txt").write_text(steps_content, encoding="utf-8")

    return base


def test_epoch_coverage_fraction_calculation() -> None:
    """Test compute_epoch_coverage_fraction logic on known timestamps."""
    ts = np.array([5.0, 15.0, 35.0, 65.0])
    # 4 epochs: [0, 30), [30, 60), [60, 90), [90, 120)
    # Epochs with data: 0, 1, 2 (3/4 = 0.75)
    cov = compute_epoch_coverage_fraction(ts, 4)
    assert np.isclose(cov, 0.75)


def test_load_subject_v2_full_coverage(mock_raw_v2: Path) -> None:
    """Test loading a subject with 100% coverage and verifying label mappings."""
    night = load_subject_v2(mock_raw_v2, "101", allow_partial=False, min_coverage=0.95)
    assert night.subject == "101"
    assert night.night == "1"
    assert night.rec_start == 0.0
    assert night.labels is not None
    assert len(night.labels) == 5

    # Check stage mapping:
    # 0 -> 0 (Wake)
    # 1 -> 1 (N1)
    # 2 -> 2 (N2)
    # 4 -> 3 (N4 merged into N3)
    # -1 -> 5 (Unknown)
    expected_labels = np.array([0, 1, 2, 3, 5])
    np.testing.assert_array_equal(night.labels, expected_labels)


def test_load_subject_v2_coverage_guard_raises(mock_raw_v2: Path) -> None:
    """Test that load_subject_v2 raises ValueError when coverage < 0.95."""
    with pytest.raises(ValueError, match="failed coverage guard"):
        load_subject_v2(mock_raw_v2, "102", allow_partial=False, min_coverage=0.95)

    # Should succeed when allow_partial=True
    night_partial = load_subject_v2(mock_raw_v2, "102", allow_partial=True, min_coverage=0.95)
    assert night_partial.subject == "102"


def test_build_subject_features_v2_canonical_schema(mock_raw_v2: Path) -> None:
    """Test that build_subject_features_v2 outputs valid canonical epoch table."""
    df = build_subject_features_v2(mock_raw_v2, "101", min_coverage=0.95)
    assert len(df) == 5

    # Canonical schema columns
    assert "subject_id" in df.columns
    assert "night_id" in df.columns
    assert "epoch_idx" in df.columns
    assert "label" in df.columns
    assert "label_binary" in df.columns
    assert "label_3class" in df.columns
    assert "label_unknown" in df.columns

    # Unknown label handling: epoch 4 had raw -1 -> label is NaN, label_unknown is True
    assert df["label_unknown"].iloc[4]
    assert np.isnan(df["label"].iloc[4])
    assert np.isnan(df["label_binary"].iloc[4])
    assert np.isnan(df["label_3class"].iloc[4])

    # Wake label handling: epoch 0 had raw 0 -> label 0, binary 0, 3class 0
    assert not df["label_unknown"].iloc[0]
    assert df["label"].iloc[0] == 0.0
    assert df["label_binary"].iloc[0] == 0.0
    assert df["label_3class"].iloc[0] == 0.0

    # Steps exclusion: no step features should exist in table
    assert not any("step" in col.lower() for col in df.columns)

    # Passes strict schema validation
    check_epoch_table(df)


def test_build_dataset_v2_skipping_flagged(mock_raw_v2: Path) -> None:
    """Test build_dataset_v2 skips flagged subjects by default."""
    combined, summary = build_dataset_v2(mock_raw_v2, min_coverage=0.95, skip_flagged=True)
    assert len(summary) == 2
    assert summary.loc[summary["subject_id"] == "101", "status"].iloc[0] == "included"
    assert summary.loc[summary["subject_id"] == "102", "status"].iloc[0] == "flagged_excluded"

    # Only subject 101 epochs are included
    assert combined["subject_id"].unique().tolist() == ["101"]
    assert len(combined) == 5


def test_build_dataset_v2_allow_partial(mock_raw_v2: Path) -> None:
    """Test build_dataset_v2 includes all subjects when allow_partial=True."""
    combined, summary = build_dataset_v2(
        mock_raw_v2, min_coverage=0.95, allow_partial=True, skip_flagged=False
    )
    assert len(summary) == 2
    assert (summary["status"] == "included").all()
    assert len(combined) == 10


def test_raw_label_map_completeness() -> None:
    """Verify raw label map contains all expected sleep stage integers."""
    expected_keys = {0, 1, 2, 3, 4, 5, -1}
    assert expected_keys.issubset(RAW_LABEL_MAP.keys())


@pytest.mark.data
def test_real_data_v2_integration_gated() -> None:
    """Integration test gated on real data/raw_v2 availability and SHA-256 verification."""
    data_dir = Path("data/raw_v2")
    sums_file = data_dir / "SHA256SUMS.txt"
    if not data_dir.exists() or not sums_file.exists():
        pytest.skip("data/raw_v2 or SHA256SUMS.txt missing; skipping real data integration test")

    # Load verified subject 1360686
    sub_id = "1360686"
    try:
        df = build_subject_features_v2(data_dir, sub_id, min_coverage=0.95)
    except Exception as exc:
        pytest.fail(f"Failed to load and process real subject {sub_id}: {exc}")

    assert len(df) == 968
    validation = validate_epoch_table(df)
    assert validation.is_valid, f"Canonical validation failed: {validation.errors}"
    assert validation.n_subjects == 1
    assert validation.n_wake == 117
