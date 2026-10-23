"""
Integration test loading and epoching the actual BIDSleep Bidslab00/3 recording.
"""

from pathlib import Path

import numpy as np
import pytest

from src.io.contracts import SignalQuality, SleepStage
from src.io.loader import load_night
from src.preprocessing.epoching import align_night_to_epochs

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "Bidslab00" / "3"


@pytest.mark.skipif(not (RAW_DIR / "labels.mat").is_file(), reason="Raw dataset not found")
def test_end_to_end_bidsleep_night():
    # 1. Load the complete raw night
    rec = load_night(RAW_DIR, subject_id="Bidslab00", night_id="3")

    assert rec.has_hr is True
    assert rec.has_labels is True
    assert rec.total_epochs == 993
    assert rec.labels is not None
    assert len(rec.labels) == 993

    # Check stage counts match data notes exactly
    unique, counts = np.unique(rec.labels, return_counts=True)
    stage_counts = dict(zip(unique, counts, strict=True))
    assert stage_counts[SleepStage.WAKE] == 35
    assert stage_counts[SleepStage.N1] == 65
    assert stage_counts[SleepStage.N2] == 451
    assert stage_counts[SleepStage.N3] == 203
    assert stage_counts[SleepStage.REM] == 239

    # 2. Align to 30-second epochs
    epoched = align_night_to_epochs(rec)
    df = epoched.epochs_df

    # Must preserve exactly 993 rows
    assert len(df) == 993
    assert list(df["epoch"]) == list(range(993))

    # HR audit verification (matching data_notes.md)
    hr_audit = epoched.metadata["hr_audit"]
    assert hr_audit["total_samples"] == 5282
    assert hr_audit["samples_before_start"] == 16
    assert hr_audit["samples_inside_window"] == 5266
    assert hr_audit["samples_after_window"] == 0
    assert hr_audit["epochs_missing"] == 2
    assert hr_audit["missing_epoch_indices"] == [991, 992]

    # Verify epochs 991 and 992 have hr_count == 0
    assert df.loc[991, "hr_count"] == 0
    assert df.loc[992, "hr_count"] == 0
    assert bool(df.loc[991, "has_hr"]) is False
    assert bool(df.loc[992, "has_hr"]) is False

    # Verify epoch 990 has hr_count == 1 and motion missing (since motion ends early)
    assert df.loc[990, "hr_count"] == 1
    assert df.loc[990, "quality_flag"] == SignalQuality.MISSING_MOTION.value

    # Verify epochs 991 and 992 have quality flag EMPTY (both sensors ended)
    assert df.loc[991, "quality_flag"] == SignalQuality.EMPTY.value
    assert df.loc[992, "quality_flag"] == SignalQuality.EMPTY.value

    # Verify duration and timestamps
    assert pytest.approx(rec.duration_seconds, 1.0) == 993 * 30.0
