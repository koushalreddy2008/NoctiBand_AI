"""
Tests for data contracts, SleepStage, and schema models.
"""

import numpy as np
import pandas as pd

from noctiband.io.contracts import EpochedNight, NightRecording, SignalQuality, SleepStage


def test_sleep_stage_names():
    assert SleepStage.to_name(SleepStage.WAKE) == "Wake"
    assert SleepStage.to_name(SleepStage.N1) == "N1 (Light NREM)"
    assert SleepStage.to_name(SleepStage.N2) == "N2 (Core NREM)"
    assert SleepStage.to_name(SleepStage.N3) == "N3 (Deep NREM)"
    assert SleepStage.to_name(SleepStage.REM) == "REM"
    assert SleepStage.to_name(SleepStage.UNKNOWN) == "Unknown/Artifact"
    assert SleepStage.to_name(SleepStage.UNLABELLED) == "Unlabelled"
    assert SleepStage.to_name(np.nan) == "Unlabelled"


def test_sleep_stage_binary_mapping():
    assert SleepStage.to_binary(0) == 0  # Wake
    assert SleepStage.to_binary(1) == 1  # N1
    assert SleepStage.to_binary(2) == 1  # N2
    assert SleepStage.to_binary(3) == 1  # N3
    assert SleepStage.to_binary(4) == 1  # REM
    assert SleepStage.to_binary(5) is None
    assert SleepStage.to_binary(-1) is None
    assert SleepStage.to_binary(np.nan) is None


def test_sleep_stage_3class_mapping():
    assert SleepStage.to_3class(0) == 0  # Wake
    assert SleepStage.to_3class(1) == 1  # NREM
    assert SleepStage.to_3class(2) == 1  # NREM
    assert SleepStage.to_3class(3) == 1  # NREM
    assert SleepStage.to_3class(4) == 2  # REM
    assert SleepStage.to_3class(5) is None
    assert SleepStage.to_3class(-1) is None
    assert SleepStage.to_3class(np.nan) is None


def test_night_recording_properties():
    rec_start = pd.Timestamp("2021-12-04 23:40:29", tz="America/New_York")
    hr_df = pd.DataFrame({"timestamp": [rec_start.timestamp() + 10], "heart_rate": [60.0]})
    labels = np.array([0, 1, 2, 3, 4])

    rec = NightRecording(
        subject_id="sub_test",
        night_id="night_01",
        rec_start=rec_start,
        hr=hr_df,
        labels=labels,
    )

    assert rec.has_hr is True
    assert rec.has_motion is False
    assert rec.has_labels is True
    assert rec.has_bp is False
    assert rec.has_spo2 is False
    assert rec.total_epochs == 5
    assert rec.duration_seconds == 150.0


def test_night_recording_unlabelled_epochs():
    rec_start = pd.Timestamp("2021-12-04 23:40:29", tz="America/New_York")
    rec_unix = rec_start.timestamp()
    hr_df = pd.DataFrame({"timestamp": [rec_unix + 10, rec_unix + 89], "heart_rate": [60.0, 62.0]})

    rec = NightRecording(
        subject_id="sub_unlabelled",
        night_id="night_01",
        rec_start=rec_start,
        hr=hr_df,
        labels=None,
    )

    assert rec.has_labels is False
    # Max timestamp is rec_unix + 89 -> (89 / 30) = 2.966 -> ceil is 3 epochs
    assert rec.total_epochs == 3
    assert rec.duration_seconds == 90.0


def test_epoched_night_methods():
    rec_start = pd.Timestamp("2021-12-04 23:40:29", tz="America/New_York")
    epochs_df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "start_time": [rec_start, rec_start + pd.Timedelta(seconds=30)],
            "sleep_stage": [0, 2],
            "quality_flag": [SignalQuality.GOOD.value, SignalQuality.GOOD.value],
        }
    )
    raw_slices = {0: {"hr": pd.DataFrame({"heart_rate": [60.0]})}}

    ep_night = EpochedNight(
        subject_id="sub_test",
        night_id="night_01",
        rec_start=rec_start,
        epoch_duration_sec=30.0,
        epochs_df=epochs_df,
        raw_slices=raw_slices,
    )

    assert ep_night.num_epochs == 2
    assert len(ep_night.to_dataframe()) == 2
    assert "hr" in ep_night.get_epoch_slice(0)
    assert ep_night.get_epoch_slice(99) == {}
