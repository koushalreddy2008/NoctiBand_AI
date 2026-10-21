"""Unit tests for external wearable data stream adapter (Stage 6)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from noctiband.adapter import (
    GRAVITY_MS2,
    _normalize_acceleration_units,
    _parse_timestamps,
    _resample_motion,
    adapt_external_stream,
)
from noctiband.types import NightData


def test_parse_timestamps_unix_seconds():
    ts = pd.Series([1600000000.0, 1600000030.0, 1600000060.0])
    parsed = _parse_timestamps(ts)
    np.testing.assert_allclose(parsed.to_numpy(), ts.to_numpy())


def test_parse_timestamps_unix_milliseconds():
    # Unix milliseconds (1.6e12)
    ts_ms = pd.Series([1600000000000.0, 1600000030000.0])
    parsed = _parse_timestamps(ts_ms)
    expected = np.array([1600000000.0, 1600000030.0])
    np.testing.assert_allclose(parsed.to_numpy(), expected)


def test_parse_timestamps_iso8601():
    dt1 = datetime(2026, 10, 20, 22, 0, 0, tzinfo=UTC)
    dt2 = datetime(2026, 10, 20, 22, 0, 30, tzinfo=UTC)
    ts_iso = pd.Series([dt1.isoformat(), dt2.isoformat()])
    parsed = _parse_timestamps(ts_iso)
    assert parsed.iloc[1] - parsed.iloc[0] == 30.0


def test_normalize_acceleration_units_ms2_to_g():
    # Accelerometer in m/s^2 around ~9.81
    df_ms2 = pd.DataFrame({
        "timestamp": [0.0, 1.0],
        "x": [0.0, 0.0],
        "y": [0.0, 0.0],
        "z": [9.80665, 9.80665],
    })
    converted = _normalize_acceleration_units(df_ms2, target_unit="g")
    np.testing.assert_allclose(converted["z"].to_numpy(), np.array([1.0, 1.0]), atol=1e-4)


def test_normalize_acceleration_units_already_g():
    df_g = pd.DataFrame({
        "timestamp": [0.0, 1.0],
        "x": [0.0, 0.0],
        "y": [0.0, 0.0],
        "z": [1.0, 1.0],
    })
    converted = _normalize_acceleration_units(df_g, target_unit="g")
    np.testing.assert_allclose(converted["z"].to_numpy(), np.array([1.0, 1.0]), atol=1e-4)


def test_resample_motion_10hz_to_50hz():
    # 10 Hz signal for 2 seconds (21 samples)
    t_10hz = np.linspace(0.0, 2.0, 21)
    x = np.sin(2 * np.pi * t_10hz)
    df_10hz = pd.DataFrame({
        "timestamp": t_10hz,
        "x": x,
        "y": np.zeros(21),
        "z": np.ones(21),
    })

    df_50hz = _resample_motion(df_10hz, target_hz=50.0)
    # 2 seconds at 50 Hz -> ~100 samples
    assert len(df_50hz) == 100
    assert np.isclose(df_50hz["timestamp"].iloc[1] - df_50hz["timestamp"].iloc[0], 0.02)


def test_adapt_external_stream_end_to_end():
    # Raw HR with timestamps in ms
    hr_raw = pd.DataFrame({
        "time": [1600000000000.0, 1600000001000.0, 1600000002000.0],
        "bpm": [65.0, 66.0, 64.0],
    })
    # Raw 2D motion (missing z) in m/s^2 at 10 Hz
    t_mot = np.linspace(1600000000.0, 1600000010.0, 101)
    mot_raw = pd.DataFrame({
        "time": t_mot * 1000.0,  # in ms
        "x": np.full(101, 0.5 * GRAVITY_MS2),
        "y": np.full(101, 0.2 * GRAVITY_MS2),
    })

    night = adapt_external_stream(
        hr_raw,
        mot_raw,
        target_hz=50.0,
        target_unit="g",
        subject_id="external_patient_01",
    )

    assert isinstance(night, NightData)
    assert night.subject == "external_patient_01"
    assert "z" in night.motion.columns
    # Check that magnitude is normalized around 1.0g
    mag = np.sqrt(night.motion["x"] ** 2 + night.motion["y"] ** 2 + night.motion["z"] ** 2)
    assert np.all(mag <= 1.05)
