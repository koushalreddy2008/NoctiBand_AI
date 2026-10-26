"""
Unit tests for NoctiBand nightly sleep report builder.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from noctiband.io.contracts import SignalQuality
from noctiband.reporting.nightly import build_report, format_report_text, plot_hypnogram


def make(y, quality="ok"):
    return pd.DataFrame({
        "epoch_idx": np.arange(len(y)),
        "y_pred": y,
        "quality_flag": quality,
    })


def test_simple_night():
    # 20 Wake epochs (10 min), then 100 Sleep epochs (50 min)
    r = build_report(make([0] * 20 + [1] * 100))
    assert r["sleep_onset_min"] == pytest.approx(10.0)
    assert r["wake_up_min"] == pytest.approx(60.0)
    assert r["total_sleep_min"] == pytest.approx(50.0)
    assert r["sleep_efficiency_pct"] == pytest.approx(83.3, abs=0.1)
    assert r["awakenings"] == 0


def test_one_awakening():
    # Wake 20, Sleep 50, Wake 4 (2 min), Sleep 50
    r = build_report(make([0] * 20 + [1] * 50 + [0] * 4 + [1] * 50))
    assert r["awakenings"] == 1
    assert r["wake_after_onset_min"] == pytest.approx(2.0)
    assert r["total_sleep_min"] == pytest.approx(50.0)


def test_short_sleep_blip_is_not_onset():
    # a 1-epoch sleep blip early must not count as sleep onset
    r = build_report(make([0] * 5 + [1] + [0] * 5 + [1] * 100))
    assert r["sleep_onset_min"] == pytest.approx(11 * 0.5)


def test_all_wake_warns():
    r = build_report(make([0] * 50))
    assert r["sleep_onset_min"] is None
    assert any("No sustained sleep" in w for w in r["warnings"])


def test_low_quality_warns():
    r = build_report(make([1] * 100, quality="bad"))
    assert r["data_coverage"] == 0.0
    assert any("signal quality" in w for w in r["warnings"])


def test_signal_quality_enum_support():
    df = pd.DataFrame({
        "epoch_idx": np.arange(100),
        "y_pred": [1] * 100,
        "quality_flag": [SignalQuality.GOOD.value] * 85 + [SignalQuality.DEGRADED.value] * 15,
    })
    r = build_report(df)
    assert r["data_coverage"] == pytest.approx(0.85)
    assert not any("signal quality" in w for w in r["warnings"])


def test_clock_times_and_format_text():
    # 20 Wake epochs (10 min), then 100 Sleep epochs (50 min), start at 22:00 UTC (unix 1700000000)
    start_ts = pd.Timestamp("2026-10-22 22:00:00", tz="UTC")
    r = build_report(make([0] * 20 + [1] * 100), rec_start=start_ts, tz="UTC")
    assert r["sleep_onset_clock"] == "22:10"
    assert r["wake_up_clock"] == "23:00"

    text = format_report_text(r)
    assert "NoctiBand nightly report" in text
    assert "Sleep onset: 0 h 10 min after recording start (22:10)" in text
    assert "Wake-up: 1 h 00 min after recording start (23:00)" in text
    assert "Sleep efficiency: 83.3%" in text
    assert "Stage distribution:" in text


def test_empty_dataframe_raises():
    with pytest.raises(ValueError, match="No epochs to report on"):
        build_report(pd.DataFrame())


def test_plot_hypnogram_structure_and_labels():
    labels = np.array([0] * 60 + [1] * 120 + [0] * 20 + [1] * 120)  # 320 epochs
    fig = plot_hypnogram(labels, rec_start=1700000000.0)

    try:
        # Exactly one axis: no empty secondary axes or unused subplots
        assert len(fig.axes) == 1
        ax = fig.axes[0]

        # Verify Y ticks and human-readable stage labels
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "Wake" in y_labels
        assert "Sleep" in y_labels

        # Verify X tick labels use relative time format (e.g. 0h, 1h, 2h)
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert any("0h" in xl for xl in x_labels)
        assert any("1h" in xl for xl in x_labels)

        # Verify title and label
        assert "Hypnogram" in ax.get_title()
    finally:
        plt.close(fig)


def test_plot_hypnogram_multiclass():
    # 3-class sequence: 0=Wake, 1=NREM, 2=REM
    labels = np.array([0] * 30 + [1] * 60 + [2] * 30)
    fig = plot_hypnogram(labels)

    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert y_labels == ["Wake", "NREM", "REM"]
    finally:
        plt.close(fig)
