"""Unit tests for longitudinal storage, sleep debt, SRI, and compression."""

import numpy as np

from noctiband.storage import (
    calculate_sleep_debt,
    calculate_social_jetlag,
    calculate_sri,
    compress_hypnogram,
    decompress_hypnogram,
    get_database_connection,
    get_longitudinal_history,
    save_longitudinal_metrics,
    save_night_summary,
)


def test_hypnogram_compression_roundtrip():
    original = np.array([0, 1, 1, 2, 2, 1, 0, 0, 1], dtype=int)
    blob = compress_hypnogram(original)
    assert isinstance(blob, bytes)

    recovered = decompress_hypnogram(blob)
    np.testing.assert_array_equal(original, recovered)


def test_calculate_sleep_debt():
    # Baseline 480 min (8h). Day 1: 420 min (-60), Day 2: 400 min (-80) -> Debt = 140
    tst_list = [420.0, 400.0]
    debt = calculate_sleep_debt(tst_list, baseline_need_min=480.0)
    assert debt == 140.0

    # Extra recovery sleep Day 3: 500 min (+20) -> Debt = 120
    tst_list.append(500.0)
    debt_recovery = calculate_sleep_debt(tst_list, baseline_need_min=480.0)
    assert debt_recovery == 120.0


def test_calculate_sri_identical_and_reversed():
    h1 = np.array([0] * 100 + [1] * 800)
    h2 = np.array([0] * 100 + [1] * 800)
    assert calculate_sri([h1, h2]) == 100.0

    # Totally opposite
    h_opp = 1 - h1
    assert calculate_sri([h1, h_opp]) == -100.0


def test_calculate_social_jetlag():
    workday = [1380.0, 1370.0, 1390.0]  # ~23:00 (1380 min)
    freeday = [1440.0, 1450.0]  # ~24:00 (1445 min)
    jetlag = calculate_social_jetlag(workday, freeday)
    assert jetlag == 65.0


def test_sqlite_longitudinal_persistence():
    conn = get_database_connection(":memory:")

    # 1. Test saving night with compressed hypnogram
    labels = np.array([0, 1, 1, 0])
    night_id = save_night_summary(
        conn,
        {
            "subject_id": "SubLongitudinal",
            "night_date": "2026-10-23",
            "tst_min": 450.0,
            "sleep_efficiency": 90.0,
            "waso_min": 20.0,
            "sol_min": 15.0,
            "noa": 2,
            "passed_qc": True,
            "hypnogram": labels,
        },
    )
    assert night_id > 0

    # 2. Test saving longitudinal health metrics
    m_id = save_longitudinal_metrics(
        conn,
        {
            "subject_id": "SubLongitudinal",
            "cumulative_sleep_debt_min": 45.0,
            "sleep_regularity_index": 88.5,
            "social_jetlag_min": 30.0,
            "n_nights_analyzed": 7,
        },
    )
    assert m_id > 0

    history = get_longitudinal_history(conn, "SubLongitudinal")
    assert len(history) == 1
    assert history[0]["cumulative_sleep_debt_min"] == 45.0
    assert history[0]["sleep_regularity_index"] == 88.5
