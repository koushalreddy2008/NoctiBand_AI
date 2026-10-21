"""Unit tests for SQLite storage, habit tracking, and safety validation (Stage 5)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from noctiband.storage import (
    export_user_data,
    get_database_connection,
    get_habit_history,
    get_night_history,
    log_habit_entry,
    save_night_summary,
    validate_suggestion_safety,
)


def test_sqlite_in_memory_and_file_creation(tmp_path: Path):
    # In-memory database
    mem_conn = get_database_connection(":memory:")
    assert isinstance(mem_conn, sqlite3.Connection)
    tables = [
        row[0]
        for row in mem_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "nights_summary" in tables
    assert "habit_logs" in tables
    mem_conn.close()

    # File-based database
    db_file = tmp_path / "test_noctiband.db"
    file_conn = get_database_connection(db_file)
    assert db_file.exists()
    file_conn.close()


def test_nights_summary_crud():
    conn = get_database_connection(":memory:")
    night_data = {
        "subject_id": "test_sub_01",
        "night_date": "2026-10-20",
        "tst_min": 420.5,
        "sleep_efficiency": 87.5,
        "waso_min": 35.0,
        "sol_min": 15.0,
        "noa": 2,
        "passed_qc": True,
    }

    row_id = save_night_summary(conn, night_data)
    assert row_id == 1

    df = get_night_history(conn, subject_id="test_sub_01")
    assert len(df) == 1
    assert df.loc[0, "subject_id"] == "test_sub_01"
    assert df.loc[0, "tst_min"] == 420.5
    assert df.loc[0, "sleep_efficiency"] == 87.5
    assert df.loc[0, "passed_qc"] == 1
    conn.close()


def test_habit_logs_crud():
    conn = get_database_connection(":memory:")
    habit_data = {
        "subject_id": "test_sub_01",
        "log_date": "2026-10-20",
        "caffeine_after_2pm": 0,
        "alcohol_evening": 1,
        "screen_time_bed_min": 25.0,
        "exercise_today": 1,
        "subjective_quality": 4,
        "notes": "Went for an evening walk; mild evening screen time.",
    }

    row_id = log_habit_entry(conn, habit_data)
    assert row_id == 1

    df = get_habit_history(conn, subject_id="test_sub_01")
    assert len(df) == 1
    assert df.loc[0, "subject_id"] == "test_sub_01"
    assert df.loc[0, "alcohol_evening"] == 1
    assert df.loc[0, "exercise_today"] == 1
    assert df.loc[0, "subjective_quality"] == 4
    conn.close()


def test_export_user_data(tmp_path: Path):
    conn = get_database_connection(":memory:")
    save_night_summary(
        conn,
        {
            "subject_id": "user_export",
            "night_date": "2026-10-21",
            "tst_min": 400.0,
            "sleep_efficiency": 85.0,
            "waso_min": 40.0,
            "sol_min": 20.0,
            "noa": 3,
            "passed_qc": True,
        },
    )
    log_habit_entry(
        conn,
        {
            "subject_id": "user_export",
            "log_date": "2026-10-21",
            "caffeine_after_2pm": 1,
            "alcohol_evening": 0,
            "screen_time_bed_min": 15.0,
            "exercise_today": 1,
            "subjective_quality": 3,
            "notes": "Good day overall",
        },
    )

    out_files = export_user_data(conn, subject_id="user_export", out_dir=tmp_path)
    assert out_files["nights_summary"].exists()
    assert out_files["habit_logs"].exists()

    content_nights = out_files["nights_summary"].read_text(encoding="utf-8")
    assert "user_export" in content_nights
    assert "400.0" in content_nights

    content_habits = out_files["habit_logs"].read_text(encoding="utf-8")
    assert "user_export" in content_habits
    assert "Good day overall" in content_habits
    conn.close()


def test_validate_suggestion_safety():
    # Safe lifestyle and hygiene suggestions
    safe_text_1 = "Keep a regular sleep schedule and wind down in dim light."
    is_safe_1, violations_1 = validate_suggestion_safety(safe_text_1)
    assert is_safe_1 is True
    assert len(violations_1) == 0

    safe_text_2 = "Limit evening liquids to minimize mid-night awakenings."
    is_safe_2, violations_2 = validate_suggestion_safety(safe_text_2)
    assert is_safe_2 is True
    assert len(violations_2) == 0

    # Unsafe text containing forbidden medical / diagnostic words
    unsafe_text_1 = "Your low sleep efficiency indicates chronic insomnia disease."
    is_safe_u1, violations_u1 = validate_suggestion_safety(unsafe_text_1)
    assert is_safe_u1 is False
    assert any("insomnia" in v for v in violations_u1)
    assert any("disease" in v for v in violations_u1)

    unsafe_text_2 = "We diagnose obstructive sleep apnea; please take this prescription pill."
    is_safe_u2, violations_u2 = validate_suggestion_safety(unsafe_text_2)
    assert is_safe_u2 is False
    assert any("diagnos" in v for v in violations_u2)
    assert any("apnea" in v for v in violations_u2)
    assert any("prescri" in v for v in violations_u2)
    assert any("pill" in v for v in violations_u2)
