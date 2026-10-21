"""SQLite persistence, habit tracking, and non-diagnostic safety validation (Person A)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

FORBIDDEN_DIAGNOSTIC_TERMS = [
    "diagnos",
    "insomnia",
    "sleep apnea",
    "apnea",
    "restless leg",
    "narcolep",
    "patholog",
    "disease",
    "cure",
    "prescrib",
    "prescription",
    "medicat",
    "pill",
    "disorder",
    "clinical",
    "rx",
]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nights_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL,
    night_date TEXT NOT NULL,
    tst_min REAL NOT NULL,
    sleep_efficiency REAL NOT NULL,
    waso_min REAL NOT NULL,
    sol_min REAL NOT NULL,
    noa INTEGER NOT NULL,
    passed_qc INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS habit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL,
    log_date TEXT NOT NULL,
    caffeine_after_2pm INTEGER DEFAULT 0,
    alcohol_evening INTEGER DEFAULT 0,
    screen_time_bed_min REAL DEFAULT 0.0,
    exercise_today INTEGER DEFAULT 0,
    subjective_quality INTEGER DEFAULT 3,
    notes TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_nights_subject ON nights_summary(subject_id);
CREATE INDEX IF NOT EXISTS idx_habits_subject ON habit_logs(subject_id);
"""


def get_database_connection(db_path: Path | str = ":memory:") -> sqlite3.Connection:
    """Open a SQLite connection and ensure schema tables are initialized."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(SCHEMA_SQL)
    return conn


def save_night_summary(conn: sqlite3.Connection, summary: dict[str, Any]) -> int:
    """Insert a night sleep architecture summary record."""
    sql = """
    INSERT INTO nights_summary (
        subject_id, night_date, tst_min, sleep_efficiency,
        waso_min, sol_min, noa, passed_qc, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    night_date = summary.get("night_date") or datetime.now().strftime("%Y-%m-%d")
    created_at = summary.get("created_at") or datetime.now().isoformat()
    passed_qc = 1 if summary.get("passed_qc", True) else 0

    with conn:
        cur = conn.execute(
            sql,
            (
                str(summary["subject_id"]),
                str(night_date),
                float(summary["tst_min"]),
                float(summary["sleep_efficiency"]),
                float(summary["waso_min"]),
                float(summary["sol_min"]),
                int(summary["noa"]),
                passed_qc,
                str(created_at),
            ),
        )
        return int(cur.lastrowid)


def get_night_history(
    conn: sqlite3.Connection,
    subject_id: str,
    limit: int = 30,
) -> pd.DataFrame:
    """Retrieve historical night summaries for a user, sorted by date ascending."""
    sql = """
    SELECT id, subject_id, night_date, tst_min, sleep_efficiency,
           waso_min, sol_min, noa, passed_qc, created_at
    FROM nights_summary
    WHERE subject_id = ?
    ORDER BY night_date ASC, id ASC
    LIMIT ?
    """
    df = pd.read_sql_query(sql, conn, params=(str(subject_id), int(limit)))
    return df


def log_habit_entry(conn: sqlite3.Connection, habit: dict[str, Any]) -> int:
    """Log daily daytime and evening sleep habits for behavioral tracking."""
    sql = """
    INSERT INTO habit_logs (
        subject_id, log_date, caffeine_after_2pm, alcohol_evening,
        screen_time_bed_min, exercise_today, subjective_quality, notes, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    log_date = habit.get("log_date") or datetime.now().strftime("%Y-%m-%d")
    created_at = habit.get("created_at") or datetime.now().isoformat()

    with conn:
        cur = conn.execute(
            sql,
            (
                str(habit["subject_id"]),
                str(log_date),
                int(habit.get("caffeine_after_2pm", 0)),
                int(habit.get("alcohol_evening", 0)),
                float(habit.get("screen_time_bed_min", 0.0)),
                int(habit.get("exercise_today", 0)),
                int(habit.get("subjective_quality", 3)),
                str(habit.get("notes", "")),
                str(created_at),
            ),
        )
        return int(cur.lastrowid)


def get_habit_history(
    conn: sqlite3.Connection,
    subject_id: str,
    limit: int = 30,
) -> pd.DataFrame:
    """Retrieve historical habit logs for a user, sorted by date ascending."""
    sql = """
    SELECT id, subject_id, log_date, caffeine_after_2pm, alcohol_evening,
           screen_time_bed_min, exercise_today, subjective_quality, notes, created_at
    FROM habit_logs
    WHERE subject_id = ?
    ORDER BY log_date ASC, id ASC
    LIMIT ?
    """
    df = pd.read_sql_query(sql, conn, params=(str(subject_id), int(limit)))
    return df


def export_user_data(
    conn: sqlite3.Connection,
    subject_id: str,
    out_dir: Path | str,
) -> dict[str, Path]:
    """Export all user night summaries and habit logs to CSV files."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    nights_df = get_night_history(conn, subject_id=subject_id, limit=100000)
    habits_df = get_habit_history(conn, subject_id=subject_id, limit=100000)

    nights_csv = dest / f"{subject_id}_nights_summary.csv"
    habits_csv = dest / f"{subject_id}_habit_logs.csv"

    nights_df.to_csv(nights_csv, index=False)
    habits_df.to_csv(habits_csv, index=False)

    return {
        "nights_summary": nights_csv,
        "habit_logs": habits_csv,
    }


def validate_suggestion_safety(text: str) -> tuple[bool, list[str]]:
    """Validate that sleep suggestions and reports contain zero medical or diagnostic claims.

    Parameters
    ----------
    text : str
        The suggestion or recommendation text to inspect.

    Returns
    -------
    tuple[bool, list[str]]
        (is_safe, list_of_violations)
    """
    text_lower = text.lower()
    violations: list[str] = []

    for term in FORBIDDEN_DIAGNOSTIC_TERMS:
        if term in text_lower:
            violations.append(f"Forbidden diagnostic or medical term detected: '{term}'")

    is_safe = len(violations) == 0
    return is_safe, violations
