"""SQLite persistence, habit tracking, and non-diagnostic safety validation (Person A)."""

from __future__ import annotations

import sqlite3
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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
    hypnogram_compressed BLOB,
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

CREATE TABLE IF NOT EXISTS longitudinal_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL,
    calculation_date TEXT NOT NULL,
    cumulative_sleep_debt_min REAL NOT NULL,
    sleep_regularity_index REAL NOT NULL,
    social_jetlag_min REAL DEFAULT 0.0,
    n_nights_analyzed INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_nights_subject ON nights_summary(subject_id);
CREATE INDEX IF NOT EXISTS idx_habits_subject ON habit_logs(subject_id);
CREATE INDEX IF NOT EXISTS idx_longitudinal_subject ON longitudinal_metrics(subject_id);
"""


def get_database_connection(db_path: Path | str = ":memory:") -> sqlite3.Connection:
    """Open a SQLite connection and ensure schema tables are initialized."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(SCHEMA_SQL)
        # Safe migration if table was created in an earlier session without hypnogram_compressed
        try:
            conn.execute("ALTER TABLE nights_summary ADD COLUMN hypnogram_compressed BLOB")
        except sqlite3.OperationalError:
            pass  # column already exists
    return conn


def compress_hypnogram(labels: np.ndarray | list[int]) -> bytes:
    """Compress integer epoch sleep stage labels into compact binary blob using zlib."""
    arr = np.asarray(labels, dtype=np.int8)
    return zlib.compress(arr.tobytes())


def decompress_hypnogram(blob: bytes) -> np.ndarray:
    """Decompress binary blob back into integer epoch sleep stage array."""
    decompressed = zlib.decompress(blob)
    return np.frombuffer(decompressed, dtype=np.int8)


def calculate_sleep_debt(
    tst_history_min: list[float] | np.ndarray,
    baseline_need_min: float = 480.0,
) -> float:
    """Compute cumulative sleep debt across consecutive nights.

    Debt accumulates when nightly Total Sleep Time (TST) is below baseline need.
    Debt decays partially on nights with recovery sleep (above baseline).
    """
    if len(tst_history_min) == 0:
        return 0.0

    debt = 0.0
    for tst in tst_history_min:
        diff = baseline_need_min - float(tst)
        debt = max(0.0, debt + diff)
    return round(debt, 1)


def calculate_sri(hypnograms: list[np.ndarray | list[int]]) -> float:
    """Calculate the Sleep Regularity Index (SRI) across consecutive 24-hour hypnograms.

    Scale is -100 to 100, where 100 is identical sleep timing across consecutive days.
    """
    if len(hypnograms) < 2:
        return 100.0

    agreements = []
    for i in range(len(hypnograms) - 1):
        h1 = np.asarray(hypnograms[i], dtype=int)
        h2 = np.asarray(hypnograms[i + 1], dtype=int)
        min_len = min(len(h1), len(h2))
        if min_len > 0:
            match = np.mean(h1[:min_len] == h2[:min_len])
            agreements.append(match)

    if not agreements:
        return 100.0

    mean_match = float(np.mean(agreements))
    sri = 100.0 * (2.0 * mean_match - 1.0)
    return round(sri, 1)


def calculate_social_jetlag(
    workday_onsets_min: list[float] | np.ndarray,
    freeday_onsets_min: list[float] | np.ndarray,
) -> float:
    """Calculate absolute difference in sleep timing between workdays and free days (min)."""
    if len(workday_onsets_min) == 0 or len(freeday_onsets_min) == 0:
        return 0.0
    w_mean = float(np.mean(workday_onsets_min))
    f_mean = float(np.mean(freeday_onsets_min))
    return round(abs(f_mean - w_mean), 1)


def save_night_summary(conn: sqlite3.Connection, summary: dict[str, Any]) -> int:
    """Insert a night sleep architecture summary record."""
    sql = """
    INSERT INTO nights_summary (
        subject_id, night_date, tst_min, sleep_efficiency,
        waso_min, sol_min, noa, passed_qc, hypnogram_compressed, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    night_date = summary.get("night_date") or datetime.now().strftime("%Y-%m-%d")
    created_at = summary.get("created_at") or datetime.now().isoformat()
    passed_qc = 1 if summary.get("passed_qc", True) else 0

    hypno_blob = summary.get("hypnogram_compressed")
    if hypno_blob is None and summary.get("hypnogram") is not None:
        hypno_blob = compress_hypnogram(summary["hypnogram"])

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
                hypno_blob,
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
           waso_min, sol_min, noa, passed_qc, hypnogram_compressed, created_at
    FROM nights_summary
    WHERE subject_id = ?
    ORDER BY night_date ASC, id ASC
    LIMIT ?
    """
    df = pd.read_sql_query(sql, conn, params=(str(subject_id), int(limit)))
    return df


def save_longitudinal_metrics(conn: sqlite3.Connection, metrics: dict[str, Any]) -> int:
    """Store multi-night longitudinal health metrics (sleep debt, SRI, social jetlag)."""
    sql = """
    INSERT INTO longitudinal_metrics (
        subject_id, calculation_date, cumulative_sleep_debt_min,
        sleep_regularity_index, social_jetlag_min, n_nights_analyzed, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    calc_date = metrics.get("calculation_date") or datetime.now().strftime("%Y-%m-%d")
    created_at = metrics.get("created_at") or datetime.now().isoformat()

    with conn:
        cur = conn.execute(
            sql,
            (
                str(metrics["subject_id"]),
                str(calc_date),
                float(metrics["cumulative_sleep_debt_min"]),
                float(metrics["sleep_regularity_index"]),
                float(metrics.get("social_jetlag_min", 0.0)),
                int(metrics.get("n_nights_analyzed", 7)),
                str(created_at),
            ),
        )
        return int(cur.lastrowid)


def get_longitudinal_history(
    conn: sqlite3.Connection,
    subject_id: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Retrieve longitudinal trend records for a user."""
    sql = """
    SELECT id, subject_id, calculation_date, cumulative_sleep_debt_min,
           sleep_regularity_index, social_jetlag_min, n_nights_analyzed, created_at
    FROM longitudinal_metrics
    WHERE subject_id = ?
    ORDER BY calculation_date ASC, id ASC
    LIMIT ?
    """
    cur = conn.execute(sql, (str(subject_id), int(limit)))
    rows = cur.fetchall()
    return [dict(r) for r in rows]


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
