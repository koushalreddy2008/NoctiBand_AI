"""
Nightly report builder for NoctiBand AI.

Transforms 30-second epoch predictions and signal quality flags into a standardized
clinical summary of nocturnal sleep metrics (SOL, WASO, sleep efficiency, awakenings,
and stage distribution).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPOCH_SEC = 30
EPOCH_MIN = EPOCH_SEC / 60.0
STAGE_NAMES = {0: "Wake", 1: "Sleep"}
GOOD_QUALITY_VALUES = {"ok", "good"}


def _runs(mask: np.ndarray | list[bool]) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) for each run of True values."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    padded = np.concatenate(([False], m, [False])).astype(int)
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True))


def _clock(rec_start: float | str | pd.Timestamp, epoch: int, tz: str = "UTC") -> str:
    """Format the clock time of an epoch given recording start timestamp and timezone."""
    if isinstance(rec_start, pd.Timestamp):
        ts = rec_start + pd.Timedelta(seconds=epoch * EPOCH_SEC)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(tz).strftime("%H:%M")

    # If numeric timestamp or string
    try:
        ts = pd.to_datetime(rec_start, unit="s", utc=True)
    except (ValueError, TypeError):
        ts = pd.to_datetime(rec_start, utc=True)

    ts = ts + pd.Timedelta(seconds=epoch * EPOCH_SEC)
    return ts.tz_convert(tz).strftime("%H:%M")


def build_report(
    pred_df: pd.DataFrame,
    rec_start: float | str | pd.Timestamp | None = None,
    tz: str = "UTC",
    min_sleep_run: int = 10,
    min_wake_run: int = 2,
    min_coverage: float = 0.8,
) -> dict[str, Any]:
    """Build a comprehensive nightly sleep report from epoch-level predictions.

    Parameters
    ----------
    pred_df : pd.DataFrame
        DataFrame containing one night of predictions with columns:
        'epoch_idx', 'y_pred' (0 = Wake, 1 = Sleep), and optional 'quality_flag' or 'q_flag'.
    rec_start : float, int, str, or pd.Timestamp, optional
        Start time of recording for calculating clock times.
    tz : str, default="UTC"
        Target timezone for clock times.
    min_sleep_run : int, default=10
        Minimum consecutive Sleep epochs (10 epochs = 5 min) required to establish sleep onset.
    min_wake_run : int, default=2
        Minimum consecutive Wake epochs (2 epochs = 1 min) to count as an awakening.
    min_coverage : float, default=0.8
        Threshold for data quality warnings (default 80%).

    Returns
    -------
    dict
        Structured nightly sleep metrics and clinical advisory warnings.
    """
    if len(pred_df) == 0:
        raise ValueError("No epochs to report on.")

    if "epoch_idx" in pred_df.columns:
        df = pred_df.sort_values("epoch_idx").reset_index(drop=True)
    else:
        df = pred_df.reset_index(drop=True)
    n = len(df)

    y = df["y_pred"].to_numpy().astype(int)
    warnings: list[str] = []

    # Data quality / coverage analysis
    coverage = None
    q_col = "quality_flag" if "quality_flag" in df.columns else (
        "q_flag" if "q_flag" in df.columns else None
    )
    if q_col is not None:
        is_good = df[q_col].astype(str).str.lower().isin(GOOD_QUALITY_VALUES)
        coverage = float(is_good.mean())
        if coverage < min_coverage:
            warnings.append(
                f"Only {coverage:.0%} of epochs had good signal quality; "
                "interpret results with caution."
            )
    else:
        warnings.append("No signal-quality information was available.")

    time_in_bed_min = n * EPOCH_MIN
    if time_in_bed_min < 240:
        warnings.append("The recording is shorter than 4 hours; the estimates may be unreliable.")

    dist = pd.Series(y).value_counts(normalize=True).sort_index()
    stage_distribution = {
        STAGE_NAMES.get(int(k), str(k)): round(float(v), 3) for k, v in dist.items()
    }

    report: dict[str, Any] = {
        "recording_minutes": time_in_bed_min,
        "data_coverage": coverage,
        "stage_distribution": stage_distribution,
        "warnings": warnings,
    }

    # Sleep onset = start of the first run of at least min_sleep_run Sleep epochs
    sleep_runs = [(s, e) for s, e in _runs(y == 1) if e - s >= min_sleep_run]
    if not sleep_runs:
        warnings.append("No sustained sleep period was detected.")
        report.update(
            sleep_onset_min=None,
            wake_up_min=None,
            total_sleep_min=0.0,
            sleep_efficiency_pct=0.0,
            awakenings=0,
            wake_after_onset_min=0.0,
            sleep_onset_clock=None,
            wake_up_clock=None,
        )
        return report

    onset = int(sleep_runs[0][0])
    wake_up = int(sleep_runs[-1][1])  # exclusive end of the last sleep run
    within = y[onset:wake_up]

    total_sleep = float((within == 1).sum() * EPOCH_MIN)
    waso = float((within == 0).sum() * EPOCH_MIN)
    awakenings = sum(1 for s, e in _runs(within == 0) if e - s >= min_wake_run)

    report.update(
        sleep_onset_min=onset * EPOCH_MIN,
        wake_up_min=wake_up * EPOCH_MIN,
        total_sleep_min=total_sleep,
        sleep_efficiency_pct=round(100.0 * total_sleep / time_in_bed_min, 1),
        awakenings=awakenings,
        wake_after_onset_min=waso,
        sleep_onset_clock=_clock(rec_start, onset, tz) if rec_start is not None else None,
        wake_up_clock=_clock(rec_start, wake_up, tz) if rec_start is not None else None,
    )
    return report


def format_report_text(r: dict[str, Any]) -> str:
    """Format dictionary report into a human-readable text document."""
    def hm(minutes: float) -> str:
        return f"{int(minutes // 60)} h {int(minutes % 60):02d} min"

    lines = [
        "NoctiBand nightly report",
        "All values are estimates. This is a research prototype, not a medical diagnosis.",
        "",
        f"Recording length: {hm(r['recording_minutes'])}",
    ]

    if r.get("sleep_onset_min") is None:
        lines.append("No sustained sleep period was detected.")
    else:
        lines.append(f"Estimated sleep time: {hm(r['total_sleep_min'])}")
        lines.append(
            f"Sleep onset: {hm(r['sleep_onset_min'])} after recording start"
            + (f" ({r['sleep_onset_clock']})" if r.get("sleep_onset_clock") else "")
        )
        lines.append(
            f"Wake-up: {hm(r['wake_up_min'])} after recording start"
            + (f" ({r['wake_up_clock']})" if r.get("wake_up_clock") else "")
        )
        lines.append(f"Sleep efficiency: {r['sleep_efficiency_pct']}%")
        lines.append(
            f"Awakenings: {r['awakenings']} (awake {hm(r['wake_after_onset_min'])} after onset)"
        )

    lines.append(
        "Stage distribution: "
        + ", ".join(f"{k} {v:.0%}" for k, v in r.get("stage_distribution", {}).items())
    )

    if r.get("data_coverage") is not None:
        lines.append(f"Data coverage (good signal): {r['data_coverage']:.0%}")

    for w in r.get("warnings", []):
        lines.append(f"WARNING: {w}")

    return "\n".join(lines)


def plot_hypnogram(
    epoch_labels: Any,
    rec_start: float | str | pd.Timestamp | None = None,
    tz: str = "UTC",
    figsize: tuple[float, float] = (10, 3.2),
) -> Any:
    """Render a clinical sleep hypnogram without empty secondary axes or unused subplots.

    Parameters
    ----------
    epoch_labels : array-like or NightPrediction or dict
        Epoch sleep stage codes (0=Wake, 1=Sleep / NREM, etc.).
    rec_start : float, str, or Timestamp, optional
        Recording start time to calculate local clock times.
    tz : str, default='UTC'
        Timezone for local clock calculations.
    figsize : tuple of float, default=(10, 3.2)
        Matplotlib figure dimensions (width, height).

    Returns
    -------
    matplotlib.figure.Figure
        Single-axis clean hypnogram figure with human-readable stage and time labels.
    """
    import matplotlib.pyplot as plt

    if hasattr(epoch_labels, "epoch_labels"):
        labels = np.asarray(epoch_labels.epoch_labels, dtype=int)
        if rec_start is None:
            rec_start = getattr(epoch_labels, "rec_start", None)
    elif isinstance(epoch_labels, dict):
        raw_seq = epoch_labels.get("epoch_labels", epoch_labels.get("hypnogram", []))
        labels = np.asarray(raw_seq, dtype=int)
        if rec_start is None:
            rec_start = epoch_labels.get("rec_start", None)
    else:
        labels = np.asarray(epoch_labels, dtype=int)

    n_epochs = len(labels)
    hours = np.arange(n_epochs) * (EPOCH_SEC / 3600.0)
    total_hours = (n_epochs * EPOCH_SEC) / 3600.0

    # Determine human-readable ticks and stage labels
    max_label = int(labels.max()) if n_epochs > 0 else 1
    if max_label <= 1:
        ticks = [0, 1]
        tick_names = ["Wake", "Sleep"]
    elif max_label == 2:
        ticks = [0, 1, 2]
        tick_names = ["Wake", "NREM", "REM"]
    else:
        ticks = [0, 1, 2, 3, 4]
        tick_names = ["Wake", "N1", "N2", "N3", "REM"]

    fig, ax = plt.subplots(figsize=figsize)

    if n_epochs > 0:
        ax.step(
            hours,
            labels,
            where="post",
            color="#2b5c8f",
            linewidth=1.8,
            label="Sleep Stage",
        )
        ax.fill_between(hours, labels, step="post", alpha=0.25, color="#4a90e2")

        tick_step = 1.0 if total_hours <= 10.0 else 2.0
        x_ticks = np.arange(0.0, total_hours + 0.001, tick_step)
        if rec_start is not None and rec_start != 0.0:
            x_labels = [
                f"{int(h)}h\n({_clock(rec_start, int(h * 3600 / EPOCH_SEC), tz)})"
                for h in x_ticks
            ]
            ax.set_xlabel(f"Elapsed Time (Hours) / Clock Time ({tz})", fontsize=9)
        else:
            x_labels = [f"{int(h)}h" for h in x_ticks]
            ax.set_xlabel("Elapsed Time (Hours)", fontsize=9)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlim(0, max(hours[-1], 0.1))

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_names, fontsize=10, fontweight="bold")
    ax.set_ylabel("Stage", fontsize=10)
    ax.set_ylim(-0.35, max(ticks) + 0.35)
    ax.grid(True, linestyle=":", alpha=0.6, axis="both")
    ax.set_title("Nocturnal Sleep Hypnogram", fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig
