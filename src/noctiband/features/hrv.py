"""Heart Rate Variability (HRV) feature extraction for sleep staging (v0.2).

Computes time-domain (RMSSD, SDNN, pNN50), frequency-domain (LF, HF, LF/HF),
and non-linear / entropy metrics from heart rate and inter-beat interval series
to distinguish Wake, NREM, SWS (Deep), and REM sleep stages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

HRV_COLUMNS = [
    "hrv_rmssd",
    "hrv_sdnn",
    "hrv_pnn50",
    "hrv_lf_power",
    "hrv_hf_power",
    "hrv_lf_hf_ratio",
    "hrv_entropy",
]


def compute_hrv_metrics(rr_intervals_ms: np.ndarray | list[float]) -> dict[str, float]:
    """Compute time-domain, spectral, and entropy HRV metrics from RR intervals.

    Parameters
    ----------
    rr_intervals_ms : array-like
        Array of normal-to-normal or beat-to-beat intervals in milliseconds.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - 'hrv_rmssd': Root Mean Square of Successive Differences (ms)
        - 'hrv_sdnn': Standard deviation of NN intervals (ms)
        - 'hrv_pnn50': Percentage of adjacent interval differences > 50 ms (%)
        - 'hrv_lf_power': Spectral power in Low Frequency band (0.04 - 0.15 Hz)
        - 'hrv_hf_power': Spectral power in High Frequency band (0.15 - 0.40 Hz)
        - 'hrv_lf_hf_ratio': Ratio of LF to HF spectral power
        - 'hrv_entropy': Shannon entropy of interval distribution
    """
    rr = np.asarray(rr_intervals_ms, dtype=float)
    rr = rr[~np.isnan(rr)]
    # Filter physiologically plausible RR intervals: 300 ms (200 bpm) to 2000 ms (30 bpm)
    rr = rr[(rr >= 300.0) & (rr <= 2000.0)]

    n_samples = len(rr)
    if n_samples < 3:
        return {
            "hrv_rmssd": np.nan,
            "hrv_sdnn": np.nan,
            "hrv_pnn50": np.nan,
            "hrv_lf_power": np.nan,
            "hrv_hf_power": np.nan,
            "hrv_lf_hf_ratio": np.nan,
            "hrv_entropy": np.nan,
        }

    # 1. Time-Domain Metrics
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs**2)))
    sdnn = float(np.std(rr, ddof=1))
    pnn50 = float(np.mean(np.abs(diffs) > 50.0) * 100.0)

    # 2. Spectral Metrics via Welch Periodogram
    # Resample or estimate pseudo-frequency with mean sampling interval
    mean_rr_s = np.mean(rr) / 1000.0
    fs = 1.0 / max(mean_rr_s, 0.2)

    try:
        nperseg = min(n_samples, 64)
        freqs, psd = signal.welch(rr - np.mean(rr), fs=fs, nperseg=nperseg)

        # LF band: 0.04 to 0.15 Hz
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        lf_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0

        # HF band: 0.15 to 0.40 Hz (Parasympathetic / Vagal tone)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
        hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0

        lf_hf_ratio = float(lf_power / max(hf_power, 1e-7))
    except Exception:
        lf_power = np.nan
        hf_power = np.nan
        lf_hf_ratio = np.nan

    # 3. Short-term Shannon Entropy
    try:
        hist, _ = np.histogram(rr, bins=10, density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))
    except Exception:
        entropy = np.nan

    return {
        "hrv_rmssd": rmssd,
        "hrv_sdnn": sdnn,
        "hrv_pnn50": pnn50,
        "hrv_lf_power": lf_power,
        "hrv_hf_power": hf_power,
        "hrv_lf_hf_ratio": lf_hf_ratio,
        "hrv_entropy": entropy,
    }


def compute_epoch_hrv_features(
    hr_df: pd.DataFrame,
    rec_start: float,
    n_epochs: int,
    epoch_s: int = 30,
) -> pd.DataFrame:
    """Compute epoch-level HRV features across an entire night of HR readings.

    Parameters
    ----------
    hr_df : pd.DataFrame
        DataFrame with columns 'timestamp' (unix seconds) and 'hr' (bpm).
    rec_start : float
        Recording start unix timestamp.
    n_epochs : int
        Total number of epochs in recording grid.
    epoch_s : int, default=30
        Duration of each epoch in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by epoch (0 to n_epochs - 1) with all HRV_COLUMNS.
    """
    epochs_idx = np.arange(n_epochs, dtype=int)
    empty_df = pd.DataFrame(
        {col: np.full(n_epochs, np.nan) for col in HRV_COLUMNS},
        index=epochs_idx,
    )

    if hr_df.empty or "hr" not in hr_df.columns or "timestamp" not in hr_df.columns:
        return empty_df

    df = hr_df.dropna(subset=["timestamp", "hr"]).copy()
    if df.empty:
        return empty_df

    # Convert HR (bpm) to inter-beat interval RR (milliseconds)
    df["rr_ms"] = 60000.0 / df["hr"].clip(lower=25.0, upper=240.0)
    df["epoch"] = np.floor((df["timestamp"] - rec_start) / float(epoch_s)).astype(int)

    # Filter within valid epoch range
    df = df[(df["epoch"] >= 0) & (df["epoch"] < n_epochs)]
    if df.empty:
        return empty_df

    rows: dict[str, list[float]] = {col: [] for col in HRV_COLUMNS}
    epoch_groups = dict(tuple(df.groupby("epoch")))

    for ep in epochs_idx:
        if ep in epoch_groups:
            metrics = compute_hrv_metrics(epoch_groups[ep]["rr_ms"].to_numpy())
            for col in HRV_COLUMNS:
                rows[col].append(metrics[col])
        else:
            for col in HRV_COLUMNS:
                rows[col].append(np.nan)

    res_df = pd.DataFrame(rows, index=epochs_idx)
    return res_df


def enrich_with_hrv_features(
    epoch_df: pd.DataFrame,
    hr_df: pd.DataFrame,
    rec_start: float,
    epoch_s: int = 30,
) -> pd.DataFrame:
    """Join epoch-level HRV features onto an existing epoch feature DataFrame."""
    n_epochs = len(epoch_df)
    hrv_df = compute_epoch_hrv_features(
        hr_df=hr_df,
        rec_start=rec_start,
        n_epochs=n_epochs,
        epoch_s=epoch_s,
    )
    return epoch_df.join(hrv_df)
