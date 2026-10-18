"""30-second epoch assignment and coverage calculations."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from noctiband.types import EPOCH_S

logger = logging.getLogger(__name__)


def assign_epochs(df: pd.DataFrame, rec_start: float, n_epochs: int) -> pd.DataFrame:
    """Map signal timestamps to 30-second epoch indices relative to rec_start.

    Readings strictly outside [0, n_epochs) are removed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a 'timestamp' column in unix seconds.
    rec_start : float
        Unix seconds marking the start of epoch 0.
    n_epochs : int
        Total number of epochs in the recording window.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with an int 'epoch' column, reset index.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")

    if df.empty or n_epochs <= 0:
        res = df.copy()
        res["epoch"] = pd.Series(dtype=int)
        return res

    epoch_series = np.floor((df["timestamp"] - rec_start) / float(EPOCH_S))
    valid_mask = (epoch_series >= 0) & (epoch_series < n_epochs)

    dropped_count = int((~valid_mask).sum())
    if dropped_count > 0:
        logger.debug(
            "assign_epochs: excluded %d readings outside [0, %d)", dropped_count, n_epochs
        )

    res = df.loc[valid_mask].copy()
    res["epoch"] = epoch_series.loc[valid_mask].astype(int)
    return res.reset_index(drop=True)


def epoch_coverage(df: pd.DataFrame, n_epochs: int) -> np.ndarray:
    """Calculate the number of signal readings in each 30-second epoch.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an 'epoch' column.
    n_epochs : int
        Total number of epochs.

    Returns
    -------
    np.ndarray
        Array of integer reading counts per epoch of shape (n_epochs,).
    """
    if "epoch" not in df.columns:
        raise ValueError("DataFrame must contain an 'epoch' column. Call assign_epochs first.")

    coverage = np.zeros(n_epochs, dtype=int)
    if not df.empty and n_epochs > 0:
        counts = df["epoch"].value_counts()
        for ep, cnt in counts.items():
            ep_int = int(ep)
            if 0 <= ep_int < n_epochs:
                coverage[ep_int] = int(cnt)

    return coverage
