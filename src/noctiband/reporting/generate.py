"""
End-to-end report generation integrating inference with clinical reporting.
"""

from __future__ import annotations

from typing import Any

from noctiband.inference.pipeline import run_night  # A's function
from noctiband.reporting.nightly import build_report, format_report_text


def generate_report(night_folder: str, **kwargs: Any) -> tuple[dict[str, Any], str]:
    """Generate an end-to-end nightly sleep report directly from raw sensor files.

    Parameters
    ----------
    night_folder : str
        Directory path containing raw overnight sensor channels.
    **kwargs : Any
        Additional keyword arguments forwarded to build_report (e.g., rec_start, tz).

    Returns
    -------
    tuple[dict, str]
        (structured_report_dict, formatted_report_text)
    """
    predictions = run_night(night_folder)  # one row per epoch, must have epoch_idx and y_pred

    # Automatically extract recording start timestamp from predictions if not supplied
    if "rec_start" not in kwargs and getattr(predictions, "attrs", {}).get("rec_start"):
        kwargs["rec_start"] = predictions.attrs["rec_start"]

    report = build_report(predictions, **kwargs)
    return report, format_report_text(report)
