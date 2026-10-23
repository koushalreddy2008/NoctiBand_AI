"""
Top-level nightly reporting module forwarding to noctiband.reporting.nightly.
"""

from noctiband.reporting.nightly import (
    EPOCH_MIN,
    EPOCH_SEC,
    STAGE_NAMES,
    build_report,
    format_report_text,
    plot_hypnogram,
)

__all__ = [
    "EPOCH_MIN",
    "EPOCH_SEC",
    "STAGE_NAMES",
    "build_report",
    "format_report_text",
    "plot_hypnogram",
]
