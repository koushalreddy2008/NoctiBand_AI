"""
Top-level reporting module for NoctiBand AI.
"""

from noctiband.reporting.nightly import (
    EPOCH_MIN,
    EPOCH_SEC,
    STAGE_NAMES,
    build_report,
    format_report_text,
)

__all__ = [
    "EPOCH_MIN",
    "EPOCH_SEC",
    "STAGE_NAMES",
    "build_report",
    "format_report_text",
]
