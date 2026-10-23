"""Feature extraction and dataset splitting for NoctiBand AI."""

from noctiband.features.epoch_features import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    MODEL_FEATURES,
    QC_FEATURE_COLUMNS,
    build_dataset,
    build_epoch_table,
    make_split,
)
from noctiband.features.hrv import (
    HRV_COLUMNS,
    compute_epoch_hrv_features,
    compute_hrv_metrics,
    enrich_with_hrv_features,
)

__all__ = [
    "FEATURE_COLUMNS",
    "HRV_COLUMNS",
    "LABEL_COLUMNS",
    "METADATA_COLUMNS",
    "MODEL_FEATURES",
    "QC_FEATURE_COLUMNS",
    "build_dataset",
    "build_epoch_table",
    "compute_epoch_hrv_features",
    "compute_hrv_metrics",
    "enrich_with_hrv_features",
    "make_split",
]
