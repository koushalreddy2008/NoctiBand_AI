# Dataset Adapter Guide: Epoch-Feature Canonical Schema

## 1. Overview
The NoctiBand AI feature extraction and evaluation pipeline is designed to be completely device- and dataset-agnostic. While BIDSleep v1.0.0 serves as an initial wearable benchmark, any future wearable sensor data (including custom headband prototypes) can interface directly with the pipeline by mapping onto the canonical epoch table schema.

---

## 2. Canonical Epoch Schema Specification

An epoch-level dataset must be structured as a table with one row per 30-second epoch conforming to the following column definitions:

| Canonical Column | Type | Description | Permissible Aliases |
|:---|:---|:---|:---|
| `subject_id` | `str` or `int` | Unique participant identifier. Folds are grouped by this column. | `subject` |
| `night_id` | `str` or `int` | Identifier for the recording night session. | `night` |
| `epoch_idx` | `int` | Zero-based sequential epoch index (0, 1, 2, ...). | `epoch` |
| `label` | `float` or `int` | Ground truth sleep stage: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, NaN=Unknown. | `label_binary` (0/1) |
| Feature Columns | `float` | At least one numeric column (e.g. HR, accelerometer ENMO, rolling context). | Any numeric columns |

### Optional Quality and Label Flags:
- `label_binary`: 0 for Wake, 1 for Sleep (stages 1–4).
- `label_3class`: 0 for Wake, 1 for NREM (stages 1–3), 2 for REM (stage 4).
- `label_unknown`: Boolean flag indicating unannotated or stage 5 (Unknown) epochs.
- `motion_missing`: Boolean flag indicating zero motion samples in the epoch.

---

## 3. Creating an Adapter for a New Dataset

To adapt a new sensor dataset (e.g., custom headband, polysomnography, or Apple Watch / Fitbit):

```python
from pathlib import Path
import pandas as pd
from noctiband.data_schema import check_epoch_table, standardize_columns

def adapt_custom_headband_data(raw_export_path: Path) -> pd.DataFrame:
    """Transform custom wearable export into the canonical epoch table schema."""
    raw_df = pd.read_csv(raw_export_path)

    # 1. Map columns to canonical identifiers
    df = raw_df.rename(
        columns={
            "participant": "subject_id",
            "session": "night_id",
            "epoch_number": "epoch_idx",
            "expert_stage": "label",
        }
    )

    # 2. Derive standardized binary and 3-class target labels
    df["label_binary"] = df["label"].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df["label_3class"] = df["label"].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 2})
    df["label_unknown"] = df["label"].isna() | (df["label"] == 5)

    # 3. Standardize and validate schema
    standardized = standardize_columns(df)
    check_epoch_table(standardized)

    return standardized
```

---

## 4. Automated Schema Validation

Before running cross-validation, the pipeline runs `check_epoch_table`:
- Verifies that `subject_id`, `night_id`, `epoch_idx`, and labels are present.
- Verifies that feature columns are numeric, non-empty, and free of infinite values.
- Throws an informative `ValueError` listing any formatting defects.
