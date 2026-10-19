# Phase 4 Handoff to Person B

- **Feature Module**: `src/noctiband/features/epoch_features.py` (exported in `src/noctiband/features/`).
- **Interfaces**:
  - `build_epoch_table(night: NightData, config=None) -> pd.DataFrame`
  - `build_dataset(raw_root, out_path=None, config=None) -> pd.DataFrame`
  - `make_split(table, seed=42, ratios=(0.7, 0.15, 0.15), out_path=None) -> dict[str, list[str]]`
- **Exported Lists**: `FEATURE_COLUMNS` (40 features), `LABEL_COLUMNS`, `METADATA_COLUMNS`.
- **Config**: `configs/features.yaml` (centered rolling windows +-2, +-5; `include_hours_since_start: false`).
- **Splitting Constraint**: `make_split` strictly requires n_subjects >= 3; raises `ValueError` on local data (n_subjects=1).
- **Missing / Partial Handling**: Zero-count epochs emit NaNs + `motion_missing=True`; counts < 500 emit `motion_partial=True`.
