# Handoff to Person B — Phase 0

- **Branch**: `phase4-A` (based on `part3-loader-A`)
- **Loader**: `from noctiband.io import load_night`
  - Signature: `load_night(night_dir: str | Path, subject: str | None = None, night: str | None = None, rec_start: float | None = None) -> NightData`
- **Epoching**: `from noctiband.preprocessing import assign_epochs, epoch_coverage`
  - Signatures: `assign_epochs(df: pd.DataFrame, rec_start: float, n_epochs: int) -> pd.DataFrame`
  - `epoch_coverage(df: pd.DataFrame, n_epochs: int) -> np.ndarray`
- **CLI Diagnostics**: `python scripts/diagnose_motion.py data/raw/Bidslab00/3`
- **Pytest**: `python -m pytest -q -rs` (19 passed, 0 skipped); `python -m pytest -m data -rs` (1 passed)
- **Status**: Phase 0 preflight passed. Awaiting "GO PHASE 4".
