# Audit: Existing Epoching Pipeline and Timestamp Alignment

## 1. What Existing Code and Docs Already Do
- `docs_timestamp_alignment.md` documents the BIDSleep 30-second epoch rule (L10-18): 1st label covers `[recStart, recStart + 30 s]`, converting 1-based index $k = \lfloor(t - \text{recStart})/30\rfloor + 1$ (L14) to 0-based `epoch = floor((t - recStart) / 30)` (L18). It documents Bidslab00/3 counts (993 labels, 5282 HR rows, 16 pre-window rows, epochs 991-992 unobserved; L21-26).
- `create_epoch_features.py` loads pre-aligned HR from flat path `data/processed/aligned_hr.csv` (L7, 12), reads labels from `data/raw/labels.mat` via `loadmat` (L8, 15-16), aggregates 8 HR summary metrics by epoch (L19-28), reindexes across `range(len(expert_labels))` to retain missing epochs (L31-33), appends expert sleep stages (L36), and exports `epoch_features.csv` (L41-42).

## 2. Reusable and Correct Components
- **Epoch Index Formula**: `epoch = floor((t - rec_start) / 30)` from `docs_timestamp_alignment.md` (L18) conforms to Rule 5 and the BIDSleep specification.
- **Label Ingestion**: `mat["expert_label"].ravel()` via `scipy.io.loadmat` in `create_epoch_features.py` (L15-16) correctly extracts per-epoch ground truth.
- **Complete Epoch Grid**: Reindexing over all `n_epochs` (L31) preserves epochs with zero readings as explicit NaNs / unmeasured epochs instead of dropping them.

## 3. Bugs, Fragile Logic, and Unsafe Assumptions
- **Hard-coded Flat Paths**: `create_epoch_features.py` assumes flat `data/raw/labels.mat` and `data/processed/aligned_hr.csv` (L7-9), breaking the canonical layout `data/raw/<subject>/<night>/` defined in `docs/INTERFACES.md` (L11).
- **Timezone & DST Blindness**: Neither file implements `recStart` string parsing or timezone localization. Downstream exploration (`01_explore_one_night.ipynb` L78-80) uses `pd.Timestamp(rec_start_text).tz_localize("America/New_York")` without resolving ambiguous or non-existent daylight saving transition times.
- **Header Handling Fragility**: `create_epoch_features.py` requires an external preprocessing step to produce `aligned_hr.csv` (L7, 12). Neither script supports raw `motion.csv` or auto-detects presence/absence of CSV headers.
- **Boundary & Off-by-One Risks**: The epoch window is half-open `[rec_start, rec_start + n_epochs * 30)`. At exact upper bound $t = \text{rec\_start} + 30 \cdot n_{\text{epochs}}$, `floor` yields $n_{\text{epochs}}$. Any inclusive filtering or clipping includes an invalid out-of-bounds epoch index $n_{\text{epochs}}$.
- **Silent Data Ingestion & Dropping**: Pre-aligned input bypasses row deduplication, NaN dropping, timestamp sorting, and logging of discarded readings.
- **Missing Alignment Guards**: No verification checks whether signal timestamps drift excessively from `recStart` (e.g., timezone offset or milliseconds vs seconds mismatch).
