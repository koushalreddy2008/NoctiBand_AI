# Data Integrity and Download Verification Policy

## 1. Overview
The PhysioNet BIDSleep v1.0.0 benchmark dataset comprises 47 subjects and 253 recording nights. Previous local explorations erroneously operated on truncated motion files (~1–1.6 MB instead of ~100 MB per night) because partial transfers were stored directly under final filenames without checksum validation.

To prevent corrupted or truncated files from entering the pipeline, all data retrieval must adhere strictly to checksum verification.

---

## 2. Downloader Utility: `scripts/download_bidsleep.py`

The script `scripts/download_bidsleep.py` provides resumable HTTP streaming downloads with mandatory SHA-256 validation against official PhysioNet checksums (`SHA256SUMS.txt`).

### The Atomic Rename Rule
1. Incoming downloads are written to temporary `.part` files: `<filename>.part`.
2. Existing `.part` files support resumed HTTP range requests (`Range: bytes=<pos>-`).
3. Only upon complete download AND matching the exact expected SHA-256 hash from `SHA256SUMS.txt` is the file atomically renamed to its target path (`dest.replace(part)`).
4. If a checksum mismatch occurs, the `.part` file is deleted immediately and an error is raised. Unverified or truncated data is NEVER stored under real benchmark filenames.

---

## 3. Usage Guide

### Verification Only
Verify existing local benchmark files against official checksums without downloading:
```bash
python scripts/download_bidsleep.py --verify-only
```
- Outputs `ok` for matching files and `BAD` for missing or corrupt files.
- Exits with return code 1 if any bad files are detected.

### Targeted Download by Subject and Night
Download specific subjects and nights:
```bash
python scripts/download_bidsleep.py --subjects Bidslab00 Bidslab01 --nights 1 2 3
```

### Full Benchmark Download
Download all 47 subjects and 253 nights (~75 GB):
```bash
python scripts/download_bidsleep.py --all
```

---

## 4. Pipeline Coverage Guards

In addition to download-time verification, the pipeline includes runtime guards:
- `load_night(..., allow_partial=False)`: Raises `ValueError` if signal coverage is below 50% of the labeled recording window:
  $$\text{covered\_fraction} = \frac{\min(t_{\text{last}}, t_{\text{end}}) - \max(t_{\text{first}}, t_{\text{start}})}{n_{\text{epochs}} \times 30} < 0.5$$
- `build_dataset(..., allow_low_motion=False)`: Raises `ValueError` if the fraction of epochs with valid motion is below 50%.
