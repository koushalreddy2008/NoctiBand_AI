# Motion & Heart Rate Findings: Measured Diagnostics and Integrity Failure

## 1. Critical Finding: Truncated Downloads and False Prior Verdict

The previous verdict in earlier reports ("benign 1-2 epoch truncated tail") was **WRONG**.
The local `motion.csv` files were truncated downloads stored directly under the real filename without checksum verification:
- **Night 1 (`data/raw/Bidslab00/1/motion.csv`)**:
  - Local size: **1.07 MB** (15,242 rows)
  - Local SHA-256: `f79ebc54d6576f27237303cfafcc85827be31087ad31fe12c19fc7cc0498f6d2`
  - Official PhysioNet `SHA256SUMS.txt` hash begins: `1ff851c08a2e1186...`
  - Local coverage: **305.67 seconds (~5.1 minutes)** out of a 7.80-hour (935 epochs) recording window (**0.7% covered fraction**).
- **Night 3 (`data/raw/Bidslab00/3/motion.csv`)**:
  - Local size: **1.64 MB** (23,251 rows)
  - Local SHA-256: `4f3b62ed5c5d46e14b0d6dfc7ea035b0aa390838fd942034ec35d6532494e8cc`
  - Official PhysioNet `SHA256SUMS.txt` hash begins: `7afec69f8cbb1bb4...`
  - Local coverage: **466.38 seconds (~7.8 minutes)** out of an 8.28-hour (993 epochs) recording window (**1.2% covered fraction**).

Both files are aborted transfers covering only the first 5–8 minutes of sleep instead of the expected ~100 MB (~1.5M rows at 50 Hz). As a consequence, Phase 4 epoch tables constructed from these files suffered from ~99% missing motion readings.

---

## 2. Measured Diagnostics on Local Files

### Night 3 (`Bidslab00/3`)
- **Labels (`labels.mat`)**:
  - `recStart`: `2021-12-04 23:40:29` Eastern Time (`1638679229.0` unix s).
  - Labeled epochs: 993 (window `[1638679229.0, 1638709019.0)`).
  - Stages: Wake (0): 35, N1 (1): 65, N2 (2): 451, N3 (3): 203, REM (4): 239.
- **Heart Rate (`hr.csv`)**:
  - Header: Absent (numeric first cell). Rows: 5,282. NaNs: 0. Duplicates: 0.
  - Time span: 29,794.92 s (8.2764 hours). Median interval: 5.0 s (implied rate: ~0.20 Hz).
  - Window coverage: 5,266 samples inside window (first ts: `rec_start - 94.92s`, last ts: `rec_start + 29700.0s`).
  - Window covered fraction: 99.80% (epochs 0..990 covered, epochs 991–992 empty).
- **Motion (`motion.csv`) [TRUNCATED]**:
  - Header: Present (`Timestamp,x,y,z`). Rows: 23,251. NaNs: 0. Duplicates: 0.
  - Time span: 466.38 s (0.1295 hours). Median interval: 0.020057 s (~49.86 Hz).
  - Window coverage: 18,242 samples inside window, 5,009 samples before window.
  - Window covered fraction: **1.2%** (covered only epochs 0 through 12; epochs 13 through 992 have zero motion readings).
  - Accelerometer magnitude (first 7.8 min): mean = 1.0270 g, median = 1.0018 g, max = 6.4485 g.

### Night 1 (`Bidslab00/1`)
- **Labels (`labels.mat`)**:
  - `recStart`: `2021-12-02 23:11:25` Eastern Time (`1638504685.0` unix s).
  - Labeled epochs: 935 (window `[1638504685.0, 1638532735.0)`).
  - Stages: Wake (0): 36, N1 (1): 31, N2 (2): 347, N3 (3): 273, REM (4): 248.
- **Heart Rate (`hr.csv`)**:
  - Header: Absent. Rows: 4,791. NaNs: 0. Duplicates: 0.
  - Time span: 28,084.21 s (7.8012 hours). Median interval: 5.0 s (~0.20 Hz).
  - Window coverage: 4,773 samples inside window (first ts: `rec_start - 91.97s`, last ts: `rec_start + 27992.24s`).
  - Window covered fraction: 99.89% (epochs 0..933 covered, epoch 934 empty).
- **Motion (`motion.csv`) [TRUNCATED]**:
  - Header: Present (`Timestamp,x,y,z`). Rows: 15,242. NaNs: 0. Duplicates: 0.
  - Time span: 305.67 s (0.0849 hours). Median interval: 0.020059 s (~49.85 Hz).
  - Window coverage: 10,233 samples inside window, 5,009 samples before window.
  - Window covered fraction: **0.7%** (covered only epochs 0 through 6; epochs 7 through 934 have zero motion readings).
  - Accelerometer magnitude (first 5.1 min): mean = 1.0222 g, median = 1.0016 g, max = 3.8225 g.

---

## 3. Conclusions and Required Actions

1. The motion signal was aborted during prior download and covers under 1.5% of the overnight sleep session.
2. Any modeling or feature evaluation relying on these files is scientifically invalid until full overnight recordings (~100 MB each) are retrieved and validated against `SHA256SUMS.txt`.
3. Added automated coverage guards (`load_night` and `build_dataset`) that strictly reject recordings with < 50% window coverage unless explicit override flags are supplied.
