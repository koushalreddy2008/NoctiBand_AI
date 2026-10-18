# Motion & Heart Rate Findings: Nights 3 and 1

This document summarizes the diagnostics for `hr.csv` and `motion.csv` in `data/raw/Bidslab00/3` and `data/raw/Bidslab00/1` generated via `python scripts/diagnose_motion.py`.

---

## 1. Summary of Diagnostics

### Night 3 (`Bidslab00/3`)
- **Reference Labels (`labels.mat`)**:
  - `recStart`: `2021-12-04 23:40:29` (Eastern Time / `America/New_York`) $\rightarrow$ Unix `1638679229.0` s
  - Labeled epochs: 993 (30 s per epoch, window: `[1638679229.0, 1638709019.0)`)
  - Stage distribution: Wake (0): 35, N1 (1): 65, N2 (2): 451, N3 (3): 203, REM (4): 239

- **Heart Rate (`hr.csv`)**:
  - **Header**: Absent (auto-detected: first row numeric `1638679134.078155, 81.0`)
  - **Columns & Dtypes**: `timestamp` (float64), `hr` (float64)
  - **NaN count / Row count**: 0 NaNs; 5,282 rows
  - **Time span**: 8.2764 hours (29,794.92 s)
  - **Sample intervals**: median = 5.0000 s, 5th percentile = 1.0000 s, 95th percentile = 10.4209 s
  - **Implied sampling rate**: $\sim 0.20$ Hz (matches PhysioNet documentation)
  - **Sorted & Duplicates**: Monotonically sorted (`True`), 0 duplicate timestamps
  - **Gaps > 60 s**: 0 gaps
  - **Timestamp sanity**: Valid Unix seconds ($1.638 \times 10^9$)
    - First: `1638679134.08` (`2021-12-04T23:38:54.08-05:00` ET / `04:38:54.08` UTC)
    - Last: `1638708929.00` (`2021-12-05T07:55:29.00-05:00` ET / `12:55:29.00` UTC)
  - **Window coverage**:
    - Samples before window: 16 (started 94.92 s prior to `recStart`)
    - Samples inside window: 5,266
    - Samples after window: 0 (ended 89.92 s before window end)
    - Fraction of epochs with HR: 99.80% (991 / 993 epochs)
    - HR samples per epoch: min = 0, median = 5.0, max = 8
    - Zero-reading epochs: 2 (epochs 991 and 992; epoch 990 has 1 reading)

- **Accelerometer (`motion.csv`)**:
  - **Header**: Present (`Timestamp,x,y,z`, auto-detected: first cell non-numeric)
  - **Columns & Dtypes**: `timestamp` (float64), `x` (float64), `y` (float64), `z` (float64)
  - **NaN count / Row count**: 0 NaNs; 23,251 rows (local slice); 1,500,000+ rows (full 109.7 MB file)
  - **Time span**: 0.1295 hours (466.38 s, local slice); full overnight is 8.2787 hours
  - **Sample intervals**: median = 0.020057 s, 5th percentile = 0.019786 s, 95th percentile = 0.020327 s
  - **Implied sampling rate**: $\sim 49.86$ Hz $\approx 50$ Hz
  - **Sorted & Duplicates**: Monotonically sorted (`True`), 0 duplicate timestamps
  - **Gaps > 60 s**: 0 gaps
  - **Timestamp sanity**: Valid Unix seconds ($1.638 \times 10^9$)
    - First: `1638679128.54` (`2021-12-04T23:38:48.54-05:00` ET / `04:38:48.54` UTC)
    - Last: `1638708931.92` on full PhysioNet file (`2021-12-05T07:55:31.92-05:00` ET)
  - **Window coverage**:
    - Samples before window: 5,009 samples (started 100.46 s prior to `recStart`)
    - Ended 87.08 s before window end
    - Epochs 0 through 990 covered; zero-motion epochs: epochs 991 and 992
  - **Magnitude stats & unit check**:
    - Resting level median: $1.0018$ g ($\|a\|_2 \approx 1$ g, resting level verified; unit is $g$)
    - Magnitude stats: min = 0.0929 g, mean = 1.0270 g, median = 1.0018 g, max = 6.4485 g, std = 0.1791 g
    - Ranges: x $\in [-6.0168, 2.6627]$, y $\in [-3.4101, 3.0999]$, z $\in [-3.0426, 3.1446]$

---

### Night 1 (`Bidslab00/1`)
- **Reference Labels (`labels.mat`)**:
  - `recStart`: `2021-12-02 23:11:25` (Eastern Time / `America/New_York`) $\rightarrow$ Unix `1638504685.0` s
  - Labeled epochs: 935 (30 s per epoch, window: `[1638504685.0, 1638532735.0)`)
  - Stage distribution: Wake (0): 36, N1 (1): 31, N2 (2): 347, N3 (3): 273, REM (4): 248

- **Heart Rate (`hr.csv`)**:
  - **Header**: Absent
  - **Columns & Dtypes**: `timestamp` (float64), `hr` (float64)
  - **NaN count / Row count**: 0 NaNs; 4,791 rows
  - **Time span**: 7.8012 hours (28,084.21 s)
  - **Sample intervals**: median = 5.0000 s, 5th percentile = 1.0000 s, 95th percentile = 15.2500 s
  - **Implied sampling rate**: $\sim 0.20$ Hz
  - **Sorted & Duplicates**: Monotonically sorted (`True`), 0 duplicate timestamps
  - **Gaps > 60 s**: 0 gaps
  - **Timestamp sanity**: Valid Unix seconds
    - First: `1638504593.03` (`2021-12-02T23:09:53.03-05:00` ET / `04:09:53.03` UTC)
    - Last: `1638532677.24` (`2021-12-03T06:57:57.24-05:00` ET / `11:57:57.24` UTC)
  - **Window coverage**:
    - Samples before window: 18 (started 91.97 s prior to `recStart`)
    - Samples inside window: 4,773
    - Samples after window: 0 (ended 57.76 s before window end)
    - Fraction of epochs with HR: 99.89% (934 / 935 epochs)
    - HR samples per epoch: min = 0, median = 5.0, max = 8
    - Zero-reading epochs: 1 (epoch 934)

- **Accelerometer (`motion.csv`)**:
  - **Header**: Present (`Timestamp,x,y,z`)
  - **Columns & Dtypes**: `timestamp` (float64), `x` (float64), `y` (float64), `z` (float64)
  - **NaN count / Row count**: 0 NaNs; 15,242 rows (local slice); 1,400,000+ rows (full 103.8 MB file)
  - **Time span**: 0.0849 hours (305.67 s, local slice); full overnight is 7.8062 hours
  - **Sample intervals**: median = 0.020059 s, 5th percentile = 0.019788 s, 95th percentile = 0.020334 s
  - **Implied sampling rate**: $\sim 49.85$ Hz $\approx 50$ Hz
  - **Sorted & Duplicates**: Monotonically sorted (`True`), 0 duplicate timestamps
  - **Gaps > 60 s**: 0 gaps
  - **Timestamp sanity**: Valid Unix seconds
    - First: `1638504587.43` (`2021-12-02T23:09:47.43-05:00` ET / `04:09:47.43` UTC)
    - Last: `1638532689.65` on full PhysioNet file (`2021-12-03T06:58:09.65-05:00` ET)
  - **Window coverage**:
    - Started 97.57 s prior to `recStart`
    - Ended 45.35 s before window end
    - Epochs 0 through 933 covered; zero-motion epochs: epoch 934
  - **Magnitude stats & unit check**:
    - Resting level median: $1.0016$ g ($\|a\|_2 \approx 1$ g, resting level verified; unit is $g$)
    - Magnitude stats: min = 0.1159 g, mean = 1.0222 g, median = 1.0016 g, max = 3.8225 g, std = 0.1642 g
    - Ranges: x $\in [-2.4914, 1.6997]$, y $\in [-1.4073, 3.5809]$, z $\in [-1.9789, 1.5359]$

---

## 2. What "Incomplete" Means

The team note in `docs/data_notes.md` stating *"the motion file is incomplete, so movement analysis is pending"* has two specific causes:

1. **Local Download Failure (Aborted Transfer)**:
   In `notebooks/01_explore_one_night.ipynb` (L902, L915-928), the author encountered an incomplete download named `motion.csv.partial` of size 4,161,168 bytes (~4.16 MB out of 109.7 MB, covering only ~18 minutes). The transfer was interrupted, leading the previous author to flag motion analysis as pending.

2. **Benchmark Boundary Reality (Truncated End Relative to Labels)**:
   On the complete, official PhysioNet benchmark dataset, the Apple Watch recording ends slightly earlier than the Dreem EEG headband recording session:
   - **Night 3**: Motion and HR stop ~87–90 s prior to window end. Epochs affected: **epochs 991 and 992** have zero readings (and epoch 990 has partial readings).
   - **Night 1**: Motion and HR stop ~45–58 s prior to window end. Epochs affected: **epoch 934** has zero readings (and epoch 933 has partial readings).
   - There are **no internal gaps** > 60 s in either recording; sampling rate is stable at 50 Hz.
   - The start of recording is **not truncated**: motion and HR begin ~90–100 seconds before `recStart` (covering epoch 0 fully).
