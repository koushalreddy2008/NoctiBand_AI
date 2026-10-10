# Timestamp Alignment Notes — NoctiBand AI

## Dataset
- Dataset: BIDSleep
- Version: 1.0.0
- Subject/night: Bidslab00/3
- Reference labels: expert_label
- Epoch duration: 30 seconds

## Documented alignment rule
The first label covers [recStart, recStart + 30 seconds].
For each signal timestamp t:

k = floor((t - recStart) / 30) + 1

The formula produces a 1-based label index.
Our Python pipeline uses 0-based epoch IDs, so:
epoch = floor((t - recStart) / 30)

## Findings for this recording
- Expert labels: 993
- HR readings: 5282
- HR readings before recStart: 16
- HR readings after the label window: 0
- HR epochs covered by current mapping: 0 through 990
- Epochs without HR readings: 991 and 992

## Limitation
The dataset documentation specifies the alignment formula but does not
explain the observed HR recording boundaries for this particular night.
No arbitrary timestamp shift has been applied.

## Next actions
1. Verify alignment assumptions against additional subject-night recordings.
2. Inspect motion.csv timestamps and coverage.
3. Preserve missing HR features rather than inventing measurements.
4. Use subject-level separation for model evaluation to reduce data leakage.
