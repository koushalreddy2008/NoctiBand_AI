# NoctiBand AI — Data Notes

## Dataset and provenance
- Dataset: PhysioNet BIDSleep
- Version: 1.0.0
- Official source: https://physionet.org/content/bidsleep-dataset/1.0.0/
- License: Open Data Commons Attribution License v1.0 (ODC-By 1.0)
- Recording inspected: Bidslab00 / night 3
- Label source: expert_label
- Epoch duration: 30 seconds

## Files inspected
- data/raw/hr.csv
- data/raw/labels.mat
- data/raw/motion.csv.partial (incomplete; not used for motion analysis)

## Heart-rate findings
- HR readings: 5,282
- HR readings inside the labeled time window: 5,266
- HR readings before the label window: 16
- HR readings after the label window: 0
- Observed HR range: 49–94 BPM
- Mean HR: approximately 60.10 BPM
- Expert-labeled epochs: 993
- Epochs without HR readings: 2 (epochs 991 and 992)
- Epoch 990 contains only one HR reading.

## Expert sleep-stage counts
- Wake (0): 35
- N1 (1): 65
- N2 (2): 451
- N3 (3): 203
- REM (4): 239

## Processing
- HR timestamps were mapped to 30-second epochs relative to recStart.
- The original timestamps were not shifted.
- HR readings outside the labeled time window were excluded from the epoch feature table.
- Missing HR features were retained as missing values.
- Epoch-level HR features were generated in data/processed/epoch_features.csv.

## Limitations and pending work
- The HR recording boundaries differ from the label window; this needs further investigation.
- The motion file is incomplete, so movement analysis is pending.
- Only one participant-night has been explored.
- No machine-learning model has been trained.
- Generalization to other participants has not been evaluated.

## Figures
- docs/figures/heart_rate.png
- docs/figures/sleep_stages.png

## License and reproducibility
- Retain the dataset's attribution requirements when reusing or distributing data.
- Record any additional terms applicable to the downloaded release.
- Re-run notebooks/01_explore_one_night.ipynb to reproduce the exploration.
