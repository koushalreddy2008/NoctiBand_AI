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
- data/raw/hr.csv (wearable optical heart rate)
- data/raw/motion.csv (3-axis accelerometry)
- data/raw/labels.mat (expert sleep stages)

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

## Synthetic signals (NOT measured data)
BP and SpO2 in `scripts/synthetic_demo/` are simulated from the expert labels for UI
demos only. They encode the target and are excluded from all modelling. BIDSleep
provides only wearable HR and 3-axis accelerometry.

## Real-world public clinical datasets with overnight SpO2 and BP
If you require additional real clinical patient datasets beyond the BIDSleep benchmark:
1. **NSRR MESA (Multi-Ethnic Study of Atherosclerosis):** https://sleepdata.org/datasets/mesa
   - 2,050+ full overnight PSG studies including continuous SpO2, plethysmography, ECG, and sleep stages.
2. **NSRR SHHS (Sleep Heart Health Study):** https://sleepdata.org/datasets/shhs
   - 5,800+ overnight PSG recordings with continuous SpO2 and cardiovascular metrics.
3. **PhysioNet BIDMC PPG and Respiration Dataset:** https://physionet.org/content/bidmc/1.0.0/
   - Synchronized PPG, SpO2, ECG, and respiration.
4. **PhysioNet MIMIC-IV Waveform Database:** https://physionet.org/content/mimic4wdb/0.1.0/
   - Continuous Arterial Blood Pressure (ABP) and SpO2 waveforms.

## Processing
- HR timestamps were mapped to 30-second epochs relative to recStart.
- The original timestamps were not shifted.
- HR readings outside the labeled time window were excluded from the epoch feature table.
- Missing HR features were retained as missing values.
- Epoch-level HR features only. A table with synthetic BP/SpO2 columns exists from `scripts/synthetic_demo/` and is excluded from modelling.

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
