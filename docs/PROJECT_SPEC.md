# NoctiBand AI — Project Specification

## 1. Project Overview

NoctiBand AI is a sleep-monitoring and improvement-support system. It uses wearable heart-rate and movement signals to estimate sleep states in 30-second epochs, generate an interpretable nightly report, identify patterns that may be worth improving, and offer practical sleep-habit suggestions that can be tracked over time. It is a non-diagnostic research prototype.

### Scope

**Our responsibility**
- Data loading, validation, synchronization, and preprocessing
- Signal-quality checks and 30-second epoching
- Feature engineering and machine-learning experiments
- Model evaluation, inference, reporting, and dashboard
- Integration with the hairband data format when it becomes available
- Sleep-pattern analysis and evidence-informed, non-diagnostic habit suggestions
- Tracking sleep metrics over multiple nights to help users review changes

**Outside our responsibility**
- Hairband design, sensors, electronics, firmware, and low-level hardware integration (handled by the PhD scholar)

### Expected inputs
- Timestamp
- Heart rate or PPG-derived metrics, when available
- Three-axis acceleration (`acc_x`, `acc_y`, `acc_z`)
- Optional raw PPG and inter-beat intervals (IBI), only if the hardware provides them
- Session/device metadata

### Expected outputs
- Predicted sleep state for each 30-second epoch
- Model class probabilities and a separate signal-quality flag
- Estimated sleep duration and sleep-stage distribution
- Estimated sleep onset/wake-up times and awakening/movement summaries
- Data coverage and reliability warnings
- Sleep-improvement suggestions based on repeated patterns, with clear explanations and progress tracking

All sleep metrics are estimates. Improvement suggestions should be conservative, transparent, and based on established sleep-hygiene guidance and observed trends—not on unsupported claims that the model can diagnose or treat a sleep disorder. This project is a research prototype, not a medical diagnostic system.

## 2. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language/environment | Python, `uv` | Implementation and dependency management |
| Data | NumPy, Pandas | Numerical computation and tabular data |
| Signal processing | SciPy; NeuroKit2 when appropriate | Filtering, signal features, physiological-signal utilities |
| Classical ML | scikit-learn | Baselines, models, preprocessing, metrics |
| Advanced ML | PyTorch, later | Neural or sequence models only if justified by experiments |
| Exploration/plots | Jupyter, Matplotlib | Inspect signals, features, and errors |
| Dashboard | Streamlit | Interactive sleep summary |
| API | FastAPI, only if needed | Serve predictions to another application |
| Experiment tracking | MLflow, when useful | Track configurations, metrics, and model artifacts |
| Code quality | Ruff, pytest | Formatting/linting and automated tests |
| Collaboration | Git, GitHub, GitHub Actions | Version control and automated checks |

We will add tools only when the project needs them, rather than installing the entire stack at once.

## 3. Data Strategy

**Primary benchmark:** [PhysioNet BIDSleep v1.0.0](https://www.physionet.org/content/bidsleep-dataset/1.0.0/).

The dataset contains wearable instantaneous heart rate and three-axis accelerometry paired with 30-second sleep-stage labels. It is a benchmark for algorithm development, not data from our custom hairband.

**Important limitation:** smartwatch and hairband signals may differ because of sensor placement, sampling, and device characteristics. We must validate the final pipeline on actual hairband data when available.

**First data task:** read the dataset documentation and license, inspect one participant/night, verify timestamps and sampling rates, and plot heart rate, movement, labels, missingness, and coverage.

## 4. Machine-Learning Plan

### Development stages

1. **Data correctness:** validate timestamps, gaps, alignment, signal quality, and epoch labels.
2. **First task:** classify Wake vs Sleep.
3. **Second task:** classify Wake / NREM / REM if the first task is reliable.
4. **Research extension:** attempt Wake / N1 / N2 / N3 / REM only when data quality and results justify it.
5. **Temporal improvements:** evaluate simple smoothing or an HMM after baseline evaluation.

### Model order

- Dummy/majority baseline
- Logistic Regression
- Random Forest
- SVM
- PyTorch MLP or sequence model only after classical baselines

### Feature groups

Start with interpretable heart-rate statistics and changes, acceleration magnitude/activity statistics, temporal context, and missingness/signal-quality indicators.

True beat-to-beat HRV metrics such as RMSSD require valid inter-beat interval data. Do not derive them from sparse heart-rate values.

### Evaluation rules

- Split data by participant, not by randomly mixing epochs from the same person across train and test.
- Keep the final test set untouched until model selection is complete.
- Report macro-F1, balanced accuracy, per-class precision/recall/F1, a confusion matrix, and Cohen's kappa.
- Inspect errors and class imbalance; do not rely on accuracy alone.

## 5. Sleep-Improvement Support

The system should do more than display a sleep report: it should help users review their patterns and try practical changes over time.

- **Phase 1 — reporting:** show estimated sleep duration, schedule consistency, awakenings/movement summaries, and data-quality limitations.
- **Phase 2 — pattern tracking:** compare multiple nights and highlight repeated changes or irregularities, while distinguishing sensor/model uncertainty from real trends.
- **Phase 3 — suggestions:** offer conservative, explainable sleep-habit suggestions based on established sleep-hygiene guidance and the user's observed patterns. Examples may include maintaining a consistent sleep/wake schedule or reviewing evening caffeine habits where relevant.
- **Phase 4 — feedback:** let users record which suggestions they tried and compare later sleep reports, without claiming that a suggestion caused an improvement.

Do not initially build an automated treatment system or claim to diagnose or treat insomnia or other sleep disorders. The first version can use transparent rules and user feedback; personalization or reinforcement learning is a later research question that requires suitable data and evaluation.

## 6. Software Architecture

**Pipeline:** raw data → schema/timestamp validation → synchronization → signal-quality checks → preprocessing → 30-second epoching → feature extraction → model inference → confidence and quality flags → nightly aggregation → sleep report → pattern analysis → practical habit suggestions → multi-night progress tracking/dashboard.

The pipeline will use a stable internal data contract so the hairband's future output can be adapted without rewriting the ML logic.

### Repository structure

```text
NoctiBand_AI/
├── artifacts/          # Generated results and model artifacts; avoid committing large files
├── configs/            # Experiment configurations
├── docs/               # Project specification and technical notes
├── notebooks/          # Data exploration and experiments
├── src/                # Reusable Python modules
├── tests/               # Automated tests
├── README.md
├── LICENSE
├── requirements.txt    # Existing file; may be replaced by pyproject.toml after setup
└── .gitignore
```

As implementation grows, `src/` can be organized into modules for data loading, preprocessing, features, models, evaluation, inference, and reporting. Preserve the current repository; do not recreate it.

## 7. Roadmap

1. Set up and verify the local Python environment.
2. Obtain BIDSleep and inspect one recording.
3. Build and test the loader and 30-second epoch alignment.
4. Create a feature table and participant-level train/validation/test split.
5. Train the baseline and classical models; record results.
6. Perform error analysis and add temporal methods only if useful.
7. Build the inference workflow and Streamlit dashboard for nightly reports.
8. Add a transparent first version of sleep-improvement support using conservative sleep-habit guidance and repeated trends; let users track suggestions and outcomes.
9. Adapt the input layer to real hairband data and evaluate domain shift.

## 8. Definition of Done

- A recording can be loaded, validated, and converted into correctly aligned 30-second epochs.
- Preprocessing and features are reproducible outside a notebook.
- Train, validation, and test participants do not overlap.
- Models are compared using the same split and clearly reported metrics.
- Error analysis, model artifacts, tests, and experiment settings are saved.
- New hairband data can be mapped into the same inference pipeline.
- The dashboard reports estimates, confidence, signal quality, and warnings.
- Sleep-improvement suggestions are explainable, non-diagnostic, and tracked over time.
- Documentation clearly states limitations and avoids unsupported clinical claims.

## 9. Immediate Next Steps

1. Create this file at `docs/PROJECT_SPEC.md`.
2. Review the existing `README.md` and keep it as a short overview with setup instructions and a link to this specification.
3. Set up the Python environment and dependencies.
4. Read the BIDSleep data documentation before writing the loader.
5. Build the first data-exploration notebook and inspect one night before training a model.
6. After the reporting pipeline is reliable, design a separate sleep-improvement module that summarizes trends and offers conservative habit suggestions.

---

**Project description:** NoctiBand AI is a transparent wearable sleep-monitoring system that estimates sleep states from heart-rate and motion signals, generates nightly reports, and supports sleep improvement through explainable habit suggestions and multi-night progress tracking. It is a non-diagnostic research prototype designed to accept data from a custom hairband.
