# NoctiBand AI 🌙

**An AI-powered wearable sleep-monitoring and sleep-improvement support system.**

NoctiBand AI aims to transform wearable heart-rate and movement signals into sleep-state estimates, understandable nightly reports, and practical recommendations that help users improve their sleep habits over time.

> **Project status:** Initial repository scaffold completed. Dataset exploration and ML pipeline development are the next milestones.

## 🎯 Project Objectives

- Estimate sleep states using wearable physiological and movement signals.
- Generate interpretable nightly sleep reports.
- Identify repeated sleep patterns and potential areas for improvement.
- Provide explainable sleep-habit suggestions.
- Track sleep metrics across multiple nights to review changes.
- Build a reproducible ML pipeline that can integrate with the custom hairband.

## ✨ Planned Features

### 1. Sleep Monitoring
- Heart-rate and movement signal processing
- Signal-quality checks and missing-data detection
- Sleep-state estimation at 30-second intervals

### 2. Sleep Analysis and Reporting
- Estimated total sleep duration
- Sleep-stage distribution
- Estimated sleep onset and wake-up times
- Awakening and movement summaries
- Prediction confidence and data-quality warnings

### 3. Sleep Improvement
- Analysis of sleep patterns across multiple nights
- Practical, explainable sleep-habit suggestions
- Tracking of suggestions and subsequent sleep reports
- Progress visualization over time

## 🧠 Machine-Learning Approach

The project will follow an incremental, experimental approach:

1. Validate and preprocess the sensor data.
2. Build a Wake vs Sleep classification baseline.
3. Compare classical ML models.
4. Explore Wake / NREM / REM classification.
5. Investigate five-stage sleep classification when justified by the data.
6. Evaluate temporal methods and more advanced models only when useful.

**Planned baseline models:**
- Dummy/majority classifier
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

PyTorch-based models may be explored after the classical baselines are established.

## 📊 Dataset

**Primary benchmark:** [PhysioNet BIDSleep v1.0.0](https://www.physionet.org/content/bidsleep-dataset/1.0.0/)

The dataset provides wearable heart-rate and three-axis accelerometry data paired with sleep-stage labels at 30-second resolution.

It will support initial model development and evaluation. Because smartwatch signals may differ from those of our custom hairband, performance must be validated on actual hairband data when available.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | NumPy, Pandas |
| Signal processing | SciPy, NeuroKit2 when appropriate |
| Classical ML | scikit-learn |
| Advanced ML | PyTorch, when justified |
| Exploration and visualization | Jupyter, Matplotlib |
| Interactive dashboard | Streamlit |
| Code quality and testing | Ruff, pytest |
| Version control | Git, GitHub |
| Environment management | uv |

FastAPI, MLflow, and other supporting tools may be introduced when the implementation requires them.

## 🏗️ System Workflow

```text
Wearable Sensor Data
        ↓
Data Validation and Synchronization
        ↓
Signal Quality Checks and Preprocessing
        ↓
30-Second Epoching
        ↓
Feature Extraction
        ↓
Sleep-State ML Model
        ↓
Nightly Sleep Report
        ↓
Sleep Pattern Analysis
        ↓
Habit Suggestions and Progress Tracking
        ↓
Interactive Dashboard
```

## 📁 Repository Structure

```text
NoctiBand_AI/
├── artifacts/
├── configs/
├── data/
├── docs/
│   └── PROJECT_SPEC.md
├── notebooks/
├── src/
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 📚 Detailed Documentation

For the complete project scope, technical architecture, dataset strategy, evaluation methodology, development roadmap, and definition of done, see:

**[Project Specification](docs/PROJECT_SPEC.md)**

## 🚀 Development Roadmap

- [ ] Set up and verify the Python development environment.
- [ ] Obtain and explore the BIDSleep dataset.
- [ ] Build the data-loading and preprocessing pipeline.
- [ ] Implement and test 30-second epoch alignment.
- [ ] Extract features and establish baseline models.
- [ ] Evaluate models using participant-level data splits.
- [ ] Build the inference pipeline and nightly sleep report.
- [ ] Develop the sleep-improvement support module.
- [ ] Build the interactive dashboard.
- [ ] Integrate and validate the pipeline with actual hairband data.

## 👥 Project Scope

The student team is responsible for machine learning, signal processing, data analysis, software development, evaluation, reporting, and visualization.

The physical hairband, sensors, electronics, firmware, and low-level hardware integration are handled by the PhD scholar.

## ⚠️ Disclaimer

NoctiBand AI is a research prototype, not a medical diagnostic system. Sleep metrics are estimates and depend on signal quality, sensor characteristics, and model performance. Recommendations are intended to support general sleep habits, not diagnose or treat sleep disorders.

## 📄 License

This project is distributed under the license specified in the repository's `LICENSE` file.