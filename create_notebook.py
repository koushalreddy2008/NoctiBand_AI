from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
notebook_dir = root / "notebooks"
notebook_dir.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
}

cells = [
    nbf.v4.new_markdown_cell("""# NoctiBand AI — One-Night Data Exploration

**Dataset:** PhysioNet BIDSleep v1.0.0  
**Recording inspected:** Bidslab00 / night 3  
**Purpose:** Inspect heart rate, expert sleep-stage labels, timestamps, coverage and data quality before machine-learning experiments.

The motion file is currently incomplete, so motion analysis is intentionally marked pending. No model is trained in this notebook.

Reference: https://physionet.org/content/bidsleep-dataset/1.0.0/
"""),
    nbf.v4.new_markdown_cell("""## A. Load the data

We use the original HR CSV and label MAT file. Paths are resolved relative to the project root, not the terminal's current directory."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT / "docs" / "PROJECT_SPEC.md").exists():
    ROOT = ROOT.parent

if not (ROOT / "docs" / "PROJECT_SPEC.md").exists():
    raise FileNotFoundError("Could not locate project root containing docs/PROJECT_SPEC.md")

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

hr_path = RAW / "hr.csv"
labels_path = RAW / "labels.mat"

hr = pd.read_csv(hr_path, header=None, names=["timestamp", "heart_rate"])
mat = loadmat(labels_path)
expert_labels = mat["expert_label"].ravel()
dreem_labels = mat["dreem_label"].ravel()

rec_start_text = str(mat["recStart"][0])
rec_start = pd.Timestamp(rec_start_text).tz_localize("America/New_York")
rec_start_unix = rec_start.timestamp()

print("Project root:", ROOT)
print("HR rows:", len(hr))
print("Expert labels:", len(expert_labels))
print("Automated labels:", len(dreem_labels))
print("Recording start:", rec_start)
print("HR missing values:")
print(hr.isna().sum())
"""),
    nbf.v4.new_markdown_cell("""## B. Validate timestamps and coverage

Each label represents a 30-second epoch. Epoch IDs below are zero-based, so epoch 0 corresponds to the first label."""),
    nbf.v4.new_code_cell("""hr["time_utc"] = pd.to_datetime(hr["timestamp"], unit="s", utc=True)
hr["epoch"] = np.floor((hr["timestamp"] - rec_start_unix) / 30).astype(int)

window_start = rec_start_unix
window_end = window_start + 30 * len(expert_labels)
inside = hr["timestamp"].between(window_start, window_end, inclusive="left")

print("Label window start UTC:", pd.to_datetime(window_start, unit="s", utc=True))
print("Label window end UTC:", pd.to_datetime(window_end, unit="s", utc=True))
print("First HR UTC:", hr["time_utc"].min())
print("Last HR UTC:", hr["time_utc"].max())
print("HR readings before window:", int((hr["timestamp"] < window_start).sum()))
print("HR readings inside window:", int(inside.sum()))
print("HR readings after window:", int((hr["timestamp"] >= window_end).sum()))
print("HR readings with valid epoch IDs:", int(hr["epoch"].between(0, len(expert_labels)-1).sum()))
"""),
    nbf.v4.new_markdown_cell("""## C. Inspect heart rate

This plot shows the original HR samples. Spikes are inspected rather than automatically removed."""),
    nbf.v4.new_code_cell("""plt.figure(figsize=(14, 4))
plt.plot(hr["time_utc"], hr["heart_rate"], linewidth=0.8)
plt.xlabel("Time (UTC)")
plt.ylabel("Heart rate (BPM)")
plt.title("Heart Rate Throughout the Recording")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "heart_rate.png", dpi=150)
plt.show()

print(hr["heart_rate"].describe())
"""),
    nbf.v4.new_markdown_cell("""## D. Inspect sleep-stage labels

Label codes: 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM, 5 = Unknown."""),
    nbf.v4.new_code_cell("""stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 5: "Unknown"}
label_times = pd.to_datetime(
    rec_start_unix + np.arange(len(expert_labels)) * 30,
    unit="s", utc=True
)

plt.figure(figsize=(14, 4))
plt.step(label_times, expert_labels, where="post")
plt.yticks(list(stage_names), [stage_names[x] for x in stage_names])
plt.xlabel("Time (UTC)")
plt.ylabel("Expert sleep stage")
plt.title("Expert Sleep Stages")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "sleep_stages.png", dpi=150)
plt.show()

counts = pd.Series(expert_labels).value_counts().sort_index()
print("Expert label counts:")
print(counts.rename(index=stage_names))
print("Total labeled epochs:", len(expert_labels))
"""),
    nbf.v4.new_markdown_cell("""## E. Build an epoch-level HR feature table

Each row represents one labeled 30-second epoch. Epochs without HR readings are retained with missing features rather than invented values."""),
    nbf.v4.new_code_cell("""valid_hr = hr.loc[hr["epoch"].between(0, len(expert_labels)-1)].copy()

features = valid_hr.groupby("epoch").agg(
    hr_count=("heart_rate", "count"),
    hr_mean=("heart_rate", "mean"),
    hr_median=("heart_rate", "median"),
    hr_std=("heart_rate", "std"),
    hr_min=("heart_rate", "min"),
    hr_max=("heart_rate", "max"),
    hr_first=("heart_rate", "first"),
    hr_last=("heart_rate", "last"),
)

features = features.reindex(range(len(expert_labels)))
features.index.name = "epoch"
features = features.reset_index()
features["sleep_stage"] = expert_labels
features["hr_range"] = features["hr_max"] - features["hr_min"]
features["hr_change"] = features["hr_last"] - features["hr_first"]

PROCESSED.mkdir(parents=True, exist_ok=True)
features.to_csv(PROCESSED / "epoch_features.csv", index=False)

print("Feature table shape:", features.shape)
print("Epochs without HR:", int(features["hr_count"].isna().sum()))
display(features.head(10))
"""),
    nbf.v4.new_markdown_cell("""## F. Missingness and signal-quality checks"""),
    nbf.v4.new_code_cell("""print("Missing values by column:")
display(features.isna().sum().to_frame("missing_count"))

print("HR feature summary:")
display(features[["hr_count", "hr_mean", "hr_std", "hr_min", "hr_max"]].describe())

print("Epochs with no HR readings:")
display(features.loc[features["hr_count"].isna(), ["epoch", "sleep_stage"]])

print("Epochs with only one HR reading:")
display(features.loc[features["hr_count"].eq(1), ["epoch", "hr_count", "hr_mean", "sleep_stage"]])
"""),
    nbf.v4.new_markdown_cell("""## G. Movement data status

The available `motion.csv.partial` is an incomplete download and must not be treated as a complete recording. Movement plots and acceleration features remain pending until the original motion file has been downloaded and verified."""),
    nbf.v4.new_code_cell("""motion_complete = RAW / "motion.csv"
motion_partial = RAW / "motion.csv.partial"

if motion_complete.exists():
    print("A motion.csv file exists. Verify its completeness and schema before analysis.")
elif motion_partial.exists():
    print("Motion analysis pending: only motion.csv.partial is present.")
    print("Partial file size:", motion_partial.stat().st_size, "bytes")
else:
    print("Motion analysis pending: motion file not found.")
"""),
    nbf.v4.new_markdown_cell("""## H. Findings and limitations

- HR and expert labels are inspected separately.
- Epoch assignment follows the dataset's documented 30-second rule relative to `recStart`.
- HR samples outside the labeled time window are excluded from the epoch feature table.
- Missing HR features are retained as missing values.
- The HR recording boundaries differ from the label window; this is documented and no arbitrary timestamp shift is applied.
- Motion analysis remains incomplete until a verified full `motion.csv` is available.
- This is one night from one participant and is not sufficient to establish model generalization.
- No model is trained in this notebook.

**Next:** verify alignment and motion coverage, then process multiple participants/nights before subject-level model evaluation.
""")
]

nb["cells"] = cells
out = notebook_dir / "01_explore_one_night.ipynb"
nbf.write(nb, out)
print("Created:", out)
print("Cells:", len(cells))
