import pandas as pd
from pathlib import Path
from scipy.io import loadmat

# Project paths
ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "processed" / "aligned_hr.csv"
LABELS = ROOT / "data" / "raw" / "labels.mat"
OUTPUT = ROOT / "data" / "processed" / "epoch_features.csv"

# Load aligned heart-rate readings
df = pd.read_csv(INPUT)

# Load all expert labels, including epochs with no HR readings
mat = loadmat(LABELS)
expert_labels = mat["expert_label"].ravel()

# Calculate features for each 30-second epoch
features = df.groupby("epoch").agg(
    hr_count=("heart_rate", "count"),
    hr_mean=("heart_rate", "mean"),
    hr_median=("heart_rate", "median"),
    hr_std=("heart_rate", "std"),
    hr_min=("heart_rate", "min"),
    hr_max=("heart_rate", "max"),
    hr_first=("heart_rate", "first"),
    hr_last=("heart_rate", "last"),
)

# Include all 993 labeled epochs
features = features.reindex(range(len(expert_labels)))
features.index.name = "epoch"
features = features.reset_index()

# Add labels and derived features
features["sleep_stage"] = expert_labels
features["hr_range"] = features["hr_max"] - features["hr_min"]
features["hr_change"] = features["hr_last"] - features["hr_first"]

# Save the epoch-level dataset
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUTPUT, index=False)

# Report data quality
print("Epoch-level dataset created.")
print("Shape:", features.shape)
print("Expected epochs:", len(expert_labels))
print("Epochs without HR:", int(features["hr_count"].isna().sum()))
print("Sleep-stage counts:")
print(features["sleep_stage"].value_counts().sort_index().to_dict())
print("Feature preview:")
print(features.head(10).to_string(index=False))
print("Saved to:", OUTPUT)
