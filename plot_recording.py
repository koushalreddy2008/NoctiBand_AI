import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find the project root from this script's location
PROJECT_ROOT = Path(__file__).resolve().parent

# Load the aligned dataset
data_path = PROJECT_ROOT / "data" / "processed" / "aligned_hr.csv"
df = pd.read_csv(data_path)

# Convert Unix timestamps to readable UTC times
df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

# Create output folder
output_dir = PROJECT_ROOT / "results" / "data_exploration"
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Heart rate over time
plt.figure(figsize=(14, 5))
plt.plot(df["time"], df["heart_rate"], linewidth=0.8)
plt.xlabel("Time (UTC)")
plt.ylabel("Heart rate (BPM)")
plt.title("Heart Rate Throughout the Recording")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "heart_rate.png", dpi=150)
plt.close()

# Plot 2: Expert sleep-stage labels over time
plt.figure(figsize=(14, 4))
plt.scatter(df["time"], df["sleep_stage"], s=5)
plt.yticks([0, 1, 2, 3, 4], ["Wake", "N1", "N2", "N3", "REM"])
plt.xlabel("Time (UTC)")
plt.ylabel("Expert sleep stage")
plt.title("Expert Sleep-Stage Labels Throughout the Recording")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "sleep_stages.png", dpi=150)
plt.close()

print("Plots saved successfully:")
print(output_dir / "heart_rate.png")
print(output_dir / "sleep_stages.png")
