"""
Plot test set confusion matrix from test_final.json.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    json_path = ROOT / "docs/results/test_final.json"
    if not json_path.exists():
        json_path = ROOT / "artifacts/results/test_final.json"

    res = json.loads(json_path.read_text())
    cm = np.array(res["confusion_matrix"])
    classes = ["Wake", "Sleep"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = count / cm[i].sum() * 100
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold",
            )

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(
        f"Final Test Confusion Matrix ({res['model']} | window={res['smoothing_window']})",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )

    plt.tight_layout()
    out_fig = ROOT / "docs/figures/confusion_matrix_test.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, bbox_inches="tight")
    print(f"Confusion matrix figure successfully saved to {out_fig}")


if __name__ == "__main__":
    main()
