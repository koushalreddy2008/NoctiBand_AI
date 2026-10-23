"""
End-to-end integration test for nightly report generator.
"""


import pytest
from scipy.io import loadmat

from noctiband.models.data import ROOT
from noctiband.reporting.generate import generate_report

DATA_DIR = ROOT / "data" / "raw" / "Bidslab00" / "1"
LABELS_FILE = DATA_DIR / "labels.mat"


@pytest.mark.skipif(not LABELS_FILE.is_file(), reason="Bidslab00/1 dataset not found")
def test_end_to_end_generate_report_on_real_night():
    report, text = generate_report(str(DATA_DIR))

    # Validate report structure
    assert isinstance(report, dict)
    assert report["recording_minutes"] == pytest.approx(993 * 0.5)
    assert report["total_sleep_min"] > 0
    assert report["sleep_onset_min"] is not None
    assert report["wake_up_min"] is not None
    assert 0.0 < report["sleep_efficiency_pct"] <= 100.0
    assert isinstance(report["stage_distribution"], dict)
    assert "Wake" in report["stage_distribution"]
    assert "Sleep" in report["stage_distribution"]

    # Validate text format
    assert isinstance(text, str)
    assert "NoctiBand nightly report" in text
    assert "Estimated sleep time:" in text
    assert "Sleep efficiency:" in text

    # Sanity check against true sleep labels
    mat = loadmat(LABELS_FILE)
    labels = mat["expert_label"].ravel()
    true_sleep_epochs = int((labels != 0).sum())
    true_sleep_min = true_sleep_epochs * 0.5

    # Estimated sleep should be within 20% of ground-truth polysomnography
    assert abs(report["total_sleep_min"] - true_sleep_min) / true_sleep_min < 0.20
