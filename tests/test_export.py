"""Unit tests for automated report export in Markdown, HTML, and vector PDF."""

from pathlib import Path

import pytest

from noctiband.reporting.export import (
    export_report_html,
    export_report_markdown,
    export_report_pdf,
)


@pytest.fixture
def sample_report():
    return {
        "subject_id": "User_Test01",
        "night_id": "Night 1",
        "rec_start": 1700000000.0,
        "metrics": {
            "tst_min": 440.0,
            "sleep_efficiency": 89.5,
            "waso_min": 25.0,
            "sol_min": 14.0,
            "noa": 2,
        },
        "hypnogram": [0] * 30 + [1] * 240 + [0] * 10 + [1] * 240,
        "recommendations": [
            "Maintain your consistent morning light exposure.",
            "Keep the bedroom cool and dim.",
        ],
        "disclaimer": "This is a non-diagnostic consumer wellness tool, not a medical evaluation.",
    }


def test_export_report_markdown(tmp_path: Path, sample_report: dict):
    out_file = tmp_path / "report.md"
    content = export_report_markdown(sample_report, output_path=out_file)

    assert out_file.is_file()
    assert "# NoctiBand AI" in content
    assert "User_Test01" in content
    assert "440.0 min" in content
    assert "Maintain your consistent morning light exposure" in content


def test_export_report_html(tmp_path: Path, sample_report: dict):
    out_file = tmp_path / "report.html"
    html = export_report_html(sample_report, output_path=out_file)

    assert out_file.is_file()
    assert "<!DOCTYPE html>" in html
    assert "NoctiBand AI Sleep Architecture Report" in html
    assert "User_Test01" in html


def test_export_report_pdf(tmp_path: Path, sample_report: dict):
    out_file = tmp_path / "report.pdf"
    pdf_path = export_report_pdf(sample_report, output_path=out_file)

    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 1000  # valid PDF generated


def test_export_safety_block_on_diagnostic_terms(sample_report: dict):
    unsafe_report = dict(sample_report)
    unsafe_report["recommendations"] = [
        "You may have severe clinical insomnia requiring prescription pills."
    ]
    with pytest.raises(ValueError, match="safety validator"):
        export_report_markdown(unsafe_report)
