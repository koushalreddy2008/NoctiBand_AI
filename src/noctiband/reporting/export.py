"""Automated Sleep Staging Report Export Engine (v0.2).

Generates exportable clinical and consumer summaries in Markdown, HTML,
and vector PDF formats with embedded hypnograms and strict non-diagnostic
safety validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from noctiband.reporting.nightly import plot_hypnogram
from noctiband.storage import validate_suggestion_safety


def _verify_report_safety(report: dict[str, Any]) -> None:
    """Assert all user-facing tips and notes pass non-diagnostic compliance."""
    recommendations = report.get("recommendations", [])
    for tip in recommendations:
        is_safe, violations = validate_suggestion_safety(tip)
        if not is_safe:
            raise ValueError(
                f"Report export blocked by safety validator: {'; '.join(violations)}"
            )


def export_report_markdown(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> str:
    """Export standardized nightly sleep report in Markdown format.

    Parameters
    ----------
    report : dict
        Structured report dictionary from make_report.
    output_path : str or Path, optional
        Target file path. If None, returns Markdown text without saving.

    Returns
    -------
    str
        Rendered Markdown document.
    """
    _verify_report_safety(report)

    subject_id = report.get("subject_id", "Anonymous")
    night_id = report.get("night_id", "Night 1")
    m = report.get("metrics", {})
    tst = m.get("tst_min", 0.0)

    lines = [
        f"# NoctiBand AI — Sleep Architecture Summary: {subject_id}",
        f"**Recording Session**: {night_id}  ",
        "**Status**: Non-Diagnostic Wellness Assessment  ",
        "",
        "---",
        "",
        "## 1. Key Sleep Metrics",
        "",
        "| Metric | Estimated Value | Reference Optimal |",
        "|:---|:---:|:---:|",
        f"| **Total Sleep Time (TST)** | {tst:.1f} min ({tst / 60.0:.1f} hrs) | 420 – 540 min |",
        f"| **Sleep Efficiency** | {m.get('sleep_efficiency', 0.0):.1f}% | >= 85% |",
        f"| **Sleep Onset Latency (SOL)** | {m.get('sol_min', 0.0):.1f} min | 10 – 30 min |",
        f"| **Wake After Sleep Onset (WASO)** | {m.get('waso_min', 0.0):.1f} min | < 40 min |",
        f"| **Awakenings (NOA)** | {m.get('noa', 0)} | < 5 awakenings |",
        "",
        "## 2. Sleep Stage Breakdown",
        "",
    ]

    dist = report.get("stage_distribution", {})
    if dist:
        lines.append("| Stage | Percentage |")
        lines.append("|:---|:---:|")
        for k, v in dist.items():
            lines.append(f"| **{k}** | {v:.1%} |")
        lines.append("")

    lines.append("## 3. Hygiene & Wellness Recommendations")
    lines.append("")
    for tip in report.get("recommendations", []):
        lines.append(f"- {tip}")
    lines.append("")

    lines.append("---")
    lines.append(f"*Disclaimer: {report.get('disclaimer', '')}*")

    md_content = "\n".join(lines)
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md_content, encoding="utf-8")

    return md_content


def export_report_html(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> str:
    """Export standardized nightly sleep report in standalone responsive HTML format."""
    _verify_report_safety(report)

    subject_id = report.get("subject_id", "Anonymous")
    night_id = report.get("night_id", "Night 1")
    m = report.get("metrics", {})

    tips_html = "".join(f"<li>{tip}</li>" for tip in report.get("recommendations", []))

    css = """
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 30px; color: #222; background: #fafafa;
  }
  .card {
    background: white; padding: 24px; border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-bottom: 24px;
  }
  h1 { color: #1e3a8a; margin-top: 0; }
  .metrics-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin: 16px 0;
  }
  .metric-box {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 12px;
  }
  .metric-title { font-size: 0.85em; color: #166534; font-weight: bold; }
  .metric-value { font-size: 1.4em; font-weight: bold; color: #14532d; margin-top: 4px; }
  ul { padding-left: 20px; }
  li { margin-bottom: 8px; line-height: 1.4; }
  .disclaimer { font-size: 0.85em; color: #666; border-top: 1px solid #ddd; padding-top: 12px; }
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>NoctiBand AI Sleep Report - {subject_id}</title>
<style>{css}</style>
</head>
<body>
<div class="card">
  <h1>🌙 NoctiBand AI Sleep Architecture Report</h1>
  <p><strong>Subject ID:</strong> {subject_id} | <strong>Recording:</strong> {night_id}</p>
  <div class="metrics-grid">
    <div class="metric-box">
      <div class="metric-title">Total Sleep Time</div>
      <div class="metric-value">{m.get('tst_min', 0.0):.1f} min</div>
    </div>
    <div class="metric-box">
      <div class="metric-title">Sleep Efficiency</div>
      <div class="metric-value">{m.get('sleep_efficiency', 0.0):.1f}%</div>
    </div>
    <div class="metric-box">
      <div class="metric-title">WASO</div>
      <div class="metric-value">{m.get('waso_min', 0.0):.1f} min</div>
    </div>
    <div class="metric-box">
      <div class="metric-title">Sleep Latency</div>
      <div class="metric-value">{m.get('sol_min', 0.0):.1f} min</div>
    </div>
  </div>
  <h3>💡 Hygiene & Lifestyle Recommendations</h3>
  <ul>
    {tips_html}
  </ul>
  <p class="disclaimer">🛡️ <strong>Safety Disclaimer:</strong> {report.get('disclaimer', '')}</p>
</div>
</body>
</html>"""

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")

    return html


def export_report_pdf(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Generate and export a publication-quality vector PDF report with embedded hypnogram.

    Parameters
    ----------
    report : dict
        Structured report dictionary from make_report.
    output_path : str or Path
        Target PDF file destination.

    Returns
    -------
    Path
        Destination Path of the created PDF.
    """
    _verify_report_safety(report)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    subject_id = report.get("subject_id", "Subject")
    night_id = report.get("night_id", "Night 1")
    m = report.get("metrics", {})
    hypnogram = report.get("hypnogram", [])
    rec_start = report.get("rec_start", 0.0)

    # 1. Setup multi-panel figure
    fig = plt.figure(figsize=(8.5, 11), constrained_layout=False)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.2, 1.0, 1.2], hspace=0.35)

    # Panel 1: Header and Clinical Metrics Table
    ax_top = fig.add_subplot(gs[0])
    ax_top.axis("off")
    ax_top.text(
        0.0,
        0.95,
        "NoctiBand AI — Nocturnal Sleep Architecture Report",
        fontsize=14,
        fontweight="bold",
        color="#1e3a8a",
    )
    ax_top.text(
        0.0,
        0.83,
        f"Subject ID: {subject_id}    |    Recording: {night_id}    |    Status: Validated",
        fontsize=10,
        color="#4b5563",
    )

    tst_val = m.get("tst_min", 0.0)
    table_data = [
        ["Total Sleep Time (TST)", f"{tst_val:.1f} min ({tst_val / 60.0:.1f} h)", "420 - 540 min"],
        ["Sleep Efficiency (SE)", f"{m.get('sleep_efficiency', 0.0):.1f}%", ">= 85.0%"],
        ["Sleep Onset Latency (SOL)", f"{m.get('sol_min', 0.0):.1f} min", "10 - 30 min"],
        ["Wake After Onset (WASO)", f"{m.get('waso_min', 0.0):.1f} min", "< 40 min"],
        ["Awakenings Count (NOA)", f"{m.get('noa', 0)}", "< 5 count"],
    ]
    col_labels = ["Sleep Architecture Metric", "Estimated Value", "Target Reference"]
    table = ax_top.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="lower center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.65],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    # Panel 2: Vector Hypnogram
    ax_mid = fig.add_subplot(gs[1])
    hypno_fig = plot_hypnogram(hypnogram, rec_start=rec_start)
    hypno_ax = hypno_fig.axes[0]

    for line in hypno_ax.lines:
        ax_mid.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linewidth=line.get_linewidth(),
        )
    for _coll in hypno_ax.collections:
        ax_mid.fill_between(
            hypno_ax.lines[0].get_xdata(),
            hypno_ax.lines[0].get_ydata(),
            step="post",
            alpha=0.25,
            color="#4a90e2",
        )
    ax_mid.set_yticks(hypno_ax.get_yticks())
    ax_mid.set_yticklabels(hypno_ax.get_yticklabels(), fontsize=9, fontweight="bold")
    ax_mid.set_xticks(hypno_ax.get_xticks())
    ax_mid.set_xticklabels(hypno_ax.get_xticklabels(), fontsize=8)
    ax_mid.set_xlim(hypno_ax.get_xlim())
    ax_mid.set_ylim(hypno_ax.get_ylim())
    ax_mid.set_title("Nocturnal Sleep Hypnogram (30s Epochs)", fontsize=10, fontweight="bold")
    ax_mid.set_xlabel(hypno_ax.get_xlabel(), fontsize=8)
    ax_mid.grid(True, linestyle=":", alpha=0.6)
    plt.close(hypno_fig)

    # Panel 3: Hygiene Tips and Safety Disclaimer
    ax_bot = fig.add_subplot(gs[2])
    ax_bot.axis("off")
    ax_bot.text(
        0.0,
        0.95,
        "Sleep Hygiene & Habit Recommendations",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )

    y_cursor = 0.82
    for tip in report.get("recommendations", [])[:4]:
        ax_bot.text(0.02, y_cursor, f"• {tip}", fontsize=9, color="#374151", wrap=True)
        y_cursor -= 0.14

    disclaimer_text = f"Safety Disclaimer: {report.get('disclaimer', '')}"
    ax_bot.text(
        0.0,
        0.08,
        disclaimer_text,
        fontsize=7.5,
        color="#6b7280",
        style="italic",
        wrap=True,
    )

    with PdfPages(out_file) as pdf:
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return out_file
