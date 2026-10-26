"""NoctiBand AI - Consumer Sleep Architecture & Wellness App."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    # CLI or script execution fallback
    print("Streamlit is not installed in the current environment.")
    st = None

from noctiband.app_glue import aggregate_multi_night, load_and_validate_upload
from noctiband.features.hrv import compute_epoch_hrv_features
from noctiband.inference.predict import predict_night
from noctiband.report_bridge import make_report
from noctiband.reporting.export import (
    export_report_html,
    export_report_markdown,
    export_report_pdf,
)
from noctiband.reporting.nightly import plot_hypnogram
from noctiband.storage import (
    calculate_sleep_debt,
    calculate_social_jetlag,
    calculate_sri,
    validate_suggestion_safety,
)
from noctiband.types import NightData, QualityControlError

# Premium Dark Mode Theme Styles
CUSTOM_CSS = """
<style>
  .stApp {
    background-color: #0b0f19;
    color: #f8fafc;
  }
  section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
  }
  .nocti-card {
    background-color: #131b2e;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
  }
  .metric-header {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 4px;
  }
  .metric-val {
    font-size: 1.7rem;
    font-weight: 800;
    color: #38bdf8;
    line-height: 1.2;
    word-break: normal;
  }
  .metric-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 4px;
  }
  .badge-optimal {
    display: inline-block;
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.35);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 6px;
  }
  .badge-alert {
    display: inline-block;
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 6px;
  }
  .badge-compliance {
    display: inline-block;
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 8px 0;
  }
</style>
"""


def metric_card(
    title: str,
    value: str,
    subtext: str = "",
    badge: str = "",
    badge_type: str = "optimal",
) -> str:
    """Render a premium non-truncating metric card using custom HTML/CSS."""
    badge_html = ""
    if badge:
        cls = "badge-alert" if badge_type == "alert" else "badge-optimal"
        badge_html = f"<div class='{cls}'>{badge}</div>"
    return (
        f"<div class='nocti-card'>"
        f"<div class='metric-header'>{title}</div>"
        f"<div class='metric-val'>{value}</div>"
        f"<div class='metric-sub'>{subtext}</div>"
        f"{badge_html}"
        f"</div>"
    )


def style_dark_figure(fig: plt.Figure) -> plt.Figure:
    """Apply dark health-tech styling to a Matplotlib figure."""
    fig.patch.set_facecolor("#131b2e")
    for ax in fig.axes:
        ax.set_facecolor("#0b0f19")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")
        ax.grid(True, linestyle=":", alpha=0.3, color="#334155")
    return fig


def generate_synthetic_demo_night(seed: int = 42, wake_glitches: int = 3) -> NightData:
    """Generate a programmatic synthetic 8-hour demo recording."""
    rng = np.random.RandomState(seed)
    n_epochs = 960  # 8 hours
    total_seconds = n_epochs * 30

    t_hr = np.arange(0, total_seconds, 1.0)
    hr_vals = 62.0 + 6.0 * np.sin(2 * np.pi * t_hr / 3600.0) + rng.normal(0, 1.0, len(t_hr))
    hr_df = pd.DataFrame({"timestamp": t_hr, "hr": hr_vals})

    t_mot = np.arange(0, total_seconds, 0.02)
    x = 0.04 * np.sin(2 * np.pi * t_mot / 10.0) + rng.normal(0, 0.01, len(t_mot))
    y = 0.04 * np.cos(2 * np.pi * t_mot / 10.0) + rng.normal(0, 0.01, len(t_mot))
    z = 0.98 + 0.02 * np.sin(2 * np.pi * t_mot / 5.0) + rng.normal(0, 0.01, len(t_mot))

    for _ in range(wake_glitches):
        burst_start = rng.uniform(3600, total_seconds - 3600)
        mask = (t_mot >= burst_start) & (t_mot < burst_start + 60)
        x[mask] += rng.uniform(0.3, 0.8, mask.sum())

    mot_df = pd.DataFrame({"timestamp": t_mot, "x": x, "y": y, "z": z})

    return NightData(
        subject="demo_subject",
        night="demo_night",
        hr=hr_df,
        motion=mot_df,
        labels=None,
        rec_start=0.0,
    )


def run_app():
    if st is None:
        print("To run interactive UI: streamlit run app/app.py")
        return

    st.set_page_config(
        page_title="NoctiBand AI - Elite Sleep Architecture",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🌙 NoctiBand AI — Sleep Architecture & Autonomic Health")
    st.markdown(
        "<span style='color: #94a3b8; font-size: 0.95rem;'>"
        "Evidence-based nocturnal sleep staging, autonomic HRV tracking, "
        "and chronic wellness analytics."
        "</span>",
        unsafe_allow_html=True,
    )

    # Sidebar mode selection
    st.sidebar.header("Mode Selection")
    mode = st.sidebar.radio(
        "Choose Data Source:",
        ["Synthetic Demo Night", "Upload Wearable Data", "Multi-Night Synthetic Demo"],
    )

    # Sidebar model selection
    st.sidebar.markdown("---")
    st.sidebar.header("Model Engine")
    root_dir = Path(__file__).resolve().parent.parent
    xgb_path = root_dir / "models" / "noctiband_xgb_v0.2.joblib"
    lr_path = root_dir / "models" / "noctiband_v0.1.joblib"

    model_choices: dict[str, Path] = {}
    if xgb_path.exists():
        model_choices["XGBoost v0.2 (Gradient Boosted)"] = xgb_path
    if lr_path.exists():
        model_choices["Logistic Regression v0.1 (Linear)"] = lr_path

    if model_choices:
        selected_model_name = st.sidebar.selectbox(
            "Classifier Architecture:",
            list(model_choices.keys()),
            index=0,
        )
        chosen_model_path = model_choices[selected_model_name]
        st.sidebar.caption(f"Active: `{selected_model_name}`")
    else:
        chosen_model_path = None

    # Session state hygiene
    if "active_mode" not in st.session_state:
        st.session_state["active_mode"] = mode
    elif st.session_state["active_mode"] != mode:
        keys_to_clear = [k for k in list(st.session_state.keys()) if k != "active_mode"]
        for k in keys_to_clear:
            del st.session_state[k]
        st.session_state["active_mode"] = mode

    night_data = None

    if mode == "Synthetic Demo Night":
        st.info("ℹ️ Running in **[SYNTHETIC DEMO]** mode using programmatic actigraphy & HR.")
        night_data = generate_synthetic_demo_night(seed=42)

    elif mode == "Multi-Night Synthetic Demo":
        st.info("ℹ️ Running in **[SYNTHETIC DEMO]** multi-night mode (7 nights).")
        multi_nights = [
            generate_synthetic_demo_night(seed=40 + i, wake_glitches=i % 4)
            for i in range(7)
        ]

        if st.sidebar.button("Run Multi-Night Analysis", type="primary"):
            predictions = []
            try:
                for n in multi_nights:
                    pred = predict_night(n, model_path=chosen_model_path, allow_partial=True)
                    predictions.append(pred)
            except QualityControlError as qc_err:
                st.error(f"⚠️ Multi-Night Quality Control Failure: {qc_err}")
                st.warning(
                    "Recommendation: Ensure wearable sensors maintain continuous skin contact "
                    "and signal quality exceeds minimum coverage thresholds."
                )
                return
            except Exception as e:
                st.error(f"⚠️ Unexpected Multi-Night Analysis Error: {e}")
                return

            agg = aggregate_multi_night(predictions)
            st.session_state["multi_night_agg"] = agg
            st.session_state["multi_night_preds"] = predictions

        # Use first night for primary tab visualization
        night_data = multi_nights[0]

    else:
        st.sidebar.subheader("Upload Sensor Files")
        hr_upload = st.sidebar.file_uploader(
            "Upload Heart Rate (CSV / TXT)",
            type=["csv", "txt"],
        )
        mot_upload = st.sidebar.file_uploader(
            "Upload Accelerometer (CSV / TXT)",
            type=["csv", "txt"],
        )

        if hr_upload and mot_upload:
            try:
                night_data = load_and_validate_upload({"hr": hr_upload, "motion": mot_upload})
                st.success("Successfully loaded and validated wearable sensor data.")
            except Exception as e:
                st.error(f"Upload validation failed: {e}")
                return

    if night_data is not None:
        try:
            prediction = predict_night(night_data, model_path=chosen_model_path, allow_partial=True)
            report = make_report(
                prediction,
                subject_id=night_data.subject,
                night_id=night_data.night,
            )
            m = report["metrics"]

            # Main Multi-Tab Interface
            tab1, tab2, tab3, tab4 = st.tabs([
                "🌙 Night Overview & Hypnogram",
                "🔬 Autonomic & HRV Telemetry",
                "📈 Longitudinal Trends & Multi-Night",
                "📥 Reports & Export Hub",
            ])

            # ---------------------------------------------------------
            # TAB 1: NIGHT OVERVIEW & HYPNOGRAM
            # ---------------------------------------------------------
            with tab1:
                st.subheader("Core Sleep Metrics")
                c1, c2, c3, c4 = st.columns(4)

                tst_min = m.get("tst_min", 0.0)
                tst_hrs = tst_min / 60.0
                se = m.get("sleep_efficiency", 0.0)
                waso = m.get("waso_min", 0.0)
                sol = m.get("sol_min", 0.0)

                c1.markdown(
                    metric_card(
                        "Total Sleep Time",
                        f"{tst_min:.1f} min",
                        f"{tst_hrs:.1f} hours asleep",
                        "Optimal" if tst_min >= 420 else "Sub-optimal",
                        "optimal" if tst_min >= 420 else "alert",
                    ),
                    unsafe_allow_html=True,
                )
                c2.markdown(
                    metric_card(
                        "Sleep Efficiency",
                        f"{se:.1f}%",
                        "Target: >= 85.0%",
                        "High Efficiency" if se >= 85.0 else "Low Efficiency",
                        "optimal" if se >= 85.0 else "alert",
                    ),
                    unsafe_allow_html=True,
                )
                c3.markdown(
                    metric_card(
                        "WASO (Wake After Onset)",
                        f"{waso:.1f} min",
                        "Target: < 40 min",
                        "Normal" if waso < 40 else "Elevated Wake",
                        "optimal" if waso < 40 else "alert",
                    ),
                    unsafe_allow_html=True,
                )
                c4.markdown(
                    metric_card(
                        "Sleep Latency (SOL)",
                        f"{sol:.1f} min",
                        "Normal: 10 – 30 min",
                        "Optimal" if 10 <= sol <= 30 else "Shifted",
                        "optimal" if 10 <= sol <= 30 else "alert",
                    ),
                    unsafe_allow_html=True,
                )

                st.subheader("Clinical Hypnogram (30s Epoch Staging)")
                fig = plot_hypnogram(prediction, rec_start=night_data.rec_start)
                style_dark_figure(fig)
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("💡 Sleep Hygiene & Lifestyle Recommendations")
                for tip in report["recommendations"]:
                    st.markdown(f"- {tip}")

                st.markdown("---")
                st.caption(f"🛡️ **Non-Diagnostic Disclaimer**: {report['disclaimer']}")

            # ---------------------------------------------------------
            # TAB 2: AUTONOMIC & HRV TELEMETRY
            # ---------------------------------------------------------
            with tab2:
                st.subheader("Autonomic Nervous System & HRV Breakdown")
                if night_data.hr is not None and not night_data.hr.empty:
                    hrv_epoch_df = compute_epoch_hrv_features(
                        hr_df=night_data.hr,
                        rec_start=night_data.rec_start,
                        n_epochs=len(prediction.epoch_labels),
                    )
                    valid_rmssd = hrv_epoch_df["hrv_rmssd"].dropna()
                    valid_lf_hf = hrv_epoch_df["hrv_lf_hf_ratio"].dropna()
                    valid_sdnn = hrv_epoch_df["hrv_sdnn"].dropna()
                    valid_hr = night_data.hr["hr"].dropna()

                    h1, h2, h3, h4 = st.columns(4)
                    rmssd_val = valid_rmssd.mean() if not valid_rmssd.empty else 0.0
                    h1.markdown(
                        metric_card(
                            "Mean RMSSD",
                            f"{rmssd_val:.1f} ms",
                            "Vagal / Parasympathetic Tone",
                            "Optimal Recovery" if rmssd_val >= 25 else "Suppressed",
                            "optimal" if rmssd_val >= 25 else "alert",
                        ),
                        unsafe_allow_html=True,
                    )
                    lf_hf_val = valid_lf_hf.mean() if not valid_lf_hf.empty else 0.0
                    h2.markdown(
                        metric_card(
                            "Mean LF/HF Ratio",
                            f"{lf_hf_val:.2f}",
                            "Sympathovagal Balance",
                            "Balanced" if 1.0 <= lf_hf_val <= 2.5 else "Sympathetic Bias",
                            "optimal" if 1.0 <= lf_hf_val <= 2.5 else "alert",
                        ),
                        unsafe_allow_html=True,
                    )
                    sdnn_val = valid_sdnn.mean() if not valid_sdnn.empty else 0.0
                    h3.markdown(
                        metric_card(
                            "Mean SDNN",
                            f"{sdnn_val:.1f} ms",
                            "Overall HRV Variability",
                            "Target: >= 40 ms",
                        ),
                        unsafe_allow_html=True,
                    )
                    hr_mean_val = valid_hr.mean() if not valid_hr.empty else 0.0
                    h4.markdown(
                        metric_card(
                            "Resting Heart Rate",
                            f"{hr_mean_val:.1f} bpm",
                            "Nocturnal Average",
                            "Normal" if hr_mean_val <= 70 else "Elevated",
                        ),
                        unsafe_allow_html=True,
                    )

                    st.subheader("Nocturnal HRV Trajectory (RMSSD & Autonomic Balance)")
                    hrv_chart_df = pd.DataFrame({
                        "Epoch": np.arange(len(hrv_epoch_df)),
                        "RMSSD (ms)": hrv_epoch_df["hrv_rmssd"],
                        "LF/HF Ratio": hrv_epoch_df["hrv_lf_hf_ratio"],
                    }).set_index("Epoch")
                    st.line_chart(hrv_chart_df)
                else:
                    st.warning("No continuous heart rate telemetry available for HRV extraction.")

            # ---------------------------------------------------------
            # TAB 3: LONGITUDINAL TRENDS & MULTI-NIGHT
            # ---------------------------------------------------------
            with tab3:
                st.subheader("Longitudinal Health Analytics & Chronic Trends")
                if "multi_night_agg" in st.session_state:
                    agg = st.session_state["multi_night_agg"]
                    preds = st.session_state.get("multi_night_preds", [])

                    tst_history = [float(n["tst_min"]) for n in agg["nights"]]
                    sleep_debt = calculate_sleep_debt(tst_history, baseline_need_min=480.0)
                    hypnos = [p.epoch_labels for p in preds] if preds else []
                    sri = calculate_sri(hypnos) if hypnos else agg["sleep_regularity_index"]

                    sol_history = [float(n.get("sol_min", 20.0)) for n in agg["nights"]]
                    workdays = sol_history[:5]
                    freedays = sol_history[5:] if len(sol_history) > 5 else sol_history[-2:]
                    social_jetlag = calculate_social_jetlag(workdays, freedays)

                    l1, l2, l3, l4 = st.columns(4)
                    l1.markdown(
                        metric_card(
                            "Sleep Debt",
                            f"{sleep_debt:.1f} min",
                            "Baseline Target: 8.0h",
                            "Minimal Debt" if sleep_debt <= 60 else "High Debt",
                            "optimal" if sleep_debt <= 60 else "alert",
                        ),
                        unsafe_allow_html=True,
                    )
                    l2.markdown(
                        metric_card(
                            "Sleep Regularity Index",
                            f"{sri:.1f} / 100",
                            "Circadian Alignment",
                            "Consistent" if sri >= 80 else "Irregular",
                            "optimal" if sri >= 80 else "alert",
                        ),
                        unsafe_allow_html=True,
                    )
                    l3.markdown(
                        metric_card(
                            "Social Jetlag",
                            f"{social_jetlag:.1f} min",
                            "Workday vs Free Day Shift",
                            "Low Jetlag" if social_jetlag <= 45 else "Elevated",
                            "optimal" if social_jetlag <= 45 else "alert",
                        ),
                        unsafe_allow_html=True,
                    )
                    l4.markdown(
                        metric_card(
                            "Mean TST",
                            f"{agg['mean_tst_min']:.1f} min",
                            f"+/- {agg['std_tst_min']:.1f} min deviation",
                        ),
                        unsafe_allow_html=True,
                    )

                    trend_df = pd.DataFrame([
                        {
                            "Night": n["night_index"],
                            "Sleep Efficiency (%)": n["sleep_efficiency"],
                            "TST (min)": n["tst_min"],
                        }
                        for n in agg["nights"]
                    ])
                    st.line_chart(trend_df.set_index("Night"))
                else:
                    st.info(
                        "Longitudinal multi-night analytics require 2 or more recording sessions. "
                        "Switch to 'Multi-Night Synthetic Demo' in the sidebar to simulate "
                        "a 7-day trend."
                    )

            # ---------------------------------------------------------
            # TAB 4: REPORTS & EXPORT HUB
            # ---------------------------------------------------------
            with tab4:
                st.subheader("Automated Export Engine & Clinical Reports")

                # Live Safety Status Banner
                tips_to_check = report.get("recommendations", [])
                violations_found = []
                for tip in tips_to_check:
                    is_safe, viols = validate_suggestion_safety(tip)
                    if not is_safe:
                        violations_found.extend(viols)

                if not violations_found:
                    st.markdown(
                        "<div class='badge-compliance'>"
                        "✅ <strong>Safety Status:</strong> 100% Non-Diagnostic Compliance "
                        "Verified "
                        "(Zero Medical / Prescription Terminology)"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Compliance Violation Detected: {'; '.join(violations_found)}")

                st.markdown("### Export Formats")
                e1, e2, e3 = st.columns(3)

                try:
                    md_text = export_report_markdown(report)
                    e1.download_button(
                        label="📄 Download Markdown (.md)",
                        data=md_text,
                        file_name=f"{night_data.subject}_{night_data.night}_report.md",
                        mime="text/markdown",
                    )
                except Exception as ex:
                    e1.error(f"Markdown export blocked: {ex}")

                try:
                    html_text = export_report_html(report)
                    e2.download_button(
                        label="🌐 Download HTML Report (.html)",
                        data=html_text,
                        file_name=f"{night_data.subject}_{night_data.night}_report.html",
                        mime="text/html",
                    )
                except Exception as ex:
                    e2.error(f"HTML export blocked: {ex}")

                try:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                        tmp_pdf_path = Path(tmp_pdf.name)
                    export_report_pdf(report, tmp_pdf_path)
                    pdf_bytes = tmp_pdf_path.read_bytes()
                    tmp_pdf_path.unlink(missing_ok=True)
                    e3.download_button(
                        label="📑 Download Vector PDF (.pdf)",
                        data=pdf_bytes,
                        file_name=f"{night_data.subject}_{night_data.night}_report.pdf",
                        mime="application/pdf",
                    )
                except Exception as ex:
                    e3.error(f"PDF export blocked: {ex}")

                with st.expander("📄 Report Preview (Markdown)", expanded=False):
                    st.code(export_report_markdown(report), language="markdown")

        except QualityControlError as qc_err:
            st.error(f"⚠️ Quality Control Guard Triggered: {qc_err}")
            st.warning(
                "Actionable Recommendations:\n"
                "- Ensure wristband is worn snugly and sensors maintain uninterrupted contact.\n"
                "- Check sensor battery levels to prevent gaps in photoplethysmography telemetry.\n"
                "- Verify recording covers at least 4 hours with valid heart rate and motion data."
            )
        except Exception as e:
            st.error(f"⚠️ Sleep Staging Error: {e}")


if __name__ == "__main__":
    run_app()
