"""NoctiBand AI - Consumer Sleep Architecture & Wellness App."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    # CLI or script execution fallback
    print("Streamlit is not installed in the current environment.")
    st = None

from noctiband.app_glue import aggregate_multi_night, load_and_validate_upload
from noctiband.inference.predict import predict_night
from noctiband.report_bridge import make_report
from noctiband.types import NightData, QualityControlError


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

    # Add wake motion spikes
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

    st.set_page_config(page_title="NoctiBand AI - Sleep Wellness", layout="wide")
    st.title("🌙 NoctiBand AI - Consumer Sleep Architecture")
    st.markdown(
        "*Evidence-based nocturnal sleep staging & wellness tracking using wearable actigraphy.*"
    )

    # Sidebar mode selection
    st.sidebar.header("Mode Selection")
    mode = st.sidebar.radio(
        "Choose Data Source:",
        ["Synthetic Demo Night", "Upload Wearable Data", "Multi-Night Synthetic Demo"],
    )

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
            for n in multi_nights:
                pred = predict_night(n, allow_partial=True)
                predictions.append(pred)

            agg = aggregate_multi_night(predictions)
            st.subheader("📊 7-Night Sleep Architecture Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Mean TST",
                f"{agg['mean_tst_min']:.1f} min",
                f"±{agg['std_tst_min']:.1f} min",
            )
            col2.metric(
                "Mean Sleep Efficiency",
                f"{agg['mean_sleep_efficiency']:.1f}%",
                f"±{agg['std_sleep_efficiency']:.1f}%",
            )
            col3.metric("Sleep Regularity Index", f"{agg['sleep_regularity_index']:.1f} / 100")
            col4.metric(
                "Efficiency Trend Slope",
                f"{agg['efficiency_trend_slope']:+.3f}% / night",
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
            return

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
            prediction = predict_night(night_data, allow_partial=True)
            report = make_report(
                prediction,
                subject_id=night_data.subject,
                night_id=night_data.night,
            )
            m = report["metrics"]

            # Key metrics cards
            c1, c2, c3, c4 = st.columns(4)
            tst_hrs = m["tst_min"] / 60.0
            c1.metric("Total Sleep Time (TST)", f"{m['tst_min']:.1f} min ({tst_hrs:.1f} hrs)")
            c2.metric("Sleep Efficiency", f"{m['sleep_efficiency']:.1f}%")
            c3.metric("Wake After Onset (WASO)", f"{m['waso_min']:.1f} min")
            c4.metric("Sleep Onset Latency (SOL)", f"{m['sol_min']:.1f} min")

            # Hypnogram
            st.subheader("📈 Sleep Hypnogram (30s Epochs)")
            hypno_df = pd.DataFrame({
                "Epoch": np.arange(len(prediction.epoch_labels)),
                "Stage": ["Sleep" if s == 1 else "Wake" for s in prediction.epoch_labels],
                "StageCode": prediction.epoch_labels,
            })
            st.area_chart(hypno_df.set_index("Epoch")["StageCode"])

            # Recommendations
            st.subheader("💡 Sleep Hygiene & Lifestyle Recommendations")
            for tip in report["recommendations"]:
                st.markdown(f"- {tip}")

            # Disclaimer
            st.markdown("---")
            st.caption(f"🛡️ **Non-Diagnostic Disclaimer**: {report['disclaimer']}")

        except QualityControlError as qc_err:
            st.error(f"Quality Control Error: {qc_err}")


if __name__ == "__main__":
    run_app()
