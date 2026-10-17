"""
30-second epoching and multimodal signal alignment for NoctiBand AI.

Transforms raw overnight sensor streams into aligned 30-second epoch tables
with signal quality verification, label mapping, and physiological feature summaries.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.io.contracts import EpochedNight, NightRecording, SignalQuality, SleepStage
from src.preprocessing.alignment import (
    audit_signal_alignment,
    epoch_to_datetimes,
    epoch_to_timestamps,
    filter_and_align_signal,
)

logger = logging.getLogger(__name__)


class EpochAligner:
    """Aligns and segments multimodal overnight sensor signals into 30-second epochs."""

    def __init__(self, epoch_duration_sec: float = 30.0) -> None:
        """Initialize the epoch aligner.

        Args:
            epoch_duration_sec: Duration of each epoch in seconds (default: 30.0).
        """
        self.epoch_duration_sec = float(epoch_duration_sec)

    def align_night(
        self,
        recording: NightRecording,
        preserve_slices: bool = False,
    ) -> EpochedNight:
        """Process a NightRecording and return an EpochedNight.

        Ensures every epoch in [0, total_epochs - 1] is present in the output,
        even if no sensor readings were recorded for that epoch.

        Args:
            recording: Loaded NightRecording instance.
            preserve_slices: If True, preserves raw signal dataframes for each epoch.

        Returns:
            EpochedNight containing the aligned epoch table and metadata.
        """
        rec_start = recording.rec_start
        rec_start_unix = rec_start.timestamp()
        total_epochs = recording.total_epochs

        if total_epochs <= 0:
            logger.warning(f"Recording {recording.subject_id} has 0 epochs.")
            empty_df = pd.DataFrame(columns=["epoch", "start_time", "end_time", "quality_flag"])
            return EpochedNight(
                subject_id=recording.subject_id,
                night_id=recording.night_id,
                rec_start=rec_start,
                epoch_duration_sec=self.epoch_duration_sec,
                epochs_df=empty_df,
                metadata=recording.metadata,
            )

        # 1. Initialize master 30s epoch grid
        epoch_indices = np.arange(total_epochs, dtype=int)
        time_bounds = [
            epoch_to_datetimes(ep, rec_start, self.epoch_duration_sec) for ep in epoch_indices
        ]
        start_times = [tb[0] for tb in time_bounds]
        end_times = [tb[1] for tb in time_bounds]

        df_epochs = pd.DataFrame(
            {
                "epoch": epoch_indices,
                "start_time": start_times,
                "end_time": end_times,
                "start_unix": [
                    epoch_to_timestamps(ep, rec_start_unix, self.epoch_duration_sec)[0]
                    for ep in epoch_indices
                ],
                "end_unix": [
                    epoch_to_timestamps(ep, rec_start_unix, self.epoch_duration_sec)[1]
                    for ep in epoch_indices
                ],
            }
        ).set_index("epoch")

        metadata: dict[str, Any] = dict(recording.metadata)

        # 2. Align and aggregate Heart Rate
        if recording.has_hr and recording.hr is not None:
            hr_aligned = filter_and_align_signal(
                recording.hr,
                rec_start_unix,
                total_epochs=total_epochs,
                epoch_duration_sec=self.epoch_duration_sec,
                drop_outside=True,
            )
            hr_audit = audit_signal_alignment(
                recording.hr,
                rec_start_unix,
                total_epochs=total_epochs,
                epoch_duration_sec=self.epoch_duration_sec,
            )
            metadata["hr_audit"] = hr_audit

            hr_grp = hr_aligned.groupby("epoch")["heart_rate"]
            hr_agg = pd.DataFrame(
                {
                    "hr_count": hr_grp.count(),
                    "hr_mean": hr_grp.mean(),
                    "hr_std": hr_grp.std(),
                    "hr_min": hr_grp.min(),
                    "hr_max": hr_grp.max(),
                    "hr_first": hr_grp.first(),
                    "hr_last": hr_grp.last(),
                }
            )
            hr_agg["hr_range"] = hr_agg["hr_max"] - hr_agg["hr_min"]
            hr_agg["hr_change"] = hr_agg["hr_last"] - hr_agg["hr_first"]
            df_epochs = df_epochs.join(hr_agg)
        else:
            df_epochs["hr_count"] = 0
            df_epochs["hr_mean"] = np.nan
            df_epochs["hr_std"] = np.nan
            df_epochs["hr_min"] = np.nan
            df_epochs["hr_max"] = np.nan
            df_epochs["hr_first"] = np.nan
            df_epochs["hr_last"] = np.nan
            df_epochs["hr_range"] = np.nan
            df_epochs["hr_change"] = np.nan

        # Fill hr_count NA with 0
        df_epochs["hr_count"] = df_epochs["hr_count"].fillna(0).astype(int)

        # 3. Align and aggregate Motion / Accelerometry
        if recording.has_motion and recording.motion is not None:
            motion_aligned = filter_and_align_signal(
                recording.motion,
                rec_start_unix,
                total_epochs=total_epochs,
                epoch_duration_sec=self.epoch_duration_sec,
                drop_outside=True,
            )
            motion_audit = audit_signal_alignment(
                recording.motion,
                rec_start_unix,
                total_epochs=total_epochs,
                epoch_duration_sec=self.epoch_duration_sec,
            )
            metadata["motion_audit"] = motion_audit

            # Compute acceleration vector magnitude (VM) and ENMO (Euclidean Norm Minus One)
            vm = np.sqrt(
                motion_aligned["x"] ** 2 + motion_aligned["y"] ** 2 + motion_aligned["z"] ** 2
            )
            motion_aligned["vm"] = vm
            motion_aligned["enmo"] = np.maximum(0.0, vm - 1.0)

            mot_grp = motion_aligned.groupby("epoch")
            mot_agg = pd.DataFrame(
                {
                    "motion_count": mot_grp["vm"].count(),
                    "vm_mean": mot_grp["vm"].mean(),
                    "vm_std": mot_grp["vm"].std(),
                    "vm_max": mot_grp["vm"].max(),
                    "enmo_mean": mot_grp["enmo"].mean(),
                    "acc_x_mean": mot_grp["x"].mean(),
                    "acc_y_mean": mot_grp["y"].mean(),
                    "acc_z_mean": mot_grp["z"].mean(),
                }
            )
            df_epochs = df_epochs.join(mot_agg)
        else:
            df_epochs["motion_count"] = 0
            df_epochs["vm_mean"] = np.nan
            df_epochs["vm_std"] = np.nan
            df_epochs["vm_max"] = np.nan
            df_epochs["enmo_mean"] = np.nan
            df_epochs["acc_x_mean"] = np.nan
            df_epochs["acc_y_mean"] = np.nan
            df_epochs["acc_z_mean"] = np.nan

        df_epochs["motion_count"] = df_epochs["motion_count"].fillna(0).astype(int)

        # 4. Align and aggregate Blood Pressure (if present)
        if recording.has_bp and recording.bp is not None:
            bp_aligned = filter_and_align_signal(
                recording.bp,
                rec_start_unix,
                total_epochs=total_epochs,
                epoch_duration_sec=self.epoch_duration_sec,
                drop_outside=True,
            )
            bp_grp = bp_aligned.groupby("epoch")
            bp_agg = pd.DataFrame(
                {
                    "bp_count": bp_grp["systolic"].count(),
                    "sbp_mean": bp_grp["systolic"].mean(),
                    "sbp_std": bp_grp["systolic"].std(),
                    "dbp_mean": bp_grp["diastolic"].mean(),
                    "dbp_std": bp_grp["diastolic"].std(),
                    "map_mean": bp_grp["map"].mean() if "map" in bp_aligned.columns else np.nan,
                    "pulse_pressure_mean": (
                        bp_grp["pulse_pressure"].mean()
                        if "pulse_pressure" in bp_aligned.columns
                        else np.nan
                    ),
                }
            )
            df_epochs = df_epochs.join(bp_agg)
            df_epochs["bp_count"] = df_epochs["bp_count"].fillna(0).astype(int)

        # 5. Align and aggregate SpO2 (if present)
        if recording.has_spo2 and recording.spo2 is not None:
            spo2_aligned = filter_and_align_signal(
                recording.spo2,
                rec_start_unix,
                total_epochs=total_epochs,
                epoch_duration_sec=self.epoch_duration_sec,
                drop_outside=True,
            )
            spo2_aligned["low_spo2"] = (spo2_aligned["spo2"] < 95.0).astype(int)
            spo2_grp = spo2_aligned.groupby("epoch")
            spo2_agg = pd.DataFrame(
                {
                    "spo2_count": spo2_grp["spo2"].count(),
                    "spo2_mean": spo2_grp["spo2"].mean(),
                    "spo2_std": spo2_grp["spo2"].std(),
                    "spo2_min": spo2_grp["spo2"].min(),
                    "spo2_sec_below_95": spo2_grp["low_spo2"].sum(),
                }
            )
            df_epochs = df_epochs.join(spo2_agg)
            df_epochs["spo2_count"] = df_epochs["spo2_count"].fillna(0).astype(int)
            df_epochs["spo2_sec_below_95"] = df_epochs["spo2_sec_below_95"].fillna(0).astype(int)

        # 6. Add ground truth sleep stages and class mappings
        if recording.has_labels and recording.labels is not None:
            labels = recording.labels[:total_epochs]
            df_epochs["sleep_stage"] = labels
            df_epochs["sleep_stage_name"] = [SleepStage.to_name(st) for st in labels]
            df_epochs["sleep_binary"] = [SleepStage.to_binary(st) for st in labels]
            df_epochs["sleep_3class"] = [SleepStage.to_3class(st) for st in labels]
        else:
            df_epochs["sleep_stage"] = SleepStage.UNLABELLED
            df_epochs["sleep_stage_name"] = "Unlabelled"
            df_epochs["sleep_binary"] = None
            df_epochs["sleep_3class"] = None

        if recording.dreem_labels is not None and len(recording.dreem_labels) >= total_epochs:
            df_epochs["dreem_stage"] = recording.dreem_labels[:total_epochs]

        # 7. Evaluate per-epoch signal quality flags
        df_epochs["has_hr"] = df_epochs["hr_count"] > 0
        df_epochs["has_motion"] = df_epochs["motion_count"] > 0

        quality_flags = []
        for _, row in df_epochs.iterrows():
            has_h = bool(row["has_hr"])
            has_m = bool(row["has_motion"])
            h_count = int(row["hr_count"])

            if not has_h and not has_m:
                quality_flags.append(SignalQuality.EMPTY.value)
            elif not has_h:
                quality_flags.append(SignalQuality.MISSING_HR.value)
            elif recording.has_motion and not has_m:
                quality_flags.append(SignalQuality.MISSING_MOTION.value)
            elif h_count < 2:  # Sparse HR within 30s epoch
                quality_flags.append(SignalQuality.DEGRADED.value)
            else:
                quality_flags.append(SignalQuality.GOOD.value)

        df_epochs["quality_flag"] = quality_flags

        # Reset index to make 'epoch' a regular column
        final_df = df_epochs.reset_index()

        # 8. Optional raw signal slicing
        raw_slices: dict[int, dict[str, pd.DataFrame]] | None = None
        if preserve_slices:
            raw_slices = {}
            hr_grp_slices = (
                hr_aligned.groupby("epoch")
                if (recording.has_hr and recording.hr is not None)
                else None
            )
            mot_grp_slices = (
                motion_aligned.groupby("epoch")
                if (recording.has_motion and recording.motion is not None)
                else None
            )

            for ep in epoch_indices:
                raw_slices[ep] = {
                    "hr": (
                        hr_grp_slices.get_group(ep).copy()
                        if (hr_grp_slices is not None and ep in hr_grp_slices.groups)
                        else pd.DataFrame()
                    ),
                    "motion": (
                        mot_grp_slices.get_group(ep).copy()
                        if (mot_grp_slices is not None and ep in mot_grp_slices.groups)
                        else pd.DataFrame()
                    ),
                }

        return EpochedNight(
            subject_id=recording.subject_id,
            night_id=recording.night_id,
            rec_start=rec_start,
            epoch_duration_sec=self.epoch_duration_sec,
            epochs_df=final_df,
            raw_slices=raw_slices,
            metadata=metadata,
        )


def align_night_to_epochs(
    recording: NightRecording,
    epoch_duration_sec: float = 30.0,
    preserve_slices: bool = False,
) -> EpochedNight:
    """Convenience function to align a NightRecording into 30-second epochs.

    Args:
        recording: Loaded NightRecording instance.
        epoch_duration_sec: Duration in seconds for each epoch (default: 30.0).
        preserve_slices: Whether to preserve raw dataframe slices per epoch.

    Returns:
        EpochedNight instance.
    """
    aligner = EpochAligner(epoch_duration_sec=epoch_duration_sec)
    return aligner.align_night(recording, preserve_slices=preserve_slices)
