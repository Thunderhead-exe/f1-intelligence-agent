"""Telemetry segment feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_COLUMNS = [
    "Driver",
    "LapNumber",
    "SegmentId",
    "DistanceStart",
    "DistanceEnd",
    "MeanSpeed",
    "MinSpeed",
    "MaxSpeed",
    "MeanThrottle",
    "FullThrottlePct",
    "BrakePct",
    "MeanRPM",
    "MaxRPM",
    "MedianGear",
    "DRSActivePct",
]


def _brake_active(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric > 0


def _drs_active(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric > 0


def build_telemetry_segment_features(
    telemetry: pd.DataFrame,
    n_segments: int = 50,
) -> pd.DataFrame:
    """Split telemetry into distance bins and summarize driving inputs."""

    if telemetry is None or telemetry.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    frame = telemetry.copy()
    n_segments = max(1, int(n_segments))
    if "Distance" not in frame.columns or frame["Distance"].isna().all():
        frame["Distance"] = np.linspace(0, 1, len(frame))
    frame["Distance"] = pd.to_numeric(frame["Distance"], errors="coerce")
    frame = frame.dropna(subset=["Distance"]).sort_values("Distance")
    if frame.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    min_distance = float(frame["Distance"].min())
    max_distance = float(frame["Distance"].max())
    if np.isclose(min_distance, max_distance):
        frame["_segment"] = 0
    else:
        bins = np.linspace(min_distance, max_distance, n_segments + 1)
        frame["_segment"] = pd.cut(
            frame["Distance"],
            bins=bins,
            include_lowest=True,
            labels=False,
            duplicates="drop",
        ).fillna(0).astype(int)

    rows: list[dict[str, object]] = []
    for segment_id, group in frame.groupby("_segment", sort=True):
        speed = pd.to_numeric(group.get("Speed"), errors="coerce")
        throttle = pd.to_numeric(group.get("Throttle"), errors="coerce")
        rpm = pd.to_numeric(group.get("RPM"), errors="coerce")
        gear = pd.to_numeric(group.get("nGear"), errors="coerce")
        brake = _brake_active(group.get("Brake", pd.Series(False, index=group.index)))
        drs = _drs_active(group.get("DRS", pd.Series(0, index=group.index)))
        rows.append(
            {
                "Driver": str(group.get("Driver", pd.Series([""])).iloc[0]),
                "LapNumber": int(pd.to_numeric(group.get("LapNumber", pd.Series([0])).iloc[0], errors="coerce")),
                "SegmentId": int(segment_id),
                "DistanceStart": float(group["Distance"].min()),
                "DistanceEnd": float(group["Distance"].max()),
                "MeanSpeed": float(speed.mean()) if speed.notna().any() else np.nan,
                "MinSpeed": float(speed.min()) if speed.notna().any() else np.nan,
                "MaxSpeed": float(speed.max()) if speed.notna().any() else np.nan,
                "MeanThrottle": float(throttle.mean()) if throttle.notna().any() else np.nan,
                "FullThrottlePct": float((throttle >= 99).mean() * 100) if throttle.notna().any() else np.nan,
                "BrakePct": float(brake.mean() * 100),
                "MeanRPM": float(rpm.mean()) if rpm.notna().any() else np.nan,
                "MaxRPM": float(rpm.max()) if rpm.notna().any() else np.nan,
                "MedianGear": float(gear.median()) if gear.notna().any() else np.nan,
                "DRSActivePct": float(drs.mean() * 100),
            }
        )
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def compare_telemetry_segments(
    anomaly_segments: pd.DataFrame,
    reference_segments: pd.DataFrame,
) -> pd.DataFrame:
    """Compare anomaly telemetry segments against reference segment medians."""

    if anomaly_segments.empty:
        return pd.DataFrame()
    if reference_segments.empty:
        return anomaly_segments.copy()

    reference = reference_segments.groupby("SegmentId", as_index=False)[
        ["MeanSpeed", "MinSpeed", "BrakePct", "FullThrottlePct", "DRSActivePct"]
    ].median()
    merged = anomaly_segments.merge(reference, on="SegmentId", suffixes=("", "Reference"), how="left")
    merged["DeltaMeanSpeed"] = merged["MeanSpeed"] - merged["MeanSpeedReference"]
    merged["DeltaMinSpeed"] = merged["MinSpeed"] - merged["MinSpeedReference"]
    merged["DeltaBrakePct"] = merged["BrakePct"] - merged["BrakePctReference"]
    merged["DeltaFullThrottlePct"] = merged["FullThrottlePct"] - merged["FullThrottlePctReference"]
    merged["DeltaDRSActivePct"] = merged["DRSActivePct"] - merged["DRSActivePctReference"]
    return merged

