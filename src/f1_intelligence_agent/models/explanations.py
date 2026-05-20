"""Feature-deviation explanations for anomalous laps."""

from __future__ import annotations

import numpy as np
import pandas as pd

EXPLANATION_FEATURES = [
    "LapTimeSeconds",
    "Sector1Seconds",
    "Sector2Seconds",
    "Sector3Seconds",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "TyreLife",
    "AirTemp",
    "TrackTemp",
    "WindSpeed",
    "LapTimeDeltaToDriverMedian",
    "Sector1DeltaToDriverMedian",
    "Sector2DeltaToDriverMedian",
    "Sector3DeltaToDriverMedian",
]

HINTS = {
    "LapTimeSeconds": ("higher", "lap was slower than baseline", "lap was faster than baseline"),
    "Sector1Seconds": ("higher", "localized time loss in sector 1", "sector 1 was quicker than baseline"),
    "Sector2Seconds": ("higher", "localized time loss in sector 2", "sector 2 was quicker than baseline"),
    "Sector3Seconds": ("higher", "localized time loss in sector 3", "sector 3 was quicker than baseline"),
    "SpeedI1": ("higher", "higher speed at intermediate 1", "lower speed at intermediate 1 than baseline"),
    "SpeedI2": ("higher", "higher speed at intermediate 2", "lower speed at intermediate 2 than baseline"),
    "SpeedFL": ("higher", "higher finish-line speed", "lower finish-line speed than baseline"),
    "SpeedST": ("higher", "higher straight-line speed", "lower straight-line speed than baseline"),
    "TyreLife": ("higher", "older tyre life than comparable laps", "newer tyre set than comparable laps"),
    "WindSpeed": ("higher", "higher wind speed than session baseline", "lower wind speed than session baseline"),
    "LapTimeDeltaToDriverMedian": ("higher", "lap was slower than the driver median", "lap was faster than the driver median"),
    "Sector1DeltaToDriverMedian": ("higher", "sector 1 was slower than the driver median", "sector 1 was faster than the driver median"),
    "Sector2DeltaToDriverMedian": ("higher", "sector 2 was slower than the driver median", "sector 2 was faster than the driver median"),
    "Sector3DeltaToDriverMedian": ("higher", "sector 3 was slower than the driver median", "sector 3 was faster than the driver median"),
}

MIN_SCALE_FLOORS = {
    "LapTimeSeconds": 0.25,
    "Sector1Seconds": 0.08,
    "Sector2Seconds": 0.08,
    "Sector3Seconds": 0.08,
    "SpeedI1": 1.0,
    "SpeedI2": 1.0,
    "SpeedFL": 1.0,
    "SpeedST": 1.0,
    "TyreLife": 1.0,
    "AirTemp": 0.5,
    "TrackTemp": 0.5,
    "WindSpeed": 0.5,
    "LapTimeDeltaToDriverMedian": 0.25,
    "Sector1DeltaToDriverMedian": 0.08,
    "Sector2DeltaToDriverMedian": 0.08,
    "Sector3DeltaToDriverMedian": 0.08,
}

MAX_DISPLAY_Z_SCORE = 25.0


def _robust_z(value: float, series: pd.Series, feature: str) -> tuple[float, float, int]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty or pd.isna(value):
        return np.nan, np.nan, 0
    median = float(numeric.median())
    mad = float(np.median(np.abs(numeric - median)))
    std = float(numeric.std(ddof=0)) if len(numeric) > 1 else 0.0
    scale_floor = MIN_SCALE_FLOORS.get(feature, 1e-9)
    scale = max(mad, std if mad == 0 else 0.0, scale_floor)
    return median, float(0.6745 * (value - median) / scale), int(len(numeric))


def _hint(feature: str, z_score: float) -> str:
    _, positive_hint, negative_hint = HINTS.get(
        feature, ("higher", "feature was higher than baseline", "feature was lower than baseline")
    )
    return positive_hint if z_score >= 0 else negative_hint


def compute_feature_deviation_explanations(
    lap_features: pd.DataFrame,
    anomaly_rows: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Compute robust-z feature deviations for each anomalous lap."""

    if lap_features is None or lap_features.empty or anomaly_rows is None or anomaly_rows.empty:
        return pd.DataFrame(
            columns=[
                "Driver",
                "LapNumber",
                "Feature",
                "Value",
                "Baseline",
                "BaselineMedian",
                "RobustZScore",
                "RawRobustZScore",
                "Direction",
                "PlainEnglishHint",
                "ReferencePopulationSize",
            ]
        )

    rows: list[dict[str, object]] = []
    source_indices = (
        anomaly_rows["SourceIndex"].dropna().astype(int).tolist()
        if "SourceIndex" in anomaly_rows.columns
        else anomaly_rows.index.tolist()
    )

    for source_index in source_indices:
        if source_index not in lap_features.index:
            continue
        lap = lap_features.loc[source_index]
        driver = str(lap.get("Driver", ""))
        baseline = lap_features
        if driver and "Driver" in lap_features.columns:
            driver_baseline = lap_features[lap_features["Driver"].astype(str) == driver]
            if len(driver_baseline) >= 3:
                baseline = driver_baseline

        scored: list[dict[str, object]] = []
        for feature in EXPLANATION_FEATURES:
            if feature not in lap_features.columns:
                continue
            value = pd.to_numeric(pd.Series([lap.get(feature)]), errors="coerce").iloc[0]
            baseline_median, z_score, reference_size = _robust_z(value, baseline[feature], feature)
            if not np.isfinite(z_score):
                continue
            display_z_score = float(np.clip(z_score, -MAX_DISPLAY_Z_SCORE, MAX_DISPLAY_Z_SCORE))
            scored.append(
                {
                    "Driver": driver or None,
                    "LapNumber": int(lap["LapNumber"]) if pd.notna(lap.get("LapNumber")) else None,
                    "Feature": feature,
                    "Value": float(value) if pd.notna(value) else None,
                    "Baseline": "driver" if baseline is not lap_features else "session",
                    "BaselineMedian": baseline_median,
                    "RobustZScore": display_z_score,
                    "RawRobustZScore": z_score,
                    "Direction": "higher" if z_score >= 0 else "lower",
                    "PlainEnglishHint": _hint(feature, z_score),
                    "ReferencePopulationSize": reference_size,
                }
            )

        scored = sorted(scored, key=lambda row: abs(float(row["RawRobustZScore"])), reverse=True)
        rows.extend(scored[:top_k])

    return pd.DataFrame(rows)
