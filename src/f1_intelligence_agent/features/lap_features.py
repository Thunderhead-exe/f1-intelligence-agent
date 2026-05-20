"""Lap-level feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_intelligence_agent.data.schemas import SessionBundle
from f1_intelligence_agent.features.weather_join import join_weather_to_laps
from f1_intelligence_agent.utils.dataframe import ensure_columns
from f1_intelligence_agent.utils.time import series_to_seconds

BASE_COLUMNS = [
    "Driver",
    "Team",
    "LapNumber",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Stint",
    "TrackStatus",
    "Position",
    "PitInTime",
    "PitOutTime",
    "Deleted",
    "IsAccurate",
    "LapStartTime",
    "Time",
]

FEATURE_COLUMNS = [
    "Driver",
    "Team",
    "LapNumber",
    "LapTimeSeconds",
    "Sector1Seconds",
    "Sector2Seconds",
    "Sector3Seconds",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Stint",
    "TrackStatus",
    "Position",
    "IsPitInLap",
    "IsPitOutLap",
    "IsDeleted",
    "IsAccurate",
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Rainfall",
    "LapTimeDeltaToDriverMedian",
    "LapTimeDeltaToSessionMedian",
    "Sector1DeltaToDriverMedian",
    "Sector2DeltaToDriverMedian",
    "Sector3DeltaToDriverMedian",
    "TyreLifeNormalized",
    "IsWetCompound",
    "IsDryCompound",
    "HasRainfall",
    "IsLikelyPushLap",
    "IsLikelyCooldownLap",
    "DriverLapIndex",
    "SessionLapPct",
    "SessionPhase",
    "IsEarlySessionLap",
    "IsRaceStartLap",
    "IsTrackStatusChangeLap",
    "IsRestartOrStatusChangeLap",
    "TrackStatusIsGreen",
    "TrackStatusHasYellow",
    "TrackStatusHasSafetyCar",
    "TrackStatusHasRedFlag",
    "TrackStatusHasVSC",
    "TrackStatusHasNonGreen",
    "WeatherRegime",
    "CompoundRegime",
    "StintCompoundKey",
    "MissingSpeedTrapCount",
    "MissingSectorTimeCount",
    "HasMissingWeather",
    "HasMissingSpeedTrap",
    "HasMissingSectorTime",
    "LapTimeDeltaToDriverRegimeMedian",
    "Sector1DeltaToDriverRegimeMedian",
    "Sector2DeltaToDriverRegimeMedian",
    "Sector3DeltaToDriverRegimeMedian",
]


def _boolean_flag(series: pd.Series) -> pd.Series:
    return series.map(lambda value: bool(value) if pd.notna(value) else False).astype(bool)


def _delta_to_driver_median(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns or "Driver" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[column] - df.groupby("Driver")[column].transform("median")


def _status_codes(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text or text.upper() == "UNKNOWN":
        return set()
    return {char for char in text if char.isdigit()}


def _track_status_flag(series: pd.Series, codes: set[str]) -> pd.Series:
    return series.map(lambda value: bool(_status_codes(value) & codes)).astype(bool)


def _session_phase(lap_pct: pd.Series) -> pd.Series:
    phase = pd.Series("middle", index=lap_pct.index, dtype="string")
    phase = phase.mask(lap_pct <= 0.25, "early")
    phase = phase.mask(lap_pct >= 0.75, "late")
    phase = phase.mask(lap_pct.isna(), "unknown")
    return phase


def _delta_to_regime_median(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns or "Driver" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    group_columns = [
        column_name
        for column_name in ["Driver", "WeatherRegime", "CompoundRegime", "TrackStatusHasNonGreen"]
        if column_name in df.columns
    ]
    if len(group_columns) <= 1:
        return _delta_to_driver_median(df, column)
    group_sizes = df.groupby(group_columns, dropna=False)[column].transform("count")
    regime_median = df.groupby(group_columns, dropna=False)[column].transform("median")
    driver_median = df.groupby("Driver")[column].transform("median")
    baseline = regime_median.where(group_sizes >= 3, driver_median)
    return df[column] - baseline


def build_lap_feature_table(session_bundle: SessionBundle) -> pd.DataFrame:
    """Create one feature row per driver-lap while preserving interpretability flags."""

    if session_bundle.laps is None or session_bundle.laps.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    laps = ensure_columns(session_bundle.laps, BASE_COLUMNS)
    laps = join_weather_to_laps(laps, session_bundle.weather)
    features = pd.DataFrame(index=laps.index)

    for column in ["Driver", "Team", "Compound", "TrackStatus"]:
        features[column] = laps[column].astype("string")

    for column in ["LapNumber", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "TyreLife", "Stint", "Position"]:
        features[column] = pd.to_numeric(laps[column], errors="coerce")

    features["LapTimeSeconds"] = series_to_seconds(laps["LapTime"])
    features["Sector1Seconds"] = series_to_seconds(laps["Sector1Time"])
    features["Sector2Seconds"] = series_to_seconds(laps["Sector2Time"])
    features["Sector3Seconds"] = series_to_seconds(laps["Sector3Time"])

    features["FreshTyre"] = _boolean_flag(laps["FreshTyre"])
    features["IsPitInLap"] = laps["PitInTime"].notna()
    features["IsPitOutLap"] = laps["PitOutTime"].notna()
    features["IsDeleted"] = _boolean_flag(laps["Deleted"])
    features["IsAccurate"] = laps["IsAccurate"].fillna(True).astype(bool)

    for column in ["AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed"]:
        features[column] = pd.to_numeric(laps.get(column), errors="coerce")
    features["Rainfall"] = _boolean_flag(laps.get("Rainfall", pd.Series(False, index=laps.index)))

    features["LapTimeDeltaToDriverMedian"] = _delta_to_driver_median(features, "LapTimeSeconds")
    features["LapTimeDeltaToSessionMedian"] = (
        features["LapTimeSeconds"] - features["LapTimeSeconds"].median(skipna=True)
    )
    features["Sector1DeltaToDriverMedian"] = _delta_to_driver_median(features, "Sector1Seconds")
    features["Sector2DeltaToDriverMedian"] = _delta_to_driver_median(features, "Sector2Seconds")
    features["Sector3DeltaToDriverMedian"] = _delta_to_driver_median(features, "Sector3Seconds")

    max_tyre_life = pd.to_numeric(features["TyreLife"], errors="coerce").max(skipna=True)
    features["TyreLifeNormalized"] = (
        features["TyreLife"] / max_tyre_life if pd.notna(max_tyre_life) and max_tyre_life else np.nan
    )

    compound = features["Compound"].fillna("").str.upper()
    features["IsWetCompound"] = compound.isin(["INTERMEDIATE", "WET"])
    features["IsDryCompound"] = compound.isin(["SOFT", "MEDIUM", "HARD", "C1", "C2", "C3", "C4", "C5"])
    features["HasRainfall"] = features["Rainfall"].astype(bool)

    max_lap = pd.to_numeric(features["LapNumber"], errors="coerce").max(skipna=True)
    features["SessionLapPct"] = (
        features["LapNumber"] / max_lap if pd.notna(max_lap) and max_lap else np.nan
    )
    features["SessionPhase"] = _session_phase(features["SessionLapPct"])
    features["DriverLapIndex"] = (
        features.sort_values(["Driver", "LapNumber"])
        .groupby("Driver", dropna=False)
        .cumcount()
        .add(1)
        .reindex(features.index)
        .astype(float)
    )
    session_type = str(session_bundle.session_type or "").upper()
    features["IsEarlySessionLap"] = features["DriverLapIndex"] <= 2
    features["IsRaceStartLap"] = (session_type == "R") & (features["LapNumber"] <= 4)

    status = features["TrackStatus"].fillna("UNKNOWN").astype(str)
    features["TrackStatusIsGreen"] = status.map(
        lambda value: bool(_status_codes(value)) and _status_codes(value) <= {"1"}
    ).astype(bool)
    features["TrackStatusHasYellow"] = _track_status_flag(status, {"2"})
    features["TrackStatusHasSafetyCar"] = _track_status_flag(status, {"4"})
    features["TrackStatusHasRedFlag"] = _track_status_flag(status, {"5"})
    features["TrackStatusHasVSC"] = _track_status_flag(status, {"6", "7"})
    features["TrackStatusHasNonGreen"] = (
        features["TrackStatusHasYellow"]
        | features["TrackStatusHasSafetyCar"]
        | features["TrackStatusHasRedFlag"]
        | features["TrackStatusHasVSC"]
    )

    lap_status = (
        features.groupby("LapNumber", dropna=False)["TrackStatus"]
        .agg(lambda values: "|".join(sorted(set(values.dropna().astype(str)))))
        .sort_index()
    )
    changed_laps = lap_status[lap_status.ne(lap_status.shift())].index
    features["IsTrackStatusChangeLap"] = features["LapNumber"].isin(changed_laps)
    features["IsRestartOrStatusChangeLap"] = (
        features["IsTrackStatusChangeLap"] & features["TrackStatusHasNonGreen"]
    )

    features["WeatherRegime"] = np.where(
        features["HasRainfall"] | features["IsWetCompound"], "wet", "dry"
    )
    features["CompoundRegime"] = np.select(
        [features["IsWetCompound"], features["IsDryCompound"]],
        ["wet_compound", "dry_compound"],
        default="unknown_compound",
    )
    features["StintCompoundKey"] = (
        features["Stint"].astype("string").fillna("unknown")
        + ":"
        + features["Compound"].astype("string").fillna("UNKNOWN")
    )

    speed_columns = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
    sector_columns = ["Sector1Seconds", "Sector2Seconds", "Sector3Seconds"]
    weather_columns = ["AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed"]
    features["MissingSpeedTrapCount"] = features[speed_columns].isna().sum(axis=1)
    features["MissingSectorTimeCount"] = features[sector_columns].isna().sum(axis=1)
    features["HasMissingWeather"] = features[weather_columns].isna().any(axis=1)
    features["HasMissingSpeedTrap"] = features["MissingSpeedTrapCount"] > 0
    features["HasMissingSectorTime"] = features["MissingSectorTimeCount"] > 0

    features["LapTimeDeltaToDriverRegimeMedian"] = _delta_to_regime_median(
        features, "LapTimeSeconds"
    )
    features["Sector1DeltaToDriverRegimeMedian"] = _delta_to_regime_median(
        features, "Sector1Seconds"
    )
    features["Sector2DeltaToDriverRegimeMedian"] = _delta_to_regime_median(
        features, "Sector2Seconds"
    )
    features["Sector3DeltaToDriverRegimeMedian"] = _delta_to_regime_median(
        features, "Sector3Seconds"
    )

    driver_rank = features.groupby("Driver")["LapTimeSeconds"].rank(pct=True, method="average")
    features["IsLikelyPushLap"] = (
        (driver_rank <= 0.25)
        & ~features["IsPitInLap"]
        & ~features["IsPitOutLap"]
        & ~features["IsDeleted"]
        & features["IsAccurate"]
    )
    features["IsLikelyCooldownLap"] = (
        (driver_rank >= 0.75) | features["IsPitInLap"] | features["IsPitOutLap"]
    )

    features = ensure_columns(features, FEATURE_COLUMNS)
    return features[FEATURE_COLUMNS].reset_index(drop=True)
