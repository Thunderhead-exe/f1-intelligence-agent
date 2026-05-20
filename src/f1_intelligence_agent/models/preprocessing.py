"""Preprocessing for lap-level unsupervised models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "LapTimeSeconds",
    "Sector1Seconds",
    "Sector2Seconds",
    "Sector3Seconds",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "TyreLife",
    "Stint",
    "Position",
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "LapTimeDeltaToDriverMedian",
    "LapTimeDeltaToSessionMedian",
    "Sector1DeltaToDriverMedian",
    "Sector2DeltaToDriverMedian",
    "Sector3DeltaToDriverMedian",
    "TyreLifeNormalized",
    "DriverLapIndex",
    "SessionLapPct",
    "MissingSpeedTrapCount",
    "MissingSectorTimeCount",
    "LapTimeDeltaToDriverRegimeMedian",
    "Sector1DeltaToDriverRegimeMedian",
    "Sector2DeltaToDriverRegimeMedian",
    "Sector3DeltaToDriverRegimeMedian",
]

CATEGORICAL_FEATURES = [
    "Compound",
    "TrackStatus",
    "SessionPhase",
    "WeatherRegime",
    "CompoundRegime",
    "StintCompoundKey",
]
BOOLEAN_FEATURES = [
    "FreshTyre",
    "Rainfall",
    "HasRainfall",
    "IsWetCompound",
    "IsDryCompound",
    "IsLikelyPushLap",
    "IsLikelyCooldownLap",
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
    "HasMissingWeather",
    "HasMissingSpeedTrap",
    "HasMissingSectorTime",
]

METADATA_COLUMNS = [
    "Driver",
    "Team",
    "LapNumber",
    "Compound",
    "Stint",
    "TrackStatus",
    "IsPitInLap",
    "IsPitOutLap",
    "IsDeleted",
    "IsAccurate",
    "SessionPhase",
    "WeatherRegime",
    "CompoundRegime",
    "TrackStatusHasNonGreen",
    "IsRaceStartLap",
    "IsEarlySessionLap",
    "IsRestartOrStatusChangeLap",
    "HasRainfall",
    "HasMissingWeather",
]


def _valid_model_rows(lap_features: pd.DataFrame) -> pd.Series:
    """Return the default training mask for laps suitable for ML."""

    if lap_features.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=lap_features.index)
    if "LapTimeSeconds" in lap_features:
        mask &= lap_features["LapTimeSeconds"].notna()
    if "IsDeleted" in lap_features:
        mask &= ~lap_features["IsDeleted"].fillna(False).astype(bool)
    if "IsAccurate" in lap_features:
        mask &= lap_features["IsAccurate"].fillna(True).astype(bool)
    if "IsPitInLap" in lap_features:
        mask &= ~lap_features["IsPitInLap"].fillna(False).astype(bool)
    if "IsPitOutLap" in lap_features:
        mask &= ~lap_features["IsPitOutLap"].fillna(False).astype(bool)
    return mask


def _as_bool(value: object, default: bool = False) -> bool:
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        return default
    return bool(value)


def build_model_row_profile(lap_features: pd.DataFrame) -> pd.DataFrame:
    """Return model inclusion/exclusion reasons for each feature row."""

    columns = [
        "SourceIndex",
        "Driver",
        "LapNumber",
        "IsModelRow",
        "ExclusionReason",
    ]
    if lap_features is None or lap_features.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for source_index, row in lap_features.iterrows():
        reasons: list[str] = []
        if pd.isna(row.get("LapTimeSeconds")):
            reasons.append("missing_lap_time")
        if _as_bool(row.get("IsDeleted", False)):
            reasons.append("deleted_lap")
        if not _as_bool(row.get("IsAccurate", True), default=True):
            reasons.append("inaccurate_lap")
        if _as_bool(row.get("IsPitInLap", False)):
            reasons.append("pit_in_lap")
        if _as_bool(row.get("IsPitOutLap", False)):
            reasons.append("pit_out_lap")
        rows.append(
            {
                "SourceIndex": source_index,
                "Driver": row.get("Driver"),
                "LapNumber": row.get("LapNumber"),
                "IsModelRow": not reasons,
                "ExclusionReason": ", ".join(reasons) if reasons else "included",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def get_model_feature_names(preprocessor: Any) -> list[str]:
    """Return transformed feature names from a fitted ColumnTransformer."""

    if preprocessor is None:
        return []
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        return []
    return [str(name) for name in names]


def prepare_lap_model_matrix(
    lap_features: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, Any]:
    """Prepare a model matrix and metadata for valid lap rows."""

    if lap_features is None or lap_features.empty:
        return np.empty((0, 0)), pd.DataFrame(columns=METADATA_COLUMNS), None

    frame = lap_features.copy()
    model_mask = _valid_model_rows(frame)
    model_frame = frame.loc[model_mask].copy()
    if model_frame.empty:
        return np.empty((0, 0)), pd.DataFrame(columns=METADATA_COLUMNS), None

    numeric = [
        column
        for column in NUMERIC_FEATURES
        if column in model_frame.columns and model_frame[column].notna().any()
    ]
    categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column in model_frame.columns and model_frame[column].notna().any()
    ]
    boolean = [column for column in BOOLEAN_FEATURES if column in model_frame.columns]

    for column in numeric:
        model_frame[column] = pd.to_numeric(model_frame[column], errors="coerce").astype(float)
    for column in boolean:
        model_frame[column] = model_frame[column].fillna(False).astype(bool).astype(int)
    for column in categorical:
        model_frame[column] = model_frame[column].astype("string").fillna("UNKNOWN").astype(str)
    for column in categorical:
        model_frame[column] = model_frame[column].astype(object).where(model_frame[column].notna(), np.nan)

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if boolean:
        transformers.append(
            (
                "boolean",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                boolean,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )

    if not transformers:
        metadata = model_frame[[column for column in METADATA_COLUMNS if column in model_frame.columns]].copy()
        return np.empty((len(model_frame), 0)), metadata.reset_index(drop=True), None

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X = preprocessor.fit_transform(model_frame)
    X = np.asarray(X, dtype=float)

    metadata = model_frame[[column for column in METADATA_COLUMNS if column in model_frame.columns]].copy()
    metadata["SourceIndex"] = model_frame.index
    metadata = metadata.reset_index(drop=True)
    return X, metadata, preprocessor
