"""Weather-to-lap joining utilities."""

from __future__ import annotations

import pandas as pd

WEATHER_COLUMNS = ["AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"]


def _seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def join_weather_to_laps(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join nearest weather samples to laps, falling back to summary values."""

    result = laps.copy()
    if result.empty:
        for column in WEATHER_COLUMNS:
            if column not in result.columns:
                result[column] = pd.NA
        return result

    if weather is None or weather.empty:
        for column in WEATHER_COLUMNS:
            if column not in result.columns:
                result[column] = pd.NA
        return result

    weather_frame = weather.copy()
    for column in WEATHER_COLUMNS:
        if column not in weather_frame.columns:
            weather_frame[column] = pd.NA

    lap_time_col = "LapStartTime" if "LapStartTime" in result.columns else "Time" if "Time" in result.columns else None
    weather_time_col = (
        "Time" if "Time" in weather_frame.columns else "SessionTime" if "SessionTime" in weather_frame.columns else None
    )

    if lap_time_col and weather_time_col:
        left = result.copy()
        right = weather_frame.copy()
        left["_source_index"] = left.index
        left["_join_seconds"] = _seconds(left[lap_time_col])
        right["_join_seconds"] = _seconds(right[weather_time_col])
        left_valid = left.dropna(subset=["_join_seconds"]).sort_values("_join_seconds")
        right_valid = right.dropna(subset=["_join_seconds"]).sort_values("_join_seconds")
        if not left_valid.empty and not right_valid.empty:
            joined = pd.merge_asof(
                left_valid,
                right_valid[["_join_seconds", *WEATHER_COLUMNS]],
                on="_join_seconds",
                direction="nearest",
            )
            for column in WEATHER_COLUMNS:
                result.loc[joined["_source_index"], column] = joined[column].to_numpy()
            return result

    summary = weather_frame[WEATHER_COLUMNS].mode(dropna=True)
    numeric_summary = weather_frame[WEATHER_COLUMNS].apply(pd.to_numeric, errors="ignore")
    means = numeric_summary.mean(numeric_only=True)
    for column in WEATHER_COLUMNS:
        if column in means:
            result[column] = means[column]
        elif not summary.empty and column in summary:
            result[column] = summary[column].iloc[0]
        else:
            result[column] = pd.NA
    return result
