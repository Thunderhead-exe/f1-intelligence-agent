"""Time conversion helpers for FastF1 and pandas objects."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_seconds(value: Any) -> float | None:
    """Convert timedeltas, timestamps, or numeric values to seconds when possible."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if isinstance(value, np.timedelta64):
        return float(pd.to_timedelta(value).total_seconds())
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    if isinstance(value, str):
        parsed = pd.to_timedelta(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return float(parsed.total_seconds())
    if isinstance(value, int | float | np.integer | np.floating):
        if np.isnan(value):
            return None
        return float(value)
    return None


def series_to_seconds(series: pd.Series) -> pd.Series:
    """Convert a pandas Series containing timedelta-like values to float seconds."""

    if series.empty:
        return pd.Series(dtype="float64", index=series.index)
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    converted = pd.to_timedelta(series, errors="coerce")
    if converted.notna().any():
        return converted.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return pd.Timestamp.utcnow().isoformat()

