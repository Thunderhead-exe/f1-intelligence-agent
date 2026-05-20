"""DataFrame utilities used across profiling, UI, and tests."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def empty_dataframe(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Return an empty DataFrame with stable columns."""

    return pd.DataFrame(columns=list(columns or []))


def safe_copy_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Copy a DataFrame-like value or return an empty frame."""

    if frame is None:
        return pd.DataFrame()
    try:
        return frame.copy()
    except AttributeError:
        return pd.DataFrame(frame)


def ensure_columns(
    frame: pd.DataFrame | None,
    columns: Iterable[str],
    default: object = pd.NA,
) -> pd.DataFrame:
    """Return a copy of `frame` with all requested columns present."""

    result = safe_copy_frame(frame)
    for column in columns:
        if column not in result.columns:
            result[column] = default
    return result


def missing_value_summary(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return a compact missing-value summary."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct"])
    missing = frame.isna().sum()
    return (
        pd.DataFrame(
            {
                "column": missing.index,
                "missing_count": missing.values,
                "missing_pct": (missing.values / max(len(frame), 1) * 100).round(2),
            }
        )
        .sort_values(["missing_pct", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def safe_head(df: pd.DataFrame | None, n: int = 25) -> pd.DataFrame:
    """Return a display-safe head of a DataFrame."""

    return dataframe_preview(df, n=n)


def dataframe_preview(frame: pd.DataFrame | None, n: int = 200) -> pd.DataFrame:
    """Return a UI-safe DataFrame preview."""

    if frame is None:
        return pd.DataFrame()
    preview = frame.head(n).copy()
    for column in preview.columns:
        if pd.api.types.is_timedelta64_dtype(preview[column]):
            preview[column] = preview[column].astype(str)
    return preview
