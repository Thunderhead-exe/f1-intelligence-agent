"""Telemetry Plotly visualizations."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


def create_speed_distance_plot(telemetry_dict_or_df: dict[str, pd.DataFrame] | pd.DataFrame) -> go.Figure:
    """Plot speed against distance for one or more telemetry traces."""

    fig = go.Figure()
    traces = _as_trace_dict(telemetry_dict_or_df)
    for name, frame in traces.items():
        if frame.empty or not {"Distance", "Speed"}.issubset(frame.columns):
            continue
        fig.add_trace(
            go.Scatter(
                x=frame["Distance"],
                y=frame["Speed"],
                mode="lines",
                name=name,
                line={"width": 2.6},
            )
        )
    fig.update_layout(
        title="Speed vs Distance",
        template="plotly_white",
        xaxis_title="Distance around lap",
        yaxis_title="Speed",
    )
    if not fig.data:
        fig.add_annotation(text="No telemetry available", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    return fig


def create_throttle_brake_distance_plot(
    telemetry_dict_or_df: dict[str, pd.DataFrame] | pd.DataFrame,
) -> go.Figure:
    """Plot throttle and brake usage against distance."""

    fig = go.Figure()
    traces = _as_trace_dict(telemetry_dict_or_df)
    for name, frame in traces.items():
        if frame.empty or "Distance" not in frame.columns:
            continue
        if "Throttle" in frame:
            fig.add_trace(
                go.Scatter(
                    x=frame["Distance"],
                    y=frame["Throttle"],
                    mode="lines",
                    name=f"{name} throttle",
                    line={"width": 2.2},
                )
            )
        if "Brake" in frame:
            brake = pd.to_numeric(frame["Brake"], errors="coerce").fillna(0)
            if brake.max() <= 1:
                brake = brake * 100
            fig.add_trace(
                go.Scatter(
                    x=frame["Distance"],
                    y=brake,
                    mode="lines",
                    name=f"{name} brake",
                    line={"width": 2.2, "dash": "dot"},
                )
            )
    fig.update_layout(
        title="Throttle/Brake vs Distance",
        template="plotly_white",
        xaxis_title="Distance around lap",
        yaxis_title="Input (%)",
    )
    if not fig.data:
        fig.add_annotation(text="No telemetry available", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    return fig


def create_telemetry_delta_plot(telemetry_results: dict[str, Any] | None) -> go.Figure:
    """Plot anomaly telemetry deltas against reference laps by distance segment."""

    fig = go.Figure()
    comparisons = (telemetry_results or {}).get("telemetry_comparison_by_lap", {})
    if not isinstance(comparisons, dict):
        comparisons = {}
    for name, frame in comparisons.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        if {"DistanceStart", "DeltaMeanSpeed"}.issubset(frame.columns):
            fig.add_trace(
                go.Scatter(
                    x=frame["DistanceStart"],
                    y=frame["DeltaMeanSpeed"],
                    mode="lines",
                    name=f"{name} mean speed delta",
                )
            )
        if {"DistanceStart", "DeltaFullThrottlePct"}.issubset(frame.columns):
            fig.add_trace(
                go.Scatter(
                    x=frame["DistanceStart"],
                    y=frame["DeltaFullThrottlePct"],
                    mode="lines",
                    name=f"{name} full-throttle delta",
                    yaxis="y2",
                )
            )
        if {"DistanceStart", "DeltaBrakePct"}.issubset(frame.columns):
            fig.add_trace(
                go.Scatter(
                    x=frame["DistanceStart"],
                    y=frame["DeltaBrakePct"],
                    mode="lines",
                    name=f"{name} brake delta",
                    yaxis="y2",
                )
            )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Telemetry Delta vs Reference Laps",
        template="plotly_white",
        xaxis_title="Distance around lap",
        yaxis_title="Speed delta",
        yaxis2={
            "title": "Brake delta (pp)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
    )
    if not fig.data:
        fig.add_annotation(
            text="No telemetry comparison available",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    return fig


def _as_trace_dict(value: dict[str, Any] | pd.DataFrame) -> dict[str, pd.DataFrame]:
    if isinstance(value, pd.DataFrame):
        return {"telemetry": value}
    if isinstance(value, dict) and isinstance(value.get("telemetry_by_lap"), dict):
        return _as_trace_dict(value.get("telemetry_by_lap", {}))
    result: dict[str, pd.DataFrame] = {}
    for key, frame in (value or {}).items():
        if isinstance(frame, pd.DataFrame):
            result[str(key)] = frame
    return result
