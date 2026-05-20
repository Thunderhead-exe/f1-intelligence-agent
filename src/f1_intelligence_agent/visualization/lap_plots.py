"""Lap-level Plotly visualizations."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_lap_time_evolution_plot(lap_features: pd.DataFrame) -> go.Figure:
    """Plot lap time evolution by driver."""

    if lap_features is None or lap_features.empty or "LapTimeSeconds" not in lap_features:
        return _empty_figure("Lap Time Evolution")
    frame = lap_features.dropna(subset=["LapNumber", "LapTimeSeconds"]).copy()
    if frame.empty:
        return _empty_figure("Lap Time Evolution")
    frame["LapQuality"] = "clean"
    if "IsPitInLap" in frame:
        frame.loc[frame["IsPitInLap"].fillna(False).astype(bool), "LapQuality"] = "pit-in"
    if "IsPitOutLap" in frame:
        frame.loc[frame["IsPitOutLap"].fillna(False).astype(bool), "LapQuality"] = "pit-out"
    if "IsDeleted" in frame:
        frame.loc[frame["IsDeleted"].fillna(False).astype(bool), "LapQuality"] = "deleted"
    if "IsAccurate" in frame:
        frame.loc[~frame["IsAccurate"].fillna(True).astype(bool), "LapQuality"] = "inaccurate"
    if "TrackStatusHasNonGreen" in frame:
        frame.loc[
            frame["TrackStatusHasNonGreen"].fillna(False).astype(bool)
            & frame["LapQuality"].eq("clean"),
            "LapQuality",
        ] = "non-green"
    fig = px.line(
        frame,
        x="LapNumber",
        y="LapTimeSeconds",
        color="Driver",
        symbol="LapQuality",
        markers=True,
        title="Lap Time Evolution (lower is faster)",
        hover_data=[
            column
            for column in ["Compound", "TyreLife", "TrackStatus", "WeatherRegime", "LapQuality"]
            if column in frame
        ],
    )
    fig.update_layout(template="plotly_white", xaxis_title="Lap", yaxis_title="Lap time (s, lower is faster)")
    return fig


def create_sector_time_plot(
    lap_features: pd.DataFrame,
    selected_drivers: list[str] | None = None,
) -> go.Figure:
    """Create a grouped sector comparison chart."""

    if lap_features is None or lap_features.empty:
        return _empty_figure("Sector Time Comparison")
    frame = lap_features.copy()
    if selected_drivers:
        frame = frame[frame["Driver"].astype(str).isin(selected_drivers)]
    sectors = ["Sector1Seconds", "Sector2Seconds", "Sector3Seconds"]
    available = [column for column in sectors if column in frame]
    if not available:
        return _empty_figure("Sector Time Comparison")
    grouped = frame.groupby("Driver", as_index=False)[available].median()
    melted = grouped.melt(id_vars="Driver", value_vars=available, var_name="Sector", value_name="MedianSeconds")
    melted["DeltaToGroupMedian"] = melted["MedianSeconds"] - melted.groupby("Sector")[
        "MedianSeconds"
    ].transform("median")
    melted["DeltaLabel"] = melted["DeltaToGroupMedian"].map(lambda value: f"{value:+.2f}s")
    fig = px.bar(
        melted,
        x="Driver",
        y="MedianSeconds",
        color="Sector",
        barmode="group",
        title="Median Sector Time by Driver",
        hover_data=["DeltaToGroupMedian"],
        text="DeltaLabel",
    )
    fig.update_layout(template="plotly_white", yaxis_title="Median sector time (s)")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def create_anomaly_score_plot(anomaly_table: pd.DataFrame) -> go.Figure:
    """Create an anomaly ranking bar chart."""

    if anomaly_table is None or anomaly_table.empty or "AnomalyScore" not in anomaly_table:
        return _empty_figure("Anomaly Ranking")
    frame = anomaly_table.copy()
    frame["FindingCategory"] = frame.get("FindingCategory", "driver_lap_anomaly")
    frame["_category_order"] = frame["FindingCategory"].map(
        {"driver_lap_anomaly": 0, "session_regime_event": 1}
    ).fillna(2)
    frame = frame.sort_values(["_category_order", "AnomalyScore"], ascending=[True, False]).head(20)
    frame["DriverLap"] = frame["Driver"].astype(str) + " L" + frame["LapNumber"].astype(str)
    fig = px.bar(
        frame.sort_values("AnomalyScore", ascending=True),
        x="AnomalyScore",
        y="DriverLap",
        color="FindingCategory",
        orientation="h",
        title="Top Lap Anomaly Scores",
        hover_data=[
            column
            for column in ["Cluster", "Compound", "TrackStatus", "WeatherRegime", "IsAnomaly"]
            if column in frame
        ],
    )
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(template="plotly_white", xaxis_title="Anomaly score", yaxis_title="Driver lap")
    return fig


def create_anomaly_timeline_plot(
    lap_features: pd.DataFrame,
    anomaly_table: pd.DataFrame,
) -> go.Figure:
    """Plot anomaly scores over lap number with session-regime category color."""

    if anomaly_table is None or anomaly_table.empty or "AnomalyScore" not in anomaly_table:
        return _empty_figure("Anomaly Timeline")
    frame = anomaly_table.dropna(subset=["LapNumber", "AnomalyScore"]).copy()
    if frame.empty:
        return _empty_figure("Anomaly Timeline")
    frame["FindingCategory"] = frame.get("FindingCategory", "driver_lap_anomaly")
    fig = px.scatter(
        frame,
        x="LapNumber",
        y="AnomalyScore",
        color="FindingCategory",
        symbol="Driver" if "Driver" in frame else None,
        hover_data=[
            column
            for column in ["Driver", "Compound", "TrackStatus", "WeatherRegime", "SessionPhase"]
            if column in frame
        ],
        title="Anomaly Timeline by Finding Category",
    )
    if lap_features is not None and not lap_features.empty and "TrackStatusHasNonGreen" in lap_features:
        non_green_laps = (
            lap_features.loc[lap_features["TrackStatusHasNonGreen"].fillna(False).astype(bool), "LapNumber"]
            .dropna()
            .unique()
        )
        for lap_number in sorted(non_green_laps):
            fig.add_vrect(
                x0=float(lap_number) - 0.5,
                x1=float(lap_number) + 0.5,
                fillcolor="rgba(255, 193, 7, 0.16)",
                line_width=0,
                layer="below",
            )
    if lap_features is not None and not lap_features.empty and "WeatherRegime" in lap_features:
        wet_laps = (
            lap_features.loc[lap_features["WeatherRegime"].astype(str).eq("wet"), "LapNumber"]
            .dropna()
            .unique()
        )
        for lap_number in sorted(wet_laps):
            fig.add_vrect(
                x0=float(lap_number) - 0.5,
                x1=float(lap_number) + 0.5,
                fillcolor="rgba(33, 150, 243, 0.10)",
                line_width=0,
                layer="below",
            )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    if "IsAnomaly" in frame:
        flagged = frame.loc[frame["IsAnomaly"].fillna(False).astype(bool), "AnomalyScore"]
        if not flagged.empty:
            fig.add_hline(
                y=float(flagged.min()),
                line_dash="dash",
                line_color="crimson",
                annotation_text="Flag threshold",
            )
    fig.update_layout(template="plotly_white", xaxis_title="Lap", yaxis_title="Anomaly score")
    return fig


def create_model_row_exclusion_plot(model_row_profile: pd.DataFrame) -> go.Figure:
    """Plot included/excluded model-row counts by reason."""

    if model_row_profile is None or model_row_profile.empty or "ExclusionReason" not in model_row_profile:
        return _empty_figure("Model Row Inclusion")
    frame = model_row_profile.copy()
    counts = frame["ExclusionReason"].fillna("unknown").value_counts().reset_index()
    counts.columns = ["Reason", "Rows"]
    total = counts["Rows"].sum()
    counts["Percent"] = counts["Rows"] / total * 100 if total else 0
    counts["Label"] = counts.apply(lambda row: f"{int(row['Rows'])} ({row['Percent']:.1f}%)", axis=1)
    fig = px.bar(counts, x="Reason", y="Rows", title="Model Row Inclusion and Exclusion", text="Label")
    fig.update_layout(template="plotly_white", xaxis_title="Reason", yaxis_title="Rows")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def create_missingness_plot(lap_features: pd.DataFrame) -> go.Figure:
    """Plot missing values by engineered feature."""

    if lap_features is None or lap_features.empty:
        return _empty_figure("Feature Missingness")
    missing = lap_features.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(30)
    if missing.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["No missing values"], y=[0], marker_color="#6c757d"))
        fig.update_layout(title="Feature Missingness", template="plotly_white")
        fig.add_annotation(
            text="No missing engineered values",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return fig
    frame = missing.reset_index()
    frame.columns = ["Feature", "MissingRows"]
    frame["MissingPercent"] = frame["MissingRows"] / len(lap_features) * 100
    frame["FeatureGroup"] = frame["Feature"].map(_feature_group)
    frame["MissingLabel"] = frame["MissingPercent"].map(lambda value: f"{value:.1f}%")
    fig = px.bar(
        frame,
        x="Feature",
        y="MissingPercent",
        color="FeatureGroup",
        title="Feature Missingness",
        hover_data=["MissingRows"],
        text="MissingLabel",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Feature", yaxis_title="Missing rows (%)")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def _feature_group(feature: str) -> str:
    if feature.startswith("Sector") or feature.startswith("LapTime"):
        return "timing"
    if feature.startswith("Speed"):
        return "speed trap"
    if feature in {"AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"}:
        return "weather"
    if "Tyre" in feature or "Compound" in feature or "Stint" in feature:
        return "tyre/stint"
    if "TrackStatus" in feature:
        return "track status"
    return "other"


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white")
    fig.add_annotation(text="No data available", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    return fig
