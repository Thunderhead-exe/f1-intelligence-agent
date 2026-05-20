"""Cluster and PCA visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_pca_cluster_plot(
    pca_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cluster_labels: np.ndarray | list[int],
    anomaly_scores: np.ndarray | list[float],
    anomaly_table: pd.DataFrame | None = None,
) -> go.Figure:
    """Create a PCA scatter plot colored by DBSCAN cluster."""

    if pca_df is None or pca_df.empty:
        return _empty_figure("PCA Cluster Projection")
    frame = pca_df.reset_index(drop=True).copy()
    metadata = metadata_df.reset_index(drop=True).copy() if metadata_df is not None else pd.DataFrame()
    for column in [
        "Driver",
        "LapNumber",
        "Compound",
        "FindingCategory",
        "WeatherRegime",
        "TrackStatus",
    ]:
        frame[column] = metadata[column] if column in metadata else ""
    if anomaly_table is not None and not anomaly_table.empty and {"Driver", "LapNumber"}.issubset(frame):
        context_columns = [
            "Driver",
            "LapNumber",
            "FindingCategory",
            "WeatherRegime",
            "TrackStatus",
            "TrackStatusHasNonGreen",
        ]
        context = anomaly_table[[column for column in context_columns if column in anomaly_table]].copy()
        frame = frame.merge(context, on=["Driver", "LapNumber"], how="left", suffixes=("", "Context"))
        for column in ["FindingCategory", "WeatherRegime", "TrackStatus"]:
            context_column = f"{column}Context"
            if context_column in frame:
                frame[column] = frame[context_column].fillna(frame[column])
    frame["Cluster"] = list(cluster_labels) if len(cluster_labels) == len(frame) else 0
    frame["ClusterLabel"] = frame["Cluster"].astype(str)
    frame["AnomalyScore"] = (
        list(anomaly_scores) if len(anomaly_scores) == len(frame) else np.zeros(len(frame))
    )
    score = pd.to_numeric(frame["AnomalyScore"], errors="coerce").fillna(0.0)
    shifted = score - score.min()
    frame["MarkerSize"] = 8 + (shifted / shifted.max() * 14 if shifted.max() > 0 else 0)
    frame["FindingCategory"] = frame["FindingCategory"].replace("", "driver_lap_anomaly").fillna("driver_lap_anomaly")
    fig = px.scatter(
        frame,
        x="PC1",
        y="PC2",
        color="ClusterLabel",
        symbol="FindingCategory",
        size="MarkerSize",
        hover_data=[
            column
            for column in [
                "Driver",
                "LapNumber",
                "Compound",
                "WeatherRegime",
                "TrackStatus",
                "FindingCategory",
                "AnomalyScore",
            ]
            if column in frame
        ],
        title="PCA Projection by Cluster and Finding Category",
    )
    fig.update_traces(marker={"opacity": 0.82, "line": {"width": 0.6, "color": "white"}})
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Cluster / finding",
        xaxis_title="PC1",
        yaxis_title="PC2",
    )
    return fig


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white")
    fig.add_annotation(text="No data available", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    return fig
