"""LangGraph state definitions."""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from f1_intelligence_agent.agents.report_schemas import (
    InsightCandidate,
    PlotInterpretation,
    ReportResult,
    SessionRegimeFinding,
    VisualAnalysisResult,
)
from f1_intelligence_agent.data.schemas import AnalysisRunConfig, SessionBundle


class F1AnalysisState(TypedDict, total=False):
    """Shared mutable workflow state passed between LangGraph nodes."""

    run_config: dict[str, Any]
    step_log: list[str]
    current_step: str
    session_bundle: SessionBundle
    data_profile: dict
    lap_features: pd.DataFrame
    model_matrix: Any
    model_metadata: pd.DataFrame
    preprocessor: Any
    clustering_result: dict
    anomaly_result: dict
    pca_projection: pd.DataFrame
    anomaly_table: pd.DataFrame
    feature_explanations: pd.DataFrame
    session_regime_findings: list[SessionRegimeFinding]
    model_diagnostics: dict
    model_row_profile: pd.DataFrame
    telemetry_results: dict
    insight_candidates: list[InsightCandidate]
    lap_plot_interpretations: list[PlotInterpretation]
    model_plot_interpretations: list[PlotInterpretation]
    telemetry_plot_interpretations: list[PlotInterpretation]
    plot_interpretations: list[PlotInterpretation]
    visual_analysis_result: VisualAnalysisResult
    retrieved_context: list[dict]
    web_context: list[dict]
    report_result: ReportResult
    memory_proposals: list[dict]
    errors: list[str]


def normalize_config(config: AnalysisRunConfig | dict) -> AnalysisRunConfig:
    """Normalize a dict or model into AnalysisRunConfig."""

    if isinstance(config, AnalysisRunConfig):
        return config
    return AnalysisRunConfig.model_validate(config)
