"""Pydantic schemas for evidence, insights, and reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    evidence_type: Literal[
        "lap_feature",
        "telemetry",
        "cluster",
        "weather",
        "race_control",
        "retrieved_context",
        "memory",
        "web",
        "visual",
    ]
    description: str
    value: str | float | int | None = None
    source: str | None = None


class InsightCandidate(BaseModel):
    id: str
    finding_category: Literal["driver_lap_anomaly", "session_regime_event"] = "driver_lap_anomaly"
    title: str
    driver: str | None
    lap_number: int | None
    cluster_id: int | str | None
    anomaly_score: float | None
    severity: Literal["low", "medium", "high"]
    confidence: Literal["low", "medium", "high"]
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    possible_explanations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)
    reference_population: str | None = None
    exclusion_reason: str | None = None
    data_quality_notes: list[str] = Field(default_factory=list)
    memory_candidate: bool = True


class SessionRegimeFinding(BaseModel):
    id: str
    finding_category: Literal["session_regime_event"] = "session_regime_event"
    title: str
    regime_type: Literal[
        "rain",
        "track_status",
        "race_start",
        "restart",
        "compound_transition",
        "session_phase",
        "mixed",
    ]
    affected_laps: list[int] = Field(default_factory=list)
    affected_drivers: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"
    confidence: Literal["low", "medium", "high"] = "medium"
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)


class PlotDescriptor(BaseModel):
    plot_id: str
    title: str
    context: str
    what_to_look_for: str
    caveats: list[str] = Field(default_factory=list)


class PlotInterpretation(BaseModel):
    plot_id: str
    title: str
    confidence: Literal["low", "medium", "high"]
    observations: list[str] = Field(default_factory=list)
    caution_notes: list[str] = Field(default_factory=list)
    report_summary: str
    source_plot_ids: list[str] = Field(default_factory=list)


class VisualAnalysisResult(BaseModel):
    confidence: Literal["low", "medium", "high"]
    summaries: list[str] = Field(default_factory=list)
    caution_notes: list[str] = Field(default_factory=list)
    source_plot_ids: list[str] = Field(default_factory=list)


class ReportResult(BaseModel):
    executive_summary: str
    top_findings: list[InsightCandidate] = Field(default_factory=list)
    session_regime_findings: list[SessionRegimeFinding] = Field(default_factory=list)
    cluster_summary: str
    telemetry_summary: str
    context_used: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    visual_analysis: VisualAnalysisResult | None = None
    markdown_report: str
