"""Guided OpenAI report generation from structured evidence."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from f1_intelligence_agent.agents.prompts import REPORT_SYSTEM_PROMPT, REPORT_USER_TEMPLATE
from f1_intelligence_agent.agents.report_schemas import (
    InsightCandidate,
    PlotInterpretation,
    ReportResult,
    SessionRegimeFinding,
    VisualAnalysisResult,
)
from f1_intelligence_agent.config import get_settings
from f1_intelligence_agent.reports.templates import REPORT_LIMITATIONS


def generate_report(
    *,
    session_metadata: dict[str, Any],
    data_profile: dict[str, Any],
    insight_candidates: list[InsightCandidate],
    retrieved_context: list[dict[str, Any]],
    web_context: list[dict[str, Any]],
    clustering_result: dict[str, Any],
    telemetry_results: dict[str, Any],
    session_regime_findings: list[SessionRegimeFinding] | None = None,
    model_diagnostics: dict[str, Any] | None = None,
    plot_interpretations: list[PlotInterpretation] | None = None,
    visual_analysis_result: VisualAnalysisResult | None = None,
) -> ReportResult:
    """Generate the final Markdown report using a guided OpenAI chat model."""

    settings = get_settings()
    api_key = settings.require_openai_api_key()
    payload = {
        "session_metadata": session_metadata,
        "data_profile": data_profile,
        "model_diagnostics": model_diagnostics or {},
        "clustering_result": _compact_dict(clustering_result),
        "driver_lap_anomalies": [insight.model_dump(mode="json") for insight in insight_candidates],
        "session_regime_findings": [
            finding.model_dump(mode="json") for finding in (session_regime_findings or [])
        ],
        "retrieved_context": retrieved_context[:12],
        "web_context": web_context[:3],
        "telemetry_results_summary": _compact_telemetry(telemetry_results),
        "visual_analysis_result": visual_analysis_result.model_dump(mode="json")
        if visual_analysis_result
        else None,
        "plot_interpretations": [
            interpretation.model_dump(mode="json")
            for interpretation in (plot_interpretations or [])
        ],
        "reporting_limitations": REPORT_LIMITATIONS,
    }
    llm = ChatOpenAI(model=settings.openai_report_model, api_key=api_key, temperature=0.2)
    response = llm.invoke(
        [
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(
                content=REPORT_USER_TEMPLATE.format(
                    payload=json.dumps(payload, indent=2, sort_keys=True, default=str)
                )
            ),
        ]
    )
    raw_markdown = str(response.content)
    quality_warnings = report_quality_warnings(raw_markdown)
    markdown = _enforce_reporting_language(raw_markdown)
    return ReportResult(
        executive_summary=_extract_section(markdown, "Executive Summary"),
        top_findings=insight_candidates,
        session_regime_findings=session_regime_findings or [],
        cluster_summary=_extract_section(markdown, "Cluster Summary"),
        telemetry_summary=_extract_section(markdown, "Telemetry Deep Dive"),
        context_used=[
            str(item.get("source") or item.get("metadata", {}).get("source") or item.get("collection", "context"))
            for item in retrieved_context[:12]
        ]
        + [str(item.get("url") or item.get("title") or "web context") for item in web_context[:3]],
        limitations=REPORT_LIMITATIONS,
        quality_warnings=quality_warnings,
        visual_analysis=visual_analysis_result,
        markdown_report=markdown,
    )


def _extract_section(markdown: str, heading: str) -> str:
    marker = f"## {heading}"
    if marker not in markdown:
        return markdown[:500]
    remainder = markdown.split(marker, 1)[1].strip()
    next_heading = remainder.find("\n## ")
    section = remainder if next_heading == -1 else remainder[:next_heading]
    return section.strip()[:1500]


def _compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, item in (value or {}).items():
        if key in {"labels", "is_anomaly"}:
            continue
        compact[key] = item
    return compact


def _compact_telemetry(value: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, item in (value or {}).items():
        if key.endswith("_rows") or key in {"reference_laps_by_lap"}:
            compact[key] = item
        elif key == "telemetry_comparison_by_lap" and isinstance(item, dict):
            compact[key] = {
                sub_key: _compact_frame(sub_value, max_rows=8)
                for sub_key, sub_value in item.items()
            }
        elif isinstance(item, dict):
            compact[key] = {sub_key: str(sub_value)[:500] for sub_key, sub_value in item.items()}
        else:
            compact[key] = str(item)[:500]
    return compact


def _compact_frame(value: Any, max_rows: int = 8) -> list[dict[str, Any]] | str:
    if hasattr(value, "head") and hasattr(value, "to_dict"):
        return value.head(max_rows).to_dict(orient="records")
    return str(value)[:500]


UNSUPPORTED_CAUSAL_PATTERNS: list[tuple[str, str, str]] = [
    (r"\bthis proves\b", "This suggests", "Rewrote unsupported proof language."),
    (r"\bthis proved\b", "This suggested", "Rewrote unsupported proof language."),
    (r"\bproves that\b", "suggests that", "Rewrote unsupported proof language."),
    (r"\bthe driver made a mistake\b", "a possible contributing factor", "Rewrote unsupported driver-error claim."),
    (r"\bdriver made a mistake\b", "possible contributing factor", "Rewrote unsupported driver-error claim."),
    (r"\bmade a mistake\b", "may have encountered a possible contributing factor", "Rewrote unsupported mistake claim."),
    (r"\bwas definitely caused\b", "may have been related to", "Rewrote unsupported causal certainty."),
    (r"\bdefinitely caused\b", "may be a possible contributing factor in", "Rewrote unsupported causal certainty."),
    (r"\bthe exact cause\b", "a possible contributing factor", "Rewrote unsupported exact-cause claim."),
    (r"\bexact cause\b", "possible contributing factor", "Rewrote unsupported exact-cause claim."),
    (r"\bconfirmed cause\b", "possible contributing factor", "Rewrote unsupported confirmed-cause claim."),
]


def report_quality_warnings(markdown: str) -> list[str]:
    """Return guardrail warnings for unsupported causal report language."""

    warnings: list[str] = []
    for pattern, _, warning in UNSUPPORTED_CAUSAL_PATTERNS:
        if re.search(pattern, markdown, flags=re.IGNORECASE):
            warnings.append(warning)
    return sorted(set(warnings))


def _enforce_reporting_language(markdown: str) -> str:
    """Rewrite unsupported causal phrasing in model-generated reports."""

    cleaned = markdown
    for pattern, replacement, _ in UNSUPPORTED_CAUSAL_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned
