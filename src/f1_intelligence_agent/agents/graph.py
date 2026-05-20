"""LangGraph workflow construction and streaming wrapper."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, StateGraph

from f1_intelligence_agent.agents.nodes import (
    context_retrieval_node,
    feature_engineering_node,
    insight_interpretation_node,
    lap_visual_interpreter_node,
    load_data_node,
    memory_proposal_node,
    ml_analysis_node,
    model_visual_interpreter_node,
    profile_data_node,
    report_generation_node,
    should_fetch_web_context,
    telemetry_analysis_node,
    telemetry_visual_interpreter_node,
    visual_context_join_node,
    web_context_node,
)
from f1_intelligence_agent.agents.state import AnalysisRunConfig, F1AnalysisState


def build_analysis_graph():
    """Build the deterministic LangGraph analysis workflow."""

    graph = StateGraph(F1AnalysisState)
    graph.add_node("load_data_node", load_data_node)
    graph.add_node("profile_data_node", profile_data_node)
    graph.add_node("feature_engineering_node", feature_engineering_node)
    graph.add_node("ml_analysis_node", ml_analysis_node)
    graph.add_node("telemetry_analysis_node", telemetry_analysis_node)
    graph.add_node("lap_visual_interpreter_node", lap_visual_interpreter_node)
    graph.add_node("model_visual_interpreter_node", model_visual_interpreter_node)
    graph.add_node("telemetry_visual_interpreter_node", telemetry_visual_interpreter_node)
    graph.add_node("insight_interpretation_node", insight_interpretation_node)
    graph.add_node("visual_context_join_node", visual_context_join_node)
    graph.add_node("context_retrieval_node", context_retrieval_node)
    graph.add_node("web_context_node", web_context_node)
    graph.add_node("report_generation_node", report_generation_node)
    graph.add_node("memory_proposal_node", memory_proposal_node)

    graph.set_entry_point("load_data_node")
    graph.add_edge("load_data_node", "profile_data_node")
    graph.add_edge("profile_data_node", "feature_engineering_node")
    graph.add_edge("feature_engineering_node", "ml_analysis_node")
    graph.add_edge("ml_analysis_node", "lap_visual_interpreter_node")
    graph.add_edge("ml_analysis_node", "model_visual_interpreter_node")
    graph.add_edge("ml_analysis_node", "telemetry_analysis_node")
    graph.add_edge("ml_analysis_node", "insight_interpretation_node")
    graph.add_edge("telemetry_analysis_node", "telemetry_visual_interpreter_node")
    graph.add_edge(
        [
            "insight_interpretation_node",
            "lap_visual_interpreter_node",
            "model_visual_interpreter_node",
            "telemetry_visual_interpreter_node",
        ],
        "visual_context_join_node",
    )
    graph.add_edge("visual_context_join_node", "context_retrieval_node")
    graph.add_conditional_edges(
        "context_retrieval_node",
        should_fetch_web_context,
        {"web": "web_context_node", "skip": "report_generation_node"},
    )
    graph.add_edge("web_context_node", "report_generation_node")
    graph.add_edge("report_generation_node", "memory_proposal_node")
    graph.add_edge("memory_proposal_node", END)
    return graph.compile()


def run_analysis(config: AnalysisRunConfig | dict[str, Any]) -> F1AnalysisState:
    """Run the full workflow and return final state."""

    run_config = (
        config.model_dump(mode="python") if isinstance(config, AnalysisRunConfig) else dict(config)
    )
    graph = build_analysis_graph()
    return graph.invoke({"run_config": run_config, "step_log": [], "errors": []})


def run_analysis_stream(config: AnalysisRunConfig | dict[str, Any]) -> Iterator[F1AnalysisState]:
    """Yield state updates after each LangGraph node for Gradio progress display."""

    run_config = (
        config.model_dump(mode="python") if isinstance(config, AnalysisRunConfig) else dict(config)
    )
    graph = build_analysis_graph()
    latest: F1AnalysisState = {"run_config": run_config, "step_log": [], "errors": []}
    for event in graph.stream(latest, stream_mode="values"):
        latest = event
        yield latest
