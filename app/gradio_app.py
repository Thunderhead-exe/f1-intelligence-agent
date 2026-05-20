"""Gradio UI for the F1 Intelligence Agent."""

from __future__ import annotations

import json
import numbers
from datetime import UTC, datetime
from typing import Any

import gradio as gr
import pandas as pd

from f1_intelligence_agent.agents.graph import run_analysis_stream
from f1_intelligence_agent.agents.state import AnalysisRunConfig, F1AnalysisState
from f1_intelligence_agent.data.data_dictionary import get_field_dictionary, get_glossary_markdown
from f1_intelligence_agent.logging_utils import configure_logging
from f1_intelligence_agent.rag.memory_store import InsightMemoryStore
from f1_intelligence_agent.utils.dataframe import missing_value_summary, safe_head
from f1_intelligence_agent.visualization.cluster_plots import create_pca_cluster_plot
from f1_intelligence_agent.visualization.descriptors import plot_descriptor_markdown
from f1_intelligence_agent.visualization.lap_plots import (
    create_anomaly_score_plot,
    create_anomaly_timeline_plot,
    create_lap_time_evolution_plot,
    create_missingness_plot,
    create_model_row_exclusion_plot,
    create_sector_time_plot,
)
from f1_intelligence_agent.visualization.telemetry_plots import (
    create_speed_distance_plot,
    create_telemetry_delta_plot,
    create_throttle_brake_distance_plot,
)

LATEST_STATE: F1AnalysisState = {}
MEMORY_DECISIONS: dict[str, dict[str, str]] = {}

SEASON_CHOICES = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018]
GRAND_PRIX_CHOICES = [
    "Bahrain",
    "Jeddah",
    "Melbourne",
    "Suzuka",
    "Shanghai",
    "Miami",
    "Imola",
    "Monaco",
    "Montreal",
    "Barcelona",
    "Spielberg",
    "Silverstone",
    "Hungaroring",
    "Spa",
    "Zandvoort",
    "Monza",
    "Baku",
    "Singapore",
    "Austin",
    "Mexico City",
    "Interlagos",
    "Las Vegas",
    "Lusail",
    "Abu Dhabi",
]
DRIVER_CHOICES = [
    "VER",
    "NOR",
    "LEC",
    "PIA",
    "SAI",
    "HAM",
    "RUS",
    "PER",
    "ALO",
    "STR",
    "GAS",
    "OCO",
    "TSU",
    "RIC",
    "LAW",
    "HUL",
    "MAG",
    "BOT",
    "ZHO",
    "ALB",
    "SAR",
    "COL",
    "BEA",
    "ANT",
]
SESSION_CHOICES = ["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"]
ANALYSIS_MODE_CHOICES = [
    "Lap-level anomaly analysis",
    "Driver comparison",
    "Telemetry deep dive",
]


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks application."""

    configure_logging()
    with gr.Blocks(title="f1-intelligence-agent") as demo:
        gr.Markdown(
            "# f1-intelligence-agent\n"
            "Agentic F1 telemetry anomaly detection with FastF1, unsupervised ML, "
            "LangGraph, RAG, and HITL memory.",
        )

        with gr.Tab("Run Analysis"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=7, min_width=620):
                    with gr.Accordion("Session", open=True):
                        with gr.Row():
                            year = gr.Dropdown(
                                choices=SEASON_CHOICES,
                                value=2024,
                                label="Season",
                                allow_custom_value=False,
                            )
                            grand_prix = gr.Dropdown(
                                choices=GRAND_PRIX_CHOICES,
                                value="Monza",
                                label="Grand Prix / circuit",
                                allow_custom_value=True,
                            )
                            session_type = gr.Dropdown(
                                choices=SESSION_CHOICES,
                                value="Q",
                                label="Session",
                                allow_custom_value=False,
                            )
                        drivers = gr.Dropdown(
                            choices=DRIVER_CHOICES,
                            value=["VER", "NOR", "LEC"],
                            multiselect=True,
                            label="Drivers",
                            allow_custom_value=True,
                        )
                with gr.Column(scale=4, min_width=360):
                    with gr.Accordion("Analysis", open=True):
                        analysis_mode = gr.Dropdown(
                            choices=ANALYSIS_MODE_CHOICES,
                            value="Lap-level anomaly analysis",
                            label="Mode",
                            allow_custom_value=False,
                        )
                        with gr.Row():
                            sensitivity = gr.Dropdown(
                                choices=["low", "medium", "high"],
                                value="medium",
                                label="Sensitivity",
                                allow_custom_value=False,
                            )
                            top_n = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Top N",
                            )
                        use_tavily = gr.Checkbox(value=False, label="Use Tavily web context")
                        run_button = gr.Button("Run analysis", variant="primary", size="lg")

            with gr.Row(equal_height=True):
                with gr.Column(scale=5, min_width=420):
                    with gr.Accordion("Workflow progress", open=True):
                        run_summary = gr.Markdown("No completed run yet.")
                        current_step = gr.Textbox(
                            label="Current step",
                            interactive=False,
                        )
                        workflow_status = gr.Textbox(
                            label="Step history",
                            lines=10,
                            interactive=False,
                        )
                with gr.Column(scale=4, min_width=360):
                    with gr.Accordion("Session metadata", open=True):
                        metadata = gr.JSON(label="Metadata")
                    with gr.Accordion("Errors", open=False):
                        errors = gr.JSON(label="Errors")

        with gr.Tab("Data Preview"):
            with gr.Tab("Lap Features"):
                lap_features = gr.Dataframe(
                    label="Lap feature preview",
                    interactive=False,
                    max_height=520,
                    max_chars=90,
                    show_search="filter",
                    pinned_columns=1,
                )
            with gr.Tab("Session Context"):
                with gr.Row():
                    weather = gr.Dataframe(
                        label="Weather preview",
                        interactive=False,
                        max_height=420,
                        max_chars=90,
                        show_search="filter",
                    )
                    race_control = gr.Dataframe(
                        label="Race control preview",
                        interactive=False,
                        max_height=420,
                        max_chars=120,
                        show_search="filter",
                    )
            with gr.Tab("Data Quality"):
                missing_summary = gr.Dataframe(
                    label="Missing value summary",
                    interactive=False,
                    max_height=520,
                    show_search="filter",
                )
            with gr.Tab("Field Dictionary"):
                gr.Dataframe(
                    value=get_field_dictionary(),
                    label="F1/FastF1 field dictionary",
                    interactive=False,
                    max_height=560,
                    wrap=True,
                    max_chars=120,
                    show_search="filter",
                    pinned_columns=1,
                )

        with gr.Tab("ML Results"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=6, min_width=560):
                    anomaly_table = gr.Dataframe(
                        label="Anomaly ranking",
                        interactive=False,
                        max_height=430,
                        max_chars=90,
                        show_search="filter",
                        pinned_columns=1,
                    )
                with gr.Column(scale=5, min_width=460):
                    feature_contributions = gr.Dataframe(
                        label="Feature contribution explanations",
                        interactive=False,
                        max_height=430,
                        max_chars=100,
                        show_search="filter",
                        pinned_columns=2,
                    )
            visual_interpretations = gr.Dataframe(
                label="Visual interpretations",
                interactive=False,
                max_height=360,
                max_chars=140,
                show_search="filter",
                pinned_columns=1,
            )
            with gr.Tab("Model Overview"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(plot_descriptor_markdown("pca_cluster"))
                        pca_plot = gr.Plot(label="PCA cluster plot")
                    with gr.Column():
                        gr.Markdown(plot_descriptor_markdown("anomaly_scores"))
                        anomaly_plot = gr.Plot(label="Anomaly scores")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(plot_descriptor_markdown("lap_time_evolution"))
                        lap_time_plot = gr.Plot(label="Lap time evolution")
                    with gr.Column():
                        gr.Markdown(plot_descriptor_markdown("sector_comparison"))
                        sector_plot = gr.Plot(label="Sector comparison")
            with gr.Tab("Diagnostics"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(plot_descriptor_markdown("anomaly_timeline"))
                        anomaly_timeline_plot = gr.Plot(label="Anomaly timeline")
                    with gr.Column():
                        gr.Markdown(plot_descriptor_markdown("model_row_inclusion"))
                        model_exclusion_plot = gr.Plot(label="Model row inclusion")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(plot_descriptor_markdown("feature_missingness"))
                        missingness_plot = gr.Plot(label="Feature missingness")
            with gr.Tab("Telemetry Deep Dive"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(plot_descriptor_markdown("telemetry_speed"))
                        speed_plot = gr.Plot(label="Telemetry speed")
                    with gr.Column():
                        gr.Markdown(plot_descriptor_markdown("telemetry_inputs"))
                        throttle_brake_plot = gr.Plot(label="Telemetry throttle/brake")
                gr.Markdown(plot_descriptor_markdown("telemetry_delta"))
                telemetry_delta_plot = gr.Plot(label="Telemetry deltas")

        with gr.Tab("Agent Report"):
            report = gr.Markdown()

        with gr.Tab("HITL Memory"):
            with gr.Row():
                with gr.Column(scale=6, min_width=560):
                    hitl_summary = gr.Markdown("No memory proposals yet.")
                    memory_table = gr.Dataframe(
                        label="Memory proposals and review status",
                        interactive=False,
                        max_height=360,
                        max_chars=100,
                        show_search="filter",
                        pinned_columns=1,
                    )
                    similar_memories = gr.JSON(label="Similar validated memories")
                with gr.Column(scale=4, min_width=380):
                    with gr.Group():
                        memory_id = gr.Dropdown(label="Selected proposal", choices=[])
                        edited_memory = gr.Textbox(label="Editable memory JSON/text", lines=12)
                    with gr.Row():
                        approve_button = gr.Button("Approve", variant="primary")
                        reject_button = gr.Button("Reject")
                    memory_status = gr.Textbox(label="Memory action status")

        with gr.Tab("F1 Glossary"):
            gr.Markdown(get_glossary_markdown())

        outputs = [
            run_summary,
            workflow_status,
            current_step,
            metadata,
            errors,
            lap_features,
            weather,
            race_control,
            missing_summary,
            anomaly_table,
            feature_contributions,
            visual_interpretations,
            pca_plot,
            lap_time_plot,
            sector_plot,
            anomaly_plot,
            anomaly_timeline_plot,
            model_exclusion_plot,
            missingness_plot,
            speed_plot,
            throttle_brake_plot,
            telemetry_delta_plot,
            report,
            memory_table,
            memory_id,
            edited_memory,
            similar_memories,
            hitl_summary,
        ]
        run_button.click(
            fn=run_ui_analysis,
            inputs=[
                year,
                grand_prix,
                session_type,
                drivers,
                analysis_mode,
                sensitivity,
                top_n,
                use_tavily,
            ],
            outputs=outputs,
        )
        approve_button.click(
            fn=approve_selected_memory,
            inputs=[memory_id, edited_memory],
            outputs=[
                memory_status,
                similar_memories,
                memory_table,
                memory_id,
                edited_memory,
                current_step,
                workflow_status,
                hitl_summary,
            ],
        )
        reject_button.click(
            fn=reject_selected_memory,
            inputs=[memory_id],
            outputs=[
                memory_status,
                similar_memories,
                memory_table,
                memory_id,
                edited_memory,
                current_step,
                workflow_status,
                hitl_summary,
            ],
        )

    return demo


def create_app() -> gr.Blocks:
    """Compatibility factory for smoke tests and external launchers."""

    return build_app()


def run_ui_analysis(
    year: int | str,
    grand_prix: str,
    session_type: str,
    drivers: list[str] | str,
    analysis_mode: str,
    sensitivity: str,
    top_n: int,
    use_tavily: bool,
):
    """Run analysis and stream UI updates."""

    config = AnalysisRunConfig(
        year=int(year),
        grand_prix=grand_prix,
        session_type=session_type,
        drivers=drivers,
        analysis_mode=analysis_mode,
        sensitivity=sensitivity,
        top_n=int(top_n),
        use_tavily=bool(use_tavily),
    )
    global LATEST_STATE, MEMORY_DECISIONS
    MEMORY_DECISIONS = {}
    for state in run_analysis_stream(config):
        LATEST_STATE = state
        yield _state_to_outputs(state)


def approve_selected_memory(memory_id: str | None, edited_text: str | None) -> tuple[Any, ...]:
    """Approve the selected memory proposal."""

    if not memory_id:
        return _memory_action_outputs("No memory proposal selected.", [])
    proposal = _find_memory(memory_id)
    if proposal is None:
        return _memory_action_outputs(f"Memory proposal {memory_id} was not found.", [])
    try:
        store = InsightMemoryStore()
        approved_id = store.approve_memory(proposal, edited_text)
        _record_memory_decision(
            memory_id,
            decision="approved",
            status=f"Approved as {approved_id}",
            persisted_id=approved_id,
        )
        _append_ui_step(f"HITL decision recorded: approved {memory_id}")
        similar = [doc.__dict__ for doc in store.query_similar_memories(json.dumps(proposal), k=5)]
        return _memory_action_outputs(f"Approved memory {approved_id}.", similar)
    except Exception as exc:
        _record_memory_decision(memory_id, decision="error", status=f"Approval failed: {exc}")
        _append_ui_step(f"HITL decision failed: approval error for {memory_id}")
        return _memory_action_outputs(f"Memory approval failed: {exc}", [])


def reject_selected_memory(memory_id: str | None) -> tuple[Any, ...]:
    """Reject the selected proposal."""

    if not memory_id:
        return _memory_action_outputs("No memory proposal selected.", [])
    try:
        InsightMemoryStore().reject_memory(memory_id)
        _record_memory_decision(memory_id, decision="rejected", status="Rejected by user")
        _append_ui_step(f"HITL decision recorded: rejected {memory_id}")
        return _memory_action_outputs(f"Rejected memory proposal {memory_id}.", [])
    except Exception as exc:
        _record_memory_decision(memory_id, decision="error", status=f"Rejection failed: {exc}")
        _append_ui_step(f"HITL decision failed: rejection error for {memory_id}")
        return _memory_action_outputs(f"Memory rejection failed: {exc}", [])


def _find_memory(memory_id: str) -> dict[str, Any] | None:
    for proposal in LATEST_STATE.get("memory_proposals", []):
        if proposal.get("memory_id") == memory_id:
            return proposal
    return None


def _state_to_outputs(state: F1AnalysisState) -> tuple[Any, ...]:
    bundle = state.get("session_bundle")
    lap_df = state.get("lap_features", pd.DataFrame())
    anomaly_df = state.get("anomaly_table", pd.DataFrame())
    metadata = dict(bundle.metadata) if bundle else {}
    if bundle:
        metadata.update(
            {
                "year": bundle.year,
                "grand_prix": bundle.grand_prix,
                "session_type": bundle.session_type,
                "drivers": bundle.drivers,
            }
        )
    workflow_status = _format_step_log(state.get("step_log", []))
    report = (
        state["report_result"].markdown_report
        if state.get("report_result")
        else _error_markdown(state.get("errors", []))
    )

    proposals = state.get("memory_proposals", [])
    proposal_ids = [str(item.get("memory_id")) for item in proposals]
    proposal_table = _memory_table(proposals)
    edited_value = json.dumps(proposals[0], indent=2) if proposals else ""

    return (
        _run_summary(state),
        workflow_status,
        state.get("current_step", ""),
        metadata,
        state.get("errors", []),
        _display_df(lap_df, 50),
        _display_df(bundle.weather if bundle else pd.DataFrame(), 50),
        _display_df(bundle.race_control_messages if bundle else pd.DataFrame(), 50),
        _display_df(pd.DataFrame(state.get("data_profile", {}).get("missing_value_summary", [])), 100)
        if state.get("data_profile")
        else _display_df(missing_value_summary(bundle.laps) if bundle else pd.DataFrame(), 100),
        _display_df(anomaly_df, 50),
        _display_df(state.get("feature_explanations", pd.DataFrame()), 100),
        _display_df(_visual_interpretation_table(state.get("plot_interpretations", [])), 50),
        create_pca_cluster_plot(
            state.get("pca_projection", pd.DataFrame()),
            state.get("model_metadata", pd.DataFrame()),
            state.get("clustering_result", {}).get("labels", []),
            state.get("anomaly_result", {}).get("anomaly_scores", []),
            anomaly_df,
        ),
        create_lap_time_evolution_plot(lap_df),
        create_sector_time_plot(lap_df),
        create_anomaly_score_plot(anomaly_df),
        create_anomaly_timeline_plot(lap_df, anomaly_df),
        create_model_row_exclusion_plot(state.get("model_row_profile", pd.DataFrame())),
        create_missingness_plot(lap_df),
        create_speed_distance_plot(state.get("telemetry_results", {})),
        create_throttle_brake_distance_plot(state.get("telemetry_results", {})),
        create_telemetry_delta_plot(state.get("telemetry_results", {})),
        report,
        _display_df(proposal_table, 20),
        gr.update(choices=proposal_ids, value=proposal_ids[0] if proposal_ids else None),
        edited_value,
        [],
        _hitl_summary(proposals),
    )


def _memory_action_outputs(status: str, similar: list[dict[str, Any]]) -> tuple[Any, ...]:
    proposals = LATEST_STATE.get("memory_proposals", [])
    proposal_ids = [str(item.get("memory_id")) for item in proposals]
    selected_id = _selected_memory_id_after_action(proposal_ids)
    selected_proposal = _find_memory(selected_id) if selected_id else None
    edited_value = json.dumps(selected_proposal, indent=2) if selected_proposal else ""
    return (
        status,
        similar,
        _display_df(_memory_table(proposals), 20),
        gr.update(choices=proposal_ids, value=selected_id),
        edited_value,
        LATEST_STATE.get("current_step", ""),
        _format_step_log(LATEST_STATE.get("step_log", [])),
        _hitl_summary(proposals),
    )


def _visual_interpretation_table(interpretations: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for interpretation in interpretations or []:
        data = (
            interpretation.model_dump(mode="python")
            if hasattr(interpretation, "model_dump")
            else dict(interpretation)
        )
        rows.append(
            {
                "PlotId": data.get("plot_id", ""),
                "Title": data.get("title", ""),
                "Confidence": data.get("confidence", ""),
                "Observations": " | ".join(str(item) for item in data.get("observations", [])[:3]),
                "Cautions": " | ".join(str(item) for item in data.get("caution_notes", [])[:3]),
                "ReportSummary": data.get("report_summary", ""),
            }
        )
    return pd.DataFrame(rows)


def _format_step_log(steps: list[str]) -> str:
    if not steps:
        return "No analysis has been run yet."
    return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))


def _run_summary(state: F1AnalysisState) -> str:
    config = AnalysisRunConfig.model_validate(state.get("run_config", {}))
    bundle = state.get("session_bundle")
    lap_df = state.get("lap_features", pd.DataFrame())
    anomaly_df = state.get("anomaly_table", pd.DataFrame())
    telemetry = state.get("telemetry_results", {})
    proposals = state.get("memory_proposals", [])
    report_status = "ready" if state.get("report_result") else "pending"
    if state.get("errors") and report_status == "ready":
        report_status = "ready with warnings"
    session_label = (
        f"{bundle.year} {bundle.grand_prix} {bundle.session_type}"
        if bundle
        else f"{config.year} {config.grand_prix} {config.session_type}"
    )
    lap_count = len(bundle.laps) if bundle else 0
    model_rows = len(anomaly_df)
    anomaly_count = int(anomaly_df.get("IsAnomaly", pd.Series(dtype=bool)).astype(bool).sum()) if not anomaly_df.empty else 0
    telemetry_loaded = int(telemetry.get("loaded_laps", 0)) if telemetry else 0
    visual_count = len(state.get("plot_interpretations", []))
    return (
        f"**Run summary**  \n"
        f"Session: {session_label} | Mode: {config.analysis_mode} | Drivers: "
        f"{', '.join(bundle.drivers if bundle else config.drivers) or 'all available'}  \n"
        f"Loaded laps: {lap_count} | Feature rows: {len(lap_df)} | Model rows: {model_rows} | "
        f"Flagged anomalies: {anomaly_count} | Telemetry traces: {telemetry_loaded} | "
        f"Visual notes: {visual_count} | Memory proposals: {len(proposals)} | "
        f"Report: {report_status} | Errors: {len(state.get('errors', []))}"
    )


def _selected_memory_id_after_action(proposal_ids: list[str]) -> str | None:
    for proposal_id in proposal_ids:
        if MEMORY_DECISIONS.get(proposal_id, {}).get("decision", "pending") == "pending":
            return proposal_id
    return proposal_ids[0] if proposal_ids else None


def _record_memory_decision(
    memory_id: str,
    *,
    decision: str,
    status: str,
    persisted_id: str | None = None,
) -> None:
    MEMORY_DECISIONS[memory_id] = {
        "decision": decision,
        "status": status,
        "persisted_id": persisted_id or "",
        "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _append_ui_step(message: str) -> None:
    LATEST_STATE["current_step"] = message
    LATEST_STATE["step_log"] = [*LATEST_STATE.get("step_log", []), message]


def _memory_table(proposals: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        memory_id = str(proposal.get("memory_id", ""))
        decision = MEMORY_DECISIONS.get(memory_id, {})
        rows.append(
            {
                "Decision": decision.get("decision", "pending"),
                "ReviewStatus": decision.get("status", "Pending review"),
                "MemoryId": memory_id,
                "Pattern": proposal.get("pattern_name", ""),
                "Confidence": proposal.get("confidence", ""),
                "Driver": proposal.get("driver", ""),
                "LapNumber": proposal.get("lap_number", ""),
                "Hypothesis": proposal.get("hypothesis", ""),
                "PersistedId": decision.get("persisted_id", ""),
                "ReviewedAt": decision.get("reviewed_at", ""),
            }
        )
    return pd.DataFrame(rows)


def _hitl_summary(proposals: list[dict[str, Any]]) -> str:
    if not proposals:
        return "No memory proposals yet."
    counts = {"pending": 0, "approved": 0, "rejected": 0, "error": 0}
    for proposal in proposals:
        memory_id = str(proposal.get("memory_id", ""))
        decision = MEMORY_DECISIONS.get(memory_id, {}).get("decision", "pending")
        counts[decision if decision in counts else "error"] += 1
    return (
        f"**Memory review:** {counts['pending']} pending | "
        f"{counts['approved']} approved | {counts['rejected']} rejected"
        + (f" | {counts['error']} errors" if counts["error"] else "")
    )


def _display_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    frame = safe_head(df, n)
    if frame.empty:
        return frame
    return frame.astype(object).map(_format_cell)


def _format_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        if 0 < abs(number) < 0.001:
            return f"{number:.2e}"
        return f"{number:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _error_markdown(errors: list[str]) -> str:
    if not errors:
        return "Report will appear after analysis completes."
    return "## Analysis Errors\n\n" + "\n".join(f"- {error}" for error in errors)


def main() -> None:
    """Launch the Gradio app."""

    demo = build_app()
    demo.queue()
    demo.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
