"""LangGraph node implementations for the F1 analysis workflow."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f1_intelligence_agent.agents.report_schemas import (
    EvidenceItem,
    InsightCandidate,
    PlotInterpretation,
    SessionRegimeFinding,
    VisualAnalysisResult,
)
from f1_intelligence_agent.agents.state import AnalysisRunConfig, F1AnalysisState
from f1_intelligence_agent.config import get_settings
from f1_intelligence_agent.data.fastf1_loader import get_lap_telemetry, load_fastf1_session
from f1_intelligence_agent.features.lap_features import build_lap_feature_table
from f1_intelligence_agent.features.telemetry_features import (
    build_telemetry_segment_features,
    compare_telemetry_segments,
)
from f1_intelligence_agent.models.anomaly_detection import run_isolation_forest_anomaly_detection
from f1_intelligence_agent.models.clustering import run_dbscan_clustering
from f1_intelligence_agent.models.dimensionality import run_pca_projection
from f1_intelligence_agent.models.explanations import compute_feature_deviation_explanations
from f1_intelligence_agent.models.preprocessing import (
    build_model_row_profile,
    get_model_feature_names,
    prepare_lap_model_matrix,
)
from f1_intelligence_agent.rag.memory_store import InsightMemoryStore
from f1_intelligence_agent.rag.retriever import retrieve_context_for_insight
from f1_intelligence_agent.rag.vector_store import RetrievedDocument, VectorStoreManager
from f1_intelligence_agent.reports.report_generator import generate_report
from f1_intelligence_agent.utils.dataframe import missing_value_summary
from f1_intelligence_agent.utils.errors import sanitize_error_message
from f1_intelligence_agent.visualization.descriptors import get_plot_descriptor


def _append_step(state: F1AnalysisState, message: str) -> F1AnalysisState:
    state["current_step"] = message
    state["step_log"] = [*state.get("step_log", []), message]
    return state


def _append_error(state: F1AnalysisState, message: str) -> F1AnalysisState:
    state["errors"] = [*state.get("errors", []), sanitize_error_message(message)]
    return state


def _config(state: F1AnalysisState) -> AnalysisRunConfig:
    return AnalysisRunConfig.model_validate(state.get("run_config", {}))


def load_data_node(state: F1AnalysisState) -> F1AnalysisState:
    """Load FastF1 session data."""

    state = _append_step(state, "Loading data")
    config = _config(state)
    settings = get_settings()
    bundle = load_fastf1_session(
        year=config.year,
        grand_prix=config.grand_prix,
        session_type=config.session_type,
        drivers=config.drivers,
        cache_dir=settings.fastf1_cache_dir,
    )
    state["session_bundle"] = bundle
    if bundle.error:
        _append_error(state, bundle.error)
    return state


def profile_data_node(state: F1AnalysisState) -> F1AnalysisState:
    """Profile lap, weather, and race-control availability."""

    state = _append_step(state, "Profiling data")
    bundle = state.get("session_bundle")
    if bundle is None:
        return _append_error(state, "No session bundle available for profiling.")
    laps = bundle.laps
    profile = {
        "number_of_laps": int(len(laps)),
        "number_of_drivers": int(laps["Driver"].nunique()) if "Driver" in laps else 0,
        "available_columns": list(laps.columns),
        "missing_value_summary": missing_value_summary(laps).to_dict(orient="records"),
        "number_of_deleted_laps": int(laps.get("Deleted", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if not laps.empty
        else 0,
        "number_of_inaccurate_laps": int((~laps.get("IsAccurate", pd.Series(True, index=laps.index)).fillna(True).astype(bool)).sum())
        if not laps.empty
        else 0,
        "number_of_pit_in_laps": int(laps.get("PitInTime", pd.Series(index=laps.index)).notna().sum())
        if not laps.empty
        else 0,
        "number_of_pit_out_laps": int(laps.get("PitOutTime", pd.Series(index=laps.index)).notna().sum())
        if not laps.empty
        else 0,
        "weather_availability": not bundle.weather.empty,
        "race_control_availability": not bundle.race_control_messages.empty,
        "telemetry_availability_note": "Telemetry is loaded only for selected top anomalies.",
    }
    state["data_profile"] = profile
    return state


def feature_engineering_node(state: F1AnalysisState) -> F1AnalysisState:
    """Build lap features."""

    state = _append_step(state, "Building features")
    bundle = state.get("session_bundle")
    if bundle is None:
        return _append_error(state, "No session bundle available for feature engineering.")
    state["lap_features"] = build_lap_feature_table(bundle)
    return state


def ml_analysis_node(state: F1AnalysisState) -> F1AnalysisState:
    """Run preprocessing, clustering, anomaly detection, PCA, and feature explanations."""

    state = _append_step(state, "Running clustering and anomaly detection")
    config = _config(state)
    lap_features = state.get("lap_features", pd.DataFrame())
    model_row_profile = build_model_row_profile(lap_features)
    X, metadata, preprocessor = prepare_lap_model_matrix(lap_features)
    clustering = run_dbscan_clustering(X, sensitivity=config.sensitivity)
    anomalies = run_isolation_forest_anomaly_detection(X, sensitivity=config.sensitivity)
    pca = run_pca_projection(X)

    anomaly_table = metadata.copy()
    anomaly_table["Cluster"] = clustering.get("labels", np.array([], dtype=int))
    anomaly_table["AnomalyLabel"] = anomalies.get("labels", np.array([], dtype=int))
    anomaly_table["IsAnomaly"] = anomalies.get("is_anomaly", np.array([], dtype=bool))
    anomaly_table["AnomalyScore"] = anomalies.get("anomaly_scores", np.array([], dtype=float))
    if not pca.empty and len(pca) == len(anomaly_table):
        anomaly_table = pd.concat([anomaly_table.reset_index(drop=True), pca.reset_index(drop=True)], axis=1)
    anomaly_table = anomaly_table.sort_values("AnomalyScore", ascending=False).reset_index(drop=True)
    anomaly_table = _attach_lap_context(anomaly_table, lap_features)
    session_regime_findings, regime_source_indices = _detect_session_regime_findings(
        anomaly_table,
        top_n=config.top_n,
        session_type=config.session_type,
    )
    anomaly_table["FindingCategory"] = np.where(
        anomaly_table["SourceIndex"].isin(regime_source_indices),
        "session_regime_event",
        "driver_lap_anomaly",
    )
    anomaly_table["DriverLapAnomalyScore"] = np.where(
        anomaly_table["FindingCategory"] == "driver_lap_anomaly",
        anomaly_table["AnomalyScore"],
        pd.to_numeric(anomaly_table["AnomalyScore"], errors="coerce").min() - 1.0,
    )
    anomaly_table = anomaly_table.sort_values(
        ["FindingCategory", "DriverLapAnomalyScore", "AnomalyScore"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    driver_lap_top = anomaly_table[anomaly_table["FindingCategory"] == "driver_lap_anomaly"].head(
        config.top_n
    )
    top = driver_lap_top if not driver_lap_top.empty else anomaly_table.head(config.top_n)
    explanations = compute_feature_deviation_explanations(lap_features, top, top_k=5)
    model_diagnostics = _build_model_diagnostics(
        lap_features=lap_features,
        model_row_profile=model_row_profile,
        X=X,
        preprocessor=preprocessor,
        clustering=clustering,
        anomalies=anomalies,
        pca=pca,
        sensitivity=config.sensitivity,
    )

    state["model_matrix"] = X
    state["model_metadata"] = metadata
    state["preprocessor"] = preprocessor
    state["clustering_result"] = clustering
    state["anomaly_result"] = anomalies
    state["pca_projection"] = pca
    state["anomaly_table"] = anomaly_table
    state["feature_explanations"] = explanations
    state["session_regime_findings"] = session_regime_findings
    state["model_diagnostics"] = model_diagnostics
    state["model_row_profile"] = model_row_profile
    return state


def telemetry_analysis_node(state: F1AnalysisState) -> F1AnalysisState:
    """Load telemetry only for top anomaly laps and summarize it."""

    config = _config(state)
    bundle = state.get("session_bundle")
    lap_features = state.get("lap_features", pd.DataFrame())
    anomaly_table = state.get("anomaly_table", pd.DataFrame())
    telemetry_results: dict[str, Any] = {
        "telemetry_by_lap": {},
        "segments_by_lap": {},
        "reference_laps_by_lap": {},
        "reference_segments_by_lap": {},
        "telemetry_comparison_by_lap": {},
        "loaded_laps": 0,
    }
    if bundle is None or anomaly_table.empty:
        return {"telemetry_results": telemetry_results}

    if "FindingCategory" in anomaly_table:
        telemetry_candidates = anomaly_table[
            anomaly_table["FindingCategory"] == "driver_lap_anomaly"
        ].head(config.top_n)
    else:
        telemetry_candidates = anomaly_table.head(config.top_n)
    if telemetry_candidates.empty:
        telemetry_candidates = anomaly_table.head(config.top_n)

    for _, row in telemetry_candidates.iterrows():
        driver = str(row.get("Driver", ""))
        lap_number = row.get("LapNumber")
        if not driver or pd.isna(lap_number):
            continue
        telemetry = get_lap_telemetry(bundle, driver, int(lap_number))
        key = f"{driver}_lap_{int(lap_number)}"
        segments = build_telemetry_segment_features(telemetry)
        telemetry_results["telemetry_by_lap"][key] = telemetry
        telemetry_results["segments_by_lap"][key] = segments
        telemetry_results["loaded_laps"] += 0 if telemetry.empty else 1

        reference_laps = _reference_laps_for_row(row, lap_features, max_laps=3)
        reference_segments: list[pd.DataFrame] = []
        reference_labels: list[str] = []
        for reference_lap in reference_laps:
            ref_telemetry = get_lap_telemetry(bundle, driver, int(reference_lap))
            ref_segments = build_telemetry_segment_features(ref_telemetry)
            if ref_segments.empty:
                continue
            ref_segments["ReferenceLapNumber"] = int(reference_lap)
            reference_segments.append(ref_segments)
            reference_labels.append(str(int(reference_lap)))
        telemetry_results["reference_laps_by_lap"][key] = reference_labels
        if reference_segments:
            reference_frame = pd.concat(reference_segments, ignore_index=True)
            telemetry_results["reference_segments_by_lap"][key] = reference_frame
            telemetry_results["telemetry_comparison_by_lap"][key] = compare_telemetry_segments(
                segments,
                reference_frame,
            )
        else:
            telemetry_results["reference_segments_by_lap"][key] = pd.DataFrame()
            telemetry_results["telemetry_comparison_by_lap"][key] = pd.DataFrame()
    return {"telemetry_results": telemetry_results}


def insight_interpretation_node(state: F1AnalysisState) -> F1AnalysisState:
    """Convert ranked anomaly rows into structured insight candidates."""

    config = _config(state)
    anomaly_table = state.get("anomaly_table", pd.DataFrame())
    explanations = state.get("feature_explanations", pd.DataFrame())
    model_diagnostics = state.get("model_diagnostics", {})
    insights: list[InsightCandidate] = []
    if "FindingCategory" in anomaly_table:
        driver_lap_table = anomaly_table[anomaly_table["FindingCategory"] == "driver_lap_anomaly"]
    else:
        driver_lap_table = anomaly_table
    if driver_lap_table.empty:
        driver_lap_table = anomaly_table
    scores = (
        driver_lap_table["AnomalyScore"]
        if "AnomalyScore" in driver_lap_table
        else pd.Series(dtype=float)
    )
    for index, row in driver_lap_table.head(config.top_n).iterrows():
        driver = str(row.get("Driver", "")) or None
        lap_number = int(row["LapNumber"]) if pd.notna(row.get("LapNumber")) else None
        score = float(row.get("AnomalyScore", 0.0)) if pd.notna(row.get("AnomalyScore")) else None
        lap_explanations = explanations
        if driver is not None and lap_number is not None and not explanations.empty:
            lap_explanations = explanations[
                (explanations["Driver"].astype(str) == driver)
                & (pd.to_numeric(explanations["LapNumber"], errors="coerce") == lap_number)
            ]
        severity = _severity(score, scores)
        data_quality_notes = _data_quality_notes_for_row(row, model_diagnostics)
        evidence = [
            EvidenceItem(
                evidence_type="cluster",
                description=f"DBSCAN cluster assignment: {row.get('Cluster')}",
                value=str(row.get("Cluster")),
                source="DBSCAN",
            ),
            EvidenceItem(
                evidence_type="lap_feature",
                description="Anomaly score from unsupervised model",
                value=score,
                source="IsolationForest",
            ),
        ]
        for _, explanation in lap_explanations.head(5).iterrows():
            evidence.append(
                EvidenceItem(
                    evidence_type="lap_feature",
                    description=f"{explanation['Feature']}: {explanation['PlainEnglishHint']}",
                    value=round(float(explanation["RobustZScore"]), 3),
                    source=f"{explanation['Baseline']} robust z-score",
                )
            )
        confidence = _insight_confidence(len(evidence), data_quality_notes, model_diagnostics)
        title = _insight_title(config.analysis_mode, driver, lap_number)
        possible = _possible_explanations(lap_explanations)
        insights.append(
            InsightCandidate(
                id=f"finding_{index + 1}",
                finding_category="driver_lap_anomaly",
                title=title,
                driver=driver,
                lap_number=lap_number,
                cluster_id=int(row["Cluster"]) if pd.notna(row.get("Cluster")) else None,
                anomaly_score=score,
                severity=severity,
                confidence=confidence,
                summary=_insight_summary(config.analysis_mode, driver, lap_number),
                evidence=evidence,
                possible_explanations=possible,
                limitations=[
                    "The model detects unusual patterns but does not prove causality.",
                    "Telemetry and official context should be checked before making causal claims.",
                ],
                suggested_followups=_suggested_followups(config.analysis_mode, driver),
                reference_population=_reference_population_label(row),
                exclusion_reason=None,
                data_quality_notes=data_quality_notes,
                memory_candidate=confidence in {"medium", "high"},
            )
        )
    return {"insight_candidates": insights}


def lap_visual_interpreter_node(state: F1AnalysisState) -> F1AnalysisState:
    """Interpret lap-level and diagnostic visualizations from their source data."""

    lap_features = state.get("lap_features", pd.DataFrame())
    anomaly_table = state.get("anomaly_table", pd.DataFrame())
    model_row_profile = state.get("model_row_profile", pd.DataFrame())
    model_diagnostics = state.get("model_diagnostics", {})
    interpretations = [
        _interpret_lap_time_evolution(lap_features),
        _interpret_sector_comparison(lap_features),
        _interpret_anomaly_timeline(anomaly_table),
        _interpret_model_row_inclusion(model_row_profile, model_diagnostics),
        _interpret_feature_missingness(lap_features),
    ]
    return {"lap_plot_interpretations": interpretations}


def model_visual_interpreter_node(state: F1AnalysisState) -> F1AnalysisState:
    """Interpret model-space visualizations from PCA and anomaly ranking outputs."""

    pca = state.get("pca_projection", pd.DataFrame())
    anomaly_table = state.get("anomaly_table", pd.DataFrame())
    clustering = state.get("clustering_result", {})
    interpretations = [
        _interpret_pca_projection(pca, anomaly_table, clustering),
        _interpret_anomaly_scores(anomaly_table),
    ]
    return {"model_plot_interpretations": interpretations}


def telemetry_visual_interpreter_node(state: F1AnalysisState) -> F1AnalysisState:
    """Interpret telemetry visualizations from loaded traces and segment comparisons."""

    telemetry_results = state.get("telemetry_results", {})
    interpretations = [
        _interpret_telemetry_speed(telemetry_results),
        _interpret_telemetry_inputs(telemetry_results),
        _interpret_telemetry_delta(telemetry_results),
    ]
    return {"telemetry_plot_interpretations": interpretations}


def visual_context_join_node(state: F1AnalysisState) -> F1AnalysisState:
    """Merge visual-agent outputs into report-ready visual context."""

    interpretations = [
        *state.get("lap_plot_interpretations", []),
        *state.get("model_plot_interpretations", []),
        *state.get("telemetry_plot_interpretations", []),
    ]
    source_plot_ids = sorted({plot_id for item in interpretations for plot_id in item.source_plot_ids})
    caution_notes = _unique(
        note for item in interpretations for note in item.caution_notes if note
    )
    summaries = [item.report_summary for item in interpretations if item.report_summary]
    confidence = _combined_confidence([item.confidence for item in interpretations])
    result = VisualAnalysisResult(
        confidence=confidence,
        summaries=summaries[:10],
        caution_notes=caution_notes[:10],
        source_plot_ids=source_plot_ids,
    )
    state["plot_interpretations"] = interpretations
    state["visual_analysis_result"] = result
    state = _append_step(state, "Interpreting visual diagnostics")
    return {
        "plot_interpretations": interpretations,
        "visual_analysis_result": result,
        "current_step": state["current_step"],
        "step_log": state["step_log"],
    }


def context_retrieval_node(state: F1AnalysisState) -> F1AnalysisState:
    """Retrieve RAG context for generated insights."""

    state = _append_step(state, "Retrieving context")
    insights = state.get("insight_candidates", [])
    session_findings = state.get("session_regime_findings", [])
    if not insights and not session_findings:
        state["retrieved_context"] = []
        return state
    try:
        vector_store = VectorStoreManager()
        vector_store.ensure_knowledge_base()
        retrieved: list[RetrievedDocument] = []
        for insight in insights[:5]:
            retrieved.extend(retrieve_context_for_insight(insight, vector_store, k=4))
        for finding in session_findings[:5]:
            query = " ".join(
                [
                    finding.title,
                    finding.summary,
                    finding.regime_type,
                    " ".join(item.description for item in finding.evidence),
                ]
            )
            retrieved.extend(vector_store.query(query, k=3))
        state["retrieved_context"] = [_doc_to_dict(doc) for doc in retrieved]
    except Exception as exc:
        state["retrieved_context"] = []
        _append_error(state, f"Context retrieval failed: {exc}")
    return state


def web_context_node(state: F1AnalysisState) -> F1AnalysisState:
    """Retrieve optional Tavily web context."""

    state = _append_step(state, "Retrieving optional web context")
    config = _config(state)
    settings = get_settings()
    if not config.use_tavily:
        state["web_context"] = []
        return state
    if not settings.tavily_api_key:
        state["web_context"] = []
        return _append_error(
            state,
            "Tavily web context was requested, but TAVILY_API_KEY is not configured.",
        )
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        query = (
            f"{config.year} {config.grand_prix} Grand Prix {config.session_type} "
            "incidents weather penalties qualifying"
        )
        result = client.search(query=query, max_results=3, include_answer=False)
        state["web_context"] = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "source": "Tavily",
            }
            for item in result.get("results", [])[:3]
        ]
    except Exception as exc:
        state["web_context"] = []
        _append_error(state, f"Tavily web context skipped: {exc}")
    return state


def report_generation_node(state: F1AnalysisState) -> F1AnalysisState:
    """Generate the final OpenAI-guided report."""

    state = _append_step(state, "Generating report")
    bundle = state.get("session_bundle")
    metadata = dict(bundle.metadata) if bundle else {}
    if bundle:
        metadata.update(
            {
                "year": bundle.year,
                "grand_prix": bundle.grand_prix,
                "session_type": bundle.session_type,
                "session_name": bundle.session_name,
                "drivers": bundle.drivers,
            }
        )
    report_kwargs = {
        "session_metadata": metadata,
        "data_profile": state.get("data_profile", {}),
        "insight_candidates": state.get("insight_candidates", []),
        "session_regime_findings": state.get("session_regime_findings", []),
        "model_diagnostics": state.get("model_diagnostics", {}),
        "retrieved_context": state.get("retrieved_context", []),
        "web_context": state.get("web_context", []),
        "clustering_result": state.get("clustering_result", {}),
        "telemetry_results": state.get("telemetry_results", {}),
        "plot_interpretations": state.get("plot_interpretations", []),
        "visual_analysis_result": state.get("visual_analysis_result"),
    }
    try:
        state["report_result"] = generate_report(**report_kwargs)
    except Exception as exc:
        _append_error(state, f"OpenAI report generation failed: {exc}")
    return state


def memory_proposal_node(state: F1AnalysisState) -> F1AnalysisState:
    """Create editable memory proposals from top insights."""

    state = _append_step(state, "Preparing memory proposals")
    insights = [
        insight
        for insight in state.get("insight_candidates", [])
        if insight.finding_category == "driver_lap_anomaly"
        and insight.memory_candidate
        and insight.confidence in {"medium", "high"}
    ]
    if not insights:
        state["memory_proposals"] = []
        state = _append_step(state, "No HITL memory review needed for this run")
        return state
    try:
        store = InsightMemoryStore()
        proposals = [store.propose_memory(insight) for insight in insights[:5]]
        state["memory_proposals"] = proposals
        if proposals:
            state = _append_step(
                state,
                "HITL review required: open the HITL Memory tab to approve, edit, or reject proposed memories",
            )
        else:
            state = _append_step(state, "No HITL memory review needed for this run")
    except Exception as exc:
        state["memory_proposals"] = []
        _append_error(state, f"Memory proposal creation failed: {exc}")
    return state


def should_fetch_web_context(state: F1AnalysisState) -> str:
    """Conditional edge selector for optional Tavily context."""

    config = _config(state)
    return "web" if config.use_tavily else "skip"


def _plot_interpretation(
    plot_id: str,
    *,
    observations: list[str],
    caution_notes: list[str] | None = None,
    confidence: str = "medium",
) -> PlotInterpretation:
    descriptor = get_plot_descriptor(plot_id)
    clean_observations = [item for item in observations if item]
    clean_cautions = [item for item in (caution_notes or []) if item]
    if not clean_observations:
        clean_observations = ["No usable data was available for this plot."]
        confidence = "low"
    return PlotInterpretation(
        plot_id=plot_id,
        title=descriptor.title,
        confidence=confidence,  # type: ignore[arg-type]
        observations=clean_observations,
        caution_notes=clean_cautions,
        report_summary=f"{descriptor.title}: {clean_observations[0]}",
        source_plot_ids=[plot_id],
    )


def _interpret_lap_time_evolution(lap_features: pd.DataFrame) -> PlotInterpretation:
    if lap_features is None or lap_features.empty or "LapTimeSeconds" not in lap_features:
        return _plot_interpretation(
            "lap_time_evolution",
            observations=[],
            caution_notes=["Lap timing data is unavailable."],
            confidence="low",
        )
    frame = lap_features.dropna(subset=["Driver", "LapNumber", "LapTimeSeconds"]).copy()
    if frame.empty:
        return _plot_interpretation("lap_time_evolution", observations=[], confidence="low")
    slowest = frame.loc[pd.to_numeric(frame["LapTimeSeconds"], errors="coerce").idxmax()]
    fastest = frame.loc[pd.to_numeric(frame["LapTimeSeconds"], errors="coerce").idxmin()]
    non_green = int(
        frame.get("TrackStatusHasNonGreen", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    observations = [
        f"Fastest visible lap is {fastest['Driver']} lap {int(fastest['LapNumber'])} at {float(fastest['LapTimeSeconds']):.3f}s; slowest is {slowest['Driver']} lap {int(slowest['LapNumber'])} at {float(slowest['LapTimeSeconds']):.3f}s.",
    ]
    if non_green:
        observations.append(f"{non_green} lap rows have non-green track-status context visible in the lap-time plot.")
    cautions = ["Lower lap time means faster pace."]
    return _plot_interpretation("lap_time_evolution", observations=observations, caution_notes=cautions)


def _interpret_sector_comparison(lap_features: pd.DataFrame) -> PlotInterpretation:
    sectors = ["Sector1Seconds", "Sector2Seconds", "Sector3Seconds"]
    if lap_features is None or lap_features.empty or not set(sectors).intersection(lap_features.columns):
        return _plot_interpretation(
            "sector_comparison",
            observations=[],
            caution_notes=["Sector timing data is unavailable."],
            confidence="low",
        )
    available = [column for column in sectors if column in lap_features]
    grouped = lap_features.groupby("Driver", as_index=False)[available].median(numeric_only=True)
    if grouped.empty:
        return _plot_interpretation("sector_comparison", observations=[], confidence="low")
    spreads = {
        column: float(grouped[column].max() - grouped[column].min())
        for column in available
        if grouped[column].notna().any()
    }
    if not spreads:
        return _plot_interpretation("sector_comparison", observations=[], confidence="low")
    sector, spread = max(spreads.items(), key=lambda item: item[1])
    driver = str(grouped.loc[grouped[sector].idxmin(), "Driver"])
    observations = [
        f"{sector.replace('Seconds', '')} has the largest median spread across drivers ({spread:.3f}s); {driver} is quickest on that sector median.",
    ]
    return _plot_interpretation("sector_comparison", observations=observations)


def _interpret_anomaly_timeline(anomaly_table: pd.DataFrame) -> PlotInterpretation:
    if anomaly_table is None or anomaly_table.empty or "AnomalyScore" not in anomaly_table:
        return _plot_interpretation("anomaly_timeline", observations=[], confidence="low")
    category_counts = anomaly_table.get(
        "FindingCategory", pd.Series("driver_lap_anomaly", index=anomaly_table.index)
    ).value_counts()
    top = anomaly_table.sort_values("AnomalyScore", ascending=False).head(1)
    observations = [
        f"The timeline contains {int(category_counts.get('driver_lap_anomaly', 0))} driver/lap rows and {int(category_counts.get('session_regime_event', 0))} session-regime rows.",
    ]
    if not top.empty:
        row = top.iloc[0]
        observations.append(
            f"Highest timeline score is {row.get('Driver')} lap {int(row.get('LapNumber')) if pd.notna(row.get('LapNumber')) else 'unknown'} ({float(row.get('AnomalyScore')):.3f})."
        )
    repeated = (
        anomaly_table.groupby("LapNumber")["Driver"].nunique()
        if {"LapNumber", "Driver"}.issubset(anomaly_table.columns)
        else pd.Series(dtype=int)
    )
    shared = repeated[repeated >= 2]
    if not shared.empty:
        observations.append(f"{len(shared)} lap number(s) appear for multiple drivers, indicating shared session context.")
    return _plot_interpretation("anomaly_timeline", observations=observations)


def _interpret_model_row_inclusion(
    model_row_profile: pd.DataFrame,
    model_diagnostics: dict[str, Any],
) -> PlotInterpretation:
    total = int(model_diagnostics.get("total_feature_rows", len(model_row_profile)))
    included = int(model_diagnostics.get("included_model_rows", 0))
    excluded = int(model_diagnostics.get("excluded_model_rows", max(total - included, 0)))
    if total == 0:
        return _plot_interpretation("model_row_inclusion", observations=[], confidence="low")
    observations = [
        f"{included}/{total} feature rows were included in the model; {excluded} were excluded before ML scoring.",
    ]
    reasons = model_diagnostics.get("exclusion_reason_counts", {})
    if reasons:
        top_reason, top_count = max(reasons.items(), key=lambda item: item[1])
        observations.append(f"Most common exclusion reason is {top_reason} ({top_count} rows).")
    cautions = []
    if "high_exclusion_rate" in model_diagnostics.get("warnings", []):
        cautions.append("High exclusion rate reduces confidence in broad model conclusions.")
    return _plot_interpretation(
        "model_row_inclusion",
        observations=observations,
        caution_notes=cautions,
        confidence="low" if cautions else "medium",
    )


def _interpret_feature_missingness(lap_features: pd.DataFrame) -> PlotInterpretation:
    if lap_features is None or lap_features.empty:
        return _plot_interpretation("feature_missingness", observations=[], confidence="low")
    missing = lap_features.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return _plot_interpretation(
            "feature_missingness",
            observations=["No engineered feature has missing values in the displayed feature table."],
            confidence="high",
        )
    top_feature = str(missing.index[0])
    top_count = int(missing.iloc[0])
    percent = top_count / len(lap_features) * 100
    return _plot_interpretation(
        "feature_missingness",
        observations=[f"{top_feature} has the highest missingness ({top_count} rows, {percent:.1f}%)."],
        caution_notes=["Concentrated missingness can bias model explanations for affected feature groups."],
    )


def _interpret_pca_projection(
    pca: pd.DataFrame,
    anomaly_table: pd.DataFrame,
    clustering: dict[str, Any],
) -> PlotInterpretation:
    if pca is None or pca.empty:
        return _plot_interpretation("pca_cluster", observations=[], confidence="low")
    variance = 0.0
    if "ExplainedVariancePC1" in pca:
        variance += float(pd.to_numeric(pca["ExplainedVariancePC1"], errors="coerce").fillna(0).iloc[0])
    if "ExplainedVariancePC2" in pca:
        variance += float(pd.to_numeric(pca["ExplainedVariancePC2"], errors="coerce").fillna(0).iloc[0])
    n_clusters = int(clustering.get("n_clusters", 0) or 0)
    n_noise = int(clustering.get("n_noise", 0) or 0)
    observations = [
        f"The PCA view explains about {variance * 100:.1f}% of transformed-model variance across PC1 and PC2.",
        f"DBSCAN reports {n_clusters} cluster(s) and {n_noise} noise row(s) in the projected model view.",
    ]
    if anomaly_table is not None and not anomaly_table.empty and {"PC1", "PC2", "AnomalyScore"}.issubset(anomaly_table.columns):
        top = anomaly_table.sort_values("AnomalyScore", ascending=False).head(1).iloc[0]
        observations.append(f"Top-scored row appears at PC1={float(top['PC1']):.2f}, PC2={float(top['PC2']):.2f}.")
    cautions = ["Use PCA as a visual diagnostic only; it does not explain physical cause."]
    return _plot_interpretation("pca_cluster", observations=observations, caution_notes=cautions)


def _interpret_anomaly_scores(anomaly_table: pd.DataFrame) -> PlotInterpretation:
    if anomaly_table is None or anomaly_table.empty or "AnomalyScore" not in anomaly_table:
        return _plot_interpretation("anomaly_scores", observations=[], confidence="low")
    top = anomaly_table.sort_values("AnomalyScore", ascending=False).head(1).iloc[0]
    category_counts = anomaly_table.get(
        "FindingCategory", pd.Series("driver_lap_anomaly", index=anomaly_table.index)
    ).value_counts()
    observations = [
        f"Highest anomaly score is {float(top['AnomalyScore']):.3f} for {top.get('Driver')} lap {int(top.get('LapNumber')) if pd.notna(top.get('LapNumber')) else 'unknown'}.",
        f"Ranking contains {int(category_counts.get('driver_lap_anomaly', 0))} driver/lap rows and {int(category_counts.get('session_regime_event', 0))} session-regime rows.",
    ]
    return _plot_interpretation("anomaly_scores", observations=observations)


def _interpret_telemetry_speed(telemetry_results: dict[str, Any]) -> PlotInterpretation:
    traces = telemetry_results.get("telemetry_by_lap", {}) if isinstance(telemetry_results, dict) else {}
    loaded = int(telemetry_results.get("loaded_laps", 0) or 0) if isinstance(telemetry_results, dict) else 0
    if not traces or loaded == 0:
        return _plot_interpretation(
            "telemetry_speed",
            observations=[],
            caution_notes=["No telemetry traces were loaded for the selected anomaly candidates."],
            confidence="low",
        )
    observations = [f"{loaded} anomaly telemetry trace(s) were loaded for speed-distance review."]
    lengths = {
        key: len(frame)
        for key, frame in traces.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    }
    if lengths:
        key, length = max(lengths.items(), key=lambda item: item[1])
        observations.append(f"{key} has the densest available telemetry trace ({length} samples).")
    return _plot_interpretation("telemetry_speed", observations=observations)


def _interpret_telemetry_inputs(telemetry_results: dict[str, Any]) -> PlotInterpretation:
    traces = telemetry_results.get("telemetry_by_lap", {}) if isinstance(telemetry_results, dict) else {}
    if not traces:
        return _plot_interpretation(
            "telemetry_inputs",
            observations=[],
            caution_notes=["Throttle/brake traces are unavailable until telemetry loads successfully."],
            confidence="low",
        )
    brake_traces = sum(
        1
        for frame in traces.values()
        if isinstance(frame, pd.DataFrame) and "Brake" in frame and frame["Brake"].notna().any()
    )
    throttle_traces = sum(
        1
        for frame in traces.values()
        if isinstance(frame, pd.DataFrame) and "Throttle" in frame and frame["Throttle"].notna().any()
    )
    observations = [
        f"Throttle data is available for {throttle_traces} trace(s), and brake data is available for {brake_traces} trace(s)."
    ]
    return _plot_interpretation(
        "telemetry_inputs",
        observations=observations,
        caution_notes=["Brake encoding can be boolean or numeric depending on source data."],
    )


def _interpret_telemetry_delta(telemetry_results: dict[str, Any]) -> PlotInterpretation:
    comparisons = (
        telemetry_results.get("telemetry_comparison_by_lap", {})
        if isinstance(telemetry_results, dict)
        else {}
    )
    frames = [frame for frame in comparisons.values() if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return _plot_interpretation(
            "telemetry_delta",
            observations=[],
            caution_notes=["No reference telemetry comparison is available for the selected laps."],
            confidence="low",
        )
    combined = pd.concat(frames, ignore_index=True)
    observations = [f"{len(frames)} anomaly lap(s) have segment-level telemetry comparisons."]
    if "DeltaMeanSpeed" in combined and combined["DeltaMeanSpeed"].notna().any():
        max_delta = combined.loc[combined["DeltaMeanSpeed"].abs().idxmax()]
        observations.append(
            f"Largest mean-speed delta is {float(max_delta['DeltaMeanSpeed']):+.2f} near distance {float(max_delta.get('DistanceStart', 0.0)):.1f}."
        )
    return _plot_interpretation("telemetry_delta", observations=observations)


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _combined_confidence(values: list[str]) -> str:
    if not values:
        return "low"
    if "low" in values:
        return "low"
    if "medium" in values:
        return "medium"
    return "high"


def _doc_to_dict(doc: RetrievedDocument) -> dict[str, Any]:
    return {
        "id": doc.id,
        "text": doc.text,
        "metadata": doc.metadata,
        "distance": doc.distance,
        "collection": doc.collection,
        "source": doc.metadata.get("source"),
    }


def _attach_lap_context(anomaly_table: pd.DataFrame, lap_features: pd.DataFrame) -> pd.DataFrame:
    if anomaly_table.empty or lap_features.empty or "SourceIndex" not in anomaly_table:
        return anomaly_table
    context_columns = [
        "SourceIndex",
        "SessionPhase",
        "WeatherRegime",
        "CompoundRegime",
        "StintCompoundKey",
        "FreshTyre",
        "HasRainfall",
        "IsWetCompound",
        "IsDryCompound",
        "IsRaceStartLap",
        "IsEarlySessionLap",
        "IsTrackStatusChangeLap",
        "IsRestartOrStatusChangeLap",
        "TrackStatusHasNonGreen",
        "TrackStatusHasYellow",
        "TrackStatusHasSafetyCar",
        "TrackStatusHasRedFlag",
        "TrackStatusHasVSC",
        "HasMissingWeather",
        "HasMissingSpeedTrap",
        "HasMissingSectorTime",
    ]
    available = [
        column
        for column in context_columns
        if column == "SourceIndex" or (column in lap_features and column not in anomaly_table.columns)
    ]
    if available == ["SourceIndex"] or not available:
        return anomaly_table
    context = lap_features[[column for column in available if column != "SourceIndex"]].copy()
    context["SourceIndex"] = context.index
    return anomaly_table.merge(context, on="SourceIndex", how="left")


def _detect_session_regime_findings(
    anomaly_table: pd.DataFrame,
    top_n: int,
    session_type: str = "",
) -> tuple[list[SessionRegimeFinding], set[int]]:
    if anomaly_table.empty or "SourceIndex" not in anomaly_table:
        return [], set()

    findings: list[SessionRegimeFinding] = []
    regime_source_indices: set[int] = set()

    def add_finding(
        *,
        finding_id: str,
        title: str,
        regime_type: str,
        rows: pd.DataFrame,
        summary: str,
        evidence_description: str,
        severity: str = "medium",
    ) -> None:
        nonlocal regime_source_indices
        if rows.empty:
            return
        drivers = sorted(rows.get("Driver", pd.Series(dtype=object)).dropna().astype(str).unique())
        laps = sorted(
            {
                int(value)
                for value in pd.to_numeric(rows.get("LapNumber", pd.Series(dtype=float)), errors="coerce").dropna()
            }
        )
        if len(drivers) < 2:
            return
        scores = pd.to_numeric(rows.get("AnomalyScore", pd.Series(dtype=float)), errors="coerce").dropna()
        evidence = [
            EvidenceItem(
                evidence_type="lap_feature",
                description=evidence_description,
                value=float(scores.max()) if not scores.empty else None,
                source="session-regime detector",
            )
        ]
        findings.append(
            SessionRegimeFinding(
                id=finding_id,
                title=title,
                regime_type=regime_type,  # type: ignore[arg-type]
                affected_laps=laps[:30],
                affected_drivers=drivers,
                severity=severity,  # type: ignore[arg-type]
                confidence="medium",
                summary=summary,
                evidence=evidence,
                limitations=[
                    "This is treated as a session-wide context pattern, not an individual driver anomaly.",
                    "Review official timing, race-control, and broadcast context before making causal claims.",
                ],
                suggested_followups=[
                    "Compare affected laps against race-control messages and weather samples.",
                    "Inspect whether the same pattern appears across multiple drivers.",
                ],
            )
        )
        regime_source_indices |= set(pd.to_numeric(rows["SourceIndex"], errors="coerce").dropna().astype(int))

    top_rows = anomaly_table.head(max(int(top_n) * 3, 12))
    is_race_session = str(session_type).upper() == "R"

    wet_mask = (
        anomaly_table.get("WeatherRegime", pd.Series("", index=anomaly_table.index)).astype(str).eq("wet")
        | anomaly_table.get("HasRainfall", pd.Series(False, index=anomaly_table.index)).fillna(False).astype(bool)
        | anomaly_table.get("IsWetCompound", pd.Series(False, index=anomaly_table.index)).fillna(False).astype(bool)
    )
    wet_rows = anomaly_table[wet_mask]
    weather_regime_count = (
        anomaly_table.get("WeatherRegime", pd.Series(dtype=object)).dropna().astype(str).nunique()
    )
    if wet_rows["Driver"].nunique() >= 2 and weather_regime_count > 1:
        add_finding(
            finding_id="session_regime_rain",
            title="Rain or wet-tyre regime affected multiple drivers",
            regime_type="rain",
            rows=wet_rows,
            summary="Wet-weather or rainfall context appears across multiple drivers and should be reported separately from individual lap anomalies.",
            evidence_description="Multiple drivers have wet-regime, rainfall, or wet-compound laps in the model rows.",
            severity="high",
        )

    non_green_mask = anomaly_table.get(
        "TrackStatusHasNonGreen", pd.Series(False, index=anomaly_table.index)
    ).fillna(False).astype(bool)
    non_green_rows = anomaly_table[non_green_mask]
    if non_green_rows["Driver"].nunique() >= 2:
        add_finding(
            finding_id="session_regime_track_status",
            title="Non-green track status affected multiple drivers",
            regime_type="track_status",
            rows=non_green_rows,
            summary="Yellow, safety-car, virtual-safety-car, or red-flag context appears across multiple drivers and should be handled as session context.",
            evidence_description="Multiple drivers have model rows with non-green track-status flags.",
            severity="high",
        )

    race_start_mask = top_rows.get(
        "IsRaceStartLap", pd.Series(False, index=top_rows.index)
    ).fillna(False).astype(bool)
    race_start_rows = top_rows[race_start_mask]
    if race_start_rows["Driver"].nunique() >= 2:
        add_finding(
            finding_id="session_regime_race_start",
            title="Race-start phase appears in the top model outliers",
            regime_type="race_start",
            rows=anomaly_table[
                anomaly_table.get("IsRaceStartLap", pd.Series(False, index=anomaly_table.index))
                .fillna(False)
                .astype(bool)
            ],
            summary="Early race laps are appearing as model outliers across drivers, so they should be interpreted as phase context.",
            evidence_description="Top-ranked outlier rows include race-start laps from multiple drivers.",
        )

    duplicate_laps = (
        top_rows.groupby("LapNumber")["Driver"].nunique()
        if "LapNumber" in top_rows and "Driver" in top_rows
        else pd.Series(dtype=int)
    )
    shared_laps = duplicate_laps[duplicate_laps >= 2].index.tolist()
    if shared_laps and is_race_session:
        shared_rows = anomaly_table[anomaly_table["LapNumber"].isin(shared_laps)]
        add_finding(
            finding_id="session_regime_shared_lap",
            title="Same-lap outliers appeared for multiple drivers",
            regime_type="session_phase",
            rows=shared_rows,
            summary="The same lap number appears as an outlier for multiple drivers, which usually indicates session phase or event context.",
            evidence_description="Multiple drivers share the same anomalous lap number among top-ranked rows.",
        )

    transition_mask = top_rows.get("FreshTyre", pd.Series(False, index=top_rows.index)).fillna(False).astype(bool) & (
        pd.to_numeric(top_rows.get("LapNumber", pd.Series(dtype=float)), errors="coerce") > 1
    )
    transition_rows = top_rows[transition_mask]
    if transition_rows["Driver"].nunique() >= 2 and is_race_session:
        add_finding(
            finding_id="session_regime_compound_transition",
            title="Compound or stint transition affected multiple drivers",
            regime_type="compound_transition",
            rows=transition_rows,
            summary="Fresh-tyre or stint-transition laps appear across multiple drivers in top-ranked rows.",
            evidence_description="Multiple top-ranked rows occur on fresh tyres after the opening lap.",
        )

    return findings, regime_source_indices


def _build_model_diagnostics(
    *,
    lap_features: pd.DataFrame,
    model_row_profile: pd.DataFrame,
    X: np.ndarray,
    preprocessor: Any,
    clustering: dict[str, Any],
    anomalies: dict[str, Any],
    pca: pd.DataFrame,
    sensitivity: str,
) -> dict[str, Any]:
    included = int(model_row_profile.get("IsModelRow", pd.Series(dtype=bool)).sum())
    total = int(len(model_row_profile))
    excluded = total - included
    reason_counts = (
        model_row_profile.loc[model_row_profile["ExclusionReason"] != "included", "ExclusionReason"]
        .value_counts()
        .to_dict()
        if not model_row_profile.empty
        else {}
    )
    warnings: list[str] = []
    if included < 8:
        warnings.append("low_model_sample")
    if total and excluded / total >= 0.4:
        warnings.append("high_exclusion_rate")
    if X is None or X.size == 0 or (X.ndim == 2 and X.shape[1] == 0):
        warnings.append("empty_model_matrix")
    missing_weather = (
        int(lap_features.get("HasMissingWeather", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if not lap_features.empty
        else 0
    )
    if total and missing_weather / total >= 0.4:
        warnings.append("high_weather_missingness")

    pca_variance = {}
    if not pca.empty:
        pca_variance = {
            "pc1": float(pca["ExplainedVariancePC1"].iloc[0]) if "ExplainedVariancePC1" in pca else 0.0,
            "pc2": float(pca["ExplainedVariancePC2"].iloc[0]) if "ExplainedVariancePC2" in pca else 0.0,
        }

    return {
        "total_feature_rows": total,
        "included_model_rows": included,
        "excluded_model_rows": excluded,
        "exclusion_reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
        "feature_count": int(X.shape[1]) if X is not None and X.ndim == 2 else 0,
        "feature_names": get_model_feature_names(preprocessor),
        "sensitivity": sensitivity,
        "anomaly_method": anomalies.get("method"),
        "contamination": anomalies.get("contamination"),
        "anomaly_threshold": anomalies.get("threshold"),
        "clustering_method": clustering.get("method"),
        "dbscan_eps": clustering.get("eps"),
        "dbscan_min_samples": clustering.get("min_samples"),
        "dbscan_noise_rows": clustering.get("n_noise"),
        "dbscan_clusters": clustering.get("n_clusters"),
        "pca_explained_variance": pca_variance,
        "warnings": warnings,
    }


def _reference_laps_for_row(row: pd.Series, lap_features: pd.DataFrame, max_laps: int = 3) -> list[int]:
    if lap_features.empty or "SourceIndex" not in row:
        return []
    driver = str(row.get("Driver", ""))
    lap_number = pd.to_numeric(row.get("LapNumber"), errors="coerce")
    if not driver or pd.isna(lap_number):
        return []
    frame = lap_features.copy()
    mask = frame["Driver"].astype(str).eq(driver)
    mask &= pd.to_numeric(frame["LapNumber"], errors="coerce").ne(float(lap_number))
    for column in ["IsDeleted", "IsPitInLap", "IsPitOutLap"]:
        if column in frame:
            mask &= ~frame[column].fillna(False).astype(bool)
    if "IsAccurate" in frame:
        mask &= frame["IsAccurate"].fillna(True).astype(bool)
    reference = frame[mask].copy()
    if reference.empty:
        return []
    for column in ["WeatherRegime", "CompoundRegime", "TrackStatusHasNonGreen"]:
        if column in reference and column in row and pd.notna(row.get(column)):
            same = reference[reference[column].astype(str) == str(row.get(column))]
            if len(same) >= max_laps:
                reference = same
    reference["_distance_to_lap"] = (
        pd.to_numeric(reference["LapNumber"], errors="coerce") - float(lap_number)
    ).abs()
    return (
        pd.to_numeric(
            reference.sort_values("_distance_to_lap")["LapNumber"].head(max_laps),
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )


def _reference_population_label(row: pd.Series) -> str:
    parts = [
        f"driver={row.get('Driver')}",
        f"weather={row.get('WeatherRegime', 'unknown')}",
        f"compound_regime={row.get('CompoundRegime', 'unknown')}",
        f"track_non_green={row.get('TrackStatusHasNonGreen', 'unknown')}",
    ]
    return ", ".join(parts)


def _row_bool(row: pd.Series, column: str) -> bool:
    value = row.get(column, False)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return bool(value)


def _data_quality_notes_for_row(row: pd.Series, model_diagnostics: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if _row_bool(row, "HasMissingWeather"):
        notes.append("Weather values are partially missing for this row.")
    if _row_bool(row, "HasMissingSpeedTrap"):
        notes.append("One or more speed-trap values are missing for this row.")
    if _row_bool(row, "HasMissingSectorTime"):
        notes.append("One or more sector times are missing for this row.")
    if _row_bool(row, "TrackStatusHasNonGreen"):
        notes.append("Track status is non-green, so session context may explain part of the pattern.")
    warnings = set(model_diagnostics.get("warnings", []))
    if "low_model_sample" in warnings:
        notes.append("The model sample is small; treat this finding as low confidence.")
    if "high_exclusion_rate" in warnings:
        notes.append("Many loaded laps were excluded from modeling due to quality or pit-lap filters.")
    return notes


def _insight_confidence(
    evidence_count: int,
    data_quality_notes: list[str],
    model_diagnostics: dict[str, Any],
) -> str:
    warnings = set(model_diagnostics.get("warnings", []))
    if "low_model_sample" in warnings or "empty_model_matrix" in warnings:
        return "low"
    if evidence_count >= 6 and not data_quality_notes:
        return "high"
    if evidence_count >= 4:
        return "medium"
    return "low"


def _severity(score: float | None, scores: pd.Series) -> str:
    if score is None or scores.empty:
        return "low"
    numeric = pd.to_numeric(scores, errors="coerce").dropna()
    if numeric.empty:
        return "low"
    high = numeric.quantile(0.9)
    medium = numeric.quantile(0.7)
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


def _possible_explanations(explanations: pd.DataFrame) -> list[str]:
    if explanations.empty:
        return ["A possible explanation is a lap pattern not captured by a single feature deviation."]
    hints = " ".join(explanations["PlainEnglishHint"].astype(str).head(3).tolist()).lower()
    possible: list[str] = []
    if "sector" in hints:
        possible.append("A possible explanation is localized time loss in one sector, such as traffic, caution, or corner-specific behavior.")
    if "speed" in hints:
        possible.append("Telemetry or speed-trap evidence may indicate a straight-line or corner-minimum-speed difference.")
    if "tyre" in hints:
        possible.append("Tyre state may be relevant, but this should be validated against stint context and telemetry.")
    if not possible:
        possible.append("The evidence suggests an unusual lap profile that needs domain validation.")
    return possible[:3]


def _insight_title(mode: str, driver: str | None, lap_number: int | None) -> str:
    driver_label = driver or "Unknown driver"
    lap_label = lap_number or "unknown"
    if mode == "Driver comparison":
        return f"{driver_label} lap {lap_label} driver comparison outlier"
    if mode == "Telemetry deep dive":
        return f"{driver_label} lap {lap_label} telemetry review candidate"
    return f"{driver_label} lap {lap_label} unusual lap pattern"


def _insight_summary(mode: str, driver: str | None, lap_number: int | None) -> str:
    base = (
        f"Lap {lap_number} for {driver} was ranked as unusual by the anomaly model. "
        "The strongest feature deviations are listed as evidence."
    )
    if mode == "Driver comparison":
        return (
            f"{base} The selected mode emphasizes how this driver-lap differs from driver "
            "and session baselines."
        )
    if mode == "Telemetry deep dive":
        return (
            f"{base} The selected mode prioritizes this lap for telemetry trace comparison."
        )
    return base


def _suggested_followups(mode: str, driver: str | None) -> list[str]:
    driver_label = driver or "the same driver"
    if mode == "Driver comparison":
        return [
            "Compare median sector and speed features against the selected driver group.",
            "Check whether the outlier remains after excluding inaccurate, pit-in, and pit-out laps.",
            "Review race-control messages around the lap time.",
        ]
    if mode == "Telemetry deep dive":
        return [
            f"Compare telemetry with nearby clean laps from {driver_label}.",
            "Align speed, throttle, brake, DRS, and gear traces by distance.",
            "Review race-control messages around the lap time.",
        ]
    return [
        "Compare against onboard footage or official session notes.",
        "Review race-control messages around the lap time.",
        "Compare telemetry with nearby clean laps from the same driver.",
    ]
