"""Plot descriptor registry used by the UI, visual agents, and reports."""

from __future__ import annotations

from f1_intelligence_agent.agents.report_schemas import PlotDescriptor

PLOT_DESCRIPTORS: dict[str, PlotDescriptor] = {
    "pca_cluster": PlotDescriptor(
        plot_id="pca_cluster",
        title="PCA Cluster Projection",
        context="Shows the model feature space compressed to two dimensions for visual inspection.",
        what_to_look_for="Look for isolated points, cluster separation, and whether top anomaly laps sit away from the main pack.",
        caveats=[
            "PCA axes are mathematical summaries, not physical causes.",
            "Overlapping points can still be meaningfully different in the full feature space.",
        ],
    ),
    "lap_time_evolution": PlotDescriptor(
        plot_id="lap_time_evolution",
        title="Lap Time Evolution",
        context="Shows lap-time progression by driver across the session; lower values are faster.",
        what_to_look_for="Look for sudden lap-time spikes, persistent pace shifts, and laps affected by pit, quality, or track-status flags.",
        caveats=["Pit laps and non-green track status should not be read as pure pace loss."],
    ),
    "sector_comparison": PlotDescriptor(
        plot_id="sector_comparison",
        title="Median Sector Comparison",
        context="Compares each driver's median sector times and deltas to the selected group median.",
        what_to_look_for="Look for sector-specific strengths or losses rather than only total lap-time differences.",
        caveats=["Median sectors hide lap-by-lap volatility and traffic effects."],
    ),
    "anomaly_scores": PlotDescriptor(
        plot_id="anomaly_scores",
        title="Top Anomaly Scores",
        context="Ranks model rows by unusualness and separates driver/lap candidates from session-regime context.",
        what_to_look_for="Look for high-scoring driver/lap rows that remain unusual after shared session effects are separated.",
        caveats=["The score is a ranking signal, not a probability or proof of cause."],
    ),
    "anomaly_timeline": PlotDescriptor(
        plot_id="anomaly_timeline",
        title="Anomaly Timeline",
        context="Places anomaly scores on the lap timeline with wet and non-green status bands.",
        what_to_look_for="Look for clusters of anomalies around shared session events versus isolated driver-specific spikes.",
        caveats=["Timeline bands are context hints and should be checked against race-control data."],
    ),
    "model_row_inclusion": PlotDescriptor(
        plot_id="model_row_inclusion",
        title="Model Row Inclusion",
        context="Shows how many loaded laps were included in or excluded from ML modeling.",
        what_to_look_for="Look for high exclusion rates caused by inaccurate, deleted, pit-in, or pit-out laps.",
        caveats=["High exclusion rates reduce confidence in model-level conclusions."],
    ),
    "feature_missingness": PlotDescriptor(
        plot_id="feature_missingness",
        title="Feature Missingness",
        context="Summarizes missing engineered features by feature group and feature name.",
        what_to_look_for="Look for concentrated missingness in speed traps, weather, or timing fields that could bias interpretation.",
        caveats=["Missingness may reflect FastF1 availability rather than an on-track phenomenon."],
    ),
    "telemetry_speed": PlotDescriptor(
        plot_id="telemetry_speed",
        title="Telemetry Speed vs Distance",
        context="Compares anomaly-lap speed traces with available reference laps over distance.",
        what_to_look_for="Look for localized speed deficits or gains at the same distance around the lap.",
        caveats=["Telemetry availability varies; distance alignment is approximate."],
    ),
    "telemetry_inputs": PlotDescriptor(
        plot_id="telemetry_inputs",
        title="Telemetry Inputs vs Distance",
        context="Shows throttle and brake traces for anomaly and reference laps.",
        what_to_look_for="Look for braking or throttle differences near speed deltas.",
        caveats=["Brake and DRS encodings can vary by source and should be interpreted cautiously."],
    ),
    "telemetry_delta": PlotDescriptor(
        plot_id="telemetry_delta",
        title="Telemetry Delta vs Reference",
        context="Shows segment-level deltas between anomaly telemetry and nearby clean reference laps.",
        what_to_look_for="Look for distance segments where speed, braking, throttle, or DRS differ most from reference laps.",
        caveats=["Reference laps are selected heuristically from nearby clean laps in similar context."],
    ),
}


def get_plot_descriptor(plot_id: str) -> PlotDescriptor:
    """Return a descriptor for a known plot id."""

    return PLOT_DESCRIPTORS[plot_id]


def all_plot_descriptors() -> list[PlotDescriptor]:
    """Return descriptors in display order."""

    return list(PLOT_DESCRIPTORS.values())


def plot_descriptor_markdown(plot_id: str) -> str:
    """Return a compact Markdown helper block for a UI plot."""

    descriptor = get_plot_descriptor(plot_id)
    caveats = " ".join(descriptor.caveats)
    return (
        f"**{descriptor.title}**  \n"
        f"{descriptor.context}  \n"
        f"Look for: {descriptor.what_to_look_for}"
        + (f"  \nCaveat: {caveats}" if caveats else "")
    )

