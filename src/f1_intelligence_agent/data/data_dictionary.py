"""Field dictionary and beginner-friendly F1 glossary."""

from __future__ import annotations

import pandas as pd

FIELD_DEFINITIONS: list[dict[str, object]] = [
    {
        "field": "Driver",
        "category": "lap",
        "plain_english_definition": "Three-letter FIA driver code.",
        "used_in_mvp": True,
        "notes": "Used as the main driver identifier.",
    },
    {
        "field": "DriverNumber",
        "category": "lap",
        "plain_english_definition": "Car number assigned to the driver.",
        "used_in_mvp": False,
        "notes": "Useful for joins when driver code is unavailable.",
    },
    {
        "field": "Team",
        "category": "lap",
        "plain_english_definition": "Constructor/team name for the lap.",
        "used_in_mvp": True,
        "notes": "Used for preview and interpretation.",
    },
    {
        "field": "LapNumber",
        "category": "lap",
        "plain_english_definition": "Lap sequence number within the session.",
        "used_in_mvp": True,
        "notes": "Primary lap identifier.",
    },
    {
        "field": "LapTime",
        "category": "timing",
        "plain_english_definition": "Total time needed to complete the lap.",
        "used_in_mvp": True,
        "notes": "Converted to seconds for modeling.",
    },
    {
        "field": "Sector1Time",
        "category": "timing",
        "plain_english_definition": "Time through the first third of the circuit.",
        "used_in_mvp": True,
        "notes": "Converted to seconds for localized time-loss checks.",
    },
    {
        "field": "Sector2Time",
        "category": "timing",
        "plain_english_definition": "Time through the second sector of the circuit.",
        "used_in_mvp": True,
        "notes": "Converted to seconds for localized time-loss checks.",
    },
    {
        "field": "Sector3Time",
        "category": "timing",
        "plain_english_definition": "Time through the final sector of the circuit.",
        "used_in_mvp": True,
        "notes": "Converted to seconds for localized time-loss checks.",
    },
    {
        "field": "SpeedI1",
        "category": "speed",
        "plain_english_definition": "Speed at intermediate speed trap 1.",
        "used_in_mvp": True,
        "notes": "May be missing for some sessions.",
    },
    {
        "field": "SpeedI2",
        "category": "speed",
        "plain_english_definition": "Speed at intermediate speed trap 2.",
        "used_in_mvp": True,
        "notes": "May be missing for some sessions.",
    },
    {
        "field": "SpeedFL",
        "category": "speed",
        "plain_english_definition": "Speed recorded at the finish line.",
        "used_in_mvp": True,
        "notes": "Useful for straight-line performance context.",
    },
    {
        "field": "SpeedST",
        "category": "speed",
        "plain_english_definition": "Speed trap measurement on a straight.",
        "used_in_mvp": True,
        "notes": "Useful at circuits where straight-line speed matters.",
    },
    {
        "field": "Compound",
        "category": "tyre",
        "plain_english_definition": "Tyre compound used for the lap.",
        "used_in_mvp": True,
        "notes": "Encoded for the model and explained in reports.",
    },
    {
        "field": "TyreLife",
        "category": "tyre",
        "plain_english_definition": "Approximate number of laps completed on the tyre set.",
        "used_in_mvp": True,
        "notes": "Used as a degradation proxy, not causal proof.",
    },
    {
        "field": "FreshTyre",
        "category": "tyre",
        "plain_english_definition": "Whether the tyre set was new when fitted.",
        "used_in_mvp": True,
        "notes": "May be unavailable for some sessions.",
    },
    {
        "field": "Stint",
        "category": "tyre",
        "plain_english_definition": "Continuous run between pit stops.",
        "used_in_mvp": True,
        "notes": "Useful for comparing similar running phases.",
    },
    {
        "field": "PitInTime",
        "category": "event",
        "plain_english_definition": "Timestamp when the car entered the pit lane.",
        "used_in_mvp": True,
        "notes": "Used to flag pit-in laps.",
    },
    {
        "field": "PitOutTime",
        "category": "event",
        "plain_english_definition": "Timestamp when the car exited the pit lane.",
        "used_in_mvp": True,
        "notes": "Used to flag pit-out laps.",
    },
    {
        "field": "TrackStatus",
        "category": "event",
        "plain_english_definition": "Encoded track condition such as green, yellow, safety car, or red flag.",
        "used_in_mvp": True,
        "notes": "Used for context; codes require session-specific care.",
    },
    {
        "field": "Position",
        "category": "lap",
        "plain_english_definition": "Running order position associated with the lap.",
        "used_in_mvp": True,
        "notes": "May be noisy in practice sessions.",
    },
    {
        "field": "Deleted",
        "category": "quality",
        "plain_english_definition": "Whether the lap was deleted by officials.",
        "used_in_mvp": True,
        "notes": "Kept visible and excluded from default ML training.",
    },
    {
        "field": "IsAccurate",
        "category": "quality",
        "plain_english_definition": "FastF1 flag for whether timing data appears accurate.",
        "used_in_mvp": True,
        "notes": "Inaccurate laps are kept visible but excluded from default ML training.",
    },
    {
        "field": "AirTemp",
        "category": "weather",
        "plain_english_definition": "Ambient air temperature in Celsius.",
        "used_in_mvp": True,
        "notes": "Joined to laps by nearest available session time.",
    },
    {
        "field": "TrackTemp",
        "category": "weather",
        "plain_english_definition": "Track surface temperature in Celsius.",
        "used_in_mvp": True,
        "notes": "Can influence tyre behavior, but not causal proof alone.",
    },
    {
        "field": "Humidity",
        "category": "weather",
        "plain_english_definition": "Relative humidity percentage.",
        "used_in_mvp": True,
        "notes": "Contextual weather input.",
    },
    {
        "field": "Pressure",
        "category": "weather",
        "plain_english_definition": "Atmospheric pressure.",
        "used_in_mvp": True,
        "notes": "Contextual weather input.",
    },
    {
        "field": "WindSpeed",
        "category": "weather",
        "plain_english_definition": "Wind speed.",
        "used_in_mvp": True,
        "notes": "Contextual weather input.",
    },
    {
        "field": "WindDirection",
        "category": "weather",
        "plain_english_definition": "Wind direction in degrees.",
        "used_in_mvp": False,
        "notes": "Displayed in dictionary; not modeled by default.",
    },
    {
        "field": "Rainfall",
        "category": "weather",
        "plain_english_definition": "Whether rainfall was recorded.",
        "used_in_mvp": True,
        "notes": "Used to flag wet-context laps.",
    },
    {
        "field": "Speed",
        "category": "telemetry",
        "plain_english_definition": "Car speed in telemetry samples.",
        "used_in_mvp": True,
        "notes": "Used in telemetry deep dives.",
    },
    {
        "field": "RPM",
        "category": "telemetry",
        "plain_english_definition": "Engine revolutions per minute.",
        "used_in_mvp": True,
        "notes": "Used in telemetry segment summaries.",
    },
    {
        "field": "nGear",
        "category": "telemetry",
        "plain_english_definition": "Selected gear number.",
        "used_in_mvp": True,
        "notes": "Summarized per telemetry segment.",
    },
    {
        "field": "Throttle",
        "category": "telemetry",
        "plain_english_definition": "Throttle application percentage.",
        "used_in_mvp": True,
        "notes": "Used to compare acceleration behavior.",
    },
    {
        "field": "Brake",
        "category": "telemetry",
        "plain_english_definition": "Brake signal, usually boolean or numeric.",
        "used_in_mvp": True,
        "notes": "Used to compare braking behavior.",
    },
    {
        "field": "DRS",
        "category": "telemetry",
        "plain_english_definition": "Drag Reduction System status.",
        "used_in_mvp": True,
        "notes": "Active states vary by FastF1 encoding.",
    },
    {
        "field": "Distance",
        "category": "telemetry",
        "plain_english_definition": "Estimated distance around the lap.",
        "used_in_mvp": True,
        "notes": "Used as the x-axis for telemetry plots.",
    },
    {
        "field": "SessionTime",
        "category": "timing",
        "plain_english_definition": "Timestamp relative to the session clock.",
        "used_in_mvp": True,
        "notes": "Used for weather joins and telemetry display.",
    },
]

GLOSSARY: list[dict[str, str]] = [
    {"term": "Grand Prix", "definition": "A Formula 1 race weekend held at a specific circuit."},
    {"term": "Session", "definition": "A practice, qualifying, sprint, or race period within a Grand Prix."},
    {"term": "FP1/FP2/FP3", "definition": "Free practice sessions used for setup, learning, and race preparation."},
    {"term": "Qualifying", "definition": "A timed session that decides the starting order for a race."},
    {"term": "Race", "definition": "The main competitive session where points are normally awarded."},
    {"term": "Sprint", "definition": "A shorter race format used at selected events."},
    {"term": "Lap", "definition": "One complete circuit around the track."},
    {"term": "Sector", "definition": "One of three timing sections that split a lap."},
    {"term": "Stint", "definition": "A continuous run on one tyre set between pit stops."},
    {"term": "Pit stop", "definition": "A visit to pit lane, usually for tyre changes or repairs."},
    {"term": "Tyre compound", "definition": "The tyre type, such as soft, medium, hard, intermediate, or wet."},
    {"term": "Tyre life", "definition": "Approximate number of laps completed on a tyre set."},
    {"term": "Push lap", "definition": "A lap where the driver appears to be driving near maximum pace."},
    {"term": "Cooldown lap", "definition": "A slower lap used to manage tyres, battery, or traffic."},
    {"term": "Track status", "definition": "Official status describing green, yellow, safety car, or red flag conditions."},
    {"term": "DRS", "definition": "A rear-wing system that reduces drag in designated zones when allowed."},
    {"term": "Speed trap", "definition": "A timing point that records car speed at a specific part of the circuit."},
    {"term": "Deleted lap", "definition": "A lap invalidated by officials, often for track limits."},
    {"term": "Safety car", "definition": "A controlled race condition that slows the field after an incident."},
    {"term": "Yellow flag", "definition": "A warning condition requiring drivers to slow or avoid hazards."},
    {"term": "Red flag", "definition": "A session stoppage, usually for safety or track condition reasons."},
]


def get_field_dictionary() -> pd.DataFrame:
    """Return the FastF1/F1 field dictionary used by the UI."""

    return pd.DataFrame(FIELD_DEFINITIONS)


def get_glossary() -> pd.DataFrame:
    """Return a beginner-friendly F1 glossary."""

    return pd.DataFrame(GLOSSARY)


def get_glossary_markdown() -> str:
    """Return the glossary as Markdown for the Gradio tab."""

    rows = [f"- **{row['term']}**: {row['definition']}" for row in GLOSSARY]
    return "# F1 Glossary\n\n" + "\n".join(rows)

