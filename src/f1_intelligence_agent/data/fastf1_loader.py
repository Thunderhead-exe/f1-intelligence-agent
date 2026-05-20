"""FastF1 session and telemetry loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from f1_intelligence_agent.data.schemas import SessionBundle
from f1_intelligence_agent.logging_utils import get_logger
from f1_intelligence_agent.utils.dataframe import ensure_columns

LOGGER = get_logger(__name__)

TELEMETRY_COLUMNS = [
    "Time",
    "SessionTime",
    "Distance",
    "Speed",
    "RPM",
    "nGear",
    "Throttle",
    "Brake",
    "DRS",
    "Source",
    "Driver",
    "LapNumber",
]


def _empty_bundle(
    year: int,
    grand_prix: str,
    session_type: str,
    drivers: list[str] | None,
    error: str,
) -> SessionBundle:
    """Build a failed-but-displayable session bundle."""

    return SessionBundle(
        year=year,
        grand_prix=grand_prix,
        session_type=session_type,
        laps=pd.DataFrame(),
        weather=pd.DataFrame(),
        race_control_messages=pd.DataFrame(),
        drivers=drivers or [],
        metadata={"error": error},
        raw_session=None,
        error=error,
    )


def _safe_frame(value: Any) -> pd.DataFrame:
    """Convert optional FastF1 data to a plain DataFrame."""

    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value).copy()
    except Exception:  # pragma: no cover - defensive against FastF1 object variants
        return pd.DataFrame()


def load_fastf1_session(
    year: int,
    grand_prix: str,
    session_type: str,
    drivers: list[str] | None = None,
    cache_dir: str | Path = ".cache/fastf1",
) -> SessionBundle:
    """Load a FastF1 session without loading full telemetry for every lap."""

    normalized_drivers = [driver.strip().upper() for driver in drivers or [] if driver.strip()]
    try:
        import fastf1
    except ImportError as exc:  # pragma: no cover - dependency installed in app env
        return _empty_bundle(year, grand_prix, session_type, normalized_drivers, str(exc))

    try:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_path))

        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=True)

        laps = _safe_frame(getattr(session, "laps", pd.DataFrame()))
        if normalized_drivers and "Driver" in laps.columns:
            laps = laps[laps["Driver"].astype(str).str.upper().isin(normalized_drivers)].copy()

        weather = _safe_frame(getattr(session, "weather_data", pd.DataFrame()))
        race_control_messages = _safe_frame(
            getattr(session, "race_control_messages", pd.DataFrame())
        )
        available_drivers = (
            sorted(laps["Driver"].dropna().astype(str).str.upper().unique().tolist())
            if "Driver" in laps.columns
            else normalized_drivers
        )

        event = getattr(session, "event", None)
        metadata = {
            "event_name": getattr(event, "EventName", None) if event is not None else None,
            "location": getattr(event, "Location", None) if event is not None else None,
            "country": getattr(event, "Country", None) if event is not None else None,
            "session_date": str(getattr(session, "date", "")),
            "lap_count": int(len(laps)),
            "driver_count": int(len(available_drivers)),
        }

        return SessionBundle(
            year=year,
            grand_prix=grand_prix,
            session_type=session_type,
            session_name=getattr(session, "name", None),
            laps=laps,
            weather=weather,
            race_control_messages=race_control_messages,
            drivers=available_drivers,
            metadata=metadata,
            raw_session=session,
        )
    except Exception as exc:
        message = f"FastF1 session load failed for {year} {grand_prix} {session_type}: {exc}"
        LOGGER.exception(message)
        return _empty_bundle(year, grand_prix, session_type, normalized_drivers, message)


def get_lap_telemetry(
    session_bundle: SessionBundle,
    driver: str,
    lap_number: int,
) -> pd.DataFrame:
    """Load telemetry for one driver-lap from an already loaded FastF1 session."""

    if session_bundle.raw_session is None:
        return pd.DataFrame(columns=TELEMETRY_COLUMNS)

    try:
        session_laps = getattr(session_bundle.raw_session, "laps", session_bundle.laps)
        telemetry = _extract_lap_telemetry(session_laps, driver, lap_number)
    except Exception as exc:
        if "has not been loaded" not in str(exc):
            LOGGER.warning("Telemetry load failed for %s lap %s: %s", driver, lap_number, exc)
            return pd.DataFrame(columns=TELEMETRY_COLUMNS)
        try:
            session_bundle.raw_session.load(
                laps=False,
                telemetry=True,
                weather=False,
                messages=False,
            )
            session_laps = getattr(session_bundle.raw_session, "laps", session_bundle.laps)
            telemetry = _extract_lap_telemetry(session_laps, driver, lap_number)
        except Exception as retry_exc:
            LOGGER.warning(
                "Telemetry load failed for %s lap %s after on-demand load: %s",
                driver,
                lap_number,
                retry_exc,
            )
            return pd.DataFrame(columns=TELEMETRY_COLUMNS)

    telemetry["Driver"] = driver.upper()
    telemetry["LapNumber"] = int(lap_number)
    telemetry["Source"] = "FastF1 car data"
    telemetry = ensure_columns(telemetry, TELEMETRY_COLUMNS)
    return telemetry[TELEMETRY_COLUMNS].copy()


def _extract_lap_telemetry(session_laps: Any, driver: str, lap_number: int) -> pd.DataFrame:
    """Extract one lap's car telemetry from a FastF1 laps object."""

    laps = session_laps
    if hasattr(laps, "pick_drivers"):
        laps = laps.pick_drivers([driver.upper()])
    elif "Driver" in laps.columns:
        laps = laps[laps["Driver"].astype(str).str.upper() == driver.upper()]
    if "LapNumber" in laps.columns:
        laps = laps[pd.to_numeric(laps["LapNumber"], errors="coerce") == int(lap_number)]
    if len(laps) == 0:
        return pd.DataFrame(columns=TELEMETRY_COLUMNS)

    lap = laps.iloc[0]
    telemetry = lap.get_car_data()
    if hasattr(telemetry, "add_distance"):
        telemetry = telemetry.add_distance()
    return pd.DataFrame(telemetry).copy()
