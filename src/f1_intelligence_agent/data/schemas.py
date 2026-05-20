"""Typed data structures for session loading and analysis configuration."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class SessionBundle(BaseModel):
    """Container for one loaded FastF1 session and the tables used by the MVP."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    year: int
    grand_prix: str
    session_type: str
    session_name: str | None = None
    laps: pd.DataFrame = Field(default_factory=pd.DataFrame)
    weather: pd.DataFrame = Field(default_factory=pd.DataFrame)
    race_control_messages: pd.DataFrame = Field(default_factory=pd.DataFrame)
    drivers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_session: Any | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the session loaded without a recoverable error."""

        return self.error is None

    @field_validator("drivers", mode="before")
    @classmethod
    def uppercase_drivers(cls, value: Any) -> list[str]:
        if value is None:
            return []
        return [str(driver).upper().strip() for driver in value if str(driver).strip()]


class AnalysisRunConfig(BaseModel):
    """User-selected options for a single Gradio/LangGraph analysis run."""

    year: int = 2024
    grand_prix: str = "Monza"
    session_type: str = "R"
    drivers: list[str] = Field(default_factory=list)
    analysis_mode: Literal[
        "Lap-level anomaly analysis", "Driver comparison", "Telemetry deep dive"
    ] = "Lap-level anomaly analysis"
    sensitivity: Literal["low", "medium", "high"] = "medium"
    top_n_anomalies: int = Field(
        default=5,
        validation_alias=AliasChoices("top_n_anomalies", "top_n"),
    )
    use_tavily: bool = False

    @property
    def top_n(self) -> int:
        """Compatibility alias used by graph nodes."""

        return self.top_n_anomalies

    @field_validator("drivers", mode="before")
    @classmethod
    def parse_drivers(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [part.strip().upper() for part in value.split(",") if part.strip()]
        return [str(part).strip().upper() for part in value if str(part).strip()]

    @field_validator("top_n_anomalies")
    @classmethod
    def clamp_top_n(cls, value: int) -> int:
        return max(1, min(int(value), 20))
