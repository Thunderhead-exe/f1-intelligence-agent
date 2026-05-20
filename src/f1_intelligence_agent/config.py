"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for local app execution."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    fastf1_cache_dir: Path = Field(default=Path(".cache/fastf1"), alias="FASTF1_CACHE_DIR")
    chroma_persist_dir: Path = Field(default=Path(".cache/chroma"), alias="CHROMA_PERSIST_DIR")
    memory_store_dir: Path = Field(default=Path(".cache/memory"), alias="MEMORY_STORE_DIR")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    openai_report_model: str = Field(default="gpt-4o-mini", alias="OPENAI_REPORT_MODEL")

    def ensure_cache_dirs(self) -> None:
        """Create local cache directories used by FastF1, Chroma, and memory storage."""

        for path in (self.fastf1_cache_dir, self.chroma_persist_dir, self.memory_store_dir):
            path.mkdir(parents=True, exist_ok=True)

    def require_openai_api_key(self) -> str:
        """Return the OpenAI key or raise a clear configuration error."""

        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for embeddings and guided report generation."
            )
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load environment-backed settings once per process."""

    load_dotenv()
    settings = Settings()
    settings.ensure_cache_dirs()
    return settings

