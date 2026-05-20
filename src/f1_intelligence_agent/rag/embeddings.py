"""OpenAI embeddings for Chroma."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from openai import OpenAI

from f1_intelligence_agent.config import get_settings


class OpenAIEmbeddingFunction:
    """Chroma-compatible embedding function backed by OpenAI embeddings."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.openai_embedding_model
        self.client = OpenAI(api_key=api_key or settings.require_openai_api_key())

    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002 - Chroma API name
        """Embed a sequence of strings."""

        texts = [str(text) for text in input]
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [item.embedding for item in response.data]
        return [_normalize(vector) for vector in vectors]

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002 - Chroma API name
        """Embed query text for newer Chroma versions."""

        return self(input)

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002 - Chroma API name
        """Embed document text for newer Chroma versions."""

        return self(input)

    def name(self) -> str:
        """Return a stable Chroma embedding function name."""

        return f"openai-{self.model}"

    def get_config(self) -> dict[str, str]:
        """Return Chroma embedding-function configuration metadata."""

        return {"model": self.model}

    def validate_config(self, config: dict[str, str]) -> None:
        """Chroma compatibility hook for persisted embedding function config."""

        return None


def _normalize(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm == 0:
        return array.tolist()
    return (array / norm).tolist()
