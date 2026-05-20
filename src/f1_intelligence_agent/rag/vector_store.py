"""Chroma vector store manager for local knowledge and validated insight memory."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from f1_intelligence_agent.config import get_settings
from f1_intelligence_agent.rag.embeddings import OpenAIEmbeddingFunction

KNOWLEDGE_COLLECTION = "f1_knowledge"
MEMORY_COLLECTION = "validated_insights"
DOCUMENT_DIR = Path(__file__).parent / "documents"


@dataclass(frozen=True)
class RetrievedDocument:
    """A document retrieved from Chroma."""

    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None
    collection: str | None = None


class VectorStoreManager:
    """Manage Chroma collections for curated knowledge and validated insights."""

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_function = _EmbeddingFunctionAdapter(OpenAIEmbeddingFunction())
        self._ensure_chroma_embedding_interface()
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.knowledge_collection = self.client.get_or_create_collection(
            KNOWLEDGE_COLLECTION,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self.memory_collection = self.client.get_or_create_collection(
            MEMORY_COLLECTION,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def ensure_knowledge_base(self) -> None:
        """Index or refresh local Markdown knowledge docs."""
        documents: list[dict[str, Any]] = []
        for path in sorted(DOCUMENT_DIR.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = path.stem.replace("_", " ").title()
            category = path.stem
            for index, chunk in enumerate(_chunk_text(text)):
                documents.append(
                    {
                        "id": _stable_id(f"{path.name}:{index}:{chunk[:80]}"),
                        "text": chunk,
                        "metadata": {
                            "source": path.name,
                            "category": category,
                            "title": title,
                            "chunk": index,
                        },
                    }
                )
        if documents:
            self.add_documents(documents, collection_name=KNOWLEDGE_COLLECTION)

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        collection_name: str = KNOWLEDGE_COLLECTION,
    ) -> None:
        """Add text documents to a Chroma collection."""

        if not documents:
            return
        collection = self._collection(collection_name)
        ids = [str(doc.get("id") or _stable_id(doc.get("text", ""))) for doc in documents]
        texts = [str(doc.get("text") or doc.get("page_content") or "") for doc in documents]
        metadatas = [_sanitize_metadata(doc.get("metadata", {})) for doc in documents]
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def query(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        collection_name: str = KNOWLEDGE_COLLECTION,
    ) -> list[RetrievedDocument]:
        """Query a collection and return normalized retrieved documents."""

        collection = self._collection(collection_name)
        if collection.count() == 0:
            return []
        result = collection.query(
            query_texts=[query],
            n_results=max(1, int(k)),
            where=_sanitize_metadata(filters) if filters else None,
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else [None] * len(ids)
        return [
            RetrievedDocument(
                id=str(doc_id),
                text=str(text),
                metadata=dict(metadata or {}),
                distance=float(distance) if distance is not None else None,
                collection=collection_name,
            )
            for doc_id, text, metadata, distance in zip(ids, docs, metadatas, distances, strict=False)
        ]

    def _collection(self, collection_name: str):
        if collection_name == MEMORY_COLLECTION:
            return self.memory_collection
        return self.knowledge_collection

    def _ensure_chroma_embedding_interface(self) -> None:
        if not hasattr(self.embedding_function, "name"):
            self.embedding_function.name = lambda: "default"
        if not hasattr(self.embedding_function, "embed_query"):
            self.embedding_function.embed_query = lambda input: self.embedding_function(input)
        if not hasattr(self.embedding_function, "embed_documents"):
            self.embedding_function.embed_documents = lambda input: self.embedding_function(input)
        if not hasattr(self.embedding_function, "is_legacy"):
            self.embedding_function.is_legacy = lambda: False
        if not hasattr(self.embedding_function, "supported_spaces"):
            self.embedding_function.supported_spaces = lambda: ["cosine", "l2", "ip"]
        if not hasattr(self.embedding_function, "get_config"):
            self.embedding_function.get_config = lambda: {}
        if not hasattr(self.embedding_function, "validate_config"):
            self.embedding_function.validate_config = lambda config: None


def _chunk_text(text: str, max_chars: int = 900) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


class _EmbeddingFunctionAdapter:
    """Adapter for Chroma versions that call embed_query/embed_documents."""

    def __init__(self, embedding_function: Any) -> None:
        self.embedding_function = embedding_function

    def __call__(self, input):  # noqa: A002 - Chroma API name
        return self.embedding_function(input)

    def embed_documents(self, input):  # noqa: A002 - Chroma API name
        return self.embedding_function(input)

    def embed_query(self, input):  # noqa: A002 - Chroma API name
        return self.embedding_function(input)


def _stable_id(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]


def _sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            sanitized[str(key)] = value
        else:
            sanitized[str(key)] = str(value)
    return sanitized
