"""Context retrieval for generated insights."""

from __future__ import annotations

from f1_intelligence_agent.agents.report_schemas import InsightCandidate
from f1_intelligence_agent.rag.vector_store import (
    MEMORY_COLLECTION,
    RetrievedDocument,
    VectorStoreManager,
)


def retrieve_context_for_insight(
    insight: InsightCandidate,
    vector_store: VectorStoreManager,
    k: int = 5,
) -> list[RetrievedDocument]:
    """Retrieve curated knowledge and similar validated memories for one insight."""

    parts = [
        insight.title,
        insight.summary,
        " ".join(insight.possible_explanations),
        " ".join(item.description for item in insight.evidence),
    ]
    query = " ".join(part for part in parts if part)
    knowledge = vector_store.query(query, k=k)
    memories = vector_store.query(query, k=max(1, min(3, k)), collection_name=MEMORY_COLLECTION)
    return [*knowledge, *memories]

