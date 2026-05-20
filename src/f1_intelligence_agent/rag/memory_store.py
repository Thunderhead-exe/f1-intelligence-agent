"""Human-in-the-loop insight memory persistence."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from f1_intelligence_agent.agents.report_schemas import InsightCandidate
from f1_intelligence_agent.config import get_settings
from f1_intelligence_agent.rag.vector_store import (
    MEMORY_COLLECTION,
    RetrievedDocument,
    VectorStoreManager,
)


class InsightMemoryStore:
    """Store validated reusable insight patterns in JSONL and Chroma."""

    def __init__(
        self,
        vector_store: VectorStoreManager | None = None,
        memory_dir: str | Path | None = None,
    ) -> None:
        settings = get_settings()
        self.vector_store = vector_store or VectorStoreManager()
        self.memory_dir = Path(memory_dir or settings.memory_store_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.memory_dir / "validated_insights.jsonl"

    def propose_memory(self, insight: InsightCandidate) -> dict[str, Any]:
        """Create an editable memory proposal from a structured insight."""

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        feature_evidence = {
            item.description.split(":")[0][:48]: str(item.value)
            for item in insight.evidence
            if item.evidence_type in {"lap_feature", "telemetry", "cluster"}
        }
        return {
            "memory_id": memory_id,
            "pattern_name": _slug(insight.title),
            "scope": "driver_lap_segment" if insight.lap_number is not None else "session_pattern",
            "features": feature_evidence,
            "hypothesis": insight.possible_explanations[0]
            if insight.possible_explanations
            else insight.summary,
            "evidence": [item.description for item in insight.evidence[:6]],
            "confidence": insight.confidence,
            "validated_by_user": False,
            "created_at": datetime.now(UTC).isoformat(),
            "source_insight_id": insight.id,
            "driver": insight.driver,
            "lap_number": insight.lap_number,
        }

    def approve_memory(self, memory: dict[str, Any], edited_text: str | None = None) -> str:
        """Approve and persist a memory, optionally parsing edited JSON or text."""

        approved = dict(memory)
        if edited_text:
            edited_text = edited_text.strip()
            if edited_text:
                try:
                    parsed = json.loads(edited_text)
                    if isinstance(parsed, dict):
                        approved.update(parsed)
                    else:
                        approved["hypothesis"] = str(parsed)
                except json.JSONDecodeError:
                    approved["hypothesis"] = edited_text

        approved["validated_by_user"] = True
        approved["created_at"] = approved.get("created_at") or datetime.now(UTC).isoformat()
        approved["memory_id"] = approved.get("memory_id") or f"mem_{uuid.uuid4().hex[:12]}"
        text = _memory_to_text(approved)

        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(approved, sort_keys=True) + "\n")

        self.vector_store.add_documents(
            [
                {
                    "id": str(approved["memory_id"]),
                    "text": text,
                    "metadata": {
                        "source": "validated_insights.jsonl",
                        "category": "validated_memory",
                        "title": approved.get("pattern_name", "validated memory"),
                        "confidence": approved.get("confidence", "medium"),
                        "driver": approved.get("driver") or "",
                    },
                }
            ],
            collection_name=MEMORY_COLLECTION,
        )
        return str(approved["memory_id"])

    def reject_memory(self, memory_id: str) -> None:
        """Rejecting a proposal has no persistence side effect in the MVP."""

        return None

    def query_similar_memories(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        """Retrieve similar validated memories."""

        return self.vector_store.query(query, k=k, collection_name=MEMORY_COLLECTION)


def _memory_to_text(memory: dict[str, Any]) -> str:
    evidence = "; ".join(str(item) for item in memory.get("evidence", []))
    features = json.dumps(memory.get("features", {}), sort_keys=True)
    return (
        f"Pattern: {memory.get('pattern_name')}. "
        f"Hypothesis: {memory.get('hypothesis')}. "
        f"Features: {features}. Evidence: {evidence}. "
        f"Confidence: {memory.get('confidence')}."
    )


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:80] or "insight_pattern"
