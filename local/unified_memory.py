"""
Unified Memory Facade — single interface for all Rio memory systems.

Merges results from:
  1. ChromaDB vector store (MemoryStore) — semantic similarity
  2. Session notes (in-memory dict) — keyword match
  3. Chat history (ChatStore) — full-text match

Exposed as a single ``search()`` call for the orchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class MemoryResult:
    """A single search result from any memory source."""
    key: str
    content: str
    score: float              # Normalised 0-1 (higher = better)
    source: str               # "vector" | "session_notes" | "chat_history"
    entry_type: str = ""
    timestamp: float = 0.0


class UnifiedMemory:
    """Facade that queries all memory stores and merges / deduplicates."""

    def __init__(
        self,
        memory_store: Any | None = None,
        session_notes: dict[str, str] | None = None,
        chat_store: Any | None = None,
    ) -> None:
        self._vector = memory_store        # MemoryStore (ChromaDB)
        self._notes = session_notes if session_notes is not None else {}
        self._chat = chat_store            # ChatStore (SQLite)

    # Allow the orchestrator to swap the notes dict reference
    def set_notes(self, notes: dict[str, str]) -> None:
        self._notes = notes

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[MemoryResult]:
        """Search across all memory stores, merge and deduplicate."""
        if not query.strip():
            return []

        results: list[MemoryResult] = []

        # 1. Hybrid search (vector + BM25) when available, else vector only
        if self._vector is not None:
            try:
                if hasattr(self._vector, "hybrid_query"):
                    entries = self._vector.hybrid_query(query, top_k=limit)
                else:
                    entries = self._vector.query(query, top_k=limit)
                for e in entries:
                    results.append(MemoryResult(
                        key=e.id,
                        content=e.content[:500],
                        score=max(0.0, 1.0 - e.distance),
                        source="vector",
                        entry_type=e.entry_type,
                        timestamp=e.timestamp,
                    ))
            except Exception as exc:
                log.debug("unified_memory.vector_error", error=str(exc))

        # 2. Keyword search on session notes
        keywords = query.lower().split()
        for k, v in self._notes.items():
            text = f"{k} {v}".lower()
            match_count = sum(1 for kw in keywords if kw in text)
            if match_count > 0:
                score = min(1.0, match_count / max(len(keywords), 1))
                results.append(MemoryResult(
                    key=k,
                    content=v[:500],
                    score=score,
                    source="session_notes",
                ))

        # 3. Full-text search on chat history
        if self._chat is not None:
            try:
                messages = self._chat.search(query, limit=limit)
                for msg in messages:
                    results.append(MemoryResult(
                        key=f"chat_{msg.id}" if hasattr(msg, "id") else f"chat_{id(msg)}",
                        content=(msg.content if hasattr(msg, "content") else str(msg))[:500],
                        score=0.5,
                        source="chat_history",
                        timestamp=msg.timestamp if hasattr(msg, "timestamp") else 0.0,
                    ))
            except Exception as exc:
                log.debug("unified_memory.chat_error", error=str(exc))

        # Deduplicate by content prefix
        seen: set[str] = set()
        unique: list[MemoryResult] = []
        for r in results:
            content_key = r.content[:100].lower().strip()
            if content_key and content_key not in seen:
                seen.add(content_key)
                unique.append(r)

        unique.sort(key=lambda r: -r.score)
        return unique[:limit]

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_note(self, key: str, value: str, persist_vector: bool = False, media_parts: list[Any] | None = None) -> None:
        """Save to session notes (and optionally to long-term vector store).

        Caps the in-memory dict at 200 entries to bound RAM usage.
        """
        _MAX_NOTES = 200
        if len(self._notes) >= _MAX_NOTES and key not in self._notes:
            # Evict the oldest entry (first inserted)
            try:
                oldest_key = next(iter(self._notes))
                del self._notes[oldest_key]
            except StopIteration:
                pass
        self._notes[key] = value
        if persist_vector and self._vector is not None:
            try:
                self._vector.add(
                    f"{key}: {value}",
                    entry_type="user_note",
                    metadata={"key": key},
                    media_parts=media_parts,
                )
            except Exception:
                pass

    def get_note(self, key: str) -> str | None:
        return self._notes.get(key)

    def get_all_notes(self) -> dict[str, str]:
        return dict(self._notes)

    def format_search_results(self, results: list[MemoryResult]) -> str:
        """Format results for injection into the model prompt."""
        if not results:
            return ""
        lines = ["=== MEMORY RECALL ==="]
        for r in results:
            tag = f"[{r.source}]"
            lines.append(f"{tag} {r.key}: {r.content}")
        lines.append("=== END MEMORY ===")
        return "\n".join(lines)
