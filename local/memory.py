"""
Rio Local — RAG Memory (L5)
ChromaDB + sentence-transformers (bge-small-en-v1.5) for cross-session recall.

Stores summaries of meaningful interactions and retrieves relevant past context
on struggle triggers or user questions.

Graceful degradation: if chromadb or sentence-transformers are not installed,
the module exposes MemoryStore = None so callers can skip memory features.
"""
from __future__ import annotations

import hashlib
import time
import os
from dataclasses import dataclass, field
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ── Optional dependency imports (graceful degradation) ────────────────
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None
    ChromaSettings = None
    log.warning("chromadb not installed — memory features disabled")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    log.warning("sentence-transformers not installed — memory features disabled")


# ── Constants ─────────────────────────────────────────────────────────
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "memory")
DEFAULT_COLLECTION = "rio_interactions"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_RECALL = 5

# Minimum content length to store (skip trivial exchanges)
MIN_CONTENT_LENGTH = 20

# ── Data types ────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single stored memory entry."""
    id: str
    content: str
    timestamp: float
    entry_type: str  # "interaction", "error_fix", "struggle", "tool_use"
    metadata: dict = field(default_factory=dict)
    distance: float = 0.0  # similarity distance (lower = more similar)


class MemoryStore:
    """
    Persistent RAG memory using ChromaDB for vector storage and
    bge-small-en-v1.5 for embeddings.

    Usage:
        store = MemoryStore(db_path="./rio_memory", max_recall=5)
        store.add("User fixed NoneType in auth.py line 47",
                   entry_type="error_fix",
                   metadata={"file": "auth.py", "error": "NoneType"})

        results = store.query("NoneType error in authentication", top_k=5)
        for entry in results:
            print(entry.content, entry.distance)
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL_NAME,
        max_recall: int = MAX_RECALL,
    ):
        if chromadb is None or SentenceTransformer is None:
            raise RuntimeError(
                "Memory requires chromadb and sentence-transformers. "
                "Install with: pip install chromadb sentence-transformers"
            )

        self.db_path = os.path.abspath(db_path)
        self.collection_name = collection_name
        self.max_recall = max_recall
        self._model_name = model_name

        # ── Lazy-load embedding model (deferred to first use) ─────
        # The SentenceTransformer model is ~130 MB in RAM.  Loading at
        # startup delays readiness.  We defer to the first add/query call
        # so the process starts faster and uses less RAM when memory
        # features aren't immediately needed.
        self._embedder = None  # type: ignore[assignment]

        # ── Initialize ChromaDB ───────────────────────────────────
        os.makedirs(self.db_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.db_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

        # ── A2: FTS5 keyword index for hybrid search ─────────────
        import sqlite3
        self._fts_db_path = os.path.join(self.db_path, "fts_index.db")
        self._fts_conn = sqlite3.connect(self._fts_db_path, check_same_thread=False)
        self._fts_conn.execute("PRAGMA journal_mode=WAL")
        self._fts_conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5"
            "(entry_id, content, entry_type, tokenize='porter unicode61')"
        )
        self._fts_conn.commit()

        count = self._collection.count()
        log.info("memory.ready", db_path=self.db_path, entries=count)

    # ── Lazy embedding model loader ───────────────────────────────

    def _get_embedder(self):
        """Load the SentenceTransformer model on first use."""
        if self._embedder is None:
            log.info("memory.loading_model", model=self._model_name)
            self._embedder = SentenceTransformer(self._model_name)
            log.info("memory.model_loaded", model=self._model_name)
        return self._embedder

    # ── Store ─────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        entry_type: str = "interaction",
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Embed and store a memory entry.

        Args:
            content: The text to store (summary of interaction/fix/error).
            entry_type: Category — "interaction", "error_fix", "struggle", "tool_use".
            metadata: Optional dict with extra info (file, error_class, etc.).

        Returns:
            The generated entry ID, or None if content is too short.
        """
        if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
            log.debug("memory.skip_short", length=len(content) if content else 0)
            return None

        content = content.strip()
        now = time.time()

        # Deduplicate: hash content to avoid storing exact duplicates
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        entry_id = f"{entry_type}_{content_hash}_{int(now)}"

        # Build metadata
        meta = {
            "entry_type": entry_type,
            "timestamp": now,
            "content_hash": content_hash,
        }
        if metadata:
            # ChromaDB metadata values must be str, int, float, or bool
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif isinstance(v, list):
                    meta[k] = ", ".join(str(item) for item in v)
                else:
                    meta[k] = str(v)

        # Check for exact duplicate (same content hash already stored)
        try:
            existing = self._collection.get(
                where={"content_hash": content_hash},
                limit=1,
            )
            if existing and existing["ids"]:
                log.debug("memory.duplicate_skipped", hash=content_hash)
                return None
        except Exception:
            pass  # If where filter fails, just proceed

        # Embed
        embedding = self._get_embedder().encode(content).tolist()

        # Store
        self._collection.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta],
        )

        # A2: Also index in FTS5 for keyword search
        try:
            self._fts_conn.execute(
                "INSERT OR IGNORE INTO memory_fts(entry_id, content, entry_type) VALUES (?, ?, ?)",
                (entry_id, content, entry_type),
            )
            self._fts_conn.commit()
        except Exception:
            pass  # Non-critical

        log.info("memory.stored",
                 id=entry_id,
                 type=entry_type,
                 length=len(content))
        return entry_id

    # ── Retrieve ──────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        entry_type: Optional[str] = None,
    ) -> list[MemoryEntry]:
        """
        Query for similar past entries.

        Args:
            query_text: The text to search for (current error, question, etc.).
            top_k: Number of results to return (default: self.max_recall).
            entry_type: Optional filter by entry type.

        Returns:
            List of MemoryEntry sorted by relevance (most similar first).
        """
        if not query_text or not query_text.strip():
            return []

        k = top_k or self.max_recall
        count = self._collection.count()
        if count == 0:
            return []

        # Don't request more than available
        k = min(k, count)

        # Embed query
        query_embedding = self._get_embedder().encode(query_text.strip()).tolist()

        # Build where filter
        where_filter = None
        if entry_type:
            where_filter = {"entry_type": entry_type}

        # Query ChromaDB
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.error("memory.query_failed", err=str(exc))
            return []

        # Parse results
        entries = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = results["documents"][0][i] if results["documents"] else ""
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0.0

                entries.append(MemoryEntry(
                    id=doc_id,
                    content=doc,
                    timestamp=meta.get("timestamp", 0.0),
                    entry_type=meta.get("entry_type", "unknown"),
                    metadata=meta,
                    distance=dist,
                ))

        log.info("memory.queried",
                 query_len=len(query_text),
                 results=len(entries))
        return entries

    # ── Utility ───────────────────────────────────────────────────

    def keyword_search(self, query_text: str, top_k: int = 5) -> list[MemoryEntry]:
        """A2: BM25 keyword search via FTS5."""
        if not query_text or not query_text.strip():
            return []
        try:
            cursor = self._fts_conn.execute(
                "SELECT entry_id, content, entry_type, rank "
                "FROM memory_fts WHERE memory_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (query_text.strip(), top_k),
            )
            entries = []
            for row in cursor.fetchall():
                entries.append(MemoryEntry(
                    id=row[0],
                    content=row[1],
                    timestamp=0.0,
                    entry_type=row[2],
                    distance=abs(row[3]),  # FTS5 rank is negative
                ))
            return entries
        except Exception:
            return []

    def hybrid_query(
        self,
        query_text: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[MemoryEntry]:
        """A2: Hybrid search combining vector similarity + BM25 keyword.

        Scores are normalised 0-1 (higher=better) then weighted-averaged.
        """
        if not query_text or not query_text.strip():
            return []

        # Vector results
        vector_results = self.query(query_text, top_k=top_k * 2)
        # Keyword results
        keyword_results = self.keyword_search(query_text, top_k=top_k * 2)

        # Build score maps (normalised 0-1, higher=better)
        scores: dict[str, dict] = {}

        for entry in vector_results:
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            vec_score = max(0.0, 1.0 - entry.distance)
            scores[entry.id] = {
                "entry": entry,
                "vector": vec_score,
                "keyword": 0.0,
            }

        for entry in keyword_results:
            # FTS5 rank: lower abs = better match; normalise loosely
            kw_score = 1.0 / (1.0 + entry.distance)
            if entry.id in scores:
                scores[entry.id]["keyword"] = kw_score
            else:
                scores[entry.id] = {
                    "entry": entry,
                    "vector": 0.0,
                    "keyword": kw_score,
                }

        # Weighted combination
        ranked: list[tuple[float, MemoryEntry]] = []
        for sid, data in scores.items():
            combined = (
                vector_weight * data["vector"]
                + keyword_weight * data["keyword"]
            )
            entry = data["entry"]
            entry.distance = 1.0 - combined  # Store as distance for compatibility
            ranked.append((combined, entry))

        ranked.sort(key=lambda x: -x[0])
        return [entry for _, entry in ranked[:top_k]]

    def count(self) -> int:
        """Return total number of stored entries."""
        return self._collection.count()

    def format_context(self, entries: list[MemoryEntry]) -> str:
        """
        Format retrieved entries as context text for injection into Gemini.

        Returns a string like:
            Past similar issues:
            1. [2 days ago] User fixed NoneType in auth.py line 47 (error_fix)
            2. [5 days ago] User debugged import error in utils.py (interaction)
        """
        if not entries:
            return ""

        lines = ["Past similar issues:"]
        now = time.time()
        for i, entry in enumerate(entries, 1):
            age = now - entry.timestamp
            age_str = _format_age(age)
            lines.append(
                f"  {i}. [{age_str}] {entry.content} ({entry.entry_type})"
            )
        return "\n".join(lines)

    def build_interaction_summary(
        self,
        user_text: str,
        rio_text: str,
        tool_calls: Optional[list[dict]] = None,
    ) -> Optional[str]:
        """
        Build a summary string from an interaction for storage.

        Returns None if the interaction is too trivial to store.
        """
        # Skip trivial interactions
        if len(user_text) < 10 and len(rio_text) < 20:
            return None

        parts = []

        # Include tool usage if any
        if tool_calls:
            tool_names = [tc.get("name", "unknown") for tc in tool_calls]
            parts.append(f"Tools used: {', '.join(tool_names)}.")

        # Truncate for embedding efficiency (keep first 500 chars)
        user_snippet = user_text[:250].strip()
        rio_snippet = rio_text[:250].strip()

        if user_snippet:
            parts.append(f"User asked: {user_snippet}")
        if rio_snippet:
            parts.append(f"Rio responded: {rio_snippet}")

        summary = " ".join(parts)
        if len(summary) < MIN_CONTENT_LENGTH:
            return None

        return summary

    def clear(self):
        """Clear all stored memories. Use with caution."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("memory.cleared")


# ── Helpers ───────────────────────────────────────────────────────────

def _format_age(seconds: float) -> str:
    """Format an age in seconds to a human-readable string."""
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins}m ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    else:
        days = int(seconds / 86400)
        return f"{days}d ago"