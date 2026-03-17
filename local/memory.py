"""
Rio Local — RAG Memory (L5)
ChromaDB + Gemini Embeddings for cross-session recall.

Stores summaries of meaningful interactions and retrieves relevant past context
on struggle triggers or user questions.

Replaced local sentence-transformers (bge-small-en-v1.5) with Gemini API
for lower RAM usage and better semantic recall.
"""
from __future__ import annotations

import hashlib
import time
import os
from dataclasses import dataclass, field
from typing import Optional, Any

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
    from google import genai
except ImportError:
    genai = None
    log.warning("google-genai not installed — memory features disabled")


# ── Constants ─────────────────────────────────────────────────────────
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "memory")
# Dimension change (768 -> 3072) requires a new v2 collection
DEFAULT_COLLECTION = "rio_interactions_gemini_v2"
# Prefer a stable default unless explicitly overridden via env.
DEFAULT_MODEL_NAME = os.environ.get("RIO_EMBEDDING_MODEL", "gemini-embedding-001")
MAX_RECALL = 8 # Higher recall possible with 3072 dims

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


import re

# ── Entity Extraction (Graph-RAG Lite) ──────────────────────────────
class EntityExtractor:
    """Extracts filenames, project names, and key terms for cross-linking."""
    
    # Matches: main.py, src/auth.py, .env, package.json
    FILE_PATTERN = re.compile(r'\b[\w\-\.\/]+\.(?:py|js|ts|json|md|txt|html|css|yaml|yml|env|sh|bat)\b')
    # Matches: project-name, Project_X (capitalized or hyphenated/underscored words)
    PROJECT_PATTERN = re.compile(r'\b[A-Z][\w\-]{3,}\b|\b[\w\-]{3,}-project\b')

    @classmethod
    def extract(cls, text: str) -> list[str]:
        entities = set()
        # Find files
        for match in cls.FILE_PATTERN.finditer(text):
            entities.add(match.group(0).lower())
        # Find potential project names
        for match in cls.PROJECT_PATTERN.finditer(text):
            entities.add(match.group(0).lower())
        return sorted(list(entities))


class MemoryStore:
    """
    Persistent RAG memory using ChromaDB for vector storage and
    Gemini embedding models for embeddings.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL_NAME,
        max_recall: int = MAX_RECALL,
        genai_client: Optional[Any] = None,
    ):
        if chromadb is None:
            raise RuntimeError("Memory requires chromadb. Install with: pip install chromadb")

        self.db_path = os.path.abspath(db_path)
        self.collection_name = collection_name
        self.max_recall = max_recall
        self._model_name = model_name
        self._notes_kv: dict[str, str] = {}
        # Fallbacks for accounts/regions where a model is unavailable.
        # Keep order deterministic and prefer stable over preview.
        self._embedding_model_fallbacks = [
            "gemini-embedding-2-preview",
            "text-embedding-004",
            "gemini-embedding-001",
        ]
        self._client_ref = genai_client
        self._vertex_client = None
        self._use_vertex_for_embeddings = os.environ.get(
            "RIO_MEMORY_USE_VERTEX",
            os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", ""),
        ).strip().lower() in {"1", "true", "yes", "on"}
        gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if gcp_project and self._use_vertex_for_embeddings:
            try:
                from google import genai as _genai
                self._vertex_client = _genai.Client(
                    vertexai=True,
                    project=gcp_project,
                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
                )
                log.info("memory.vertex_client_initialized", project=gcp_project)
            except Exception as e:
                log.debug("memory.vertex_client_failed", error=str(e))
        elif gcp_project and not self._use_vertex_for_embeddings:
            log.info(
                "memory.vertex_client_skipped",
                project=gcp_project,
                reason="RIO_MEMORY_USE_VERTEX/GOOGLE_GENAI_USE_VERTEXAI not enabled",
            )

        self._extractor = EntityExtractor()

        # ── Initialize ChromaDB ───────────────────────────────────
        os.makedirs(self.db_path, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=self.db_path)
        self._collection = self._chroma.get_or_create_collection(
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
        log.info("memory.ready", 
                 db_path=self.db_path, 
                 entries=count, 
                 model=self._model_name,
                 collection=self.collection_name)

    # ── Backward-compatible Notes API ─────────────────────────────

    def save_note(
        self,
        key: str,
        value: str,
        persist_vector: bool = True,
        media_parts: Optional[list[Any]] = None,
    ) -> bool:
        """Compatibility layer for orchestrator memory tools.

        Stores key/value notes in-memory for fast retrieval and optionally
        persists them to the vector store for cross-session recall.
        """
        k = (key or "").strip()
        v = (value or "").strip()
        if not k or not v:
            return False

        self._notes_kv[k] = v

        if persist_vector:
            note_text = f"{k}: {v}"
            try:
                self.add(
                    content=note_text,
                    entry_type="note",
                    metadata={"note_key": k, "note_value": v},
                    media_parts=media_parts,
                )
            except Exception as exc:
                log.warning("memory.save_note.persist_failed", key=k, error=str(exc))

        return True

    def get_notes(self, key: str = "") -> dict[str, str] | str | None:
        """Compatibility layer for orchestrator memory tools.

        Returns a single value for a key when provided, otherwise returns all
        notes captured in this process.
        """
        k = (key or "").strip()
        if k:
            return self._notes_kv.get(k)
        return dict(self._notes_kv)

    def set_client(self, client: Any) -> None:
        """Dynamically update the GenAI client reference."""
        self._client_ref = client

    # ── Embedding logic (Gemini API) ───────────────────────────────

    def _get_embedding(self, contents: Any, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Call Gemini API to generate embeddings for multimodal contents."""
        if not getattr(self, "_client_ref", None) and not getattr(self, "_vertex_client", None):
            # Fallback to dummy zero vector if no client (should not happen in prod)
            log.warning("memory.no_client_for_embedding")
            return [0.0] * 3072

        candidate_models: list[str] = [self._model_name]
        for fallback in self._embedding_model_fallbacks:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_exc: Exception | None = None
        for idx, model_name in enumerate(candidate_models):
            try:
                # 3072 is specific to gemini-embedding-2. text-embedding-004 defaults to 768.
                config = {"task_type": task_type}
                if "gemini-embedding-2" in model_name:
                    config["output_dimensionality"] = 3072
                elif "text-embedding-004" in model_name:
                    config["output_dimensionality"] = 768
                
                # Use synchronous embed_content if client is synchronous, or wrap it.
                # adk_server uses the sync client for orchestrator tasks sometimes.
                # Here we use the standard sync client pattern.
                active_client = getattr(self, "_client_ref", None)
                if "gemini-embedding-2" in model_name and getattr(self, "_vertex_client", None):
                    active_client = self._vertex_client
                    
                if active_client is None:
                    raise ValueError(f"No valid client available for model {model_name}")

                result = active_client.models.embed_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                
                # Persist the working model to avoid repeated failed attempts.
                if model_name != self._model_name:
                    prev = self._model_name
                    self._model_name = model_name
                    log.warning(
                        "memory.embedding_model_switched",
                        from_model=prev,
                        to_model=model_name,
                    )
                    
                # Pad to 3072 if we fell back to a 768-dim model, 
                # because the ChromaDB collection was initialized with 3072.
                # ChromaDB requires all vectors in a collection to have the same length.
                vals = result.embeddings[0].values
                if len(vals) < 3072:
                    vals = vals + [0.0] * (3072 - len(vals))
                    
                return vals
            except Exception as exc:
                last_exc = exc
                err = str(exc)
                retriable = (
                    "NOT_FOUND" in err
                    or "not found" in err.lower()
                    or "INVALID_ARGUMENT" in err
                    or "not supported" in err.lower()
                    or "404" in err
                )
                log.error(
                    "memory.embedding_attempt_failed",
                    model=model_name,
                    attempt=idx + 1,
                    error=err,
                )
                if not retriable:
                    break

        log.error("memory.embedding_failed", error=str(last_exc) if last_exc else "unknown")
        return [0.0] * 3072

    # ── Store ─────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        entry_type: str = "interaction",
        metadata: Optional[dict] = None,
        media_parts: Optional[list[Any]] = None,
    ) -> Optional[str]:
        """
        Embed and store a memory entry, optionally with multimodal media parts.
        """
        if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
            log.debug("memory.skip_short", length=len(content) if content else 0)
            return None

        content = content.strip()
        now = time.time()

        # Extract entities for Graph-RAG (linking)
        entities = self._extractor.extract(content)
        entities_str = ";".join(entities) if entities else ""

        # Deduplicate: hash content to avoid storing exact duplicates
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        entry_id = f"{entry_type}_{content_hash}_{int(now)}"

        # Build metadata
        meta = {
            "entry_type": entry_type,
            "timestamp": now,
            "content_hash": content_hash,
            "entities": entities_str,
            "has_media": bool(media_parts),
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)

        # Check for exact duplicate
        try:
            existing = self._collection.get(
                where={"content_hash": content_hash},
                limit=1,
            )
            if existing and existing["ids"]:
                log.debug("memory.duplicate_skipped", hash=content_hash)
                return None
        except Exception:
            pass

        # Prepare multimodal contents for embedding
        embed_contents = [content]
        if media_parts:
            embed_contents.extend(media_parts)

        # Embed using RETRIEVAL_DOCUMENT task type
        embedding = self._get_embedding(embed_contents, task_type="RETRIEVAL_DOCUMENT")

        # Store in ChromaDB (only the text content is stored as the document for hybrid search,
        # but the embedding represents the multimodal combination)
        self._collection.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta],
        )

        # Index in FTS5
        try:
            self._fts_conn.execute(
                "INSERT OR IGNORE INTO memory_fts(entry_id, content, entry_type) VALUES (?, ?, ?)",
                (entry_id, content, entry_type),
            )
            self._fts_conn.commit()
        except Exception:
            pass

        log.info("memory.stored", id=entry_id, type=entry_type, length=len(content), has_media=bool(media_parts))
        return entry_id

    # ── Retrieve ──────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        entry_type: Optional[str] = None,
        include_entities: bool = True,
    ) -> list[MemoryEntry]:
        """
        Query for similar past entries.
        """
        if not query_text or not query_text.strip():
            return []

        k = top_k or self.max_recall
        count = self._collection.count()
        if count == 0:
            return []

        k = min(k, count)

        # Embed query using RETRIEVAL_QUERY task type
        query_embedding = self._get_embedding(query_text.strip(), task_type="RETRIEVAL_QUERY")

        where_filter: dict[str, Any] = {}
        if entry_type:
            where_filter["entry_type"] = entry_type

        # Graph-RAG: Find linked entities in query
        if include_entities:
            entities = self._extractor.extract(query_text)
            if entities:
                # If we have multiple entities, we'll try to find any that match
                # ChromaDB doesn't support easy OR for $contains, so we prioritize the first one
                # or we can use a $or block if supported by the version
                if len(entities) == 1:
                    where_filter["entities"] = {"$contains": entities[0]}
                else:
                    where_filter["$or"] = [{"entities": {"$contains": e}} for e in entities]

        if not where_filter:
            where_filter = None

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.error("memory.query_failed", err=str(exc))
            # Fallback to no filter if entity filter was too restrictive
            if where_filter:
                return self.query(query_text, top_k=top_k, entry_type=entry_type, include_entities=False)
            return []

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

        log.info("memory.queried", query_len=len(query_text), results=len(entries))
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
                    distance=abs(row[3]),
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
        recency_decay: float = 0.95,
    ) -> list[MemoryEntry]:
        """A2: Hybrid search combining vector similarity + BM25 keyword.

        Includes recency-weighted conflict resolution: newer memories
        are ranked higher via exponential decay.  This ensures that
        when conflicting information exists (e.g. 'user prefers dark mode'
        vs 'user switched to light mode'), the most recent entry wins.

        recency_decay: score multiplier per hour of age (0.95 = 5% penalty/hour).
        """
        if not query_text or not query_text.strip():
            return []

        vector_results = self.query(query_text, top_k=top_k * 2)
        keyword_results = self.keyword_search(query_text, top_k=top_k * 2)

        scores: dict[str, dict] = {}

        for entry in vector_results:
            vec_score = max(0.0, 1.0 - entry.distance)
            scores[entry.id] = {"entry": entry, "vector": vec_score, "keyword": 0.0}

        for entry in keyword_results:
            kw_score = 1.0 / (1.0 + entry.distance)
            if entry.id in scores:
                scores[entry.id]["keyword"] = kw_score
            else:
                scores[entry.id] = {"entry": entry, "vector": 0.0, "keyword": kw_score}

        now = time.time()
        ranked: list[tuple[float, MemoryEntry]] = []
        for sid, data in scores.items():
            combined = (vector_weight * data["vector"] + keyword_weight * data["keyword"])
            entry = data["entry"]
            # Recency weighting: newer memories get higher scores
            age_hours = max(0.0, (now - entry.timestamp) / 3600.0) if entry.timestamp else 0.0
            recency_factor = recency_decay ** age_hours  # e.g. 0.95^24 ≈ 0.29 for day-old
            combined *= recency_factor
            entry.distance = 1.0 - combined
            ranked.append((combined, entry))

        ranked.sort(key=lambda x: -x[0])
        return [entry for _, entry in ranked[:top_k]]

    def embed(self, text: str) -> list[float]:
        """Public embedding method for semantic similarity (e.g. agent routing)."""
        return self._get_embedding(text, task_type="RETRIEVAL_QUERY")

    def count(self) -> int:
        return self._collection.count()

    def format_context(self, entries: list[MemoryEntry]) -> str:
        if not entries: return ""
        lines = ["Past similar issues:"]
        now = time.time()
        for i, entry in enumerate(entries, 1):
            age_str = _format_age(now - entry.timestamp)
            lines.append(f"  {i}. [{age_str}] {entry.content} ({entry.entry_type})")
        return "\n".join(lines)

    def build_interaction_summary(
        self,
        user_text: str,
        rio_text: str,
        tool_calls: Optional[list[dict]] = None,
    ) -> Optional[str]:
        if len(user_text) < 10 and len(rio_text) < 20:
            return None
        parts = []
        if tool_calls:
            tool_names = [tc.get("name", "unknown") for tc in tool_calls]
            parts.append(f"Tools used: {', '.join(tool_names)}.")
        user_snippet = user_text[:250].strip()
        rio_snippet = rio_text[:250].strip()
        if user_snippet: parts.append(f"User asked: {user_snippet}")
        if rio_snippet: parts.append(f"Rio responded: {rio_snippet}")
        summary = " ".join(parts)
        return summary if len(summary) >= MIN_CONTENT_LENGTH else None

    def clear(self):
        """Clear all stored memories."""
        self._chroma.delete_collection(self.collection_name)
        self._collection = self._chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("memory.cleared")

    def get_recent_memories(self, limit: int = 50) -> list[MemoryEntry]:
        """Fetch the most recent memories for compaction analysis."""
        try:
            results = self._collection.get(
                limit=limit,
                include=["documents", "metadatas"],
            )
            entries = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    doc = results["documents"][i]
                    meta = results["metadatas"][i]
                    entries.append(MemoryEntry(
                        id=doc_id,
                        content=doc,
                        timestamp=meta.get("timestamp", 0.0),
                        entry_type=meta.get("entry_type", "unknown"),
                        metadata=meta,
                    ))
            # Sort by timestamp descending
            entries.sort(key=lambda x: -x.timestamp)
            return entries
        except Exception as exc:
            log.error("memory.get_recent_failed", err=str(exc))
            return []

    def delete_entries(self, ids: list[str]) -> bool:
        """Delete specific memory entries (used after compaction)."""
        if not ids: return True
        try:
            self._collection.delete(ids=ids)
            # Also delete from FTS5
            for doc_id in ids:
                self._fts_conn.execute("DELETE FROM memory_fts WHERE entry_id = ?", (doc_id,))
            self._fts_conn.commit()
            log.info("memory.deleted", count=len(ids))
            return True
        except Exception as exc:
            log.error("memory.delete_failed", err=str(exc))
            return False


# ── Helpers ───────────────────────────────────────────────────────────

def _format_age(seconds: float) -> str:
    if seconds < 60: return "just now"
    if seconds < 3600: return f"{int(seconds / 60)}m ago"
    if seconds < 86400: return f"{int(seconds / 3600)}h ago"
    return f"{int(seconds / 86400)}d ago"
