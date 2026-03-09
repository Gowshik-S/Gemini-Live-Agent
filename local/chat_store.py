"""
Rio Local — Chat Store (SQLite)

Persists all conversation messages (user + Rio) to a local SQLite database.
Provides query, export, and history retrieval capabilities.

Thread-safe: uses a single connection with WAL mode for concurrent reads.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "rio_chats.db")


@dataclass
class ChatMessage:
    """A single chat message."""
    id: int
    session_id: str
    speaker: str          # "user" | "rio" | "system"
    content: str
    timestamp: float
    metadata: dict = field(default_factory=dict)

    @property
    def time_str(self) -> str:
        """Human-readable timestamp."""
        import datetime
        dt = datetime.datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class ChatStore:
    """SQLite-backed chat message storage.
    
    Usage::
    
        store = ChatStore(db_path="./rio_chats.db")
        store.add_message("session-1", "user", "Hello Rio")
        store.add_message("session-1", "rio", "Hello! How can I help?")
        
        messages = store.get_session("session-1")
        recent = store.get_recent(limit=20)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # We handle thread safety manually
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        
        self._create_tables()
        log.info("chat_store.init", db_path=self.db_path)

    def _create_tables(self) -> None:
        """Create the messages table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                speaker TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp
            ON messages(timestamp)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                ended_at REAL,
                message_count INTEGER DEFAULT 0
            )
        """)
        self._conn.commit()

    def start_session(self, session_id: str) -> None:
        """Record the start of a new chat session."""
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, started_at, message_count) VALUES (?, ?, 0)",
            (session_id, time.time()),
        )
        self._conn.commit()
        log.info("chat_store.session_started", session_id=session_id)

    def end_session(self, session_id: str) -> None:
        """Record the end of a chat session."""
        self._conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
            (time.time(), session_id),
        )
        self._conn.commit()
        log.info("chat_store.session_ended", session_id=session_id)

    def add_message(
        self,
        session_id: str,
        speaker: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """Store a chat message and return its ID."""
        if not content or not content.strip():
            return -1
        
        ts = time.time()
        meta_json = json.dumps(metadata or {})
        
        cursor = self._conn.execute(
            "INSERT INTO messages (session_id, speaker, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
            (session_id, speaker, content.strip(), ts, meta_json),
        )
        self._conn.execute(
            "UPDATE sessions SET message_count = message_count + 1 WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()
        
        msg_id = cursor.lastrowid
        log.debug(
            "chat_store.message_added",
            id=msg_id,
            speaker=speaker,
            length=len(content),
        )
        return msg_id

    def get_session(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[ChatMessage]:
        """Get all messages for a session."""
        rows = self._conn.execute(
            "SELECT id, session_id, speaker, content, timestamp, metadata "
            "FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def get_recent(self, limit: int = 50) -> list[ChatMessage]:
        """Get the most recent messages across all sessions."""
        rows = self._conn.execute(
            "SELECT id, session_id, speaker, content, timestamp, metadata "
            "FROM messages ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_message(r) for r in reversed(rows)]

    def get_context_window(
        self,
        session_id: str,
        max_messages: int = 20,
    ) -> list[ChatMessage]:
        """Return the last *max_messages* for a session (sliding window).

        This is the method used to build Gemini context — capping the
        history prevents token bloat and 400 errors on long sessions.
        """
        rows = self._conn.execute(
            "SELECT id, session_id, speaker, content, timestamp, metadata "
            "FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, max_messages),
        ).fetchall()
        # Reverse so they're in chronological order
        return [self._row_to_message(r) for r in reversed(rows)]

    def get_history_for_dashboard(self, limit: int = 100) -> list[dict]:
        """Get recent messages formatted for dashboard consumption."""
        rows = self._conn.execute(
            "SELECT id, session_id, speaker, content, timestamp, metadata "
            "FROM messages ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for r in reversed(rows):
            result.append({
                "id": r[0],
                "speaker": r[2],
                "text": r[3],
                "timestamp": r[4],
            })
        return result

    def search(self, query: str, limit: int = 20) -> list[ChatMessage]:
        """Search messages by content (basic LIKE search)."""
        rows = self._conn.execute(
            "SELECT id, session_id, speaker, content, timestamp, metadata "
            "FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def count(self) -> int:
        """Total message count."""
        row = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return row[0] if row else 0

    def session_count(self) -> int:
        """Total session count."""
        row = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def _row_to_message(self, row: tuple) -> ChatMessage:
        """Convert a database row to a ChatMessage."""
        return ChatMessage(
            id=row[0],
            session_id=row[1],
            speaker=row[2],
            content=row[3],
            timestamp=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
        )
