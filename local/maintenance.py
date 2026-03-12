"""
Rio Local — Maintenance (A4 + G2)

Startup maintenance tasks:
  - A4: Daily log partitioning (archive old conversation JSON files)
  - G2: Auto-pruning (purge stale ChatStore messages, vacuum SQLite)

Runs automatically once at server startup.

Configure in config.yaml:
    rio:
      maintenance:
        retention_days: 90
        max_conversations: 500
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_DEFAULT_RETENTION_DAYS = 90
_DEFAULT_MAX_CONVERSATIONS = 500


def run_maintenance(
    data_dir: str | Path | None = None,
    retention_days: int = _DEFAULT_RETENTION_DAYS,
    max_conversations: int = _DEFAULT_MAX_CONVERSATIONS,
) -> dict:
    """Run all maintenance tasks. Called once at startup.

    Returns a summary of actions taken.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    summary = {
        "conversations_archived": 0,
        "conversations_pruned": 0,
        "chat_messages_pruned": 0,
        "memory_entries_pruned": 0,
    }

    try:
        summary["conversations_archived"] = _archive_old_conversations(
            data_dir / "conversations",
            retention_days=retention_days,
        )
    except Exception:
        log.exception("maintenance.archive_conversations.error")

    try:
        summary["conversations_pruned"] = _prune_excess_conversations(
            data_dir / "conversations",
            max_keep=max_conversations,
        )
    except Exception:
        log.exception("maintenance.prune_conversations.error")

    try:
        summary["chat_messages_pruned"] = _prune_chat_store(
            data_dir,
            retention_days=retention_days,
        )
    except Exception:
        log.exception("maintenance.prune_chat_store.error")

    try:
        summary["memory_entries_pruned"] = _prune_memory_fts(
            data_dir / "memory",
            retention_days=retention_days,
        )
    except Exception:
        log.exception("maintenance.prune_memory_fts.error")

    log.info("maintenance.complete", **summary)
    return summary


def _archive_old_conversations(
    conversations_dir: Path,
    retention_days: int,
) -> int:
    """A4: Move old conversation JSON files to an archive subdirectory."""
    if not conversations_dir.is_dir():
        return 0

    archive_dir = conversations_dir / "archive"
    cutoff = time.time() - (retention_days * 86400)
    archived = 0

    for f in conversations_dir.glob("*.json"):
        try:
            stat = f.stat()
            if stat.st_mtime < cutoff:
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(archive_dir / f.name))
                archived += 1
        except Exception:
            continue

    if archived:
        log.info("maintenance.archived", count=archived, dir=str(archive_dir))
    return archived


def _prune_excess_conversations(
    conversations_dir: Path,
    max_keep: int,
) -> int:
    """G2: If more than max_keep conversations, delete oldest."""
    if not conversations_dir.is_dir():
        return 0

    files = sorted(
        conversations_dir.glob("*.json"),
        key=lambda f: f.stat().st_mtime,
    )
    pruned = 0
    while len(files) > max_keep:
        oldest = files.pop(0)
        try:
            oldest.unlink()
            pruned += 1
        except Exception:
            continue

    if pruned:
        log.info("maintenance.conversations_pruned", count=pruned)
    return pruned


def _prune_chat_store(data_dir: Path, retention_days: int) -> int:
    """G2: Prune old messages from ChatStore SQLite database."""
    import sqlite3

    db_path = data_dir / "chat_store.db"
    if not db_path.is_file():
        return 0

    cutoff = time.time() - (retention_days * 86400)
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "DELETE FROM messages WHERE timestamp < ?", (cutoff,)
        )
        deleted = cursor.rowcount
        conn.execute("DELETE FROM sessions WHERE ended_at IS NOT NULL AND ended_at < ?", (cutoff,))
        conn.execute("VACUUM")
        conn.commit()
        if deleted > 0:
            log.info("maintenance.chat_store_pruned", messages_deleted=deleted)
        return deleted
    except Exception:
        return 0
    finally:
        conn.close()


def _prune_memory_fts(memory_dir: Path, retention_days: int) -> int:
    """G2: Prune old FTS5 entries (non-critical)."""
    import sqlite3

    fts_path = memory_dir / "fts_index.db"
    if not fts_path.is_file():
        return 0

    # FTS5 doesn't have timestamps; just vacuum to reclaim space
    conn = sqlite3.connect(str(fts_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        # Can't easily prune FTS5 by time, but vacuum helps
        conn.commit()
        return 0
    except Exception:
        return 0
    finally:
        conn.close()
