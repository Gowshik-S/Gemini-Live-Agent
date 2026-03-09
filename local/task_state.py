"""
Rio Local — Task State Machine

Manages the lifecycle of autonomous tasks: planning, step execution,
verification, retry, and completion.  Task state persists to SQLite
for crash recovery.

State transitions:
    pending → running → verifying → done | failed | retrying
    retrying → running   (up to MAX_RETRIES per step)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

log = structlog.get_logger(__name__)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rio_tasks.db")
MAX_STEP_RETRIES = 3


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"
    PARTIAL = "partial"       # partially completed
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class StepType(str, Enum):
    BROWSER = "browser"       # Gemini Computer Use / Playwright
    SYSTEM = "system"         # pyautogui / pywinauto / shell commands
    CREATIVE = "creative"     # text generation, images, etc.
    TOOL = "tool"             # read_file, write_file, run_command
    VERIFY = "verify"         # screenshot + vision verification


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Step:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    step_type: StepType = StepType.TOOL
    action: str = ""              # Human-readable description of what to do
    expected_outcome: str = ""    # What success looks like
    tool_name: str = ""           # Tool / agent to invoke
    tool_args: dict = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    result: str = ""              # Output from execution
    error: str = ""               # Last error message
    started_at: float = 0.0
    completed_at: float = 0.0

    def mark_running(self) -> None:
        self.status = StepStatus.RUNNING
        self.attempts += 1
        self.started_at = time.time()
        self.error = ""

    def mark_done(self, result: str = "") -> None:
        self.status = StepStatus.DONE
        self.result = result
        self.completed_at = time.time()

    def mark_failed(self, error: str = "") -> None:
        self.error = error
        if self.attempts < MAX_STEP_RETRIES:
            self.status = StepStatus.RETRYING
        else:
            self.status = StepStatus.FAILED
            self.completed_at = time.time()

    def mark_verifying(self) -> None:
        self.status = StepStatus.VERIFYING

    @property
    def can_retry(self) -> bool:
        return self.status == StepStatus.RETRYING and self.attempts < MAX_STEP_RETRIES

    def to_dict(self) -> dict:
        d = asdict(self)
        d["step_type"] = self.step_type.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        d = dict(d)
        d["step_type"] = StepType(d.get("step_type", "tool"))
        d["status"] = StepStatus(d.get("status", "pending"))
        return cls(**d)


@dataclass
class Task:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str = ""                  # User's original request
    plan_summary: str = ""          # Brief plan overview from Gemini Pro
    steps: list[Step] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    scratchpad: dict = field(default_factory=dict)  # inter-step context

    @property
    def current_step(self) -> Optional[Step]:
        """Return the first non-completed step, or None if all done."""
        for s in self.steps:
            if s.status not in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED):
                return s
        return None

    @property
    def progress(self) -> str:
        done = sum(1 for s in self.steps if s.status in (StepStatus.DONE, StepStatus.SKIPPED))
        total = len(self.steps)
        return f"{done}/{total}"

    @property
    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED,
                               TaskStatus.PARTIAL, TaskStatus.CANCELLED)

    def mark_running(self) -> None:
        self.status = TaskStatus.RUNNING

    def mark_done(self) -> None:
        self.status = TaskStatus.DONE
        self.completed_at = time.time()

    def mark_failed(self) -> None:
        # Partial if some steps succeeded
        done = sum(1 for s in self.steps if s.status == StepStatus.DONE)
        if done > 0:
            self.status = TaskStatus.PARTIAL
        else:
            self.status = TaskStatus.FAILED
        self.completed_at = time.time()

    def mark_cancelled(self) -> None:
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()

    def advance(self) -> Optional[Step]:
        """Move to the next pending step and mark it running. Returns it, or None."""
        step = self.current_step
        if step is None:
            # All steps are terminal — decide task status
            failed = any(s.status == StepStatus.FAILED for s in self.steps)
            if failed:
                self.mark_failed()
            else:
                self.mark_done()
            return None
        step.mark_running()
        return step

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "plan_summary": self.plan_summary,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "scratchpad": self.scratchpad,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        steps = [Step.from_dict(s) for s in d.get("steps", [])]
        return cls(
            id=d["id"],
            goal=d.get("goal", ""),
            plan_summary=d.get("plan_summary", ""),
            steps=steps,
            status=TaskStatus(d.get("status", "pending")),
            created_at=d.get("created_at", 0.0),
            completed_at=d.get("completed_at", 0.0),
            scratchpad=d.get("scratchpad", {}),
        )


# ---------------------------------------------------------------------------
# SQLite Persistence
# ---------------------------------------------------------------------------

class TaskStore:
    """Persists tasks to SQLite for crash recovery and history."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        log.info("task_store.init", db_path=self.db_path)

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status
            ON tasks(status)
        """)
        self._conn.commit()

    def save(self, task: Task) -> None:
        """Insert or update a task."""
        data = json.dumps(task.to_dict())
        self._conn.execute(
            "INSERT OR REPLACE INTO tasks (id, data, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task.id, data, task.status.value, task.created_at, time.time()),
        )
        self._conn.commit()

    def load(self, task_id: str) -> Optional[Task]:
        """Load a task by ID."""
        row = self._conn.execute(
            "SELECT data FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return None
        return Task.from_dict(json.loads(row[0]))

    def load_active(self) -> list[Task]:
        """Load all non-terminal tasks (for crash recovery)."""
        rows = self._conn.execute(
            "SELECT data FROM tasks WHERE status IN ('pending', 'running', 'verifying') "
            "ORDER BY created_at DESC",
        ).fetchall()
        return [Task.from_dict(json.loads(r[0])) for r in rows]

    def load_recent(self, limit: int = 20) -> list[Task]:
        """Load the most recent tasks."""
        rows = self._conn.execute(
            "SELECT data FROM tasks ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Task.from_dict(json.loads(r[0])) for r in rows]

    def get_status_summary(self) -> str:
        """Return a human-readable summary of all tasks for show-and-tell.

        This is the persistent memory system: it provides a snapshot of
        what's pending, in progress, completed, and failed — across sessions.
        """
        active = self.load_active()
        recent = self.load_recent(limit=10)

        lines = ["# Rio Task Status\n"]

        # Active tasks (in progress right now)
        if active:
            lines.append("## Currently Active")
            for t in active:
                lines.append(f"- [{t.status.value.upper()}] {t.goal}")
                lines.append(f"  Progress: {t.progress}")
                step = t.current_step
                if step:
                    lines.append(f"  Current step: {step.action} ({step.status.value})")
                lines.append("")
        else:
            lines.append("## No Active Tasks\n")

        # Recent completed/failed tasks
        terminal = [t for t in recent if t.is_terminal]
        if terminal:
            lines.append("## Recent Completed/Failed")
            for t in terminal[:5]:
                emoji = "done" if t.status == TaskStatus.DONE else t.status.value
                lines.append(f"- [{emoji.upper()}] {t.goal}")
                done_steps = sum(1 for s in t.steps if s.status == StepStatus.DONE)
                lines.append(f"  Steps: {done_steps}/{len(t.steps)} completed")
                if t.status == TaskStatus.FAILED:
                    failed_step = next((s for s in t.steps if s.status == StepStatus.FAILED), None)
                    if failed_step:
                        lines.append(f"  Error: {failed_step.error[:100]}")
                lines.append("")

        # Stats
        total = self._conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        done = self._conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'done'"
        ).fetchone()[0]
        failed = self._conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status IN ('failed', 'partial')"
        ).fetchone()[0]
        lines.append(f"## Stats: {total} total, {done} done, {failed} failed")

        return "\n".join(lines)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Session Notes — persistent memory across sessions
# ---------------------------------------------------------------------------

class SessionMemory:
    """Persistent key-value notes that survive across sessions.

    Stores session notes in SQLite alongside tasks. This enables
    openclaw-style show-and-tell: "What are you working on?",
    "What's left to do?", "What happened last session?"
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS session_notes (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def set(self, key: str, value: str) -> None:
        """Store or update a note."""
        self._conn.execute(
            "INSERT OR REPLACE INTO session_notes (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        self._conn.commit()

    def get(self, key: str, default: str = "") -> str:
        """Retrieve a note by key."""
        row = self._conn.execute(
            "SELECT value FROM session_notes WHERE key = ?", (key,),
        ).fetchone()
        return row[0] if row else default

    def get_all(self) -> dict[str, str]:
        """Return all notes as a dict."""
        rows = self._conn.execute(
            "SELECT key, value FROM session_notes ORDER BY updated_at DESC"
        ).fetchall()
        return {k: v for k, v in rows}

    def delete(self, key: str) -> bool:
        """Delete a note. Returns True if it existed."""
        cur = self._conn.execute("DELETE FROM session_notes WHERE key = ?", (key,))
        self._conn.commit()
        return cur.rowcount > 0

    def get_summary(self) -> str:
        """Return all session notes as a formatted string."""
        notes = self.get_all()
        if not notes:
            return "No session notes saved."
        lines = ["# Session Notes\n"]
        for key, value in notes.items():
            lines.append(f"## {key}")
            lines.append(value)
            lines.append("")
        return "\n".join(lines)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
