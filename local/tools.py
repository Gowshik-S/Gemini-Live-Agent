"""
Rio Local -- Tool Executor (Day 8 / L3 + Skills)

Executes tool calls from Gemini on the local machine.

Available tools:
  Core:
  - read_file(path)         -- Read file contents (auto-approve)
  - write_file(path, content) -- Backup to .rio.bak, then write
  - patch_file(path, old_text, new_text) -- Find-and-replace with backup
  - run_command(command)     -- Shell command with 30s timeout + blocklist

  Customer Care skill:
  - create_ticket(title, ...) -- Create a support ticket (JSON file)

  Tutor skill:
  - generate_quiz(topic, ...) -- Generate quiz questions for a topic
  - track_progress(action, ...) -- Record/query student learning progress
  - explain_concept(concept, ...) -- Structured explanation scaffold

Security:
  - Dangerous commands (rm -rf /, format, dd, fork bombs) are blocked
  - File writes always create .rio.bak backups
  - Command output is truncated at 100KB to prevent memory issues
  - All operations are logged via structlog
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

# Shell commands that are always blocked (case-insensitive regex patterns)
COMMAND_BLOCKLIST = [
    r"\brm\s+(-\w+\s+)*-r",                   # rm -r (any variant: -rf, -r /, -rf /*, etc.)
    r"\brm\s+(-\w+\s+)*--no-preserve-root",  # rm --no-preserve-root
    r"\bmkfs\b",                              # format filesystem
    r"\bformat\b",                            # format (Windows)
    r"\bdd\s+.*of=/dev/",                     # dd to a device
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:",       # fork bomb
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpoweroff\b",
    r"\binit\s+0\b",
    r"\bchmod\s+(-\w+\s+)*777\s+/",          # chmod 777 /
    r">\s*/dev/sd[a-z]",                      # write to disk device
    r">\s*/etc/passwd",                       # overwrite passwd
    r"\bcurl\b.*\|\s*(ba)?sh",                # curl | bash / curl | sh
    r"\bwget\b.*\|\s*(ba)?sh",                # wget | bash / wget | sh
    r"\bsudo\s+rm\b",                         # sudo rm (any variant)
    r"\bpython[23]?\s+-c\s+.*\bos\.system\b", # python -c inject via os.system
    r"\bpython[23]?\s+-c\s+.*\bsubprocess\b", # python -c inject via subprocess
    r"\b>\.\w+rc\b",                          # overwrite shell rc files
    r">\s*/etc/",                             # overwrite any /etc/ file
]

COMMAND_TIMEOUT = 30  # seconds
MAX_OUTPUT_SIZE = 100_000  # bytes — truncate large outputs
MAX_FILE_READ = 100_000  # chars — truncate large file reads


class ToolExecutor:
    """Executes tool calls locally with safety checks.

    Usage::

        tools = ToolExecutor(working_dir="/home/dev/project")
        result = await tools.execute("read_file", {"path": "main.py"})
        # result = {"success": True, "content": "...", "path": "..."}
    """

    def __init__(self, working_dir: str | None = None) -> None:
        self._cwd = working_dir or os.getcwd()
        log.info("tools.init", working_dir=self._cwd)

    @property
    def working_dir(self) -> str:
        return self._cwd

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call by name. Returns a result dict."""
        handlers = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "patch_file": self._patch_file,
            "run_command": self._run_command,
            "create_ticket": self._create_ticket,
            "update_ticket": self._update_ticket,
            "generate_quiz": self._generate_quiz,
            "track_progress": self._track_progress,
            "explain_concept": self._explain_concept,
        }

        handler = handlers.get(name)
        if handler is None:
            log.warning("tool.unknown", name=name)
            return {"success": False, "error": f"Unknown tool: {name}"}

        try:
            result = await handler(**args)
            log.info(
                "tool.executed",
                name=name,
                success=result.get("success", False),
            )
            return result
        except TypeError as exc:
            # Missing or wrong arguments
            log.warning("tool.bad_args", name=name, error=str(exc))
            return {"success": False, "error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:
            log.exception("tool.error", name=name)
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to the working directory.

        Raises ValueError if the resolved path escapes the working directory
        (path traversal protection).
        """
        p = Path(path)
        if not p.is_absolute():
            p = Path(self._cwd) / p
        resolved = p.resolve()
        cwd_resolved = Path(self._cwd).resolve()
        # Use is_relative_to (Python 3.9+) for correct path containment check.
        # The old startswith approach had false positives:
        # e.g. /home/user/project_evil passes startswith(/home/user/project)
        try:
            resolved.relative_to(cwd_resolved)
        except ValueError:
            raise ValueError(
                f"Path traversal blocked: {path!r} resolves outside working directory"
            )
        return resolved

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------

    async def _read_file(self, path: str) -> dict[str, Any]:
        """Read the contents of a file."""
        resolved = self._resolve(path)
        log.info("tool.read_file", path=str(resolved))

        if not resolved.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if not resolved.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
            truncated = False
            if len(content) > MAX_FILE_READ:
                content = content[:MAX_FILE_READ] + "\n... [truncated at 100K chars]"
                truncated = True
            return {
                "success": True,
                "content": content,
                "path": str(resolved),
                "lines": content.count("\n") + 1,
                "truncated": truncated,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------

    async def _write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write content to a file, creating a .rio.bak backup if it exists."""
        resolved = self._resolve(path)
        log.info("tool.write_file", path=str(resolved), content_len=len(content))

        # Create backup of existing file
        if resolved.exists():
            backup = resolved.parent / (resolved.name + ".rio.bak")
            shutil.copy2(str(resolved), str(backup))
            log.info("tool.write_file.backup", backup=str(backup))

        # Ensure parent directory exists
        resolved.parent.mkdir(parents=True, exist_ok=True)

        resolved.write_text(content, encoding="utf-8")
        size = len(content.encode("utf-8"))
        return {
            "success": True,
            "path": str(resolved),
            "bytes_written": size,
        }

    # ------------------------------------------------------------------
    # patch_file
    # ------------------------------------------------------------------

    async def _patch_file(
        self, path: str, old_text: str, new_text: str,
    ) -> dict[str, Any]:
        """Apply a find-and-replace edit to a file."""
        resolved = self._resolve(path)
        log.info("tool.patch_file", path=str(resolved))

        if not resolved.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if not resolved.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        content = resolved.read_text(encoding="utf-8")

        if old_text not in content:
            return {
                "success": False,
                "error": f"old_text not found in {path}. Make sure it matches exactly.",
            }

        # Backup before patching
        backup = resolved.parent / (resolved.name + ".rio.bak")
        shutil.copy2(str(resolved), str(backup))

        # Replace first occurrence only
        new_content = content.replace(old_text, new_text, 1)
        resolved.write_text(new_content, encoding="utf-8")

        return {
            "success": True,
            "path": str(resolved),
            "replaced": True,
        }

    # ------------------------------------------------------------------
    # run_command
    # ------------------------------------------------------------------

    async def _run_command(self, command: str) -> dict[str, Any]:
        """Execute a shell command with timeout and safety blocklist."""
        log.info("tool.run_command", command=command)

        # Check blocklist
        for pattern in COMMAND_BLOCKLIST:
            if re.search(pattern, command, re.IGNORECASE):
                log.warning("tool.run_command.blocked", command=command, pattern=pattern)
                return {
                    "success": False,
                    "error": f"Command blocked by safety filter: {command}",
                }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_sync, command)

    def _run_sync(self, command: str) -> dict[str, Any]:
        """Blocking subprocess execution (runs in executor)."""
        try:
            # Use shell=True so pipes, redirects, &&, and env vars work.
            # Dangerous commands are already blocked by the blocklist above.
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self._cwd,
                timeout=COMMAND_TIMEOUT,
            )
            output = proc.stdout or ""
            if proc.stderr:
                output += "\n[stderr]\n" + proc.stderr

            # Truncate large outputs
            if len(output) > MAX_OUTPUT_SIZE:
                output = output[:MAX_OUTPUT_SIZE] + "\n... [truncated]"

            return {
                "success": proc.returncode == 0,
                "exit_code": proc.returncode,
                "output": output,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {COMMAND_TIMEOUT}s",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # create_ticket  (Customer Care skill)
    # ------------------------------------------------------------------

    async def _create_ticket(
        self,
        title: str,
        category: str = "general",
        priority: str = "medium",
        description: str = "",
        customer_id: str = "",
        tags: str = "",
    ) -> dict[str, Any]:
        """Create a support ticket and persist it as a JSON file.

        Tickets are stored in ``<working_dir>/rio_tickets/`` with UUID-based
        filenames so they survive across sessions and can be queried later.
        """
        ticket_dir = Path(self._cwd) / "rio_tickets"
        ticket_dir.mkdir(parents=True, exist_ok=True)

        ticket_id = str(uuid.uuid4())[:8].upper()
        now = datetime.now(timezone.utc).isoformat()

        ticket = {
            "id": f"RIO-{ticket_id}",
            "title": title,
            "category": category,
            "priority": priority,
            "description": description,
            "customer_id": customer_id or "anonymous",
            "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
            "status": "open",
            "created_at": now,
            "updated_at": now,
            "history": [{"action": "created", "timestamp": now}],
        }

        ticket_path = ticket_dir / f"{ticket['id']}.json"
        ticket_path.write_text(json.dumps(ticket, indent=2), encoding="utf-8")

        log.info(
            "tool.create_ticket",
            ticket_id=ticket["id"],
            category=category,
            priority=priority,
        )
        return {
            "success": True,
            "ticket_id": ticket["id"],
            "path": str(ticket_path),
            "message": f"Ticket {ticket['id']} created — {title}",
        }

    # ------------------------------------------------------------------
    # update_ticket  (Customer Care skill — escalation + status changes)
    # ------------------------------------------------------------------

    async def _update_ticket(
        self,
        ticket_id: str,
        status: str = "",
        priority: str = "",
        escalation_tier: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        """Update an existing ticket: status, priority, escalation tier, or notes.

        Supports the escalation workflow:
          tier_0 → tier_1 → tier_2 → tier_3
        Each update appends to the ticket's history log for auditability.
        """
        ticket_dir = Path(self._cwd) / "rio_tickets"
        now = datetime.now(timezone.utc).isoformat()

        # Normalize ticket_id (accept with or without prefix)
        tid = ticket_id.upper()
        if not tid.startswith("RIO-"):
            tid = f"RIO-{tid}"

        # Find the ticket file
        ticket_path = ticket_dir / f"{tid}.json"
        if not ticket_path.exists():
            # Try glob search in case of format mismatch
            candidates = list(ticket_dir.glob(f"*{ticket_id.upper()}*"))
            if candidates:
                ticket_path = candidates[0]
            else:
                return {
                    "success": False,
                    "error": f"Ticket not found: {ticket_id}. Check the ticket ID.",
                }

        # Load ticket
        ticket = json.loads(ticket_path.read_text(encoding="utf-8"))

        # Apply updates
        changes: list[str] = []

        valid_statuses = {"open", "in-progress", "escalated", "resolved", "closed"}
        if status and status.lower() in valid_statuses:
            old = ticket.get("status", "unknown")
            ticket["status"] = status.lower()
            changes.append(f"status: {old} → {status.lower()}")

        valid_priorities = {"low", "medium", "high", "critical"}
        if priority and priority.lower() in valid_priorities:
            old = ticket.get("priority", "unknown")
            ticket["priority"] = priority.lower()
            changes.append(f"priority: {old} → {priority.lower()}")

        valid_tiers = {"tier_0", "tier_1", "tier_2", "tier_3"}
        if escalation_tier and escalation_tier.lower() in valid_tiers:
            old = ticket.get("escalation_tier", "none")
            ticket["escalation_tier"] = escalation_tier.lower()
            # Auto-set status to escalated if moving to tier_2+
            if escalation_tier.lower() in ("tier_2", "tier_3"):
                ticket["status"] = "escalated"
            changes.append(f"escalation: {old} → {escalation_tier.lower()}")

        if notes:
            ticket.setdefault("notes_log", []).append({
                "timestamp": now,
                "note": notes,
            })
            changes.append("notes added")

        if not changes:
            return {
                "success": False,
                "error": "No valid updates provided. Specify status, priority, escalation_tier, or notes.",
            }

        # Append to history
        ticket["updated_at"] = now
        ticket.setdefault("history", []).append({
            "action": "updated",
            "changes": changes,
            "timestamp": now,
        })

        # Save
        ticket_path.write_text(json.dumps(ticket, indent=2), encoding="utf-8")

        log.info(
            "tool.update_ticket",
            ticket_id=ticket["id"],
            changes=changes,
        )
        return {
            "success": True,
            "ticket_id": ticket["id"],
            "changes": changes,
            "current_status": ticket.get("status"),
            "current_priority": ticket.get("priority"),
            "escalation_tier": ticket.get("escalation_tier", "none"),
            "message": f"Ticket {ticket['id']} updated: {', '.join(changes)}",
        }

    # ------------------------------------------------------------------
    # generate_quiz  (Tutor skill)
    # ------------------------------------------------------------------

    async def _generate_quiz(
        self,
        topic: str,
        difficulty: str = "intermediate",
        num_questions: int = 5,
        question_types: str = "multiple_choice,short_answer",
        focus_areas: str = "",
    ) -> dict[str, Any]:
        """Build a prompt for Gemini to generate a quiz on any topic.

        No hardcoded question banks — the LLM handles all subjects:
        math, science, history, literature, languages, law, medicine,
        programming, music, philosophy, or anything else.
        """
        types_readable = question_types.replace("_", " ").replace(",", ", ")
        focus_clause = f" Focus specifically on: {focus_areas}." if focus_areas else ""

        prompt = (
            f"Generate exactly {num_questions} quiz questions about '{topic}' "
            f"at {difficulty} level.{focus_clause}\n"
            f"Question types to use: {types_readable}.\n\n"
            f"Rules:\n"
            f"- Each question must test genuine understanding, not just memorisation.\n"
            f"- For multiple-choice: provide 4 options (A-D), mark the correct one.\n"
            f"- For short-answer / problem-solving: give the expected answer clearly.\n"
            f"- For true-false: state why the false option is wrong.\n"
            f"- Include a one-line hint per question (don't give the answer away).\n"
            f"- Include a one-line explanation after the answer.\n"
            f"- Difficulty guide: "
            f"beginner=recall, novice=basic application, "
            f"intermediate=analysis, advanced=synthesis/edge-cases.\n\n"
            f"Format each question as:\n"
            f"Q<n>. [question text]\n"
            f"Options: A) ... B) ... C) ... D) ...  (omit for non-MC)\n"
            f"Answer: [correct answer]\n"
            f"Hint: [hint]\n"
            f"Explanation: [why this is the answer]\n"
        )

        log.info("tool.generate_quiz", topic=topic, difficulty=difficulty, n=num_questions)
        return {
            "success": True,
            "source": "llm",
            "prompt": prompt,
            "meta": {
                "topic": topic,
                "difficulty": difficulty,
                "num_questions": num_questions,
                "question_types": question_types,
                "focus_areas": focus_areas,
            },
        }

    # ------------------------------------------------------------------
    # track_progress  (Tutor skill)
    # ------------------------------------------------------------------

    async def _track_progress(
        self,
        action: str,
        subject: str = "",
        topic: str = "",
        score: float = 0.0,
        notes: str = "",
        student_id: str = "default",
    ) -> dict[str, Any]:
        """Track student learning progress.

        Supported actions:
          - ``record``  — Add a new progress entry (quiz score, topic mastery, etc.)
          - ``query``   — Retrieve recent progress for a subject/topic
          - ``summary`` — Get an overall learning summary
        """
        progress_dir = Path(self._cwd) / "rio_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)

        progress_file = progress_dir / f"{student_id}.json"
        now = datetime.now(timezone.utc).isoformat()

        # Load existing progress
        if progress_file.exists():
            data = json.loads(progress_file.read_text(encoding="utf-8"))
        else:
            data = {"student_id": student_id, "entries": [], "created_at": now}

        if action == "record":
            entry = {
                "subject": subject,
                "topic": topic,
                "score": score,
                "notes": notes,
                "timestamp": now,
            }
            data["entries"].append(entry)
            data["updated_at"] = now
            progress_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            log.info("tool.track_progress.record", student=student_id, subject=subject, topic=topic)
            return {
                "success": True,
                "action": "record",
                "entry": entry,
                "total_entries": len(data["entries"]),
            }

        elif action == "query":
            # Filter entries by subject/topic
            filtered = data["entries"]
            if subject:
                filtered = [e for e in filtered if e.get("subject", "").lower() == subject.lower()]
            if topic:
                filtered = [e for e in filtered if topic.lower() in e.get("topic", "").lower()]
            # Return last 10 entries
            recent = filtered[-10:]
            avg_score = sum(e.get("score", 0) for e in recent) / max(len(recent), 1)
            log.info("tool.track_progress.query", student=student_id, results=len(recent))
            return {
                "success": True,
                "action": "query",
                "entries": recent,
                "count": len(recent),
                "average_score": round(avg_score, 2),
            }

        elif action == "summary":
            entries = data["entries"]
            if not entries:
                return {"success": True, "action": "summary", "message": "No progress data yet."}

            # Build per-subject summary
            subjects: dict[str, list[float]] = {}
            for e in entries:
                subj = e.get("subject", "unknown")
                subjects.setdefault(subj, []).append(e.get("score", 0))

            summary = {}
            for subj, scores in subjects.items():
                summary[subj] = {
                    "sessions": len(scores),
                    "average_score": round(sum(scores) / len(scores), 2),
                    "latest_score": scores[-1],
                    "trend": "improving" if len(scores) >= 2 and scores[-1] > scores[-2] else "stable",
                }

            log.info("tool.track_progress.summary", student=student_id, subjects=list(summary.keys()))
            return {
                "success": True,
                "action": "summary",
                "subjects": summary,
                "total_sessions": len(entries),
            }

        else:
            return {"success": False, "error": f"Unknown action: {action}. Use record/query/summary."}

    # ------------------------------------------------------------------
    # explain_concept  (Tutor skill)
    # ------------------------------------------------------------------

    async def _explain_concept(
        self,
        concept: str,
        level: str = "intermediate",
        context: str = "",
    ) -> dict[str, Any]:
        """Return a structured explanation scaffold for a concept.

        This doesn't generate the full explanation itself — it returns a
        structured template that guides Gemini to produce a well-organized,
        level-appropriate explanation using the Socratic method.
        """
        level_guidance = {
            "beginner": {
                "vocabulary": "everyday language, no jargon",
                "analogies": True,
                "depth": "surface-level intuition, concrete examples",
                "prereqs": "assume no prior knowledge",
            },
            "novice": {
                "vocabulary": "simple terms, define any jargon",
                "analogies": True,
                "depth": "basic understanding with 1-2 examples",
                "prereqs": "assume minimal background",
            },
            "intermediate": {
                "vocabulary": "standard technical terms OK",
                "analogies": True,
                "depth": "conceptual + procedural, why + how",
                "prereqs": "assume foundational knowledge",
            },
            "advanced": {
                "vocabulary": "full technical vocabulary",
                "analogies": False,
                "depth": "theory, edge cases, connections to other concepts",
                "prereqs": "assume strong background",
            },
        }

        guidance = level_guidance.get(level, level_guidance["intermediate"])

        log.info("tool.explain_concept", concept=concept, level=level)
        return {
            "success": True,
            "concept": concept,
            "level": level,
            "guidance": guidance,
            "structure": [
                f"1. Hook: Ask a thought-provoking question about '{concept}'",
                f"2. Connect: Link to something the student already knows",
                f"3. Build: Explain the core idea ({guidance['depth']})",
                "4. Example: Walk through a concrete example together",
                "5. Check: Ask the student to explain it back in their own words",
                "6. Extend: Pose a 'what if' question to deepen understanding",
            ],
            "context": context,
            "anti_patterns": [
                "Don't lecture — ask questions that lead to discovery",
                "Don't give the answer — guide toward it",
                "Don't overwhelm — one concept at a time",
                "Don't use 'it's easy' or 'obviously' — these shame learners",
            ],
        }
