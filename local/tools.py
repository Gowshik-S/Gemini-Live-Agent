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

    # Screen action tools that trigger auto-capture when autonomous mode is on
    SCREEN_ACTION_TOOLS = frozenset({
        "screen_click", "screen_type", "screen_scroll",
        "screen_hotkey", "screen_move", "screen_drag",
    })

    # Stuck detection: max identical consecutive actions before warning
    MAX_REPEATED_ACTIONS = 3

    def __init__(self, working_dir: str | None = None) -> None:
        self._cwd = working_dir or os.getcwd()
        self._screen_navigator = None  # Set via set_screen_navigator()
        self._screen_capture = None    # Set via set_screen_capture()
        self._ws_send_binary = None    # Set via set_ws_sender()
        self._last_actions: list[tuple[str, str]] = []  # (name, args_key) ring buffer
        log.info("tools.init", working_dir=self._cwd)

    @property
    def working_dir(self) -> str:
        return self._cwd

    def set_screen_navigator(self, navigator) -> None:
        """Attach a ScreenNavigator instance for screen interaction tools."""
        self._screen_navigator = navigator
        log.info("tools.screen_navigator_attached")

    def set_screen_capture(self, screen_capture) -> None:
        """Attach a ScreenCapture instance for auto-capture after screen actions."""
        self._screen_capture = screen_capture
        log.info("tools.screen_capture_attached")

    def set_ws_sender(self, send_binary_fn) -> None:
        """Attach an async callable to send binary frames to the cloud.

        Used by auto-capture to send screenshots after screen actions.
        Signature: ``async def send_binary(data: bytes) -> None``
        """
        self._ws_send_binary = send_binary_fn
        log.info("tools.ws_sender_attached")

    async def execute_with_auto_capture(
        self, name: str, args: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a screen action tool, then auto-capture a screenshot.

        This closes the autonomous agent feedback loop:
        1. Execute the screen action (click, type, scroll, etc.)
        2. Wait briefly for the UI to update (300ms)
        3. Capture a new screenshot
        4. Send the screenshot to the cloud as a binary image frame

        Gemini sees: tool result + fresh screenshot of the result
        → can decide the next action without another user request.
        """
        # Stuck detection: track repeated identical actions
        action_key = (name, json.dumps(args, sort_keys=True))
        self._last_actions.append(action_key)
        if len(self._last_actions) > self.MAX_REPEATED_ACTIONS:
            self._last_actions = self._last_actions[-self.MAX_REPEATED_ACTIONS:]

        # Step 1: Execute the actual screen action
        result = await self.execute(name, args)

        if not result.get("success", False):
            return result  # Action failed — skip auto-capture

        # Check if stuck (same action repeated MAX_REPEATED_ACTIONS times)
        if (
            len(self._last_actions) >= self.MAX_REPEATED_ACTIONS
            and len(set(self._last_actions[-self.MAX_REPEATED_ACTIONS:])) == 1
        ):
            result["warning"] = (
                f"You've repeated the same action ({name}) {self.MAX_REPEATED_ACTIONS} "
                "times with identical arguments. The UI may not be responding "
                "as expected. Try a different approach or ask the user for help."
            )
            log.warning("tool.stuck_detected", action=name, repeats=self.MAX_REPEATED_ACTIONS)

        # Step 2: Brief pause for UI to settle (reduced from 300ms for B-13)
        await asyncio.sleep(0.15)

        # Step 3 + 4: Auto-capture and send
        if self._screen_capture is not None and self._ws_send_binary is not None:
            try:
                jpeg = await self._screen_capture.capture_async(force=True)
                if jpeg is not None:
                    # Send as image frame (0x02 prefix) to cloud
                    await self._ws_send_binary(b"\x02" + jpeg)
                    result["auto_capture"] = True
                    result["auto_capture_note"] = (
                        "A screenshot was automatically taken after this action "
                        "and sent to your vision context. Analyze it to verify "
                        "the action succeeded and decide your next step."
                    )
                    log.info(
                        "tool.auto_capture.sent",
                        action=name,
                        size_kb=round(len(jpeg) / 1024, 1),
                    )
                else:
                    result["auto_capture"] = False
            except Exception as exc:
                log.warning("tool.auto_capture.failed", action=name, error=str(exc))
                result["auto_capture"] = False
                result["auto_capture_error"] = str(exc)
        else:
            result["auto_capture"] = False

        return result

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
            # Screen navigation
            "screen_click": self._screen_click,
            "screen_type": self._screen_type,
            "screen_scroll": self._screen_scroll,
            "screen_hotkey": self._screen_hotkey,
            "screen_move": self._screen_move,
            "screen_drag": self._screen_drag,
            "find_window": self._find_window,
            "focus_window": self._focus_window,
            # Persistent memory
            "get_task_status": self._get_task_status,
            "save_note": self._save_note,
            "get_notes": self._get_notes,
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

    # ------------------------------------------------------------------
    # Screen Navigation tools
    # ------------------------------------------------------------------

    def _nav_or_error(self) -> dict[str, Any] | None:
        """Return error dict if screen navigator is not attached."""
        if self._screen_navigator is None or not self._screen_navigator.available:
            return {
                "success": False,
                "error": "Screen navigator not available. Install pyautogui.",
            }
        return None

    async def _screen_click(
        self, x: int, y: int, button: str = "left", clicks: int = 1,
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.click(
            int(x), int(y), button=button, clicks=int(clicks),
        )

    async def _screen_type(
        self, text: str, interval: float = 0.02,
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.type_text(text, interval=float(interval))

    async def _screen_scroll(
        self, x: int, y: int, clicks: int,
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.scroll(int(x), int(y), int(clicks))

    async def _screen_hotkey(self, keys: str) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.hotkey(keys)

    async def _screen_move(self, x: int, y: int) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.move(int(x), int(y))

    async def _screen_drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.drag(
            int(start_x), int(start_y),
            int(end_x), int(end_y),
            duration=float(duration),
        )

    async def _find_window(self, title_contains: str) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.find_window(title_contains)

    async def _focus_window(self, title_contains: str) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.focus_window(title_contains)

    # ------------------------------------------------------------------
    # Persistent Memory Tools
    # ------------------------------------------------------------------

    def set_task_store(self, task_store) -> None:
        """Attach a TaskStore for task status queries."""
        self._task_store = task_store

    def set_session_memory(self, session_memory) -> None:
        """Attach a SessionMemory for persistent notes."""
        self._session_memory = session_memory

    async def _get_task_status(self) -> dict[str, Any]:
        """Return a summary of all tasks — pending, active, completed, failed."""
        if not hasattr(self, '_task_store') or self._task_store is None:
            return {"success": False, "error": "Task store not available"}
        summary = self._task_store.get_status_summary()
        return {"success": True, "summary": summary}

    async def _save_note(self, key: str, value: str) -> dict[str, Any]:
        """Save a persistent session note that survives across sessions."""
        if not hasattr(self, '_session_memory') or self._session_memory is None:
            return {"success": False, "error": "Session memory not available"}
        self._session_memory.set(key, value)
        return {"success": True, "key": key, "message": f"Note '{key}' saved."}

    async def _get_notes(self, key: str = "") -> dict[str, Any]:
        """Retrieve session notes. If key is empty, returns all notes."""
        if not hasattr(self, '_session_memory') or self._session_memory is None:
            return {"success": False, "error": "Session memory not available"}
        if key:
            value = self._session_memory.get(key)
            if value:
                return {"success": True, "key": key, "value": value}
            return {"success": False, "error": f"Note '{key}' not found"}
        summary = self._session_memory.get_summary()
        return {"success": True, "notes": summary}
