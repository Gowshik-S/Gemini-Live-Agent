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
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def _get_env_value(name: str) -> str:
    """Read an env var, falling back to rio/cloud/.env when needed."""
    value = os.environ.get(name, "").strip()
    if value:
        return value

    env_file = Path(__file__).resolve().parent.parent / "cloud" / ".env"
    if not env_file.is_file():
        return ""

    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            if key.strip() != name:
                continue
            value = raw_value.strip().strip('"').strip("'")
            if value:
                os.environ[name] = value
            return value
    except OSError as exc:
        log.warning("tools.env_read_failed", path=str(env_file), error=str(exc))

    return ""

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

# Tool output truncation (B3) — prevent single tool calls from
# consuming the entire model context window.
MAX_TOOL_OUTPUT_CHARS = 8000


def _truncate_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate tool output preserving head and tail with informative trailer."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2 - 50
    return (
        text[:half]
        + f"\n\n... [{len(text)} chars total, middle truncated] ...\n\n"
        + text[-half:]
    )


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
        "smart_click",  # Computer Use visual-grounding click
        "open_application", "focus_window",
        "minimize_window", "maximize_window", "close_window",
    })

    # Stuck detection: max identical consecutive actions before warning
    MAX_REPEATED_ACTIONS = 3

    def __init__(self, working_dir: str | None = None) -> None:
        self._cwd = working_dir or os.getcwd()
        self._creative_agent = None  # Lazy-loaded for GenMedia tools
        self._screen_navigator = None  # Set via set_screen_navigator()
        self._screen_capture = None    # Set via set_screen_capture()
        self._ws_send_binary = None    # Set via set_ws_sender()
        self._last_actions: list[tuple[str, str]] = []  # (name, args_key) ring buffer
        self._fs_policy = self._load_filesystem_policy()
        self._workspace_cli_bin = os.environ.get("RIO_WORKSPACE_CLI_BIN", "workspace-cli")
        log.info("tools.init", working_dir=self._cwd)

    @property
    def working_dir(self) -> str:
        return self._cwd

    def _load_filesystem_policy(self) -> dict[str, list[Path]]:
        """Load filesystem read/write policy from config.yaml."""
        policy: dict[str, list[Path]] = {"read": [Path(self._cwd).resolve()], "write": [Path(self._cwd).resolve()]}
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        if not cfg_path.is_file():
            return policy
        try:
            import yaml

            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            fs = raw.get("rio", {}).get("filesystem", {}) or {}
            for mode in ("read_paths", "write_paths"):
                roots = []
                for entry in fs.get(mode, []) or []:
                    p = Path(str(entry))
                    if not p.is_absolute():
                        p = Path(self._cwd) / p
                    roots.append(p.resolve())
                if roots:
                    key = "read" if mode == "read_paths" else "write"
                    policy[key] = roots
        except Exception:
            pass
        return policy

    def _allowed_by_policy(self, resolved: Path, mode: str) -> bool:
        """Return True if resolved path is allowed for read or write mode."""
        roots = self._fs_policy.get(mode, [])
        for root in roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

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

    # Actions that modify window state — these get post-action verification
    _WINDOW_VERIFY_ACTIONS = frozenset({
        "open_application", "focus_window", "close_window",
        "minimize_window", "maximize_window",
    })

    async def _verify_action(
        self, name: str, args: dict[str, Any], result: dict[str, Any],
    ) -> dict[str, Any]:
        """Post-action verification for window management actions.

        Checks whether the expected window state change actually occurred.
        Returns the result dict enriched with verification data.
        """
        if name not in self._WINDOW_VERIFY_ACTIONS:
            return result
        if self._screen_navigator is None:
            return result

        try:
            if name == "open_application":
                # open_application already has built-in verification
                # Just add a window list snapshot for the model
                win_result = await self._screen_navigator.list_all_windows()
                if win_result.get("success"):
                    top_windows = [
                        w["title"] for w in win_result.get("windows", [])[:10]
                    ]
                    result["visible_windows"] = top_windows

            elif name == "focus_window":
                title_query = (args.get("title") or args.get("title_contains") or "").lower()
                if title_query:
                    active = await self._screen_navigator.get_active_window()
                    if active.get("success"):
                        active_title = (active.get("window", {}) or {}).get("title", "")
                        if not active_title:
                            active_title = active.get("title", "")
                        if title_query in active_title.lower():
                            result["verification"] = "Window is now in foreground."
                        else:
                            result["verification_warning"] = (
                                f"Focus requested for '{title_query}' but active window "
                                f"is '{active_title}'. The window may not have focused correctly."
                            )

            elif name == "close_window":
                title_query = (args.get("title") or args.get("title_contains") or "").lower()
                if title_query:
                    await asyncio.sleep(0.3)
                    win_result = await self._screen_navigator.list_all_windows()
                    if win_result.get("success"):
                        still_open = [
                            w["title"] for w in win_result.get("windows", [])
                            if title_query in w["title"].lower()
                        ]
                        if still_open:
                            result["verification_warning"] = (
                                f"Window matching '{title_query}' is still visible: {still_open}. "
                                "Close may have been blocked by an unsaved-changes dialog."
                            )
                        else:
                            result["verification"] = "Window closed successfully."

            elif name in ("minimize_window", "maximize_window"):
                title_query = (args.get("title") or args.get("title_contains") or "").lower()
                if title_query:
                    win_result = await self._screen_navigator.list_all_windows()
                    if win_result.get("success"):
                        for w in win_result.get("windows", []):
                            if title_query in w.get("title", "").lower():
                                if name == "minimize_window" and w.get("minimized"):
                                    result["verification"] = "Window minimized."
                                elif name == "maximize_window" and w.get("maximized"):
                                    result["verification"] = "Window maximized."
                                else:
                                    result["verification_warning"] = (
                                        f"Window state may not have changed as expected."
                                    )
                                break

        except Exception as exc:
            log.debug("tool.verify.error", action=name, error=str(exc))
            # Verification is best-effort — don't fail the action
        return result

    async def execute_with_auto_capture(
        self, name: str, args: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a screen action tool, verify results, then auto-capture.

        Feedback loop (OpenClaw-inspired):
        1. Execute the screen action (click, type, scroll, etc.)
        2. Verify the action result (window state checks for window tools)
        3. Wait briefly for the UI to update
        4. Capture a new screenshot
        5. Send the screenshot to the cloud as a binary image frame

        Gemini sees: tool result + verification status + fresh screenshot
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
            return result  # Action failed — skip verification and capture

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

        # Step 2: Verify action result (window management actions)
        result = await self._verify_action(name, args, result)

        # Step 3: Brief pause for UI to settle
        await asyncio.sleep(0.15)

        # Step 4 + 5: Auto-capture and send
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
            # Vision-grounded navigation (Computer Use model)
            "smart_click": self._smart_click,
            # Windows power tools
            "open_application": self._open_application,
            "list_all_windows": self._list_all_windows,
            "get_active_window": self._get_active_window,
            "minimize_window": self._minimize_window,
            "maximize_window": self._maximize_window,
            "close_window": self._close_window,
            "resize_window": self._resize_window,
            "move_window": self._move_window,
            "list_processes": self._list_processes,
            "kill_process": self._kill_process,
            "get_clipboard": self._get_clipboard,
            "set_clipboard": self._set_clipboard,
            "get_screen_info": self._get_screen_info,
            # Persistent memory
            "get_task_status": self._get_task_status,
            "save_note": self._save_note,
            "get_notes": self._get_notes,
            "search_notes": self._search_notes,
            "export_context": self._export_context,
            "memory_stats": self._memory_stats,
            # GenMedia (Imagen 3 + Veo 2)
            "generate_image": self._generate_image,
            "generate_video": self._generate_video,
            # Web tools (E3)
            "web_search": self._web_search,
            "web_fetch": self._web_fetch,
            "web_cache_get": self._web_cache_get,
            # Long-running processes (E2)
            "start_process": self._start_process,
            "check_process": self._check_process,
            "stop_process": self._stop_process,
            # Browser automation (E1: Playwright CDP)
            "browser_connect": self._browser_connect,
            "browser_evaluate": self._browser_evaluate,
            "browser_fill_form": self._browser_fill_form,
            "browser_click_element": self._browser_click_element,
            "browser_extract_text": self._browser_extract_text,
            "browser_wait_for": self._browser_wait_for,
            "browser_navigate": self._browser_navigate,
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
        if not self._allowed_by_policy(resolved, "read"):
            return {"success": False, "error": f"Read denied by filesystem policy: {path}"}
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
            # B3: Truncate for model context window
            content = _truncate_output(content)
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
        if not self._allowed_by_policy(resolved, "write"):
            return {"success": False, "error": f"Write denied by filesystem policy: {path}"}
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
        if not self._allowed_by_policy(resolved, "write"):
            return {"success": False, "error": f"Patch denied by filesystem policy: {path}"}
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
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            output = stdout
            if stderr:
                output += "\n[stderr]\n" + stderr

            # Truncate large outputs
            if len(output) > MAX_OUTPUT_SIZE:
                output = output[:MAX_OUTPUT_SIZE] + "\n... [truncated]"
            # B3: Truncate for model context window
            output = _truncate_output(output)

            result: dict[str, Any] = {
                "success": proc.returncode == 0,
                "exit_code": proc.returncode,
                "output": output,
                "platform": sys.platform,
            }
            # Always include "error" key on failure so the agent can self-correct
            # (main.py display and the model both look for result["error"])
            if proc.returncode != 0:
                result["error"] = output or f"Command failed with exit code {proc.returncode}"
            return result
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {COMMAND_TIMEOUT}s",
            }
        except Exception as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

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

    async def _find_window(
        self,
        title: str = "",
        title_contains: str = "",
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        query = (title or title_contains).strip()
        if not query:
            return {"success": False, "error": "Missing required argument: title"}
        return await self._screen_navigator.find_window(query)

    async def _focus_window(
        self,
        title: str = "",
        title_contains: str = "",
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        query = (title or title_contains).strip()
        if not query:
            return {"success": False, "error": "Missing required argument: title"}
        return await self._screen_navigator.focus_window(query)

    # ------------------------------------------------------------------
    # Windows Power Tools (delegate to ScreenNavigator)
    # ------------------------------------------------------------------

    async def _open_application(self, name_or_path: str) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.open_application(name_or_path)

    async def _list_all_windows(self) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.list_all_windows()

    async def _get_active_window(self) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.get_active_window()

    async def _minimize_window(
        self, title: str = "", title_contains: str = "",
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        query = (title or title_contains).strip()
        if not query:
            return {"success": False, "error": "Missing required argument: title"}
        return await self._screen_navigator.minimize_window(query)

    async def _maximize_window(
        self, title: str = "", title_contains: str = "",
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        query = (title or title_contains).strip()
        if not query:
            return {"success": False, "error": "Missing required argument: title"}
        return await self._screen_navigator.maximize_window(query)

    async def _close_window(
        self, title: str = "", title_contains: str = "",
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        query = (title or title_contains).strip()
        if not query:
            return {"success": False, "error": "Missing required argument: title"}
        return await self._screen_navigator.close_window(query)

    async def _resize_window(
        self, title: str = "", title_contains: str = "",
        width: int = 800, height: int = 600,
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        query = (title or title_contains).strip()
        if not query:
            return {"success": False, "error": "Missing required argument: title"}
        return await self._screen_navigator.resize_window(query, int(width), int(height))

    async def _move_window(
        self, title: str = "", title_contains: str = "",
        x: int = 0, y: int = 0,
    ) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        query = (title or title_contains).strip()
        if not query:
            return {"success": False, "error": "Missing required argument: title"}
        return await self._screen_navigator.move_window(query, int(x), int(y))

    async def _list_processes(self, name_filter: str = "") -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.list_processes(name_filter)

    async def _kill_process(self, name_or_pid: str = "") -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        if not name_or_pid.strip():
            return {"success": False, "error": "Missing required argument: name_or_pid"}
        return await self._screen_navigator.kill_process(name_or_pid)

    async def _get_clipboard(self) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.get_clipboard()

    async def _set_clipboard(self, text: str) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.set_clipboard(text)

    async def _get_screen_info(self) -> dict[str, Any]:
        if err := self._nav_or_error():
            return err
        return await self._screen_navigator.get_screen_info()

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

    async def _save_note(self, key: str, value: str, category: str = "general") -> dict[str, Any]:
        """Save a persistent session note that survives across sessions."""
        if not hasattr(self, '_session_memory') or self._session_memory is None:
            return {"success": False, "error": "Session memory not available"}
        self._session_memory.set(key, value, category=category)
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

    async def _search_notes(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Search persistent notes by keyword. Returns matching notes ranked by relevance."""
        if not hasattr(self, '_session_memory') or self._session_memory is None:
            return {"success": False, "error": "Session memory not available"}
        results = self._session_memory.search(query, limit=limit)
        if not results:
            return {"success": True, "results": [], "message": "No matching notes found."}
        return {"success": True, "results": results, "count": len(results)}

    async def _export_context(self) -> dict[str, Any]:
        """Export all session memory to a compact context.txt file."""
        if not hasattr(self, '_session_memory') or self._session_memory is None:
            return {"success": False, "error": "Session memory not available"}
        from pathlib import Path
        ctx_path = str(Path(self._cwd) / "context.txt")
        content = self._session_memory.export_context(filepath=ctx_path)
        return {
            "success": True,
            "path": ctx_path,
            "size": len(content),
            "message": f"Context exported to {ctx_path} ({len(content)} chars)",
        }

    async def _memory_stats(self) -> dict[str, Any]:
        """Get memory system statistics: note count, total size, compaction status."""
        if not hasattr(self, '_session_memory') or self._session_memory is None:
            return {"success": False, "error": "Session memory not available"}
        stats = self._session_memory.get_stats()
        return {"success": True, **stats}

    # ------------------------------------------------------------------
    # Computer Use — official predefined-actions API
    # ------------------------------------------------------------------

    def _get_computer_use_client(self):
        """Lazy-create a genai Client for the computer-use model."""
        if hasattr(self, "_cu_client") and self._cu_client is not None:
            return self._cu_client

        from google import genai as _genai

        gcp_project = _get_env_value("GOOGLE_CLOUD_PROJECT")
        gcp_location = _get_env_value("GOOGLE_CLOUD_LOCATION") or "global"

        if gcp_project:
            self._cu_client = _genai.Client(
                vertexai=True,
                project=gcp_project,
                location=gcp_location,
            )
        else:
            api_key = _get_env_value("GEMINI_API_KEY")
            if not api_key:
                return None
            self._cu_client = _genai.Client(api_key=api_key)

        return self._cu_client

    def _get_screen_size(self) -> tuple[int, int]:
        """Get real screen resolution from screen capture metadata."""
        if self._screen_capture is not None:
            cr = self._screen_capture.get_last_capture_result()
            if cr is not None:
                return cr.original_width, cr.original_height
        # Fallback: try mss
        try:
            import mss
            with mss.mss() as sct:
                m = sct.monitors[1]
                return m["width"], m["height"]
        except Exception:
            return 1920, 1080  # Safe default

    def _denormalize_x(self, x: int) -> int:
        """Convert 0-1000 normalized X → real screen pixel X."""
        w, _ = self._get_screen_size()
        return int(x / 1000 * w)

    def _denormalize_y(self, y: int) -> int:
        """Convert 0-1000 normalized Y → real screen pixel Y."""
        _, h = self._get_screen_size()
        return int(y / 1000 * h)

    async def _capture_png_for_cu(self) -> bytes | None:
        """Capture a full-resolution screenshot as PNG for computer-use model feedback."""
        if self._screen_capture is None:
            return None
        try:
            jpeg = await self._screen_capture.capture_full_resolution_async()
            if jpeg is None:
                return None
            # Computer-use model prefers PNG for response; convert JPEG→PNG
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(jpeg))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:
            log.warning("cu.capture_png_failed", error=str(exc))
            return None

    async def _execute_cu_action(
        self, action_name: str, args: dict,
    ) -> dict[str, Any]:
        """Execute a single predefined computer-use action on the local screen.

        Handles all 13 official predefined actions + coordinate denormalization.
        Returns the action result dict.
        """
        nav = self._screen_navigator
        if nav is None or not nav.available:
            return {"success": False, "error": "Screen navigator not available"}

        log.info("cu.action", name=action_name, args=args)

        if action_name == "click_at":
            rx = self._denormalize_x(int(args.get("x", 0)))
            ry = self._denormalize_y(int(args.get("y", 0)))
            return await nav.click_absolute(rx, ry)

        elif action_name == "hover_at":
            rx = self._denormalize_x(int(args.get("x", 0)))
            ry = self._denormalize_y(int(args.get("y", 0)))
            return await nav.move(rx, ry)

        elif action_name == "type_text_at":
            rx = self._denormalize_x(int(args.get("x", 0)))
            ry = self._denormalize_y(int(args.get("y", 0)))
            text = args.get("text", "")
            press_enter = args.get("press_enter", False)
            clear_first = args.get("clear_before_typing", True)
            # Click the target position first
            click_result = await nav.click_absolute(rx, ry)
            if not click_result.get("success"):
                return click_result
            await asyncio.sleep(0.1)
            # Clear existing text if requested
            if clear_first:
                await nav.hotkey("ctrl+a")
                await asyncio.sleep(0.05)
            # Type the text
            result = await nav.type_text(text)
            if press_enter and result.get("success"):
                await asyncio.sleep(0.05)
                await nav.hotkey("enter")
            return result

        elif action_name == "scroll_at":
            rx = self._denormalize_x(int(args.get("x", 0)))
            ry = self._denormalize_y(int(args.get("y", 0)))
            direction = args.get("direction", "down")
            amount = int(args.get("amount", 3))
            clicks = amount if direction == "up" else -amount
            return await nav.scroll(rx, ry, clicks)

        elif action_name == "scroll_document":
            direction = args.get("direction", "down")
            amount = int(args.get("amount", 3))
            clicks = amount if direction == "up" else -amount
            # Scroll at screen center
            w, h = self._get_screen_size()
            return await nav.scroll(w // 2, h // 2, clicks)

        elif action_name == "key_combination":
            keys = args.get("keys", [])
            if isinstance(keys, list):
                key_str = "+".join(keys)
            else:
                key_str = str(keys)
            return await nav.hotkey(key_str)

        elif action_name == "drag_and_drop":
            sx = self._denormalize_x(int(args.get("startX", args.get("start_x", 0))))
            sy = self._denormalize_y(int(args.get("startY", args.get("start_y", 0))))
            ex = self._denormalize_x(int(args.get("endX", args.get("end_x", 0))))
            ey = self._denormalize_y(int(args.get("endY", args.get("end_y", 0))))
            return await nav.drag(sx, sy, ex, ey)

        elif action_name == "wait_5_seconds":
            await asyncio.sleep(5)
            return {"success": True, "action": "wait", "duration": 5}

        elif action_name == "navigate":
            url = args.get("url", "")
            if url:
                await nav.hotkey("ctrl+l")
                await asyncio.sleep(0.2)
                await nav.type_text(url)
                await asyncio.sleep(0.1)
                await nav.hotkey("enter")
            return {"success": True, "action": "navigate", "url": url}

        elif action_name == "go_back":
            return await nav.hotkey("alt+left")

        elif action_name == "go_forward":
            return await nav.hotkey("alt+right")

        elif action_name == "search":
            query = args.get("query", "")
            await nav.hotkey("ctrl+l")
            await asyncio.sleep(0.2)
            await nav.type_text(query)
            await asyncio.sleep(0.1)
            return await nav.hotkey("enter")

        elif action_name == "open_web_browser":
            import subprocess as _sp
            try:
                _sp.Popen("start msedge", shell=True, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            except Exception:
                _sp.Popen("start chrome", shell=True, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            await asyncio.sleep(1.5)
            return {"success": True, "action": "open_web_browser"}

        else:
            log.warning("cu.unknown_action", name=action_name)
            return {"success": False, "error": f"Unknown computer-use action: {action_name}"}

    async def _computer_use_ground(
        self, description: str,
    ) -> dict[str, Any]:
        """Use the official Computer Use predefined-actions API to locate
        an element on screen.

        Follows the official Google agentic loop pattern:
          1. Take screenshot → send with task description
          2. Model returns predefined actions (click_at, type_text_at, etc.)
             with 0-1000 normalized coordinates
          3. Execute actions, capture new screenshot
          4. Send FunctionResponse with screenshot as FunctionResponseBlob
          5. Loop until model returns no more actions or max turns reached
          6. Return coordinates of the first click_at action
        """
        if self._screen_capture is None:
            return {"success": False, "error": "Screen capture not attached"}

        # Take initial screenshot
        try:
            screenshot = await self._screen_capture.capture_full_resolution_async()
        except Exception as exc:
            return {"success": False, "error": f"Screenshot failed: {exc}"}
        if screenshot is None:
            return {"success": False, "error": "Screenshot returned None"}

        client = self._get_computer_use_client()
        if client is None:
            return {
                "success": False,
                "error": (
                    "Neither GOOGLE_CLOUD_PROJECT (for Vertex AI) nor "
                    "GEMINI_API_KEY is set. Computer Use model requires "
                    "Vertex AI or paid billing."
                ),
            }

        try:
            from google.genai import types as _gtypes

            # Resolve CU model robustly across run modes:
            # - package import (rio.local.config)
            # - local import (config.py beside main runtime)
            # - env/default fallback
            try:
                from rio.local.config import get_model as _get_model  # type: ignore
            except Exception:
                try:
                    from config import get_model as _get_model  # type: ignore
                except Exception:
                    _get_model = None

            if callable(_get_model):
                cu_model = _get_model("computer_use")
            else:
                cu_model = os.environ.get(
                    "COMPUTER_USE_MODEL",
                    os.environ.get("CU_MODEL", "gemini-2.5-computer-use-preview-10-2025"),
                )

            # Build initial contents with screenshot + task description
            contents: list = [
                _gtypes.Content(role="user", parts=[
                    _gtypes.Part(text=(
                        f"Find and click on: {description}\n"
                        f"If you can see the element, use click_at to click it."
                    )),
                    _gtypes.Part.from_bytes(
                        data=screenshot, mime_type="image/jpeg",
                    ),
                ])
            ]

            # Official Computer Use config (matches Google reference impl)
            config = _gtypes.GenerateContentConfig(
                temperature=1.0,
                top_p=0.95,
                max_output_tokens=8192,
                tools=[
                    _gtypes.Tool(
                        computer_use=_gtypes.ComputerUse(
                            environment=_gtypes.Environment.ENVIRONMENT_BROWSER,
                        ),
                    ),
                ],
            )

            first_click = None
            max_turns = 3  # Safety limit — most clicks resolve in 1 turn

            for turn in range(max_turns):
                # --- Call model with retry for transient 500s ---
                response = None
                last_exc = None
                for attempt in range(3):
                    try:
                        response = await client.aio.models.generate_content(
                            model=cu_model,
                            contents=contents,
                            config=config,
                        )
                        break
                    except Exception as retry_exc:
                        last_exc = retry_exc
                        err_str = str(retry_exc)
                        if "500" in err_str or "503" in err_str or "INTERNAL" in err_str:
                            wait = (attempt + 1) * 0.5
                            log.warning("cu.retry", attempt=attempt + 1, wait=wait, error=err_str[:200])
                            await asyncio.sleep(wait)
                            continue
                        raise
                else:
                    log.error("cu.ground_error", description=description, model=cu_model, error=str(last_exc))
                    return {
                        "success": False,
                        "error": f"Computer Use model error after 3 attempts ({cu_model}): {last_exc}",
                    }

                # --- Parse model response ---
                candidate = response.candidates[0] if response.candidates else None
                if candidate is None or candidate.content is None:
                    return {"success": False, "error": "No response from Computer Use model"}

                # Append model response to conversation history
                contents.append(candidate.content)

                # Extract function calls and thoughts
                function_calls = []
                thoughts = []
                for part in candidate.content.parts or []:
                    fc = getattr(part, "function_call", None)
                    if fc:
                        function_calls.append(fc)
                    elif hasattr(part, "text") and part.text:
                        thoughts.append(part.text)

                if thoughts:
                    log.debug("cu.reasoning", text=" ".join(thoughts)[:300])

                # No actions → model is done (or element not found)
                if not function_calls:
                    if first_click:
                        break  # We already found what we needed
                    text = " ".join(thoughts) if thoughts else "(no explanation)"
                    return {
                        "success": False,
                        "error": f"Element not found on screen: {description}. Model said: {text[:300]}",
                    }

                # --- Execute actions and capture feedback ---
                fn_response_parts = []
                any_action_executed = False
                for fc in function_calls:
                    log.info("cu.action", name=fc.name, args=fc.args)

                    # Record first click_at coordinates
                    if fc.name == "click_at" and first_click is None:
                        nx = int(fc.args.get("x", 0))
                        ny = int(fc.args.get("y", 0))
                        first_click = {
                            "normalized_x": nx,
                            "normalized_y": ny,
                            "x": self._denormalize_x(nx),
                            "y": self._denormalize_y(ny),
                        }

                    # Execute the model-requested UI action so follow-up model
                    # reasoning sees true post-action state.
                    action_result = await self._execute_cu_action(fc.name, dict(fc.args or {}))
                    any_action_executed = True

                    # Build FunctionResponse with screenshot blob
                    # (matches official Google pattern)
                    fn_response_parts.append(
                        _gtypes.Part(
                            function_response=_gtypes.FunctionResponse(
                                name=fc.name,
                                response=action_result,
                            ),
                        ),
                    )

                # Capture a fresh screenshot AFTER executing action(s).
                # If no action was executed (rare), keep the previous frame.
                if any_action_executed:
                    try:
                        new_screenshot = await self._screen_capture.capture_full_resolution_async()
                    except Exception:
                        new_screenshot = screenshot
                else:
                    new_screenshot = screenshot

                # Append function responses + new screenshot to history
                # (Separate screenshot part — more reliable than FunctionResponseBlob
                #  which may not be supported by all model versions)
                feedback_parts = fn_response_parts + [
                    _gtypes.Part.from_bytes(
                        data=new_screenshot if new_screenshot else screenshot,
                        mime_type="image/jpeg",
                    ),
                ]
                contents.append(_gtypes.Content(role="user", parts=feedback_parts))

                # If we found a click target, we're done
                if first_click:
                    break

            if first_click is None:
                return {
                    "success": False,
                    "error": f"Element not found after {max_turns} turns: {description}",
                }

            return {
                "success": True,
                "x": first_click["x"],
                "y": first_click["y"],
                "normalized": {
                    "x": first_click["normalized_x"],
                    "y": first_click["normalized_y"],
                },
                "method": "computer_use_predefined_actions",
            }

        except Exception as exc:
            log.exception("cu.ground_error", description=description, error=str(exc))
            return {"success": False, "error": f"Computer Use model error: {exc}"}

    async def _smart_click(
        self,
        target: str,
        action: str = "click",
        clicks: int = 1,
    ) -> dict[str, Any]:
        """Visual-grounding click using the official Computer Use predefined-actions API.

        Describe the element you want to click in natural language
        (e.g. 'the Save button', 'search input field').
        Uses gemini-computer-use-preview with the official
        types.Tool(computer_use=ComputerUse(...)) API and 0-1000 normalized
        coordinates for reliable element detection.
        """
        if err := self._nav_or_error():
            return err

        log.info("tool.smart_click", target=target, action=action)

        ground = await self._computer_use_ground(target)
        if not ground.get("success"):
            return ground

        x, y = ground["x"], ground["y"]
        button = "right" if action == "right_click" else "left"
        n_clicks = 2 if action == "double_click" else int(clicks)

        # Coordinates are already denormalized to real screen pixels
        result = await self._screen_navigator.click_absolute(x, y, button=button, clicks=n_clicks)
        result["grounded_by"] = "computer_use_official_api"
        result["found_at"] = {"x": x, "y": y}
        if "normalized" in ground:
            result["normalized_coords"] = ground["normalized"]
        result["target"] = target
        return result

    # ------------------------------------------------------------------
    # GenMedia — Imagen 3 + Veo 2 (proxied through CreativeAgent)
    # ------------------------------------------------------------------

    def _get_creative(self):
        """Lazy-load CreativeAgent for GenMedia tools."""
        if self._creative_agent is None:
            try:
                from creative_agent import CreativeAgent
                self._creative_agent = CreativeAgent()
            except ImportError:
                return None
        return self._creative_agent

    async def _generate_image(
        self, prompt: str, aspect_ratio: str = "1:1",
        style: str = "", negative_prompt: str = "",
    ) -> dict[str, Any]:
        """Generate an image using Imagen 3."""
        agent = self._get_creative()
        if agent is None or not agent.available:
            return {"success": False, "error": "CreativeAgent not available"}
        full_prompt = prompt
        if negative_prompt:
            full_prompt += f". Avoid: {negative_prompt}"
        return await agent.generate_image(full_prompt, style=style)

    async def _generate_video(
        self, prompt: str, duration_seconds: int = 5,
        aspect_ratio: str = "16:9",
    ) -> dict[str, Any]:
        """Generate a short video using Veo 2."""
        agent = self._get_creative()
        if agent is None or not agent.available:
            return {"success": False, "error": "CreativeAgent not available"}
        return await agent.generate_video(
            prompt, duration_seconds=duration_seconds, aspect_ratio=aspect_ratio,
        )

    # ------------------------------------------------------------------
    # Web tools (E3)
    # ------------------------------------------------------------------

    async def _web_search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the web via DuckDuckGo."""
        from web_tools import web_search
        return web_search(query, max_results=max_results)

    async def _web_fetch(self, url: str, max_chars: int = 8000) -> dict[str, Any]:
        """Fetch a web page and return text content."""
        from web_tools import web_fetch
        return web_fetch(url, max_chars=max_chars)

    async def _web_cache_get(self, url: str) -> dict[str, Any]:
        """Get a cached web page or fetch and cache it."""
        from web_tools import web_cache_get
        return web_cache_get(url)

    # ------------------------------------------------------------------
    # Long-running process management (E2)
    # ------------------------------------------------------------------

    _background_procs: dict[str, subprocess.Popen] = {}

    async def _start_process(self, command: str, label: str = "") -> dict[str, Any]:
        """Start a long-running process in the background (servers, watchers).
        Returns a process ID for status checks and cleanup."""
        # Safety: apply command blocklist
        for pattern in COMMAND_BLOCKLIST:
            if re.search(pattern, command, re.IGNORECASE):
                return {"success": False, "error": f"Blocked command pattern: {pattern}"}

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            pid = str(proc.pid)
            ToolExecutor._background_procs[pid] = proc
            log.info("process.started", pid=pid, command=command[:80], label=label)
            return {
                "success": True,
                "pid": pid,
                "label": label or command[:40],
                "message": f"Process started with PID {pid}",
            }
        except Exception as exc:
            return {"success": False, "error": f"Failed to start: {exc}"}

    async def _check_process(self, pid: str) -> dict[str, Any]:
        """Check the status of a background process."""
        proc = ToolExecutor._background_procs.get(pid)
        if proc is None:
            return {"success": False, "error": f"No tracked process with PID {pid}"}

        poll = proc.poll()
        if poll is None:
            return {"success": True, "pid": pid, "status": "running"}
        else:
            # Process has finished — read output
            stdout, stderr = "", ""
            try:
                stdout = proc.stdout.read()[:4000] if proc.stdout else ""
                stderr = proc.stderr.read()[:2000] if proc.stderr else ""
            except Exception:
                pass
            ToolExecutor._background_procs.pop(pid, None)
            return {
                "success": True,
                "pid": pid,
                "status": "exited",
                "exit_code": poll,
                "stdout": stdout,
                "stderr": stderr,
            }

    async def _stop_process(self, pid: str) -> dict[str, Any]:
        """Stop a background process by PID."""
        proc = ToolExecutor._background_procs.get(pid)
        if proc is None:
            return {"success": False, "error": f"No tracked process with PID {pid}"}

        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            ToolExecutor._background_procs.pop(pid, None)
            log.info("process.stopped", pid=pid)
            return {"success": True, "pid": pid, "message": f"Process {pid} stopped"}
        except Exception as exc:
            return {"success": False, "error": f"Failed to stop: {exc}"}

    def cleanup_processes(self) -> None:
        """Kill all tracked background processes (call on session disconnect)."""
        for pid, proc in list(ToolExecutor._background_procs.items()):
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            log.info("process.cleanup", pid=pid)
        ToolExecutor._background_procs.clear()

    # ------------------------------------------------------------------
    # Browser automation tools (E1: Playwright CDP)
    # ------------------------------------------------------------------

    def _get_browser_mod(self):
        """Lazy-import browser_tools module."""
        if not hasattr(self, "_browser_mod"):
            try:
                from browser_tools import (
                    browser_connect, browser_evaluate, browser_fill_form,
                    browser_click_element, browser_extract_text,
                    browser_wait_for, browser_screenshot, browser_navigate,
                )
                self._browser_mod = {
                    "connect": browser_connect,
                    "evaluate": browser_evaluate,
                    "fill_form": browser_fill_form,
                    "click_element": browser_click_element,
                    "extract_text": browser_extract_text,
                    "wait_for": browser_wait_for,
                    "screenshot": browser_screenshot,
                    "navigate": browser_navigate,
                }
            except ImportError:
                self._browser_mod = None
                log.info("browser_tools.unavailable", note="Install playwright for browser automation")
        return self._browser_mod

    async def _browser_connect(
        self,
        cdp_url: str = "http://127.0.0.1:9222",
        browser: str = "",
        profile: str = "",
    ) -> dict[str, Any]:
        mod = self._get_browser_mod()
        if mod is None:
            return {"success": False, "error": "Playwright not installed. Run: pip install playwright && playwright install chromium"}
        # Apply defaults from config.yaml browser section if not explicitly provided
        if not browser or not profile:
            try:
                from config import _load_raw_browser_config
                browser_cfg = _load_raw_browser_config()
            except Exception:
                browser_cfg = {}
            if not browser:
                browser = browser_cfg.get("default_browser", "auto")
            if not profile:
                profile = browser_cfg.get("default_profile", "rio")
        return await mod["connect"](cdp_url, browser=browser, profile=profile)

    async def _browser_evaluate(self, javascript: str, cdp_url: str = "http://localhost:9222") -> dict[str, Any]:
        mod = self._get_browser_mod()
        if mod is None:
            return {"success": False, "error": "Playwright not installed"}
        return await mod["evaluate"](javascript, cdp_url)

    async def _browser_fill_form(self, selector: str, value: str, cdp_url: str = "http://localhost:9222") -> dict[str, Any]:
        mod = self._get_browser_mod()
        if mod is None:
            return {"success": False, "error": "Playwright not installed"}
        return await mod["fill_form"](selector, value, cdp_url)

    async def _browser_click_element(self, selector: str, cdp_url: str = "http://localhost:9222") -> dict[str, Any]:
        mod = self._get_browser_mod()
        if mod is None:
            return {"success": False, "error": "Playwright not installed"}
        return await mod["click_element"](selector, cdp_url)

    async def _browser_extract_text(self, selector: str, cdp_url: str = "http://localhost:9222") -> dict[str, Any]:
        mod = self._get_browser_mod()
        if mod is None:
            return {"success": False, "error": "Playwright not installed"}
        return await mod["extract_text"](selector, cdp_url)

    async def _browser_wait_for(self, selector: str, timeout: int = 30000, cdp_url: str = "http://localhost:9222") -> dict[str, Any]:
        mod = self._get_browser_mod()
        if mod is None:
            return {"success": False, "error": "Playwright not installed"}
        return await mod["wait_for"](selector, timeout, cdp_url)

    async def _browser_navigate(self, url: str, cdp_url: str = "http://localhost:9222") -> dict[str, Any]:
        mod = self._get_browser_mod()
        if mod is None:
            return {"success": False, "error": "Playwright not installed"}
        return await mod["navigate"](url, cdp_url)
