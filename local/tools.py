"""
Rio Local -- Tool Executor (Day 8 / L3)

Executes tool calls from Gemini on the local machine.

Available tools:
  - read_file(path)         -- Read file contents (auto-approve)
  - write_file(path, content) -- Backup to .rio.bak, then write
  - patch_file(path, old_text, new_text) -- Find-and-replace with backup
  - run_command(command)     -- Shell command with 30s timeout + blocklist

Security:
  - Dangerous commands (rm -rf /, format, dd, fork bombs) are blocked
  - File writes always create .rio.bak backups
  - Command output is truncated at 100KB to prevent memory issues
  - All operations are logged via structlog
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

# Shell commands that are always blocked (case-insensitive regex patterns)
COMMAND_BLOCKLIST = [
    r"\brm\s+(-\w+\s+)*-r\w*\s+/\s*$",     # rm -rf /
    r"\brm\s+(-\w+\s+)*-rf\s+/\s*$",        # rm -rf /
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
        """Resolve a path relative to the working directory."""
        p = Path(path)
        if not p.is_absolute():
            p = Path(self._cwd) / p
        return p.resolve()

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
