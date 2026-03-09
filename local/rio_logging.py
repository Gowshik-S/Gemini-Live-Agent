"""
Rio — Structured Logging System

Provides subsystem-aware logging with file output, console formatting,
and diagnostic severity classification.

Inspired by OpenClaw's logging architecture:
  - Subsystem tags (e.g., "orchestrator", "browser_agent", "fallback")
  - File-based rotating log output
  - Console color formatting by severity
  - Session-scoped diagnostic state

Usage::

    from rio_logging import setup_logging, get_logger

    setup_logging(log_dir="rio/logs", verbose=True)
    log = get_logger("browser_agent")
    log.info("started", model="gemini-2.5-flash")
    log.error("failed", error="timeout", diagnostic="Check internet connection")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog


# ---------------------------------------------------------------------------
# Log directory management
# ---------------------------------------------------------------------------

_log_dir: Optional[Path] = None
_log_file = None
_verbose = False
_initialized = False


def setup_logging(
    log_dir: str | Path = "rio/logs",
    verbose: bool = False,
    max_files: int = 7,
) -> Path:
    """Initialize the Rio logging system.

    - Creates log directory if needed
    - Opens today's log file
    - Configures structlog processors
    - Cleans up old log files (keeps max_files days)

    Returns the path to today's log file.
    """
    global _log_dir, _log_file, _verbose, _initialized

    _log_dir = Path(log_dir)
    _log_dir.mkdir(parents=True, exist_ok=True)
    _verbose = verbose

    # Today's log file
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = _log_dir / f"rio-{today}.log"
    _log_file = open(log_path, "a", encoding="utf-8", buffering=1)  # Line buffered

    # Clean up old logs
    _cleanup_old_logs(_log_dir, max_files)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_subsystem,
            _rio_renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if verbose else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _initialized = True
    return log_path


def _cleanup_old_logs(log_dir: Path, keep: int) -> None:
    """Remove log files older than `keep` days."""
    log_files = sorted(log_dir.glob("rio-*.log"))
    while len(log_files) > keep:
        old = log_files.pop(0)
        try:
            old.unlink()
        except OSError:
            pass


def _add_subsystem(logger, method_name, event_dict):
    """Ensure subsystem tag is present."""
    if "subsystem" not in event_dict:
        event_dict["subsystem"] = event_dict.get("_name", "rio")
    return event_dict


# ---------------------------------------------------------------------------
# Console color formatting
# ---------------------------------------------------------------------------

_COLORS = {
    "debug": "\033[90m",     # Gray
    "info": "\033[36m",      # Cyan
    "warning": "\033[33m",   # Yellow
    "error": "\033[31m",     # Red
    "critical": "\033[91m",  # Bright red
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _rio_renderer(logger, method_name, event_dict):
    """Custom renderer that outputs to both console and file."""
    level = event_dict.pop("level", method_name)
    timestamp = event_dict.pop("timestamp", "")
    subsystem = event_dict.pop("subsystem", "rio")
    event = event_dict.pop("event", "")

    # Remove internal keys
    event_dict.pop("_name", None)

    # Build structured line for file
    file_entry = {
        "ts": timestamp,
        "level": level,
        "sub": subsystem,
        "event": event,
        **event_dict,
    }

    # Write to log file
    if _log_file is not None:
        try:
            _log_file.write(json.dumps(file_entry, default=str) + "\n")
        except Exception:
            pass

    # Console output with colors
    color = _COLORS.get(level, "")
    level_tag = f"[{level.upper():>7s}]"

    # Format key=value pairs
    extras = ""
    if event_dict:
        # Show diagnostic prominently
        diag = event_dict.pop("diagnostic", None)
        pairs = " ".join(f"{k}={v}" for k, v in event_dict.items())
        if pairs:
            extras = f" {_DIM}{pairs}{_RESET}"
        if diag:
            extras += f"\n           {_BOLD}{diag}{_RESET}"

    return f"{_DIM}{timestamp[11:19]}{_RESET} {color}{level_tag}{_RESET} {_DIM}[{subsystem}]{_RESET} {event}{extras}"


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def get_logger(subsystem: str) -> structlog.BoundLogger:
    """Get a logger bound to a subsystem name.

    If logging hasn't been initialized, returns a basic structlog logger.
    """
    return structlog.get_logger(_name=subsystem, subsystem=subsystem)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

class DiagnosticLevel:
    """Severity levels for diagnostic messages."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def log_diagnostic(
    subsystem: str,
    level: str,
    title: str,
    detail: str = "",
    suggestion: str = "",
) -> None:
    """Log a structured diagnostic message.

    These are high-level actionable messages meant for the user.
    They appear prominently in console and are tagged in the log file.
    """
    logger = get_logger(subsystem)
    msg = title
    if detail:
        msg += f" — {detail}"

    log_method = getattr(logger, level, logger.info)
    log_method(msg, diagnostic=suggestion if suggestion else None)

    # Also print to stderr for immediate visibility
    if level in ("error", "critical"):
        color = _COLORS.get(level, "")
        print(f"\n  {color}[RIO {level.upper()}] {title}{_RESET}", file=sys.stderr)
        if detail:
            print(f"  {detail}", file=sys.stderr)
        if suggestion:
            print(f"  {_BOLD}Fix: {suggestion}{_RESET}", file=sys.stderr)


def get_recent_logs(n: int = 50) -> list[dict]:
    """Read the last N entries from today's log file for dashboard display."""
    if _log_dir is None:
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = _log_dir / f"rio-{today}.log"

    if not log_path.exists():
        return []

    entries = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        entries.append({"event": line, "level": "info"})
    except OSError:
        return []

    return entries[-n:]


def get_log_file_path() -> Optional[str]:
    """Return the current log file path."""
    if _log_dir is None:
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return str(_log_dir / f"rio-{today}.log")
