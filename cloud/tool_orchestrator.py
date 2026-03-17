"""
Rio Tool Orchestrator — Multi-Agent Tool Execution Engine.

When the native audio Live model cannot invoke function calls itself
(e.g. ``gemini-2.5-flash-native-audio-preview-*``), this orchestrator
provides full agentic tool execution using a capable model.

Architecture
------------
  [Live Audio Model]  ← voice I/O only  (native-audio Live session)
        ↕  transcription
  [ToolOrchestrator]  ← agentic executor (gemini-3-flash-preview, full tool calling)
        ↕  ToolBridge
  [Local Client]      ← actual tool execution on the user's machine
        ↕
  [Computer Use Model]← visual grounding for smart_click
                        (gemini-computer-use-preview)

Flow
----
  1. User speech is transcribed by the live session (input_audio_transcription).
  2. _is_task_request(text) decides whether execution is needed.
  3. ToolOrchestrator.run_task(goal, inject_fn) is spawned as a background Task.
  4. The orchestrator calls gemini-3-flash-preview via generate_content with the
     full tool list.  generate_content() auto-converts callables → schemas.
  5. Model returns function_call parts → dispatched through ToolBridge → client.
  6. Tool result returned to model as function_response → next iteration.
  7. Loop until model returns a final text response (no more function_calls).
  8. inject_fn sends "[SYSTEM: task complete. Result: ...]" into the live session
     so the audio model speaks the result naturally.
  9. Live audio model says "Done — here's what I did..." to the user.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import os
import pathlib
import time
from enum import Enum
from typing import Any, Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)

# Default orchestrator model.  Must support function calling + vision.
# gemini-3-flash-preview has full tool support and is the agentic workhorse.
# Override with ORCHESTRATOR_MODEL env var.
_DEFAULT_ORCHESTRATOR_MODEL = "gemini-3-flash-preview"

# Hard safety cap: maximum tool-call iterations per task
_MAX_ITERATIONS = 50

# How many past task summaries to keep in the session context
_MAX_TASK_MEMORY = 20

# ---------------------------------------------------------------------------
# Context compaction thresholds (A6)
# ---------------------------------------------------------------------------
# Triggered earlier (50% of model context) so the model never approaches
# the hard limit. At ~4 chars/token a 1M-token model gives ~4M chars;
# 400K chars ≈ 100K tokens — comfortable headroom for long sessions.
_CONTEXT_WARN_CHARS = 300_000     # ~75K tokens  — log a warning
_CONTEXT_COMPACT_CHARS = 400_000  # ~100K tokens — trigger compaction

# ---------------------------------------------------------------------------
# Strip old tool results from context (A7)
# ---------------------------------------------------------------------------
_MAX_TOOL_RESULT_CHARS = 4000     # Full result limit per tool call
_RESULT_STRIP_AFTER_TURNS = 3     # Strip results older than N turns

_DEFAULT_TOOL_TIMEOUT_SECONDS = 120
_DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 5
_DEFAULT_MAX_TOOL_CALLS_PER_TASK = 120
_DEFAULT_MAX_COST_POINTS_PER_TASK = 200
_DEFAULT_SESSION_PERSIST_INTERVAL_SECONDS = 3
_DEFAULT_APPROVAL_TIMEOUT_SECONDS = 0


def _load_orchestrator_settings() -> dict[str, Any]:
    """Load orchestrator settings from config.yaml with safe defaults."""
    from pathlib import Path as _P

    defaults: dict[str, Any] = {
        "tool_timeout_seconds": _DEFAULT_TOOL_TIMEOUT_SECONDS,
        "heartbeat_interval_seconds": _DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        "approval_timeout_seconds": _DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        "session_persist_interval_seconds": _DEFAULT_SESSION_PERSIST_INTERVAL_SECONDS,
        "tool_policy": {
            "default_per_minute": 20,
            "max_calls_per_task": _DEFAULT_MAX_TOOL_CALLS_PER_TASK,
            "max_cost_points_per_task": _DEFAULT_MAX_COST_POINTS_PER_TASK,
            "cost_points": {},
            "per_tool_per_minute": {},
        },
        "composite_routing": {
            "enabled": True,
            "simple_model": "gemini-3-flash-preview",
            "complex_model": "gemini-3-pro-preview",
            "simple_max_words": 22,
            "complexity_keywords": [
                "analyze", "architecture", "tradeoff", "security", "research",
                "refactor", "optimize", "multi-step", "design", "plan",
            ],
        },
        "prompt_fragments": {
            "base": "",
            "routing": "",
            "safety": "",
            "verification": "",
            "compaction": "",
        },
        "loop_detection": {
            "warning_at": 2,
            "strategy_at": 3,
            "stop_at": 4,
        },
        "semantic_compaction": {
            "keep_middle_items": 8,
            "high_value_keywords": [
                "error", "failed", "path", "note", "decision", "todo", "retry",
                "blocked", "fix", "approved", "denied", "security", "quota",
            ],
        },
    }

    try:
        import yaml

        for candidate in (
            _P(__file__).resolve().parent.parent / "config.yaml",
            _P(os.getcwd()) / "config.yaml",
        ):
            if not candidate.is_file():
                continue
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            loaded = data.get("rio", {}).get("orchestrator", {}) or {}
            merged = dict(defaults)
            for key in (
                "tool_timeout_seconds",
                "heartbeat_interval_seconds",
                "approval_timeout_seconds",
                "session_persist_interval_seconds",
            ):
                if key in loaded:
                    merged[key] = loaded[key]
            for key in (
                "tool_policy",
                "composite_routing",
                "prompt_fragments",
                "loop_detection",
                "semantic_compaction",
            ):
                merged[key] = {
                    **defaults.get(key, {}),
                    **(loaded.get(key, {}) or {}),
                }
            return merged
    except Exception:
        pass

    return defaults


def _load_prompt_fragment_files(config_fragments: dict[str, str]) -> dict[str, str]:
    """Load prompt fragments from inline strings or prompt fragment files."""
    out: dict[str, str] = {}
    base_dir = pathlib.Path(__file__).resolve().parent.parent / "prompts" / "orchestrator"
    for key in ("base", "routing", "safety", "verification", "compaction"):
        raw = str(config_fragments.get(key, "") or "").strip()
        if not raw:
            out[key] = ""
            continue
        if "\n" in raw or len(raw.split()) > 8:
            out[key] = raw
            continue
        candidate = base_dir / raw
        if candidate.is_file():
            try:
                out[key] = candidate.read_text(encoding="utf-8").strip()
                continue
            except Exception:
                pass
        out[key] = raw
    return out


def _strip_tool_result(result: dict, max_chars: int = _MAX_TOOL_RESULT_CHARS) -> dict:
    """Reduce a tool result to essential info for compacted context."""
    stripped = {"success": result.get("success", True)}

    # Preserve error messages in full
    if "error" in result:
        stripped["error"] = str(result["error"])[:500]
        return stripped

    # For successful results, summarize long content
    content = str(result.get("result", result.get("content", "")))
    if len(content) > max_chars:
        stripped["result_summary"] = (
            content[:200]
            + f"... [{len(content)} chars total, truncated]"
        )
    else:
        stripped["result"] = content

    return stripped


def _format_for_voice(text: str) -> str:
    """Strip markdown formatting and filter technical terms so TTS reads text naturally.

    Removes: headers (#), bold/italic (*_), code fences (```), bullet
    dashes/stars at line start, and URL markdown [text](url) → text.

    Filters technical terms: agent names, tool names, model references, and
    technical terminology, replacing them with natural conversational language.

    The result is plain prose suitable for voice output.
    """
    import re

    # First, strip markdown formatting
    # Unwrap markdown links: [label](url) → label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove raw URLs so they are not spoken.
    text = re.sub(r"https?://\S+", "", text)
    # Remove code fences
    text = re.sub(r"```[^\n]*\n?", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("`", "")
    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\s*#+\s*$", "", text)
    # Remove bold/italic markers without treating in-word underscores as markdown.
    text = re.sub(r"(?<!\w)(\*{1,3}|_{1,3})([\s\S]*?)\1(?!\w)", r"\2", text)
    # Remove bullet points at line starts
    text = re.sub(r"^\s*[-*+•]\s+", "", text, flags=re.MULTILINE)
    # Remove standalone leftover emphasis markers and collapse extra blank lines.
    text = re.sub(r"(?m)^\s*(\*{1,3}|_{1,3})\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove structured REPORT schema lines (machine-facing, not user-facing).
    text = re.sub(
        r"(?im)^\s*(SCREEN|USER_INPUT|INTERRUPT|CONTEXT|GROUNDED|AUTH_REQUIRED|AUTH_RESOLVED)\s*:\s*.*$",
        "",
        text,
    )

    # Remove leaked bare status tokens if they appear in prose.
    text = re.sub(r"(?i)\b(?:grounded|interrupt|auth_required|auth_resolved)\s*[:=]?\s*(?:true|false)\b", "", text)

    # Now filter technical terms for natural voice output.
    natural_language_map = {
        r'\bbrowser_connect\b': 'opening the browser',
        r'\bsmart_click\b': 'clicking',
        r'\bbrowser_navigate\b': 'navigating to',
        r'\bcapture_screen\b': 'taking a screenshot',
        r'\bbrowser_click_element\b': 'clicking',
        r'\bbrowser_type_text\b': 'typing',
        r'\bbrowser_scroll\b': 'scrolling',
        r'\bbrowser_wait\b': 'waiting',
        r'\bbrowser_close\b': 'closing the browser',
        r'\btool execution\b': 'action',
        r'\bfunction calling\b': 'action',
        r'\bCSS selector\b': 'element',
        r'\bPlaywright\b': 'browser automation',
        r'\bCDP\b': 'browser control',
    }
    for pattern, replacement in natural_language_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Remove technical labels and internal implementation identifiers.
    text = re.sub(r'\bAgent:\s*', 'Agent ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:browser_agent|computer_use_agent|task_executor|code_agent)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:browseragent|computeruseagent|taskexecutor|codeagent)\b', '', text, flags=re.IGNORECASE)

    # Remove model references like "(model: gemini-...)" and "using model ...".
    text = re.sub(r'\(model:\s*[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\busing\s+model\b[^,.;\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmodel:\s*[^,.;\n]*', 'Model', text, flags=re.IGNORECASE)
    text = re.sub(r'\bgemini-[\w.-]+\b', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\btool\b', '', text, flags=re.IGNORECASE)

    # Clean up punctuation/whitespace left after removals.
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()




# ---------------------------------------------------------------------------
# Tool risk classification (B1)
# ---------------------------------------------------------------------------

class ToolRisk(str, Enum):
    SAFE = "safe"           # Read-only, no side effects
    MODERATE = "moderate"   # Writes/clicks, reversible
    DANGEROUS = "dangerous" # Shell exec, process kill, irreversible
    CRITICAL = "critical"   # Reserved for future destructive ops


TOOL_RISK_MAP: dict[str, ToolRisk] = {
    # Safe — read-only
    "read_file": ToolRisk.SAFE,
    "capture_screen": ToolRisk.SAFE,
    "get_notes": ToolRisk.SAFE,
    "search_notes": ToolRisk.SAFE,
    "get_clipboard": ToolRisk.SAFE,
    "get_screen_info": ToolRisk.SAFE,
    "list_all_windows": ToolRisk.SAFE,
    "get_active_window": ToolRisk.SAFE,
    "find_window": ToolRisk.SAFE,
    "list_processes": ToolRisk.SAFE,
    "get_task_status": ToolRisk.SAFE,
    "memory_stats": ToolRisk.SAFE,
    "export_context": ToolRisk.SAFE,
    # Moderate — writes, UI interaction
    "write_file": ToolRisk.MODERATE,
    "patch_file": ToolRisk.MODERATE,
    "save_note": ToolRisk.MODERATE,
    "screen_click": ToolRisk.MODERATE,
    "screen_type": ToolRisk.MODERATE,
    "screen_scroll": ToolRisk.MODERATE,
    "screen_hotkey": ToolRisk.MODERATE,
    "screen_move": ToolRisk.MODERATE,
    "screen_drag": ToolRisk.MODERATE,
    "smart_click": ToolRisk.MODERATE,
    "set_clipboard": ToolRisk.MODERATE,
    "focus_window": ToolRisk.MODERATE,
    "open_application": ToolRisk.MODERATE,
    "minimize_window": ToolRisk.MODERATE,
    "maximize_window": ToolRisk.MODERATE,
    "resize_window": ToolRisk.MODERATE,
    "move_window": ToolRisk.MODERATE,
    "generate_image": ToolRisk.MODERATE,
    "generate_video": ToolRisk.MODERATE,
    # Dangerous — shell exec, process control, destructive
    "run_command": ToolRisk.DANGEROUS,
    "kill_process": ToolRisk.DANGEROUS,
    "close_window": ToolRisk.DANGEROUS,
    # Web tools (E3) — network access, safe-ish but external
    "web_search": ToolRisk.SAFE,
    "web_fetch": ToolRisk.MODERATE,
    "web_cache_get": ToolRisk.SAFE,
    # Google Workspace CLI tools (P2.2)
    "gmail_search": ToolRisk.SAFE,
    "gmail_send": ToolRisk.MODERATE,
    "drive_list": ToolRisk.SAFE,
    "calendar_list_events": ToolRisk.SAFE,
    "sheets_read": ToolRisk.SAFE,
    "docs_create": ToolRisk.MODERATE,
    # Long-running processes (E2)
    "start_process": ToolRisk.DANGEROUS,
    "check_process": ToolRisk.SAFE,
    "stop_process": ToolRisk.MODERATE,
}


# ---------------------------------------------------------------------------
# Multi-agent configuration loader
# ---------------------------------------------------------------------------

def _load_agent_configs() -> dict[str, dict]:
    """Load multi-agent configs from config.yaml.

    Returns a dict of agent_name → config dict with keys:
    enabled, model, description, tools, max_iterations, deny_tools,
    system_instruction, capabilities, tags, schedule (D4 extended manifest)
    """
    import os
    from pathlib import Path as _P

    defaults = {
        "task_executor": {
            "enabled": True,
            "model": _DEFAULT_ORCHESTRATOR_MODEL,
            "description": "Executes computer tasks",
            "tools": "all",
            "max_iterations": 25,
            "deny_tools": [],
            "system_instruction": "",
            "capabilities": [],
            "tags": [],
            "schedule": None,
        },
    }

    try:
        import yaml
        for candidate in (
            _P(__file__).resolve().parent.parent / "config.yaml",
            _P(os.getcwd()) / "config.yaml",
        ):
            if candidate.is_file():
                data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                agents = data.get("rio", {}).get("agents", {})
                if agents:
                    return {
                        name: {
                            "enabled": cfg.get("enabled", True),
                            "model": cfg.get("model", _DEFAULT_ORCHESTRATOR_MODEL),
                            "description": cfg.get("description", ""),
                            "tools": cfg.get("tools", "all"),
                            "max_iterations": cfg.get("max_iterations", 25),
                            "deny_tools": cfg.get("deny_tools", []),
                            "system_instruction": cfg.get("system_instruction", ""),
                            "capabilities": cfg.get("capabilities", []),
                            "tags": cfg.get("tags", []),
                            "schedule": cfg.get("schedule", None),
                        }
                        for name, cfg in agents.items()
                    }
    except Exception:
        pass

    return defaults


def _load_global_deny_tools() -> list[str]:
    """Load global deny list from config.yaml (B4 Policy Pipeline)."""
    from pathlib import Path as _P
    try:
        import yaml
        for candidate in (
            _P(__file__).resolve().parent.parent / "config.yaml",
            _P(os.getcwd()) / "config.yaml",
        ):
            if candidate.is_file():
                data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                return data.get("rio", {}).get("global_deny_tools", []) or []
    except Exception:
        pass
    return []


def _load_filesystem_policy() -> dict[str, Any]:
    """Load filesystem read/write policy from config.yaml with safe defaults."""
    from pathlib import Path as _P

    defaults: dict[str, Any] = {
        "enabled": True,
        "read_paths": ["."],
        "write_paths": ["."],
    }

    try:
        import yaml
        for candidate in (
            _P(__file__).resolve().parent.parent / "config.yaml",
            _P(os.getcwd()) / "config.yaml",
        ):
            if not candidate.is_file():
                continue
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            fs = data.get("rio", {}).get("filesystem", {}) or {}
            merged = dict(defaults)
            merged.update(fs)
            merged["read_paths"] = list(merged.get("read_paths") or defaults["read_paths"])
            merged["write_paths"] = list(merged.get("write_paths") or defaults["write_paths"])
            return merged
    except Exception:
        pass
    return defaults


def _load_modes() -> dict[str, dict]:
    """D2: Load packaged mode definitions from rio/modes/*.yaml."""
    from pathlib import Path as _P
    modes: dict[str, dict] = {}
    modes_dir = _P(__file__).resolve().parent.parent / "modes"
    if not modes_dir.is_dir():
        return modes
    try:
        import yaml
        for f in modes_dir.glob("*.yaml"):
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            name = data.get("name", f.stem)
            modes[name] = {
                "name": name,
                "description": data.get("description", ""),
                "tools": data.get("tools", "all"),
                "max_iterations": data.get("max_iterations", 25),
                "system_instruction_prefix": data.get("system_instruction_prefix", ""),
                "deny_tools": data.get("deny_tools", []),
            }
    except Exception:
        pass
    return modes


# Tool sets for multi-agent routing
_TOOL_SETS = {
    "all": None,  # All tools (default)
    "browser": frozenset({
        "browser_connect", "browser_navigate", "browser_click_element",
        "browser_fill_form", "browser_extract_text", "browser_evaluate",
        "browser_wait_for", "browser_screenshot",
        "web_search", "web_fetch", "web_cache_get",
        "save_note", "get_notes", "search_notes",
        "capture_screen",
    }),
    "screen": frozenset({
        "smart_click", "screen_click", "screen_type", "screen_scroll",
        "screen_hotkey", "screen_move", "screen_drag", "find_window",
        "focus_window", "open_application", "list_all_windows",
        "get_active_window", "minimize_window", "maximize_window",
        "close_window", "resize_window", "move_window",
        "capture_screen", "get_screen_info",
        "save_note", "get_notes", "search_notes",
    }),
    "dev": frozenset({
        "read_file", "write_file", "patch_file", "run_command",
        "capture_screen", "save_note", "get_notes", "search_notes",
        "web_search", "web_fetch", "web_cache_get",
        "gmail_search", "gmail_send", "drive_list",
        "calendar_list_events", "sheets_read", "docs_create",
        "start_process", "check_process", "stop_process",
        "screen_hotkey",
    }),
    "memory": frozenset({
        "save_note", "get_notes", "search_notes", "export_context",
        "memory_stats", "read_file",
        "web_search", "web_fetch", "web_cache_get",
    }),
    "creative": frozenset({
        "generate_image", "generate_video",
        "save_note", "get_notes", "search_notes",
    }),
}


def _select_agent(goal: str, agent_configs: dict[str, dict],
                  memory_store: Any | None = None) -> str:
    """Select the best agent for a given task goal.

    Uses semantic embedding similarity when available, falling back to
    keyword matching.  The embedding approach resolves ambiguity like
    'write an email about debugging code' which would match both
    email_drafter and code_agent under keyword routing.
    """
    goal_lower = goal.lower()

    # --- High-priority deterministic routing ---
    # Keep explicit browser-launch intents ahead of semantic routing so
    # "open browser ..." does not get diverted into generic search/research flows.
    browser_launch_keywords = (
        "open browser", "launch browser", "start browser",
        "open chrome", "open firefox", "open edge",
        "launch chrome", "launch firefox", "launch edge",
    )
    if any(kw in goal_lower for kw in browser_launch_keywords):
        if "browser_agent" in agent_configs and agent_configs["browser_agent"].get("enabled"):
            return "browser_agent"

    # --- Semantic routing via embeddings (when available) ---
    if memory_store is not None and hasattr(memory_store, 'embed'):
        try:
            candidates = {
                name: cfg.get("description", "")
                for name, cfg in agent_configs.items()
                if cfg.get("enabled", True) and cfg.get("description")
            }
            if candidates:
                goal_emb = memory_store.embed(goal)
                best_agent = ""
                best_score = -1.0
                for name, desc in candidates.items():
                    desc_emb = memory_store.embed(desc)
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(goal_emb, desc_emb))
                    norm_a = sum(a * a for a in goal_emb) ** 0.5
                    norm_b = sum(b * b for b in desc_emb) ** 0.5
                    sim = dot / (norm_a * norm_b) if (norm_a and norm_b) else 0.0
                    if sim > best_score:
                        best_score = sim
                        best_agent = name
                if best_agent and best_score > 0.3:
                    return best_agent
        except Exception:
            pass  # Fall through to keyword routing

    # --- Keyword fallback routing ---
    # Browser automation keywords → browser_agent (checked BEFORE screen keywords)
    # These indicate web automation tasks that should use Playwright/CDP instead of screen automation
    browser_automation_keywords = (
        "search for", "search on", "google search", "web search", "bing search",
        "navigate to", "go to url", "visit url", "open url", "browse to",
        "fill form", "submit form", "click link", "click button on page",
        "extract text from", "scrape", "get page content", "read webpage",
        "browser automation", "playwright", "cdp", "devtools protocol",
        "inspect element", "css selector", "xpath", "dom",
        "login to website", "authenticate on", "sign in to site",
        "scroll up", "scroll down", "scroll to", "scroll on page", "scroll on website",
        "on the page", "on the website", "on this page", "on this website",
        "web page", "webpage", "website", "browser page",
        # Browser-opening keywords (Bug 1 fix)
        "open browser", "launch browser", "start browser",
        "open chrome", "open firefox", "open edge",
        "launch chrome", "launch firefox", "launch edge",
    )
    if any(kw in goal_lower for kw in browser_automation_keywords):
        if "browser_agent" in agent_configs and agent_configs["browser_agent"].get("enabled"):
            return "browser_agent"
        # Fallback to task_executor if browser_agent not available
        log.info("orchestrator.browser_automation_detected", 
                 note="browser_agent not configured, using task_executor with browser tools")

    # Screen/GUI/computer-use keywords → computer_use_agent
    # Note: Browser-related keywords removed to prevent incorrect routing (Bug 1 fix)
    screen_keywords = (
        "click", "smart_click", "screen", "scroll", "navigate", "open app",
        "open application", "launch", "window",
        "type in", "go to", "goto", "visit",
        "screenshot", "capture screen", "minimize", "maximize", "close window",
        "drag", "hotkey", "press", "switch window", "focus window",
        "open file", "open folder",
        "desktop", "taskbar", "start menu", "notification",
    )
    if any(kw in goal_lower for kw in screen_keywords):
        if "computer_use_agent" in agent_configs and agent_configs["computer_use_agent"].get("enabled"):
            return "computer_use_agent"

    # Code/dev keywords → code_agent
    code_keywords = (
        "code", "debug", "fix", "refactor", "function", "class", "variable",
        "import", "error", "traceback", "syntax", "compile", "build", "test",
        "read file", "write file", "patch", "git", "commit", "deploy",
        ".py", ".js", ".ts", ".html", ".css", ".json", ".yaml",
    )
    if any(kw in goal_lower for kw in code_keywords):
        if "code_agent" in agent_configs and agent_configs["code_agent"].get("enabled"):
            return "code_agent"

    # Creative keywords → creative_agent
    creative_keywords = (
        "generate image", "create image", "draw", "design", "generate video",
        "imagen", "veo", "picture", "illustration", "photo",
    )
    if any(kw in goal_lower for kw in creative_keywords):
        if "creative_agent" in agent_configs and agent_configs["creative_agent"].get("enabled"):
            return "creative_agent"

    # Research/explanation keywords → research_agent (uses pro model)
    research_keywords = (
        "research", "analyze", "analyse", "explain", "compare", "summarize",
        "what is", "how does", "why", "investigate", "study",
        "deep dive", "thoroughly", "in detail", "comprehensive",
        "architecture", "tradeoff", "trade-off", "security audit",
        "plan", "strategy", "evaluate", "review",
    )
    if any(kw in goal_lower for kw in research_keywords):
        if "research_agent" in agent_configs and agent_configs["research_agent"].get("enabled"):
            return "research_agent"

    # Default → task_executor for mixed/general tasks
    return "task_executor"

# ---------------------------------------------------------------------------
# Task-request detector (mirrors the one in local/main.py)
# ---------------------------------------------------------------------------

_TASK_ACTION_VERBS = (
    "open", "go to", "goto", "navigate", "search", "create", "download",
    "install", "click", "close", "launch", "start", "browse", "find",
    "visit", "play", "stop", "delete", "move", "copy", "paste", "save",
    "upload", "send", "book", "order", "sign in", "sign out", "log in",
    "log out", "register", "fill", "submit", "scroll", "type", "write",
    "run", "execute", "build", "deploy", "setup", "configure", "update",
    "uninstall", "rename", "drag", "edit", "fix", "read", "show", "switch",
)

_TASK_PHRASES = (
    "for me", "please do", "can you do", "i need you to", "i want you to",
    "go ahead and", "take over", "handle this", "do this", "complete this",
    "finish this", "make it happen", "do it", "get started",
)

_NON_TASK_STARTS = (
    "what", "why", "how", "when", "where", "who", "which", "is ", "are ",
    "was ", "were ", "do you", "does ", "did ", "can you explain",
    "tell me about", "explain", "describe", "hey rio", "hello", "hi ",
    "thanks", "thank you", "good", "great", "nice", "cool", "yeah", "yes",
    "no ", "okay", "ok ", "sure", "sounds good", "perfect", "awesome",
)


_CONVERSATIONAL_PREFIXES = (
    "hey rio", "hey,", "hey ", "hi rio", "rio,", "rio ", "ok rio",
    "okay rio", "yo rio", "alright rio", "so,", "so ", "now,", "now ",
    "can you ", "could you ", "would you ", "will you ",
    "i need you to ", "i want you to ", "i'd like you to ",
    "please ", "can you please ", "could you please ", "would you please ",
)


def _is_task_request(text: str) -> bool:
    """Return True when the transcribed utterance looks like an executable task.

    Improved detection: strips multiple prefixes, handles 'please' recursively,
    and checks for a wider range of action verbs and patterns.
    """
    text_lower = text.strip().lower().rstrip(".!?")

    if len(text_lower) < 4:
        return False

    # URL → navigation task (ALWAYS)
    if any(s in text_lower for s in ("http://", "https://", "www.", ".com", ".org", ".io")):
        return True

    # Explicit task markers
    if text_lower.startswith(
        ("task:", "do:", "execute:", "automate:", "please open",
         "please click", "please type", "please go", "please create",
         "please search", "please run", "please write", "please edit",
         "please find", "please close", "please launch", "please start",
         "please delete", "please move", "please copy", "please save",
         "please send", "please fill", "please submit", "please scroll",
         "please navigate", "please download", "please install",
         "please browse", "please visit", "please play", "please stop",
         "please build", "please deploy", "please setup", "please configure",
         "please update", "please uninstall", "please rename", "please drag",
         "please fix", "please show", "please switch", "please message",
         "please tell", "please notify", "please whatsapp")
    ):
        return True

    # Resume commands
    if text_lower in ("resume", "continue", "go on", "keep going", "go ahead"):
        return True

    # Recursive prefix stripping (handle: "Hey Rio, can you please...")
    core = text_lower
    changed = True
    while changed:
        changed = False
        for prefix in _CONVERSATIONAL_PREFIXES:
            if core.startswith(prefix):
                core = core[len(prefix):].lstrip(" ,")
                changed = True
                break

    # Very short core = unlikely to be an executable task
    if len(core.split()) < 2:
        return False

    # Action verb check on the core command
    for verb in _TASK_ACTION_VERBS:
        if (
            core.startswith(verb + " ")
            or core.startswith(verb + ":")
            or core == verb
        ):
            return True

    # Task-indicating phrases anywhere in the core
    for phrase in _TASK_PHRASES:
        if phrase in core:
            return True

    return False


# ---------------------------------------------------------------------------
# Load agents.md system prompt (user-customizable orchestration instructions)
# ---------------------------------------------------------------------------

def _load_agents_md() -> str:
    """Load agents.md from the project root as additional system instructions."""
    use_agents_md = os.environ.get("RIO_ORCHESTRATOR_USE_AGENTS_MD", "").strip().lower()
    if use_agents_md not in {"1", "true", "yes", "on"}:
        logger.info("agents_md.disabled", env_var="RIO_ORCHESTRATOR_USE_AGENTS_MD")
        return ""

    from pathlib import Path as _P
    # Search common locations for agents.md
    _base = _P(__file__).resolve().parent.parent  # rio/
    _project_root = _base.parent  # Rio-Agent/
    candidates = [
        _project_root / ".claude" / "agents" / "agents.md",
        _project_root / "agents.md",
        _base / "agents.md",
    ]
    for path in candidates:
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    logger.info("agents_md.loaded", path=str(path))
                    return content
            except Exception:
                pass
    return ""

_AGENTS_MD_CONTENT = _load_agents_md()

_ORCHESTRATOR_SYSTEM_INSTRUCTION = (
    # Include agents.md as the top-level orchestration philosophy
    ((_AGENTS_MD_CONTENT + "\n\n---\n\n") if _AGENTS_MD_CONTENT else "")
    +
    "You are Rio's autonomous task execution engine. "
    "The user has asked you to complete a specific task on their computer. "
    "Execute it fully using the available tools — do NOT stop to ask for "
    "confirmation mid-task. Complete the ENTIRE task autonomously.\n\n"
    "MEMORY — RECALL BEFORE RESPOND:\n"
    "- For browser-launch tasks (open/launch/start browser, open chrome/firefox/edge): "
    "call browser_connect FIRST, then call search_notes(query) if context is needed.\n"
    "- For non-browser tasks: call search_notes(query) early to recall relevant "
    "context from previous sessions before major actions.\n"
    "- Use save_note(key, value, media_paths=[]) to persist important information (e.g. file paths, "
    "user preferences, progress) for future tasks. You can optionally include paths to images, video, or audio to attach multimodal context to the memory.\n"
    "- Use get_notes() to retrieve all saved notes when you need full context.\n"
    "- The PREVIOUS TASKS section shows what was already done — avoid repeating work.\n"
    "- Save a concise summary note after completing each task.\n\n"
    "ACTION VERIFICATION:\n"
    "- After EVERY screen action, verify the result using the auto-captured screenshot.\n"
    "- For open_application: check the result's 'verified' field. If false, try again "
    "or use an alternative approach.\n"
    "- For window actions: check 'verification' or 'verification_warning' fields.\n"
    "- If verification fails, retry with a different approach (max 2 retries per step).\n"
    "- Use list_all_windows() to confirm window state when unsure.\n\n"
    "THE LOOP for every computer task:\n"
    "1. RECALL: Search notes for relevant context from previous work.\n"
    "2. PLAN: Silently break the task into steps.\n"
    "3. CAPTURE: Call capture_screen first if you need to see the current state.\n"
    "4. ANALYZE: Look at the screenshot. Identify what needs to be clicked/typed.\n"
    "5. ACT: Call the appropriate tool (smart_click, screen_type, screen_hotkey…)\n"
    "6. VERIFY: After each screen action, analyze the auto-captured screenshot "
    "AND the verification fields in the tool result to confirm success.\n"
    "7. CONTINUE: Repeat 4-6 until the task is fully complete.\n"
    "8. SAVE: Save a summary note of what was accomplished.\n"
    "9. REPORT: Return a concise 1-2 sentence summary of what you accomplished.\n\n"
    "RULES:\n"
    "- ALWAYS use smart_click(target='description') instead of screen_click when "
    "you know what UI element to click but not its exact pixel coordinates.\n"
    "- smart_click takes a fresh screenshot internally — no need to call "
    "capture_screen before it.\n"
    "- After every screen action, an auto-captured screenshot is sent to your "
    "vision context. USE IT to decide the next step.\n"
    "- Maximum 25 tool calls per task.\n"
    "- If a step fails, try an alternative approach before giving up.\n"
    "- When done, return ONLY the summary text (no markdown, no bullet points).\n\n"
    "DELEGATION:\n"
    "- For complex tasks requiring specialized expertise (e.g. research, code analysis, "
    "image generation), use delegate_to_agent(agent_name, sub_goal) to hand off sub-tasks.\n"
    "- Available agents: computer_use_agent (screen/GUI/clicking), code_agent (coding/debugging), "
    "research_agent (deep analysis/reasoning with pro model), "
    "creative_agent (image/video generation).\n"
    "- For screen tasks (open apps, click buttons, navigate), ALWAYS delegate to computer_use_agent.\n"
    "- For research/analysis that needs deep thinking, delegate to research_agent (uses pro model).\n"
    "- Use delegate_parallel to run multiple independent sub-tasks simultaneously.\n"
    "- Delegate when a sub-task clearly matches another agent's specialty.\n"
    "- You can continue your own work after delegation completes.\n"
)


class ToolOrchestrator:
    """Agentic tool execution engine for a single WebSocket session.

    Uses ``generate_content`` (not Live) with full function calling support.
    Runs concurrently alongside the native audio Live session.

    Tool functions from ``_make_tools(bridge)`` are passed directly —
    ``generate_content`` auto-converts callables to FunctionDeclarations,
    so no manual schema building is required here.
    """

    def __init__(
        self,
        genai_client: Any | None = None,
        tool_fns: list | None = None,
        model: str | None = None,
        memory_store: Any | None = None,
        broadcast_fn: Any | None = None,
        inject_context: Any | None = None,
        request_approval: Any | None = None,
        chat_store: Any | None = None,
        **_: Any,
    ) -> None:
        self._client = genai_client
        self._active_inject_context = inject_context
        # Session re-binding
        self._active_inject_context: Callable[[str], Awaitable[None]] | None = inject_context

        # Ensure no duplicate function names enter the tool declaration set.
        # Duplicate names cause Gemini API 400 INVALID_ARGUMENT.
        self._tool_fns = []
        self._tool_map: dict[str, Any] = {}
        for fn in (tool_fns or []):
            name = fn.__name__
            if name in self._tool_map:
                logger.warning("tools.duplicate_skipped", name=name)
                continue
            self._tool_fns.append(fn)
            self._tool_map[name] = fn

        # Auto-register Google Workspace tools (Gmail, Drive, Calendar, etc.)
        try:
            from workspace_tools import get_workspace_tools
            for ws_fn in get_workspace_tools():
                ws_name = ws_fn.__name__
                if ws_name not in self._tool_map:
                    self._tool_fns.append(ws_fn)
                    self._tool_map[ws_name] = ws_fn
            if any(ws_fn.__name__ in self._tool_map for ws_fn in get_workspace_tools()):
                logger.info("workspace_tools.registered")
        except ImportError:
            logger.debug("workspace_tools.not_available")
        except Exception as ws_exc:
            logger.debug("workspace_tools.registration_error", error=str(ws_exc))
        # Priority 1: Check environment variable for global model override
        env_model = os.environ.get("ORCHESTRATOR_MODEL")
        self._env_model_override = env_model is not None
        
        self._model = (
            model
            or env_model
            or _DEFAULT_ORCHESTRATOR_MODEL
        )
        self._log = logger.bind(component="tool_orchestrator", model=self._model)
        self._broadcast_fn = broadcast_fn
        # Track running tasks so they can be cancelled on disconnect
        self._active_tasks: set[asyncio.Task] = set()
        # Session-level task memory: ring buffer of (goal, result) summaries
        # so subsequent tasks in the same session can recall what was done before.
        self._task_history: collections.deque[tuple[str, str]] = collections.deque(
            maxlen=_MAX_TASK_MEMORY,
        )
        # Persistent notes the model can save/retrieve during the session
        # D1: Namespaced by agent — each agent sees its own + "global" namespace
        # G1: Session-scoped — notes survive per session_id
        self._session_id: str = f"session_{int(time.time())}"
        self._notes: dict[str, str] = {}           # global notes
        self._agent_notes: dict[str, dict[str, str]] = {}  # per-agent notes
        self._current_agent: str = "task_executor"  # active agent for note scoping
        # Multi-agent configs
        self._agent_configs = _load_agent_configs()

        # B4 Policy Pipeline — layered deny: global → agent → session
        self._global_deny_tools: set[str] = set(_load_global_deny_tools())
        self._session_deny_tools: set[str] = set()  # runtime overrides via dashboard

        # D2: Packaged modes
        self._modes = _load_modes()
        self._active_mode: dict | None = None

        # Priority 2/3 runtime settings
        self._orchestrator_settings = _load_orchestrator_settings()
        self._tool_timeout_seconds = int(
            self._orchestrator_settings.get("tool_timeout_seconds", _DEFAULT_TOOL_TIMEOUT_SECONDS)
        )
        self._heartbeat_interval_seconds = int(
            self._orchestrator_settings.get("heartbeat_interval_seconds", _DEFAULT_HEARTBEAT_INTERVAL_SECONDS)
        )
        self._approval_timeout_seconds = float(
            self._orchestrator_settings.get(
                "approval_timeout_seconds",
                _DEFAULT_APPROVAL_TIMEOUT_SECONDS,
            )
        )
        self._tool_policy = self._orchestrator_settings.get("tool_policy", {}) or {}
        self._loop_cfg = self._orchestrator_settings.get("loop_detection", {}) or {}
        self._composite_routing_cfg = self._orchestrator_settings.get("composite_routing", {}) or {}
        self._semantic_compaction_cfg = self._orchestrator_settings.get("semantic_compaction", {}) or {}
        self._session_persist_interval_seconds = float(
            self._orchestrator_settings.get(
                "session_persist_interval_seconds",
                _DEFAULT_SESSION_PERSIST_INTERVAL_SECONDS,
            )
        )
        self._last_session_persist_at = 0.0
        self._prompt_fragments = _load_prompt_fragment_files(
            self._orchestrator_settings.get("prompt_fragments", {}) or {}
        )
        self._filesystem_policy = _load_filesystem_policy()
        self._workspace_root = pathlib.Path(__file__).resolve().parent.parent

        # Priority 3.1: rate limits, quotas, and lightweight cost tracking.
        self._tool_call_times: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)
        self._tool_usage_total: collections.Counter[str] = collections.Counter()
        self._tool_cost_total: collections.Counter[str] = collections.Counter()

        # Priority 1.4: richer task snapshots for resume.
        self._active_task_state: dict[asyncio.Task, dict[str, Any]] = {}
        self._pending_resume_snapshots: list[dict[str, Any]] = []

        # Priority 3.2: delegated sub-task lifecycle registry.
        self._subagent_registry: dict[str, dict[str, Any]] = {}
        self._subagent_counter: int = 0

        # D3: Skill packaging
        try:
            import sys as _sys
            _cloud_dir = str(pathlib.Path(__file__).resolve().parent)
            if _cloud_dir not in _sys.path:
                _sys.path.insert(0, _cloud_dir)
            from skill_loader import load_skills
            self._skills = load_skills()
        except Exception:
            self._skills = {}

        # Long-term memory store (ChromaDB) for semantic search (A1)
        self._memory_store = memory_store

        # Memory compaction lock — prevents concurrent compaction (race condition fix)
        self._compaction_lock = asyncio.Lock()

        # Evaluation infrastructure (Agent Factory Podcast framework)
        self._evaluation_store = None
        self._llm_judge = None
        self._eval_enabled = bool(os.environ.get("RIO_EVAL_ENABLED", ""))
        try:
            from evaluation import EvaluationStore, LLMJudge
            self._evaluation_store = EvaluationStore()
            if genai_client and self._eval_enabled:
                self._llm_judge = LLMJudge(genai_client)
        except ImportError:
            pass

        # Unified memory facade (A5) — wraps vector + notes + chat
        self._unified_memory = None
        try:
            import sys
            _local_dir = str(pathlib.Path(__file__).resolve().parent.parent / "local")
            if _local_dir not in sys.path:
                sys.path.insert(0, _local_dir)
            from unified_memory import UnifiedMemory
            self._unified_memory = UnifiedMemory(
                memory_store=memory_store,
                session_notes=self._notes,
                chat_store=None,  # set later via set_chat_store()
            )
        except Exception:
            pass

        # Loop detection with graduated escalation (C6 + P3.4)
        self._recent_calls: collections.deque = collections.deque(maxlen=10)

        # Approval queue (B2) — gate dangerous tools via dashboard
        self._approval_queue: asyncio.Queue = asyncio.Queue()
        self._approval_timeout = self._approval_timeout_seconds

        # C2: Message queue — steer/cancel running tasks
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._cancel_patterns = frozenset({
            "stop", "cancel", "never mind", "nevermind", "abort",
            "forget it", "quit", "halt",
        })

        # JSONL transcript log (G3)
        self._transcript_path = self._init_transcript_log()

        # C4: Workflow objects — create Task records for structured tracking
        self._task_store = None
        try:
            import sys
            import pathlib as _pl
            _ld = str(_pl.Path(__file__).resolve().parent.parent / "local")
            if _ld not in sys.path:
                sys.path.insert(0, _ld)
            from task_state import TaskStore
            self._task_store = TaskStore()
        except Exception:
            pass

        # Register built-in memory tools so the model can save/recall notes
        self._register_memory_tools()

        # F4: Plugin/Hook system — before_tool and after_tool hooks
        # Each hook is an async callable:
        #   before_tool: (tool_name, args) → args (can modify args)
        #   after_tool:  (tool_name, args, result) → result (can modify result)
        self._before_tool_hooks: list = []
        self._after_tool_hooks: list = []

    def _select_model(self, goal: str) -> str:
        """Dynamic model selection based on task complexity (Composite Routing)."""
        # If model is forced via ORCHESTRATOR_MODEL env var, always use it.
        if self._env_model_override:
            return self._model

        cfg = self._composite_routing_cfg
        if not cfg.get("enabled", True):
            return self._model

        goal_lower = goal.lower()
        complex_keywords = cfg.get("complexity_keywords", [])
        
        # 1. Complexity by keywords
        is_complex = any(kw in goal_lower for kw in complex_keywords)
        
        # 2. Complexity by length (long prompts usually need more reasoning)
        if not is_complex:
            max_words = cfg.get("simple_max_words", 25)
            if len(goal.split()) > max_words:
                is_complex = True
        
        selected = cfg.get("complex_model") if is_complex else cfg.get("simple_model")
        return selected or self._model

    async def _analyze_request_relationship(self, new_goal: str, active_goal: str) -> str:
        """Analyze if a new request is related to the running task.
        Returns: 'RELATED', 'CANCEL', or 'UNRELATED'.
        """
        new_lower = new_goal.lower()
        # Fast keyword heuristics
        if any(p in new_lower for p in self._cancel_patterns):
            return "CANCEL"
        
        # Simple similarity check
        new_words = set(new_lower.split())
        active_words = set(active_goal.lower().split())
        overlap = new_words.intersection(active_words)
        if len(overlap) >= 2 or (len(new_words) < 4 and overlap):
            return "RELATED"

        # Fallback to LLM for nuanced analysis — use the simplest/fastest model
        try:
            from google.genai import types as _types
            # Orchestrator decides: use simple_model for analysis
            analysis_model = self._composite_routing_cfg.get("simple_model", "gemini-2.5-flash")
            
            prompt = (
                f"Active Task: \"{active_goal}\"\n"
                f"New Request: \"{new_goal}\"\n\n"
                "Is the New Request RELATED to the Active Task (modifying it or adding steps), "
                "is it a request to CANCEL/stop, or is it a completely UNRELATED new task?\n"
                "Reply with exactly one word: RELATED, CANCEL, or UNRELATED."
            )
            response = await self._client.aio.models.generate_content(
                model=analysis_model,
                contents=prompt,
                config=_types.GenerateContentConfig(temperature=0.0, max_output_tokens=10)
            )
            ans = response.text.strip().upper()
            if ans in ("RELATED", "CANCEL", "UNRELATED"):
                return ans
        except Exception:
            pass
        
        return "UNRELATED"

    @property
    def is_busy(self) -> bool:
        """Return True if any task is currently running."""
        return any(not t.done() for t in self._active_tasks)

    @property
    def model(self) -> str:
        """Return the current orchestrator model name."""
        return self._model

    def set_model(self, model_name: str) -> None:
        """Update the primary orchestrator model at runtime."""
        self._model = model_name
        self._log = logger.bind(component="tool_orchestrator", model=self._model)
        self._log.info("orchestrator.model_updated", model=model_name)

    def reset(self) -> None:
        """Reset session-level task history and active notes."""
        self._task_history.clear()
        self._notes.clear()
        self._agent_notes.clear()
        self._log.info("orchestrator.reset", session_id=self._session_id)

    def get_ui_agent_status(self) -> list[dict[str, str]]:
        """Return a list of agents with their current state for the UI AgentTable/Graph."""
        results = []
        # Main orchestrator
        results.append({
            "name": "Orchestrator",
            "model": self._model,
            "status": "active" if self.is_busy else "idle"
        })
        # Specialists from config
        for name, cfg in self._agent_configs.items():
            if not cfg.get("enabled"): continue
            
            # Find if this agent is currently running as a subagent
            is_running = any(
                s.get("agent_name") == name and s.get("status") == "running"
                for s in self._subagent_registry.values()
            )
            
            results.append({
                "name": name.replace("_", " ").title(),
                "model": cfg.get("model", "default"),
                "status": "active" if is_running else "idle"
            })
        return results

    def register_hook(self, event: str, callback) -> None:
        """Register a hook callback.

        Args:
            event: 'before_tool' or 'after_tool'
            callback: Async callable matching the hook signature.
        """
        if event == "before_tool":
            self._before_tool_hooks.append(callback)
        elif event == "after_tool":
            self._after_tool_hooks.append(callback)
        else:
            self._log.warning("hook.unknown_event", event=event)

    # G1: Session scoping ─────────────────────────────────────────────────────
    @property
    def session_id(self) -> str:
        return self._session_id

    def new_session(self, session_id: str | None = None) -> str:
        """Start a new session, persisting the old one first."""
        self._persist_session()
        self._session_id = session_id or f"session_{int(time.time())}"
        self._notes.clear()
        self._agent_notes.clear()
        self._task_history.clear()
        self._recent_calls.clear()
        self._log.info("session.new", session_id=self._session_id)
        return self._session_id

    def _persist_session(self) -> None:
        """Save current session state (notes + task_history + active goals) to disk."""
        try:
            import json
            data_dir = pathlib.Path(__file__).resolve().parent.parent / "data" / "sessions"
            data_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "session_id": self._session_id,
                "notes": dict(self._notes),
                "agent_notes": {k: dict(v) for k, v in self._agent_notes.items()},
                "task_history": list(self._task_history),
                "current_agent": self._current_agent,
                "active_tasks": self._active_task_snapshots(),
            }
            (data_dir / f"{self._session_id}.json").write_text(
                json.dumps(state, indent=2, default=str), encoding="utf-8",
            )
        except Exception:
            self._log.debug("session.persist_skip")

    def _persist_session_if_due(self, force: bool = False) -> None:
        """Persist session snapshots on an interval for robust task resumes."""
        now = time.time()
        if not force and (now - self._last_session_persist_at) < self._session_persist_interval_seconds:
            return
        self._persist_session()
        self._last_session_persist_at = now

    def load_session(self, session_id: str) -> bool:
        """Restore a previously persisted session."""
        try:
            import json
            path = pathlib.Path(__file__).resolve().parent.parent / "data" / "sessions" / f"{session_id}.json"
            if not path.is_file():
                return False
            state = json.loads(path.read_text(encoding="utf-8"))
            self._session_id = state["session_id"]
            self._notes.update(state.get("notes", {}))
            for agent, notes in state.get("agent_notes", {}).items():
                self._agent_notes.setdefault(agent, {}).update(notes)
            for entry in state.get("task_history", []):
                self._task_history.append(tuple(entry))
            self._current_agent = state.get("current_agent", "task_executor")
            self._pending_resume_snapshots = state.get("active_tasks", [])
            self._log.info("session.loaded", session_id=session_id,
                           pending_resume=len(self._pending_resume_snapshots))
            return True
        except Exception:
            return False

    def resume_interrupted_tasks(
        self,
        inject_context: Callable[[str], Awaitable[None]],
    ) -> list[asyncio.Task]:
        """Re-spawn tasks that were running when the previous session ended.

        Call this after load_session() once inject_context is available.
        Returns list of spawned asyncio.Tasks.
        """
        snapshots = getattr(self, "_pending_resume_snapshots", [])
        if not snapshots:
            return []
        spawned = []
        for snapshot in snapshots:
            goal = str(snapshot.get("goal", "")).strip()
            if not goal:
                continue
            self._log.info("session.resume_task", goal=goal[:80], task_id=snapshot.get("task_id"))
            # Prefix so the model knows this is a resumed task
            resume_goal = (
                f"[RESUMED from previous session]\n"
                f"Original goal: {goal}\n"
                f"Last step: {snapshot.get('current_step', 'unknown')}"
            )
            task = self.spawn_task(resume_goal, inject_context, resume_snapshot=snapshot)
            spawned.append(task)
        self._pending_resume_snapshots = []
        self._persist_session()
        return spawned

    def set_chat_store(self, chat_store: Any) -> None:
        """Attach a ChatStore for unified memory search (A5)."""
        if self._unified_memory is not None:
            self._unified_memory._chat = chat_store

    def set_session_deny_tools(self, tools: list[str]) -> None:
        """Update session-level deny list at runtime (B4 Policy Pipeline)."""
        self._session_deny_tools = set(tools)

    def _apply_policy_pipeline(
        self, tool_fns: list, agent_deny: list[str],
    ) -> list:
        """Apply layered deny policy: global → agent → session.  Deny-wins."""
        denied = self._global_deny_tools | set(agent_deny) | self._session_deny_tools
        if not denied:
            return tool_fns
        return [fn for fn in tool_fns if fn.__name__ not in denied]

    def switch_mode(self, mode_name: str) -> str:
        """D2: Switch to a named mode. Returns confirmation or error."""
        if mode_name in self._modes:
            self._active_mode = self._modes[mode_name]
            self._log.info("orchestrator.mode_switch", mode=mode_name)
            return f"Switched to {mode_name} mode."
        available = list(self._modes.keys())
        return f"Unknown mode '{mode_name}'. Available: {available}"

    def get_active_mode(self) -> dict | None:
        """Return the currently active mode config, or None."""
        return self._active_mode

    def get_skills(self) -> dict:
        """D3: Return loaded skill definitions."""
        return self._skills

    def _estimate_goal_complexity(self, goal: str) -> int:
        """Estimate task complexity for composite model routing."""
        words = goal.split()
        score = 1
        if len(words) > int(self._composite_routing_cfg.get("simple_max_words", 22)):
            score += 1
        keywords = [
            kw.lower().strip()
            for kw in (self._composite_routing_cfg.get("complexity_keywords", []) or [])
            if str(kw).strip()
        ]
        goal_lower = goal.lower()
        for kw in keywords:
            if kw in goal_lower:
                score += 1
        if " and " in goal_lower or " then " in goal_lower:
            score += 1
        return min(score, 5)

    def _route_model_for_goal(self, goal: str, agent_model: str) -> str:
        """Priority 2.1: choose simple/complex model based on goal complexity."""
        if not bool(self._composite_routing_cfg.get("enabled", True)):
            return agent_model
        complexity = self._estimate_goal_complexity(goal)
        if complexity >= 3:
            routed = str(self._composite_routing_cfg.get("complex_model", "") or "").strip()
        else:
            routed = str(self._composite_routing_cfg.get("simple_model", "") or "").strip()
        return routed or agent_model

    def _compose_system_instruction(self, instruction_suffix: str) -> str:
        """Priority 2.1/2.3: compose orchestrator prompt from modular fragments."""
        pieces: list[str] = [_ORCHESTRATOR_SYSTEM_INSTRUCTION]
        for key in ("base", "routing", "safety", "verification", "compaction"):
            fragment = self._prompt_fragments.get(key, "")
            if fragment:
                pieces.append(fragment)
        if instruction_suffix:
            pieces.append(instruction_suffix)
        return "\n\n".join(p for p in pieces if p.strip())

    def _enforce_tool_budget(
        self,
        tool_name: str,
        task_cost_points: int,
        task_tool_calls: int,
    ) -> tuple[bool, str, int, int]:
        """Priority 3.1: enforce per-tool rate limits and per-task quotas."""
        now = time.time()
        per_tool = self._tool_policy.get("per_tool_per_minute", {}) or {}
        default_per_minute = int(self._tool_policy.get("default_per_minute", 20))
        max_calls_per_task = int(self._tool_policy.get("max_calls_per_task", _DEFAULT_MAX_TOOL_CALLS_PER_TASK))
        max_cost_points = int(self._tool_policy.get("max_cost_points_per_task", _DEFAULT_MAX_COST_POINTS_PER_TASK))
        cost_points_cfg = self._tool_policy.get("cost_points", {}) or {}

        max_per_minute = int(per_tool.get(tool_name, default_per_minute))
        bucket = self._tool_call_times[tool_name]
        while bucket and (now - bucket[0]) > 60.0:
            bucket.popleft()
        if len(bucket) >= max_per_minute:
            return False, f"Rate limit exceeded for {tool_name}: {max_per_minute}/min", task_cost_points, 0

        next_calls = task_tool_calls + 1
        if next_calls > max_calls_per_task:
            return False, f"Task quota exceeded: max {max_calls_per_task} tool calls", task_cost_points, 0

        point_cost = int(cost_points_cfg.get(tool_name, 1))
        next_cost = task_cost_points + point_cost
        if next_cost > max_cost_points:
            return False, f"Task cost budget exceeded: {next_cost}/{max_cost_points} points", task_cost_points, 0

        bucket.append(now)
        self._tool_usage_total[tool_name] += 1
        self._tool_cost_total[tool_name] += point_cost
        return True, "", next_cost, point_cost

    def _compact_runtime_context_window(self, lines: list[str]) -> list[str]:
        """Compact runtime context strings for task resume snapshots."""
        compacted: list[str] = []
        for line in lines:
            text = str(line).strip()
            if not text:
                continue
            compacted.append(text[:220])
        return compacted[-10:]

    def _semantic_compaction_score(self, item: Any) -> int:
        """Score context items so compaction keeps semantically important turns."""
        score = 0
        text_blob = ""
        keywords = [
            str(k).lower().strip()
            for k in (self._semantic_compaction_cfg.get("high_value_keywords", []) or [])
            if str(k).strip()
        ]

        if hasattr(item, "parts"):
            for part in (item.parts or []):
                txt = getattr(part, "text", "") or ""
                if txt:
                    text_blob += " " + txt
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    score += 1
                fr = getattr(part, "function_response", None)
                if fr and hasattr(fr, "response") and isinstance(fr.response, dict):
                    response = fr.response
                    if response.get("error"):
                        score += 4
                    if response.get("_loop_warning"):
                        score += 3
                    if response.get("success") is False:
                        score += 2
                    text_blob += " " + str(response.get("result", ""))

        low = text_blob.lower()
        if any(tok in low for tok in ("todo", "next", "remaining", "decision", "approved", "denied")):
            score += 2
        if any(tok in low for tok in ("/", "\\", ".py", ".yaml", ".md")):
            score += 1
        score += sum(1 for kw in keywords if kw in low)
        return score

    def _resolve_policy_path(self, raw: Any) -> pathlib.Path | None:
        """Resolve a candidate file path argument for policy checks."""
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        path = pathlib.Path(text).expanduser()
        if not path.is_absolute():
            path = self._workspace_root / path
        try:
            return path.resolve(strict=False)
        except Exception:
            return None

    @staticmethod
    def _is_relative_to(child: pathlib.Path, parent: pathlib.Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    def _path_allowed(self, candidate: pathlib.Path, roots: list[str]) -> bool:
        for root in roots:
            base = self._resolve_policy_path(root)
            if base is None:
                continue
            if self._is_relative_to(candidate, base):
                return True
        return False

    def _enforce_filesystem_policy(self, tool_name: str, tool_args: dict[str, Any]) -> tuple[bool, str]:
        """Priority 3.5: enforce configurable read/write filesystem boundaries."""
        if not bool(self._filesystem_policy.get("enabled", True)):
            return True, ""

        intent_by_tool = {
            "read_file": "read",
            "write_file": "write",
            "patch_file": "write",
        }
        intent = intent_by_tool.get(tool_name)
        if not intent:
            return True, ""

        candidate = None
        for key in ("path", "file_path", "filepath", "target", "filename", "file"):
            if key in tool_args:
                candidate = self._resolve_policy_path(tool_args.get(key))
                if candidate is not None:
                    break

        if candidate is None:
            return True, ""

        roots = list(self._filesystem_policy.get("read_paths" if intent == "read" else "write_paths", []) or [])
        if self._path_allowed(candidate, roots):
            return True, ""

        return False, (
            f"Filesystem policy blocked {tool_name} for path '{candidate}'. "
            f"Allowed {intent} roots: {roots}"
        )

    def _active_task_snapshots(self) -> list[dict[str, Any]]:
        """Priority 1.4: collect resumable snapshots for active tasks."""
        snapshots: list[dict[str, Any]] = []
        for task, state in self._active_task_state.items():
            if task.done():
                continue
            snapshots.append(
                {
                    "task_id": state.get("task_id"),
                    "goal": state.get("goal", ""),
                    "agent": state.get("agent", "task_executor"),
                    "model": state.get("model", self._model),
                    "started_at": state.get("started_at", 0),
                    "current_step": state.get("current_step", ""),
                    "partial_results": state.get("partial_results", [])[-8:],
                    "context_window": state.get("context_window", [])[-10:],
                    "tool_calls": state.get("tool_calls", 0),
                    "cost_points": state.get("cost_points", 0),
                    "tool_usage": dict(state.get("tool_usage", {})),
                    "cost_by_tool": dict(state.get("cost_by_tool", {})),
                }
            )
        return snapshots

    def _register_memory_tools(self) -> None:
        """Add save_note, get_notes, and search_notes as callable tools for the model."""

        async def save_note(key: str, value: str, media_paths: list[str] = None) -> dict:
            """Save a persistent note for this session. Use this to remember
            important information like file paths, user preferences, or task
            progress that may be needed by future tasks.
            Optionally provide media_paths (e.g. ['path/to/image.png']) to attach multimodal context."""
            # D1: Save to both agent-specific and global namespace
            agent = self._current_agent
            if agent not in self._agent_notes:
                self._agent_notes[agent] = {}
            self._agent_notes[agent][key] = value
            self._notes[key] = value  # also accessible globally
            
            # Persist to UnifiedMemory for cross-session multimodal recall
            media_parts = None
            if media_paths:
                media_parts = []
                from google.genai import types
                import mimetypes
                for path in media_paths:
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                        mime_type, _ = mimetypes.guess_type(path)
                        if not mime_type:
                            mime_type = "application/octet-stream"
                        media_parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
                    except Exception as e:
                        self._log.warning("memory.save_note.media_read_failed", path=path, error=str(e))

            if self._memory_store is not None:
                self._memory_store.save_note(key, value, persist_vector=True, media_parts=media_parts)
                
            self._log.info("memory.save_note", key=key, agent=agent, has_media=bool(media_parts))
            return {"success": True, "key": key, "message": f"Note '{key}' saved."}

        async def get_notes(key: str = "") -> dict:
            """Retrieve saved notes. If key is provided, return that specific note.
            If key is empty, return all saved notes visible to the current agent."""
            agent = self._current_agent
            agent_ns = self._agent_notes.get(agent, {})
            # D1: Merge global + agent-specific (agent takes precedence)
            merged = {**self._notes, **agent_ns}
            if key:
                val = merged.get(key)
                if val is None:
                    return {"success": False, "error": f"No note found for key '{key}'."}
                return {"success": True, "key": key, "value": val}
            return {"success": True, "notes": merged}

        async def search_notes(query: str, limit: int = 5) -> dict:
            """Search saved notes AND long-term memory by keyword/semantic
            similarity. ALWAYS call this before starting a task to recall
            relevant context from previous work."""
            if not query.strip():
                return {"success": True, "results": [], "message": "Empty query."}

            # Unified memory facade (A5) — single search across all stores
            if self._unified_memory is not None:
                try:
                    entries = self._unified_memory.search(query, limit=limit)
                    results = [
                        {
                            "key": e.key,
                            "value": e.content[:500],
                            "score": round(e.score, 3),
                            "source": e.source,
                        }
                        for e in entries
                    ]
                    return {"success": True, "results": results}
                except Exception:
                    pass  # Fall back to legacy below

            # Legacy fallback: keyword search on session notes only
            keywords = query.lower().split()
            results = []
            for k, v in self._notes.items():
                text = f"{k} {v}".lower()
                score = sum(1 for kw in keywords if kw in text)
                if score > 0:
                    results.append({"key": k, "value": v[:500], "score": score, "source": "session"})
            # Semantic search on ChromaDB (A1)
            if self._memory_store:
                try:
                    entries = self._memory_store.query(query, top_k=limit)
                    for entry in entries:
                        results.append({
                            "key": entry.id,
                            "value": entry.content[:500],
                            "score": max(0, 1.0 - entry.distance),
                            "source": "long_term",
                            "type": entry.entry_type,
                        })
                except Exception:
                    pass
            results.sort(key=lambda r: -r["score"])
            return {"success": True, "results": results[:limit]}

        for fn in (save_note, get_notes, search_notes):
            if fn.__name__ in self._tool_map:
                self._log.warning("tools.duplicate_skipped", name=fn.__name__)
                continue
            self._tool_fns.append(fn)
            self._tool_map[fn.__name__] = fn

        # G4: Context export tool — snapshot current session state
        async def export_context(format: str = "summary") -> dict:
            """Export the current session context as a snapshot.
            format: 'summary' (compact) or 'full' (all details).
            Useful for handing off context or reviewing what happened."""
            snapshot = {
                "session_id": self._session_id,
                "agent": self._current_agent,
                "task_count": len(self._task_history),
                "note_count": len(self._notes),
                "subagent_count": len(self._subagent_registry),
            }
            if format == "full":
                snapshot["tasks"] = [
                    {"goal": g, "result": r[:300]} for g, r in self._task_history
                ]
                snapshot["notes"] = dict(self._notes)
                snapshot["subagents"] = list(self._subagent_registry.values())[-20:]
                snapshot["tool_usage"] = dict(self._tool_usage_total)
                snapshot["tool_cost"] = dict(self._tool_cost_total)
            else:
                snapshot["recent_tasks"] = [
                    {"goal": g, "result": r[:100]}
                    for g, r in list(self._task_history)[-5:]
                ]
            # Also persist to disk
            self._persist_session()
            return {"success": True, "snapshot": snapshot}

        if export_context.__name__ not in self._tool_map:
            self._tool_fns.append(export_context)
            self._tool_map[export_context.__name__] = export_context

        # C3: Subagent delegation tool — allows the orchestrator to
        # delegate sub-goals to specialized agents with depth limits.
        _orchestrator_self = self
        _MAX_DELEGATION_DEPTH = 3
        _DELEGATION_TIMEOUT = 120  # seconds

        # System tools that should be available to ALL agents
        _SYSTEM_TOOL_NAMES = {
            "delegate_to_agent", "delegate_parallel",
            "save_note", "get_notes", "search_notes",
            "export_context", "memory_stats"
        }

        async def delegate_to_agent(
            sub_goal: str,
            agent_name: str = "",
            timeout_seconds: int = _DELEGATION_TIMEOUT,
        ) -> dict:
            """Delegate a sub-task to another agent. Use this when the current
            task would benefit from a specialist (e.g. delegate web scraping
            to the browser agent, coding to the coder agent, etc.).

            Args:
                sub_goal: A clear description of what the sub-agent should do.
                agent_name: Optional agent name. Auto-selected if empty.
                timeout_seconds: Max seconds to wait (default 120).
            """
            # Track delegation depth via attribute on task
            current_depth = getattr(_orchestrator_self, "_delegation_depth", 0)
            if current_depth >= _MAX_DELEGATION_DEPTH:
                return {
                    "success": False,
                    "error": f"Maximum delegation depth ({_MAX_DELEGATION_DEPTH}) reached. "
                             f"Complete the sub-task directly instead of delegating further.",
                }

            # Select agent
            agent_name_for_delegate = agent_name
            if not agent_name_for_delegate:
                agent_name_for_delegate = _select_agent(
                    sub_goal, _orchestrator_self._agent_configs,
                    _orchestrator_self._memory_store,
                )
            agent_cfg = _orchestrator_self._agent_configs.get(agent_name_for_delegate, {})
            agent_model = agent_cfg.get("model", _orchestrator_self._model)
            agent_model = _orchestrator_self._route_model_for_goal(sub_goal, agent_model)
            max_iters = min(agent_cfg.get("max_iterations", 15), 15)  # Cap sub-agent iterations
            _orchestrator_self._subagent_counter += 1
            subagent_id = f"subagent-{_orchestrator_self._subagent_counter:04d}"
            _orchestrator_self._subagent_registry[subagent_id] = {
                "id": subagent_id,
                "goal": sub_goal,
                "agent": agent_name_for_delegate,
                "model": agent_model,
                "status": "spawned",
                "started_at": time.time(),
                "depth": current_depth + 1,
            }

            _orchestrator_self._log.info(
                "subagent.delegating",
                sub_goal=sub_goal[:80],
                agent=agent_name,
                depth=current_depth + 1,
            )

            # Build tool set for sub-agent
            agent_tools_key = agent_cfg.get("tools", "all")
            tool_set = _TOOL_SETS.get(agent_tools_key)
            if tool_set is not None:
                # Include specialty tools + global system tools
                sub_tool_fns = [
                    fn for fn in _orchestrator_self._tool_fns 
                    if fn.__name__ in tool_set or fn.__name__ in _SYSTEM_TOOL_NAMES
                ]
                sub_tool_map = {fn.__name__: fn for fn in sub_tool_fns}
            else:
                sub_tool_fns = _orchestrator_self._tool_fns
                sub_tool_map = dict(_orchestrator_self._tool_map)

            # Apply deny policy
            deny_list = agent_cfg.get("deny_tools", [])
            sub_tool_fns = _orchestrator_self._apply_policy_pipeline(sub_tool_fns, deny_list)
            sub_tool_map = {fn.__name__: fn for fn in sub_tool_fns}

            # Shared memory: pass parent's notes + memory_store to subagent context
            # so delegated agents can access what the parent already discovered.
            shared_context_parts = []
            if _orchestrator_self._notes:
                shared_context_parts.append("=== PARENT AGENT NOTES ===")
                for k, v in list(_orchestrator_self._notes.items())[:10]:
                    shared_context_parts.append(f"- {k}: {v}")
            if _orchestrator_self._task_history:
                shared_context_parts.append("=== PARENT TASK HISTORY ===")
                for prev_goal, prev_result in list(_orchestrator_self._task_history)[-5:]:
                    shared_context_parts.append(f"- {prev_goal}: {prev_result[:100]}")
            shared_prefix = "\n".join(shared_context_parts)
            enriched_sub_goal = (
                (shared_prefix + "\n\n" + sub_goal) if shared_prefix else sub_goal
            )

            # Run the sub-agent loop with incremented depth
            old_depth = getattr(_orchestrator_self, "_delegation_depth", 0)
            _orchestrator_self._delegation_depth = old_depth + 1
            
            # Emit delegation event for UI
            if _orchestrator_self._broadcast_fn:
                asyncio.create_task(_orchestrator_self._broadcast_fn({
                    "type": "delegation",
                    "payload": {
                        "from": "orchestrator",
                        "to": agent_name_for_delegate,
                        "task_label": sub_goal[:100],
                        "delegation_id": subagent_id,
                    }
                }))

            try:
                _orchestrator_self._subagent_registry[subagent_id]["status"] = "running"
                result = await asyncio.wait_for(
                    _orchestrator_self._agent_loop(
                        enriched_sub_goal,
                        model=agent_model,
                        tool_fns=sub_tool_fns,
                        tool_map=sub_tool_map,
                        max_iterations=max_iters,
                        system_instruction=_orchestrator_self._compose_system_instruction(
                            f"You are a sub-agent (depth {current_depth + 1}). "
                            f"Complete the sub-goal and return a concise result. "
                            f"You have access to the parent agent's notes and task history above."
                        ),
                    ),
                    timeout=timeout_seconds,
                )
                _orchestrator_self._log.info(
                    "subagent.completed",
                    agent=agent_name,
                    result_len=len(result),
                )
                
                # Emit completion event for UI
                if _orchestrator_self._broadcast_fn:
                    asyncio.create_task(_orchestrator_self._broadcast_fn({
                        "type": "delegation_complete",
                        "payload": {
                            "delegation_id": subagent_id,
                            "agent": agent_name_for_delegate,
                            "status": "success"
                        }
                    }))

                _orchestrator_self._subagent_registry[subagent_id].update(
                    {
                        "status": "completed",
                        "completed_at": time.time(),
                        "result": result[:500],
                    }
                )

                # Direct Handoff (Anti-Telephone Game)
                # Instead of returning the full result for the supervisor to summarize,
                # we directly inject the sub-agent's final output to the user.
                parent_task = asyncio.current_task()
                parent_state = _orchestrator_self._active_task_state.get(parent_task)
                if parent_state and "inject_context" in parent_state:
                    handoff_msg = (
                        f"[SYSTEM: Direct handoff from {agent_name_for_delegate}]:\n"
                        f"{_format_for_voice(result)}"
                    )
                    asyncio.create_task(parent_state["inject_context"](handoff_msg))
                elif _orchestrator_self._broadcast_fn:
                    # Fallback to dashboard if inject_context isn't available
                    asyncio.create_task(_orchestrator_self._broadcast_fn({
                        "type": "dashboard",
                        "subtype": "direct_handoff",
                        "message": result
                    }))

                return {"status": "success", "message": f"Sub-agent {agent_name_for_delegate} completed the task and responded directly to the user."}
            except asyncio.TimeoutError:
                _orchestrator_self._subagent_registry[subagent_id].update(
                    {
                        "status": "timeout",
                        "completed_at": time.time(),
                    }
                )
                return {"success": False, "error": f"Sub-agent timed out after {timeout_seconds}s"}
            except Exception as exc:
                _orchestrator_self._subagent_registry[subagent_id].update(
                    {
                        "status": "failed",
                        "completed_at": time.time(),
                        "error": str(exc),
                    }
                )
                return {"success": False, "error": f"Sub-agent failed: {exc}"}
            finally:
                _orchestrator_self._delegation_depth = old_depth

        if "delegate_to_agent" not in self._tool_map:
            self._tool_fns.append(delegate_to_agent)
            self._tool_map["delegate_to_agent"] = delegate_to_agent

        # ---- Parallel sub-agent delegation (Team Lead pattern) ----

        async def delegate_parallel(
            sub_tasks: str = "",
        ) -> dict:
            """Run multiple sub-agent tasks in parallel using asyncio.gather.

            This is the "Team Lead" pattern: delegate several independent
            sub-tasks to specialist agents simultaneously, then collect
            all results when they complete.

            Args:
                sub_tasks: JSON string containing a list of sub-task objects.
                    Each object should have:
                    - "goal": description of the sub-task
                    - "agent_name": (optional) target agent name
                    Example: '[{"goal": "research X"}, {"goal": "write code for Y"}]'
            """
            try:
                tasks_list = json.loads(sub_tasks)
            except (json.JSONDecodeError, TypeError):
                return {"success": False, "error": "sub_tasks must be a valid JSON list of {goal, agent_name?} objects"}

            if not isinstance(tasks_list, list) or len(tasks_list) == 0:
                return {"success": False, "error": "sub_tasks must be a non-empty list"}

            if len(tasks_list) > 5:
                return {"success": False, "error": "Maximum 5 parallel sub-tasks allowed"}

            _orchestrator_self._log.info(
                "parallel_delegation.start",
                count=len(tasks_list),
                goals=[t.get("goal", "?")[:40] for t in tasks_list],
            )

            # Run all sub-tasks concurrently
            async def _run_one(task_spec: dict) -> dict:
                goal = task_spec.get("goal", "")
                agent = task_spec.get("agent_name", "")
                if not goal:
                    return {"success": False, "error": "missing goal"}
                return await delegate_to_agent(
                    sub_goal=goal,
                    agent_name=agent,
                )

            results = await asyncio.gather(
                *[_run_one(t) for t in tasks_list],
                return_exceptions=True,
            )

            # Package results
            output = []
            for i, result in enumerate(results):
                goal = tasks_list[i].get("goal", "?")[:80]
                if isinstance(result, Exception):
                    output.append({"goal": goal, "success": False, "error": str(result)})
                else:
                    output.append({"goal": goal, **result})

            succeeded = sum(1 for r in output if r.get("success"))
            _orchestrator_self._log.info(
                "parallel_delegation.complete",
                total=len(output), succeeded=succeeded,
            )

            return {
                "success": succeeded > 0,
                "total": len(output),
                "succeeded": succeeded,
                "results": output,
            }

        if "delegate_parallel" not in self._tool_map:
            self._tool_fns.append(delegate_parallel)
            self._tool_map["delegate_parallel"] = delegate_parallel

    # ------------------------------------------------------------------
    # Approval queue (B2) — gate DANGEROUS/CRITICAL tools
    # ------------------------------------------------------------------

    async def request_approval(
        self,
        tool_name: str,
        tool_args: dict,
        inject_fn: Any | None = None,
        telegram_bot: Any | None = None,
    ) -> bool:
        """Request user approval for a dangerous tool call.

        Sends an approval request via dashboard broadcast, voice prompt,
        and optionally Telegram (P1.1).

        If ``_approval_timeout`` is <= 0, waits indefinitely for human approval.
        Otherwise waits up to ``_approval_timeout`` seconds.
        Returns True if approved, False if denied or timed out.
        """
        request_id = hashlib.md5(
            f"{tool_name}:{json.dumps(tool_args, default=str)}:{id(self)}".encode()
        ).hexdigest()[:8]

        # Try to notify the dashboard
        if self._broadcast_fn is not None:
            try:
                await self._broadcast_fn({
                    "type": "dashboard",
                    "subtype": "approval_request",
                    "request_id": request_id,
                    "tool": tool_name,
                    "args": {k: str(v)[:100] for k, v in tool_args.items()},
                    "risk": TOOL_RISK_MAP.get(tool_name, ToolRisk.DANGEROUS).value,
                    "timeout": None if self._approval_timeout <= 0 else self._approval_timeout,
                })
            except Exception:
                pass

        # P1.1: Notify via Telegram so user can approve from phone
        if telegram_bot is not None and getattr(telegram_bot, "enabled", False):
            try:
                telegram_bot.notify_approval_pending(tool_name)
            except Exception:
                pass

        # Try to send a voice prompt
        if inject_fn is not None:
            try:
                await inject_fn(
                    f"[SYSTEM: Approval required — I'm about to call {tool_name}. "
                    f"Say 'yes' or 'approve' to allow, or 'no' or 'deny' to block.]"
                )
            except Exception:
                pass

        # Wait for approval response (set via resolve_approval)
        try:
            if self._approval_timeout <= 0:
                approved = await self._approval_queue.get()
            else:
                approved = await asyncio.wait_for(
                    self._approval_queue.get(),
                    timeout=self._approval_timeout,
                )
            return bool(approved)
        except asyncio.TimeoutError:
            self._log.info("approval.timeout", tool=tool_name, request_id=request_id)
            return False  # Auto-deny on timeout

    def resolve_approval(self, approved: bool) -> None:
        """Resolve a pending approval request (called from dashboard/voice/Telegram)."""
        try:
            self._approval_queue.put_nowait(approved)
        except asyncio.QueueFull:
            pass

    # ------------------------------------------------------------------
    # C2: Message queue — steer / cancel running tasks
    # ------------------------------------------------------------------

    def inject_user_message(self, message: str) -> None:
        """Inject a user message into the running task (steer or cancel)."""
        self._message_queue.put_nowait(message)

    def _drain_messages(self) -> tuple[bool, list[str]]:
        """Drain all queued messages.  Returns (should_cancel, steer_messages)."""
        messages: list[str] = []
        cancel = False
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                msg_lower = msg.strip().lower().rstrip(".!?")
                if msg_lower in self._cancel_patterns:
                    cancel = True
                else:
                    messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return cancel, messages

    # ------------------------------------------------------------------
    # JSONL transcript logging (G3)
    # ------------------------------------------------------------------

    def _init_transcript_log(self) -> str | None:
        """Create a JSONL transcript file for this session."""
        from pathlib import Path as _P
        import time
        log_dir = _P(__file__).resolve().parent.parent / "data" / "transcripts"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"session_{ts}.jsonl"
            return str(path)
        except Exception:
            return None

    def _log_transcript(self, event: str, **data: Any) -> None:
        """Append a JSON line to the session transcript."""
        if not self._transcript_path:
            return
        import time
        entry = {"ts": time.time(), "event": event, **data}
        try:
            with open(self._transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Non-critical — don't break execution

    # ------------------------------------------------------------------
    # Loop detection (C6)
    # ------------------------------------------------------------------

    def _check_for_loop(self, tool_name: str, args: dict) -> str | None:
        """Detect repetitive tool calls. Returns warning message or None."""
        call_sig = (
            f"{tool_name}:"
            f"{hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:8]}"
        )

        # Count consecutive identical calls
        consecutive = 0
        for prev in reversed(self._recent_calls):
            if prev == call_sig:
                consecutive += 1
            else:
                break

        self._recent_calls.append(call_sig)

        warning_at = int(self._loop_cfg.get("warning_at", 2))
        strategy_at = int(self._loop_cfg.get("strategy_at", 3))
        stop_at = int(self._loop_cfg.get("stop_at", 4))
        repeats = consecutive + 1

        if repeats >= stop_at:
            self._log.warning(
                "orchestrator.loop.force_stop",
                tool=tool_name, consecutive=repeats,
            )
            return "STOP"
        if repeats >= strategy_at:
            self._log.warning(
                "orchestrator.loop.strategy_change",
                tool=tool_name,
                consecutive=repeats,
            )
            return (
                f"STRATEGY: {tool_name} has repeated {repeats} times with the same arguments. "
                "Change approach: inspect state first, then try a different tool or parameters."
            )
        if repeats >= warning_at:
            self._log.warning(
                "orchestrator.loop.warning",
                tool=tool_name, consecutive=repeats,
            )
            return (
                f"WARNING: You've called {tool_name} with the same arguments "
                f"{repeats} times consecutively."
            )
        return None

    # ------------------------------------------------------------------
    # Context compaction (A6) + pre-compaction flush (A3)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_context_chars(contents: list) -> int:
        """Rough estimate of total context size in characters."""
        total = 0
        for item in contents:
            if hasattr(item, "parts"):
                for part in (item.parts or []):
                    if hasattr(part, "text") and part.text:
                        total += len(part.text)
                    elif hasattr(part, "function_response"):
                        total += len(str(part.function_response))
            elif isinstance(item, str):
                total += len(item)
        return total

    def _compact_context(self, contents: list) -> list:
        """Compact context while preserving semantically important turns.
        
        Improved: Pins the original user goal (the first user message) 
        to prevent 'forgetting' during long tasks.
        """
        from google.genai import types as _types

        if len(contents) <= 6:
            return contents

        # Keep first turn (system instruction)
        head = [contents[0]]
        
        # Identify the original goal (first User message after system)
        goal_msg = None
        for i in range(1, len(contents)):
            if getattr(contents[i], "role", "") == "user":
                goal_msg = contents[i]
                # Ensure it's not a tool result or system message
                has_text = any(getattr(p, "text", None) for p in (getattr(goal_msg, "parts", []) or []))
                if has_text:
                    break
        
        # If found, add to head (pinned)
        if goal_msg and goal_msg not in head:
            head.append(goal_msg)

        # Keep last 5 turns verbatim.
        # From the middle, keep high-value items (errors, decisions, paths, notes).
        middle = contents[len(head):-5]
        scored_middle: list[tuple[int, Any]] = []
        keep_middle_items = int(self._semantic_compaction_cfg.get("keep_middle_items", 8))
        for item in middle:
            score = self._semantic_compaction_score(item)
            scored_middle.append((score, item))

        keep_middle = [
            item
            for score, item in sorted(scored_middle, key=lambda x: x[0], reverse=True)[:keep_middle_items]
            if score > 0
        ]

        compacted = head
        for item in keep_middle:
            if hasattr(item, "parts"):
                new_parts = []
                for part in (item.parts or []):
                    if hasattr(part, "function_response"):
                        fr = part.function_response
                        name = getattr(fr, "name", "unknown")
                        summary_text = f"[Tool result: {name} — completed]"
                        new_parts.append(_types.Part(text=summary_text))
                    else:
                        new_parts.append(part)
                compacted.append(_types.Content(role=item.role, parts=new_parts))
            else:
                compacted.append(item)

        compacted.extend(contents[-5:])
        self._log.info(
            "orchestrator.context.compacted",
            before=len(contents), after=len(compacted),
        )
        return compacted

    def _pre_compaction_flush(self, contents: list) -> None:
        """Inject a message asking the model to save critical state before compaction."""
        from google.genai import types as _types
        flush_prompt = (
            "[SYSTEM: Context window is approaching limit. Before compaction, "
            "save any critical information using save_note(). Include: "
            "1) Current task status and remaining steps, "
            "2) Important file paths or values discovered, "
            "3) User preferences learned this session. "
            "Call save_note() now, then continue your task.]"
        )
        contents.append(_types.Content(
            role="user",
            parts=[_types.Part(text=flush_prompt)],
        ))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compact_memory(self, limit: int = 50, cluster_size: int = 10) -> bool:
        """Phase 4: Memory Compaction — summarize fragmented memories into lessons.

        Uses a lock to prevent concurrent compaction (race condition fix).
        """
        if not self._memory_store or not self._client:
            return False

        if self._compaction_lock.locked():
            self._log.debug("memory.compaction_skipped_locked")
            return False

        async with self._compaction_lock:
            try:
                recent = self._memory_store.get_recent_memories(limit=limit)
                # Find tool use or failure clusters that aren't already lessons
                fragments = [m for m in recent if m.entry_type in ("tool_use", "failure_path", "interaction")]
                if len(fragments) < cluster_size:
                    return False

                self._log.info("memory.compaction_start", fragments=len(fragments))

                # Format fragment text for LLM
                fragment_text = "\n".join([f"- [{f.entry_type}] {f.content}" for f in fragments[:cluster_size]])

                prompt = (
                    "Analyze the following fragmented logs from an AI agent's computer session. "
                    "Synthesize them into one single, high-level 'Lesson Learned' or 'Project Insight'. "
                    "Focus on what worked, what failed, and the final state. "
                    "The result must be a concise, one-sentence memory.\n\n"
                    f"FRAGMENTED LOGS:\n{fragment_text}"
                )

                from google.genai import types as _types
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=256,
                    )
                )

                lesson = response.text.strip()
                if lesson:
                    # Store the distilled lesson
                    self._memory_store.add(lesson, entry_type="lesson_learned")
                    # Delete the original fragments
                    self._memory_store.delete_entries([f.id for f in fragments[:cluster_size]])
                    self._log.info("memory.compacted", lesson=lesson[:100])
                    return True

                return False
            except Exception as exc:
                self._log.warning("memory.compaction_failed", error=str(exc))
                return False

    def rebind(self, inject_context: Callable[[str], Awaitable[None]]) -> None:
        """Update the active inject_context callback for running tasks."""
        self._active_inject_context = inject_context
        self._log.info("orchestrator.rebound")

    def get_evaluation_stats(self) -> dict:
        """Return aggregate evaluation statistics.

        Data source for /api/evaluation/stats endpoint.
        Returns overall score, per-agent breakdown, and trajectory metrics.
        """
        if self._evaluation_store is not None:
            return self._evaluation_store.get_stats()
        return {"total_tasks": 0, "error": "evaluation not initialized"}

    async def spawn_task(
        self,
        goal: str,
        inject_context: Callable[[str], Awaitable[None]],
        resume_snapshot: dict[str, Any] | None = None,
        queue_mode: str = "auto",
    ) -> asyncio.Task | None:
        """Launch run_task as a background Task. Analyzes relationship if busy."""
        # 1. Parse Inline Directives
        try:
            from directive_parser import parse_directives
            clean_goal, directives = parse_directives(goal)
            goal = clean_goal
            if "model" in directives:
                self.set_model(directives["model"])
        except Exception:
            pass

        # 2. Smart Concurrency Handling (Lane Queuing)
        if self._active_tasks:
            active_task = list(self._active_tasks)[0]
            active_goal = self._active_task_state.get(active_task, {}).get("goal", "current task")
            
            rel = await self._analyze_request_relationship(goal, active_goal)
            self._log.info("orchestrator.concurrency.analysis", relationship=rel, new_goal=goal[:40])

            if rel == "CANCEL":
                self.cancel_all()
                await inject_context("[SYSTEM: All tasks cancelled as requested.]")
                return None
            
            if rel == "RELATED":
                self.inject_user_message(goal)
                self._log.info("orchestrator.lane.steer", goal=goal[:40])
                return active_task
            
            # UNRELATED - Ask and Delegate
            if queue_mode == "auto":
                await inject_context(
                    f"[SYSTEM: I'm already working on \"{active_goal[:60]}...\". "
                    f"Should I **cancel** it, or run this new task in the **background**? "
                    f"Please say 'cancel' or 'background'.]"
                )
                
                # Wait for choice from queue
                try:
                    # We peek the message queue for the next 15 seconds
                    start_wait = time.time()
                    choice = None
                    while time.time() - start_wait < 15:
                        if not self._message_queue.empty():
                            msg = await self._message_queue.get()
                            msg_clean = msg.lower().strip().rstrip(".!?")
                            if "cancel" in msg_clean or "stop" in msg_clean:
                                choice = "cancel"
                                break
                            elif "background" in msg_clean or "parallel" in msg_clean or "both" in msg_clean:
                                choice = "background"
                                break
                            else:
                                # Not a choice, maybe steering for the old task? 
                                # Put it back and continue waiting for choice
                                self._message_queue.put_nowait(msg)
                        await asyncio.sleep(0.5)
                    
                    if choice == "cancel":
                        self.cancel_all()
                        self._log.info("orchestrator.lane.interrupt_after_ask", goal=goal[:40])
                    elif choice == "background":
                        self._log.info("orchestrator.lane.parallel", goal=goal[:40])
                        # Proceed to spawn as parallel
                    else:
                        await inject_context("[SYSTEM: No clear choice received. I'll continue the current task.]")
                        return active_task
                except Exception:
                    return active_task

        # 3. Block Coalescing (UX Polisher)
        try:
            from block_coalescer import BlockCoalescer
            # Keep flush short so voice updates don't lag behind tool execution.
            coalescer = BlockCoalescer(inject_context, flush_interval=0.35)
            wrapped_inject = coalescer.push
        except Exception:
            wrapped_inject = inject_context

        # Keep a direct low-latency inject path for urgent voice cues.
        self._active_inject_context = inject_context

        task_id = (resume_snapshot or {}).get("task_id") or hashlib.md5(
            f"{goal}:{time.time()}".encode()
        ).hexdigest()[:8]
        task = asyncio.create_task(
            self.run_task(goal, wrapped_inject, resume_snapshot=resume_snapshot),
            name=f"orchestrator-{task_id}",
        )
        self._active_tasks.add(task)
        self._active_task_state[task] = {
            "task_id": task_id,
            "goal": goal,
            "inject_context": wrapped_inject,
            "started_at": time.time(),
            "resume_snapshot": resume_snapshot or {},
            "partial_results": list((resume_snapshot or {}).get("partial_results", [])),
            "context_window": list((resume_snapshot or {}).get("context_window", [])),
            "current_step": str((resume_snapshot or {}).get("current_step", "")),
            "tool_calls": int((resume_snapshot or {}).get("tool_calls", 0)),
            "cost_points": int((resume_snapshot or {}).get("cost_points", 0)),
            "tool_usage": dict((resume_snapshot or {}).get("tool_usage", {})),
            "cost_by_tool": dict((resume_snapshot or {}).get("cost_by_tool", {})),
        }
        self._persist_session_if_due(force=True)
        task.add_done_callback(self._active_tasks.discard)
        task.add_done_callback(lambda t: self._active_task_state.pop(t, None))
        
        # Ensure coalescer is flushed when task ends
        if wrapped_inject != inject_context:
            task.add_done_callback(lambda t: asyncio.create_task(coalescer.force_flush()))
            
        return task

    def cancel_all(self) -> None:
        """Cancel all running orchestrator tasks (call on session disconnect)."""
        count = len(self._active_tasks)
        for t in list(self._active_tasks):
            t.cancel()
        if count:
            self._log.info("orchestrator.cancel_all", count=count)

    async def run_task(
        self,
        goal: str,
        inject_context: Callable[[str], Awaitable[None]],
        resume_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Run the agentic loop to completion and inject the result.

        Uses IntentRouter to route to deterministic pipelines if applicable.
        Otherwise falls back to dynamic multi-agent routing.
        """
        # 1. Intent Routing (Fast Lane vs Slow Lane)
        try:
            from orchestrator_core.intent_router import IntentRouter
            from orchestrator_core.registry import execute_pipeline
            router = IntentRouter()
            lane = await router.route(goal, self._client)
            
            if lane != "dynamic":
                self._log.info("orchestrator.deterministic_lane", pipeline=lane)
                await inject_context(f"[SYSTEM: Executing deterministic pipeline '{lane}']")
                
                # Execute the strict DAG pipeline
                final_result = await execute_pipeline(lane, goal, self._client)
                
                await inject_context(f"Task completed via strict pipeline: {lane}\n\nI have finished the structured workflow. Here is the result:\n{_format_for_voice(final_result)}")
                return

        except Exception as e:
            self._log.error("orchestrator.routing_failed", error=str(e))
            # Fall through to dynamic lane

        # Select the best agent for this task (semantic routing when available)
        agent_name = _select_agent(goal, self._agent_configs, self._memory_store)
        agent_cfg = self._agent_configs.get(agent_name, {})
        agent_model = agent_cfg.get("model", self._model)
        agent_model = self._route_model_for_goal(goal, agent_model)
        agent_tools_key = agent_cfg.get("tools", "all")
        max_iters = agent_cfg.get("max_iterations", _MAX_ITERATIONS)

        # D2: Active mode overrides agent config
        mode_instruction_prefix = ""
        if self._active_mode:
            agent_tools_key = self._active_mode.get("tools", agent_tools_key)
            max_iters = self._active_mode.get("max_iterations", max_iters)
            mode_instruction_prefix = self._active_mode.get("system_instruction_prefix", "")

        # D3: Build instruction suffix from mode prefix + loaded skills
        instruction_suffix_parts: list[str] = []
        if mode_instruction_prefix:
            instruction_suffix_parts.append(mode_instruction_prefix)
        # D4: Agent-level system instruction from manifest
        agent_system_instruction = agent_cfg.get("system_instruction", "")
        if agent_system_instruction:
            instruction_suffix_parts.append(agent_system_instruction)
        if self._skills:
            try:
                import sys as _sys
                _cloud_dir = str(pathlib.Path(__file__).resolve().parent)
                if _cloud_dir not in _sys.path:
                    _sys.path.insert(0, _cloud_dir)
                from skill_loader import get_skill_instruction_fragment
                skill_fragment = get_skill_instruction_fragment(self._skills)
                if skill_fragment:
                    instruction_suffix_parts.append(skill_fragment)
            except Exception:
                pass
        instruction_suffix = "\n".join(instruction_suffix_parts)
        system_instruction = self._compose_system_instruction(instruction_suffix)

        runtime_state = self._active_task_state.get(asyncio.current_task())
        if runtime_state is not None:
            runtime_state["agent"] = agent_name
            runtime_state["model"] = agent_model
            if resume_snapshot:
                runtime_state["resume_snapshot"] = resume_snapshot

        self._log.info(
            "orchestrator.task.start",
            goal=goal[:120],
            agent=agent_name,
            model=agent_model,
        )
        
        # Broadcast agent status for UI
        if self._broadcast_fn:
            asyncio.create_task(self._broadcast_fn({
                "type": "agent_status",
                "payload": {"agents": self.get_ui_agent_status()}
            }))

        # D1: Set current agent for note scoping
        self._current_agent = agent_name
        self._log_transcript("task_start", goal=goal, agent=agent_name, model=agent_model)

        # C4: Create a Task record for structured tracking
        _current_task = None
        if self._task_store is not None:
            try:
                from task_state import Task
                _current_task = Task(goal=goal)
                _current_task.mark_running()
                self._task_store.save(_current_task)
            except Exception:
                pass
        try:
            # Filter tools based on agent's tool set
            tool_set = _TOOL_SETS.get(agent_tools_key)
            if tool_set is not None:
                agent_tool_fns = [
                    fn for fn in self._tool_fns if fn.__name__ in tool_set
                ]
                agent_tool_map = {fn.__name__: fn for fn in agent_tool_fns}
            else:
                agent_tool_fns = self._tool_fns
                agent_tool_map = self._tool_map

            # Per-agent deny lists (B5) + layered policy pipeline (B4)
            deny_list = agent_cfg.get("deny_tools", [])
            # D2: Mode deny_tools merge
            if self._active_mode:
                deny_list = list(set(deny_list) | set(self._active_mode.get("deny_tools", [])))
            agent_tool_fns = self._apply_policy_pipeline(agent_tool_fns, deny_list)
            agent_tool_map = {fn.__name__: fn for fn in agent_tool_fns}
            if deny_list or self._global_deny_tools or self._session_deny_tools:
                self._log.info(
                    "orchestrator.policy_applied",
                    agent=agent_name,
                    agent_denied=deny_list,
                    global_denied=list(self._global_deny_tools),
                    session_denied=list(self._session_deny_tools),
                )

            result_text = await self._agent_loop(
                goal,
                model=agent_model,
                tool_fns=agent_tool_fns,
                tool_map=agent_tool_map,
                max_iterations=max_iters,
                task_obj=_current_task,
                system_instruction=system_instruction,
                inject_context=inject_context,
                runtime_state=runtime_state,
            )
            # Record in session memory so future tasks have context
            self._task_history.append((goal, result_text))
            self._log_transcript("task_complete", goal=goal, result=result_text[:500])

            # --- Evaluation: LLM-as-judge post-task scoring ---
            if self._llm_judge and self._evaluation_store:
                try:
                    from evaluation import TrajectoryRecorder
                    # Get the trajectory recorder from the agent loop result
                    recorder = getattr(self, '_last_trajectory_recorder', None)
                    if recorder:
                        eval_result = recorder.finalize(result_text)
                        # Get recalled context for judge evaluation
                        recalled = await self._recall_past_context(goal)
                        judge_scores = await self._llm_judge.evaluate(eval_result, recalled)
                        eval_result.judge_scores = judge_scores
                        self._evaluation_store.record(eval_result)
                        self._log_transcript(
                            "evaluation",
                            scores=judge_scores.to_dict(),
                            overall=judge_scores.overall,
                        )
                        self._log.info(
                            "orchestrator.evaluation",
                            overall=round(judge_scores.overall, 2),
                            task_completion=judge_scores.task_completion,
                            efficiency=judge_scores.efficiency,
                        )
                except Exception as eval_exc:
                    self._log.debug("orchestrator.evaluation.failed", error=str(eval_exc))

            # G4: Auto-persist session state after each task completion
            self._persist_session()

            # Phase 4: Memory Compaction (Learning Loop)
            # Try to summarize recent tool/failure fragments into lessons.
            asyncio.create_task(self.compact_memory())

            # C4: Mark task done
            if _current_task is not None:
                try:
                    _current_task.mark_done()
                    self._task_store.save(_current_task)
                except Exception:
                    pass
            completion_msg = (
                f"[SYSTEM: The autonomous task executor has completed the task.\n"
                f"What was done: {_format_for_voice(result_text)}\n"
                f"Acknowledge completion in 1-2 sentences. Be concise and natural.]"
            )
            await inject_context(completion_msg)
            self._log.info(
                "orchestrator.task.complete",
                goal=goal[:80],
                agent=agent_name,
            )

        except asyncio.CancelledError:
            self._log.info("orchestrator.task.cancelled", goal=goal[:60])
            self._log_transcript("task_cancelled", goal=goal)
            if _current_task is not None:
                try:
                    _current_task.mark_cancelled()
                    self._task_store.save(_current_task)
                except Exception:
                    pass
            # Notify user that task was cancelled (prevents silent disappearance)
            try:
                await inject_context(
                    f"[SYSTEM: The task was cancelled.\n"
                    f"Goal: {goal}\n"
                    f"Tell the user the task was stopped. Be brief.]"
                )
            except Exception:
                pass
            raise  # Re-raise so the Task is properly cancelled

        except Exception as exc:
            self._log.exception("orchestrator.task.error", goal=goal[:60])

            # C4: Mark task failed
            if _current_task is not None:
                try:
                    _current_task.mark_failed()
                    self._task_store.save(_current_task)
                except Exception:
                    pass

            # Error classification (C1) — determine recovery strategy
            from error_classifier import classify_error, get_strategy
            category = classify_error(exc)
            strategy = get_strategy(category)
            self._log.info(
                "orchestrator.error_classified",
                category=category.value,
                action=strategy.action,
                retries=strategy.max_retries,
            )
            self._log_transcript(
                "task_error", goal=goal, error=str(exc),
                category=category.value, action=strategy.action,
            )

            error_msg = (
                f"[SYSTEM: The autonomous task executor encountered an error.\n"
                f"Goal: {goal}\n"
                f"Error: {exc}\n"
                f"Error type: {category.value} — {strategy.message}\n"
                f"Briefly explain what went wrong in 1-2 sentences and suggest "
                f"what the user can try instead.]"
            )
            try:
                await inject_context(error_msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal: ReAct agent loop
    # ------------------------------------------------------------------

    async def _recall_past_context(self, goal: str) -> str:
        """Phase 1 & 2: Automatically query long-term memory for past lessons/failures."""
        if not self._memory_store:
            return ""

        try:
            # Query memory store for related past interactions
            memories = self._memory_store.hybrid_query(goal, top_k=5)
            if not memories:
                return ""

            lines = ["=== RECALLED MEMORIES & LESSONS ==="]
            failures = []
            general = []

            for m in memories:
                if m.entry_type == "failure_path":
                    failures.append(f"• [FAILED PATH] {m.content}")
                else:
                    general.append(f"• {m.content}")

            if failures:
                lines.append("\nCRITICAL: Avoid these previously failed approaches:")
                lines.extend(failures)
            
            if general:
                lines.append("\nRelevant past context:")
                lines.extend(general)

            return "\n".join(lines) + "\n"
        except Exception as exc:
            self._log.warning("orchestrator.recall_failed", error=str(exc))
            return ""

    async def _agent_loop(
        self,
        goal: str,
        model: str | None = None,
        tool_fns: list | None = None,
        tool_map: dict | None = None,
        max_iterations: int = _MAX_ITERATIONS,
        task_obj: Any | None = None,
        system_instruction: str = "",
        inject_context: Callable[[str], Awaitable[None]] | None = None,
        runtime_state: dict[str, Any] | None = None,
    ) -> str:
        """ReAct loop: think → call tools → observe → repeat until done.

        Includes: loop detection (C6 + P3.4), risk logging (B1),
        context compaction (A6), pre-compaction flush (A3),
        strip old tool results (A7), JSONL transcript (G3),
        and policy quotas/rate limits (P3.1).

        Returns the final summary text from the orchestrator model.
        Raises on unrecoverable errors.
        """
        from google.genai import types as _types

        use_model = model or self._model
        use_tool_fns = tool_fns if tool_fns is not None else self._tool_fns
        use_tool_map = tool_map if tool_map is not None else self._tool_map
        effective_instruction = system_instruction or _ORCHESTRATOR_SYSTEM_INSTRUCTION

        task_cost_points = int((runtime_state or {}).get("cost_points", 0))
        task_tool_calls = int((runtime_state or {}).get("tool_calls", 0))

        # Final guard: dedupe by callable name before passing to model API.
        _seen_tool_names: set[str] = set()
        _unique_tool_fns: list = []
        for _fn in use_tool_fns:
            _n = _fn.__name__
            if _n in _seen_tool_names:
                continue
            _seen_tool_names.add(_n)
            _unique_tool_fns.append(_fn)
        use_tool_fns = _unique_tool_fns

        # --- Trajectory recording (Agent Factory: Tool Utilization eval) ---
        _trajectory_recorder = None
        try:
            from evaluation import TrajectoryRecorder
            agent_name = (runtime_state or {}).get("agent", "unknown")
            _trajectory_recorder = TrajectoryRecorder(goal, agent_name, use_model)
        except ImportError:
            pass

        # Build context preamble from past tasks and saved notes
        context_parts: list[str] = []
        if self._task_history:
            context_parts.append("=== PREVIOUS TASKS IN THIS SESSION ===")
            for i, (prev_goal, prev_result) in enumerate(self._task_history, 1):
                context_parts.append(f"{i}. Goal: {prev_goal}\n   Result: {prev_result}")
        if self._notes:
            context_parts.append("=== SAVED NOTES ===")
            for key, val in self._notes.items():
                context_parts.append(f"- {key}: {val}")

        initial_text = goal
        resume_snapshot = (runtime_state or {}).get("resume_snapshot", {})
        resume_prefix = ""
        if resume_snapshot:
            resume_lines = [
                "=== RESUME SNAPSHOT ===",
                f"Task ID: {resume_snapshot.get('task_id', 'unknown')}",
                f"Last step: {resume_snapshot.get('current_step', 'unknown')}",
            ]
            for item in list(resume_snapshot.get("partial_results", []))[-5:]:
                resume_lines.append(f"- {item}")
            if resume_snapshot.get("context_window"):
                resume_lines.append("Recent context:")
                for line in list(resume_snapshot.get("context_window", []))[-5:]:
                    resume_lines.append(f"* {line}")
            resume_prefix = "\n".join(resume_lines) + "\n\n"
        if context_parts:
            initial_text = resume_prefix + "\n".join(context_parts) + "\n\n=== CURRENT TASK ===\n" + goal
        else:
            initial_text = resume_prefix + initial_text

        # Phase 1: Recall-Before-Respond
        recalled_context = await self._recall_past_context(goal)
        if recalled_context:
            initial_text = recalled_context + "\n" + initial_text

        history: list[_types.Content] = [
            _types.Content(
                role="user",
                parts=[_types.Part(text=initial_text)],
            )
        ]

        # Reset loop detection for this task
        self._recent_calls.clear()
        self._tool_usage_total.clear()
        self._tool_cost_total.clear()
        _recent_thoughts = []

        for iteration in range(max_iterations):
            # --- Wrap-up warning: alert the model it's running low on steps ---
            remaining = max_iterations - iteration
            if remaining == 5:
                history.append(_types.Content(
                    role="user",
                    parts=[_types.Part(text=(
                        "[SYSTEM WARNING: You have 5 iterations remaining. "
                        "Finish your current step, then provide a clear summary of: "
                        "(1) what was completed, (2) what still needs to be done. "
                        "If the task is not finishable in 5 steps, say so explicitly.]"
                    ))],
                ))
                self._log.info("orchestrator.wrap_up_warning", remaining=remaining)
            self._log.debug(
                "orchestrator.loop", iteration=iteration, goal=goal[:60],
            )
            if runtime_state is not None:
                recent: list[str] = []
                for content in history[-10:]:
                    if not hasattr(content, "parts"):
                        continue
                    for part in (content.parts or []):
                        if getattr(part, "text", None):
                            recent.append(str(part.text)[:300])
                            break
                        fr = getattr(part, "function_response", None)
                        if fr is not None:
                            recent.append(f"[tool:{getattr(fr, 'name', 'unknown')}]")
                            break
                runtime_state["context_window"] = self._compact_runtime_context_window(recent)
                self._persist_session_if_due()

            # --- C2: Check message queue for steer/cancel ---
            should_cancel, steer_msgs = self._drain_messages()
            if should_cancel:
                self._log.info("orchestrator.task.cancelled_by_user", goal=goal[:60])
                self._log_transcript("task_cancelled_by_user", goal=goal)
                return "Task cancelled by user."
            if steer_msgs:
                steer_text = " | ".join(steer_msgs)
                history.append(_types.Content(
                    role="user",
                    parts=[_types.Part(text=f"[USER UPDATE: {steer_text}]")],
                ))
                self._log.info("orchestrator.steer", messages=len(steer_msgs))

            # --- Context compaction check (A6) ---
            ctx_chars = self._estimate_context_chars(history)
            if ctx_chars > _CONTEXT_COMPACT_CHARS:
                self._log.warning(
                    "orchestrator.context.compacting",
                    chars=ctx_chars, threshold=_CONTEXT_COMPACT_CHARS,
                )
                # Pre-compaction flush (A3) — let model save state
                self._pre_compaction_flush(history)
                history = self._compact_context(history)
            elif ctx_chars > _CONTEXT_WARN_CHARS:
                self._log.warning(
                    "orchestrator.context.large",
                    chars=ctx_chars, threshold=_CONTEXT_WARN_CHARS,
                )

            # Call the orchestrator model with full tool list.
            # Wrap in a cancel-aware task so user cancels are processed
            # even while waiting for the model response.
            async def _cancel_watcher() -> None:
                """Poll message queue every 2s; raise CancelledError if cancel detected."""
                while True:
                    await asyncio.sleep(2)
                    if not self._message_queue.empty():
                        # Peek — only cancel if it's actually a cancel keyword
                        try:
                            msg = self._message_queue.get_nowait()
                            msg_lower = msg.strip().lower().rstrip(".!?")
                            if msg_lower in self._cancel_patterns:
                                raise asyncio.CancelledError("user_cancel")
                            else:
                                # Not a cancel — put it back for drain
                                self._message_queue.put_nowait(msg)
                        except asyncio.QueueEmpty:
                            pass

            # Model Fallback Resilience (OpenClaw style)
            async def _generate_with_fallback():
                fallback_chain = list(self._orchestrator_settings.get("models", {}).get("fallback_chain", ["gemini-2.5-flash"]))
                models_to_try = [use_model] + fallback_chain
                
                last_exc = None
                for model_name in models_to_try:
                    try:
                        return await self._client.aio.models.generate_content(
                            model=model_name,
                            contents=history,
                            config=_types.GenerateContentConfig(
                                system_instruction=effective_instruction,
                                tools=use_tool_fns,
                                temperature=0.2,
                                max_output_tokens=4096,
                                automatic_function_calling=_types.AutomaticFunctionCallingConfig(
                                    disable=True,
                                ),
                            ),
                        )
                    except Exception as api_exc:
                        last_exc = api_exc
                        err_str = str(api_exc).lower()
                        # Only fallback on transient/quota errors
                        if any(code in err_str for code in ("429", "503", "500", "quota", "exhausted", "limit")):
                            self._log.warning("orchestrator.model_fallback", attempted=model_name, error=str(api_exc))
                            if inject_context:
                                asyncio.create_task(inject_context(f"[SYSTEM: Model {model_name} busy. Trying fallback...]"))
                            continue
                        raise # Rethrow permanent errors (e.g. 400 Bad Request)
                
                if last_exc:
                    raise last_exc

            model_task = asyncio.ensure_future(_generate_with_fallback())
            cancel_task = asyncio.ensure_future(_cancel_watcher())

            try:
                done, pending = await asyncio.wait(
                    {model_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

                if model_task in done:
                    response = model_task.result()
                else:
                    # Cancel watcher fired — user said cancel
                    self._log.info("orchestrator.task.cancelled_during_model_call")
                    self._log_transcript("task_cancelled_by_user", goal=goal)
                    if task_obj:
                        try:
                            task_obj.mark_cancelled()
                            self._task_store.save(task_obj)
                        except Exception:
                            pass
                    return "Task cancelled by user."
            except asyncio.CancelledError:
                cancel_task.cancel()
                raise

            # --- Parse response into function calls + text ---
            function_calls: list[Any] = []
            text_parts: list[str] = []
            thinking_parts: list[str] = []

            candidates = response.candidates or []
            if not candidates:
                break  # No response — stop

            model_content = candidates[0].content
            for part in (model_content.parts or []) if model_content else []:
                if getattr(part, "function_call", None):
                    function_calls.append(part.function_call)
                elif getattr(part, "text", None):
                    raw_text = part.text
                    # Parse Architectural CoT <thinking> blocks
                    import re
                    thoughts = re.findall(r'<thinking>(.*?)</thinking>', raw_text, re.DOTALL)
                    if thoughts:
                        for t in thoughts:
                            thinking_parts.append(t.strip())
                        # Remove thinking blocks from what the user "sees"
                        clean_text = re.sub(r'<thinking>.*?</thinking>', '', raw_text, flags=re.DOTALL).strip()
                        if clean_text:
                            text_parts.append(clean_text)
                    else:
                        text_parts.append(raw_text)

            # --- Reasoning trace capture & Streaming (Architectural CoT) ---
            if thinking_parts:
                full_thought = "\n".join(thinking_parts)
                if _trajectory_recorder:
                    _trajectory_recorder.record_reasoning(full_thought, iteration)
                self._log_transcript("reasoning", text=full_thought[:500], iteration=iteration)
                
                # Pillar 4: Semantic Loop Breaker
                import difflib
                _recent_thoughts.append(full_thought)
                if len(_recent_thoughts) >= 3:
                    # Check similarity between the last 3 thoughts
                    sim1 = difflib.SequenceMatcher(None, _recent_thoughts[-1], _recent_thoughts[-2]).ratio()
                    sim2 = difflib.SequenceMatcher(None, _recent_thoughts[-2], _recent_thoughts[-3]).ratio()
                    if sim1 > 0.85 and sim2 > 0.85:
                        self._log.warning("orchestrator.semantic_loop_detected", similarity=sim1)
                        history.append(_types.Content(
                            role="user",
                            parts=[_types.Part(text="[SYSTEM: YOU ARE STUCK IN A REASONING LOOP. Your last 3 thoughts were nearly identical. ABORT YOUR CURRENT STRATEGY OR TRY A DIFFERENT TOOL.]")]
                        ))
                        _recent_thoughts.clear() # Reset after warning

                # Stream the reasoning to the UI/Telegram
                thought_preview = full_thought.split('\n')[0][:60]
                if self._inject_fn:
                    # Non-blocking dispatch to adk_server's progress handler
                    asyncio.create_task(self._inject_fn(f"[REASONING] 🤔 {thought_preview}..."))

            # --- Planning Turn Enforcement (Active Steering) ---
            # If this is the first turn (iteration 0) and the model tries to call tools
            # WITHOUT a thinking block, intercept it.
            if iteration == 0 and function_calls and not thinking_parts:
                self._log.warning("orchestrator.planning_violation", goal=goal[:60])
                history.append(model_content)
                history.append(
                    _types.Content(
                        role="user",
                        parts=[_types.Part(text="SYSTEM: You MUST generate a step-by-step plan inside `<thinking>` tags BEFORE executing any tools. Cancelled tool calls. Please think and plan first.")],
                    )
                )
                continue  # Skip tool execution, force model to plan

            # Append model turn to history
            history.append(model_content)

            # No function calls → model gave a final text answer
            if not function_calls:
                final = " ".join(text_parts).strip() or "Task completed successfully."
                self._log_transcript("final_answer", text=final[:500])
                # Stash recorder on self for run_task LLM-as-judge hook
                if _trajectory_recorder:
                    self._last_trajectory_recorder = _trajectory_recorder
                return final

            # --- Execute all function calls and collect responses ---
            fn_response_parts: list[_types.Part] = []

            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                # Loop detection (C6) — check for repetitive calls
                loop_status = self._check_for_loop(tool_name, tool_args)
                if loop_status == "STOP":
                    self._log_transcript(
                        "loop_force_stop", tool=tool_name, iteration=iteration,
                    )
                    return (
                        f"Task stopped: detected infinite loop calling {tool_name} "
                        f"with the same arguments. Try rephrasing the task."
                    )

                # Risk logging (B1)
                risk = TOOL_RISK_MAP.get(tool_name, ToolRisk.MODERATE)
                self._log_transcript(
                    "tool_call", tool=tool_name, risk=risk.value,
                    args={k: str(v)[:80] for k, v in tool_args.items()},
                )

                allowed, budget_error, next_cost_points, point_cost = self._enforce_tool_budget(
                    tool_name,
                    task_cost_points,
                    task_tool_calls,
                )
                if not allowed:
                    exec_result = {"success": False, "error": budget_error}
                    fn_response_parts.append(
                        _types.Part(
                            function_response=_types.FunctionResponse(
                                name=tool_name,
                                response=exec_result,
                            )
                        )
                    )
                    continue

                fs_allowed, fs_error = self._enforce_filesystem_policy(tool_name, tool_args)
                if not fs_allowed:
                    exec_result = {"success": False, "error": fs_error}
                    fn_response_parts.append(
                        _types.Part(
                            function_response=_types.FunctionResponse(
                                name=tool_name,
                                response=exec_result,
                            )
                        )
                    )
                    continue

                task_cost_points = next_cost_points
                task_tool_calls += 1
                if runtime_state is not None:
                    runtime_state["cost_points"] = task_cost_points
                    runtime_state["tool_calls"] = task_tool_calls
                    runtime_state["current_step"] = f"{tool_name}({', '.join(tool_args.keys())})"
                    tool_usage = runtime_state.setdefault("tool_usage", {})
                    tool_usage[tool_name] = int(tool_usage.get(tool_name, 0)) + 1
                    cost_by_tool = runtime_state.setdefault("cost_by_tool", {})
                    cost_by_tool[tool_name] = int(cost_by_tool.get(tool_name, 0)) + int(point_cost)
                    self._persist_session_if_due()

                # Approval gate (B2) — require approval for DANGEROUS/CRITICAL
                if risk in (ToolRisk.DANGEROUS, ToolRisk.CRITICAL):
                    approved = await self.request_approval(
                        tool_name, tool_args,
                        inject_fn=inject_context,
                        telegram_bot=getattr(self, "_telegram_bot", None),
                    )
                    if not approved:
                        self._log.info("orchestrator.tool.denied", name=tool_name)
                        exec_result = {
                            "success": False,
                            "error": (
                                f"Tool '{tool_name}' was denied by the user "
                                f"(risk level: {risk.value}). Try a safer alternative."
                            ),
                        }
                        fn_response_parts.append(
                            _types.Part(
                                function_response=_types.FunctionResponse(
                                    name=tool_name,
                                    response=exec_result,
                                )
                            )
                        )
                        continue

                fn = use_tool_map.get(tool_name)
                if fn is None:
                    self._log.warning(
                        "orchestrator.unknown_tool", name=tool_name,
                    )
                    exec_result: dict = {
                        "success": False,
                        "error": f"Unknown tool: '{tool_name}'. "
                                 f"Available: {sorted(use_tool_map.keys())}",
                    }
                else:
                    # C4: Create Step record
                    _step = None
                    if task_obj is not None:
                        try:
                            from task_state import Step, StepType
                            _step = Step(
                                tool_name=tool_name,
                                action=f"{tool_name}({', '.join(f'{k}={str(v)[:40]}' for k, v in tool_args.items())})",
                                step_type=StepType.TOOL,
                            )
                            _step.mark_running()
                            task_obj.steps.append(_step)
                        except Exception:
                            pass

                    self._log.info(
                        "orchestrator.tool_call",
                        name=tool_name,
                        risk=risk.value,
                        args={k: str(v)[:80] for k, v in tool_args.items()},
                    )
                    # F4: Run before_tool hooks
                    for hook in self._before_tool_hooks:
                        try:
                            modified = await hook(tool_name, tool_args)
                            if isinstance(modified, dict):
                                tool_args = modified
                        except Exception as hook_exc:
                            self._log.debug("hook.before_tool.error", error=str(hook_exc))
                    # Emit tool_event for UI
                    if self._broadcast_fn:
                        asyncio.create_task(self._broadcast_fn({
                            "type": "tool_event",
                            "payload": {
                                "name": tool_name,
                                "status": "running",
                                "latency_ms": 0,
                                "timestamp": time.strftime("%H:%M:%S"),
                            }
                        }))

                    # Task 15: Human-sounding voice narration during tool execution
                    if inject_context:
                        # Internal memory/context tools are useful but noisy for voice UX.
                        quiet_tools = {
                            "search_notes", "get_notes", "save_note", "memory_stats", "export_context",
                            "web_cache_get",
                        }
                        messages = {
                            "browser_connect": "I'm opening the browser now.",
                            "browser_navigate": "Taking you to that page now.",
                            "browser_click_element": "Clicking that on the page.",
                            "browser_fill_form": "Filling that form now.",
                            "browser_extract_text": "Reading the page details now.",
                            "open_application": "Opening that app now.",
                            "read_file": "Checking that file now.",
                            "write_file": "Saving those changes now.",
                            "patch_file": "Applying that update now.",
                            "smart_click": "Clicking that now.",
                            "screen_type": "Typing that now.",
                        }
                        narration = messages.get(tool_name, "I'm on it.")
                        if tool_name not in quiet_tools:
                            # Use direct inject path (non-coalesced) and wait briefly so
                            # narration starts before the tool work to avoid perceived lag.
                            try:
                                low_latency_inject = self._active_inject_context or inject_context
                                await asyncio.wait_for(
                                    low_latency_inject(
                                        f"[SYSTEM: Tell the user exactly this, in your own voice, right now: '{narration}']"
                                    ),
                                    timeout=1.2,
                                )
                            except Exception:
                                pass

                    try:
                        # Wrap tool execution with progress heartbeat + timeout.
                        # If a tool takes longer than heartbeat interval, inject
                        # an in-progress message instead of silence.
                        _TOOL_TIMEOUT = self._tool_timeout_seconds
                        _HEARTBEAT_INTERVAL = self._heartbeat_interval_seconds
                        
                        start_time = time.time()

                        async def _heartbeat_while_running(
                            coro,
                            tool_name_for_hb: str,
                        ):
                            """Run coro; inject progress heartbeats every 5s."""
                            tool_task = asyncio.ensure_future(coro)
                            elapsed = 0
                            heartbeat_sent = False
                            while not tool_task.done():
                                try:
                                    await asyncio.wait_for(
                                        asyncio.shield(tool_task),
                                        timeout=_HEARTBEAT_INTERVAL,
                                    )
                                except asyncio.TimeoutError:
                                    elapsed += _HEARTBEAT_INTERVAL
                                    if inject_context and not heartbeat_sent:
                                        heartbeat_sent = True
                                        try:
                                            await inject_context(
                                                f"[SYSTEM: Still working on {tool_name_for_hb}... "
                                                f"({elapsed}s elapsed). Tell the user you're still "
                                                f"working on it briefly.]"
                                            )
                                        except Exception:
                                            pass
                                    if elapsed >= _TOOL_TIMEOUT:
                                        tool_task.cancel()
                                        raise asyncio.TimeoutError(
                                            f"{tool_name_for_hb} timed out after {_TOOL_TIMEOUT}s"
                                        )
                            return tool_task.result()

                        exec_result = await _heartbeat_while_running(
                            fn(**tool_args), tool_name,
                        )
                        
                        latency = int((time.time() - start_time) * 1000)
                        
                        # Emit completion tool_event for UI
                        if self._broadcast_fn:
                            asyncio.create_task(self._broadcast_fn({
                                "type": "tool_event",
                                "payload": {
                                    "name": tool_name,
                                    "status": "ok" if exec_result.get("success", True) else "fail",
                                    "latency_ms": latency,
                                    "timestamp": time.strftime("%H:%M:%S"),
                                }
                            }))
                        if runtime_state is not None:
                            runtime_state.setdefault("partial_results", []).append(
                                f"{tool_name}: {str(exec_result)[:240]}"
                            )
                            runtime_state["partial_results"] = runtime_state["partial_results"][-8:]
                            self._persist_session_if_due()
                    except TypeError as exc:
                        self._log.warning(
                            "orchestrator.tool_bad_args",
                            name=tool_name, error=str(exc),
                        )
                        exec_result = {
                            "success": False,
                            "error": f"Bad arguments for {tool_name}: {exc}",
                        }
                        if _step: _step.mark_failed(str(exc))
                    except (asyncio.TimeoutError, TimeoutError) as exc:
                        self._log.warning(
                            "orchestrator.tool_timeout",
                            name=tool_name, error=str(exc),
                        )
                        exec_result = {
                            "success": False,
                            "error": f"Tool {tool_name} timed out: {exc}. Try a different approach.",
                        }
                        if _step: _step.mark_failed(f"timeout: {exc}")
                    except Exception as exc:
                        self._log.exception(
                            "orchestrator.tool_error", name=tool_name,
                        )
                        exec_result = {"success": False, "error": str(exc)}
                        if _step: _step.mark_failed(str(exc))
                    else:
                        if _step:
                            if exec_result.get("success", True):
                                _step.mark_done(str(exec_result.get("result", ""))[:200])
                            else:
                                _step.mark_failed(str(exec_result.get("error", ""))[:200])

                    # Phase 2: Negative Memory — Capture failures
                    if not exec_result.get("success", True) and self._memory_store:
                        try:
                            fail_summary = (
                                f"Failed to call {tool_name} with args {tool_args}. "
                                f"Error: {exec_result.get('error', 'unknown')}"
                            )
                            self._memory_store.add(
                                fail_summary,
                                entry_type="failure_path",
                                metadata={"tool": tool_name, "error": str(exec_result.get("error", ""))}
                            )
                        except Exception:
                            pass

                    self._log.debug(
                        "orchestrator.tool_result",
                        name=tool_name,
                        success=exec_result.get("success"),
                    )
                    # --- Trajectory recording: record this tool call ---
                    if _trajectory_recorder:
                        _trajectory_recorder.record_tool_call(
                            tool=tool_name,
                            args=tool_args,
                            result=exec_result if isinstance(exec_result, dict) else {"result": str(exec_result)[:200]},
                            latency_ms=0.0,  # Latency already tracked in heartbeat
                        )
                    # F4: Run after_tool hooks
                    for hook in self._after_tool_hooks:
                        try:
                            modified = await hook(tool_name, tool_args, exec_result)
                            if isinstance(modified, dict):
                                exec_result = modified
                        except Exception as hook_exc:
                            self._log.debug("hook.after_tool.error", error=str(hook_exc))

                # Inject loop warning if detected (C6)
                if loop_status and loop_status != "STOP":
                    exec_result["_loop_warning"] = loop_status

                fn_response_parts.append(
                    _types.Part(
                        function_response=_types.FunctionResponse(
                            name=tool_name,
                            response=exec_result,
                        )
                    )
                )

            # Append tool results as a new user turn
            history.append(
                _types.Content(role="user", parts=fn_response_parts)
            )

            # C4: Persist task state periodically
            if task_obj is not None and self._task_store is not None:
                try:
                    self._task_store.save(task_obj)
                except Exception:
                    pass

            self._persist_session_if_due()

            # --- Strip old tool results from context (A7) ---
            # After _RESULT_STRIP_AFTER_TURNS iterations, compress older results
            if iteration >= _RESULT_STRIP_AFTER_TURNS and len(history) > 6:
                cutoff = len(history) - (_RESULT_STRIP_AFTER_TURNS * 2)
                for idx in range(1, max(1, cutoff)):
                    item = history[idx]
                    if not hasattr(item, "parts"):
                        continue
                    new_parts = []
                    for part in (item.parts or []):
                        fr = getattr(part, "function_response", None)
                        if fr and hasattr(fr, "response") and isinstance(fr.response, dict):
                            stripped = _strip_tool_result(fr.response)
                            new_parts.append(_types.Part(
                                function_response=_types.FunctionResponse(
                                    name=getattr(fr, "name", "unknown"),
                                    response=stripped,
                                )
                            ))
                        else:
                            new_parts.append(part)
                    history[idx] = _types.Content(role=item.role, parts=new_parts)

        # Build a partial-results summary so the user knows what was done
        partial = ""
        if runtime_state and runtime_state.get("partial_results"):
            partial = "\nPartial results:\n" + "\n".join(
                f"  - {r}" for r in runtime_state["partial_results"][-8:]
            )
        self._log.warning(
            "orchestrator.iteration_limit",
            max=max_iterations,
            goal=goal[:80],
        )
        return (
            f"Task reached the {max_iterations}-step safety limit and could not "
            f"fully complete.{partial}\n"
            f"Original goal: {goal[:200]}\n"
            f"Please break this into smaller tasks or increase the iteration limit."
        )
