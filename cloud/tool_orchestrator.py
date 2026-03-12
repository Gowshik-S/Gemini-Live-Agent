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
_MAX_ITERATIONS = 25

# How many past task summaries to keep in the session context
_MAX_TASK_MEMORY = 20

# ---------------------------------------------------------------------------
# Context compaction thresholds (A6)
# ---------------------------------------------------------------------------
_CONTEXT_WARN_CHARS = 800_000     # ~200K tokens — log a warning
_CONTEXT_COMPACT_CHARS = 900_000  # ~225K tokens — trigger compaction

# ---------------------------------------------------------------------------
# Strip old tool results from context (A7)
# ---------------------------------------------------------------------------
_MAX_TOOL_RESULT_CHARS = 4000     # Full result limit per tool call
_RESULT_STRIP_AFTER_TURNS = 3     # Strip results older than N turns


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
        "start_process", "check_process", "stop_process",
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


def _select_agent(goal: str, agent_configs: dict[str, dict]) -> str:
    """Select the best agent for a given task goal.

    Keyword-based routing — maps task content to agent specializations.
    """
    goal_lower = goal.lower()

    # Code/dev keywords → code_agent
    code_keywords = (
        "code", "debug", "fix", "refactor", "function", "class", "variable",
        "import", "error", "traceback", "syntax", "compile", "build", "test",
        "read file", "write file", "patch", "git", "commit", "deploy",
        ".py", ".js", ".ts", ".html", ".css", ".json", ".yaml",
    )
    if any(kw in goal_lower for kw in code_keywords):
        if "code_agent" in agent_configs and agent_configs["code_agent"]["enabled"]:
            return "code_agent"

    # Creative keywords → creative_agent
    creative_keywords = (
        "generate image", "create image", "draw", "design", "generate video",
        "imagen", "veo", "picture", "illustration", "photo",
    )
    if any(kw in goal_lower for kw in creative_keywords):
        if "creative_agent" in agent_configs and agent_configs["creative_agent"]["enabled"]:
            return "creative_agent"

    # Research/explanation keywords → research_agent
    research_keywords = (
        "research", "analyze", "explain", "compare", "summarize",
        "what is", "how does", "why", "investigate", "study",
    )
    if any(kw in goal_lower for kw in research_keywords):
        if "research_agent" in agent_configs and agent_configs["research_agent"]["enabled"]:
            return "research_agent"

    # Default → task_executor for screen/computer tasks
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


def _is_task_request(text: str) -> bool:
    """Return True when the transcribed utterance looks like an executable task.

    Conservative by design — only returns True for clear action requests so
    the orchestrator does not fire on every conversational turn.
    """
    text_lower = text.strip().lower().rstrip(".!?")

    if len(text_lower) < 4:
        return False

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
         "please fix", "please show", "please switch")
    ):
        return True

    # Resume commands
    if text_lower in ("resume", "continue", "go on", "keep going", "go ahead"):
        return True

    # Skip greetings and questions
    for start in _NON_TASK_STARTS:
        if text_lower.startswith(start):
            return False

    # Very short = unlikely to be an executable task
    if len(text_lower.split()) < 3:
        return False

    # Action verb at the start of the sentence
    for verb in _TASK_ACTION_VERBS:
        if (
            text_lower.startswith(verb + " ")
            or text_lower.startswith(verb + ":")
        ):
            return True

    # Task-indicating phrases anywhere in the utterance
    for phrase in _TASK_PHRASES:
        if phrase in text_lower:
            return True

    # URL → navigation task
    if (
        "http://" in text_lower
        or "https://" in text_lower
        or "www." in text_lower
    ):
        return True

    return False


# ---------------------------------------------------------------------------
# ToolOrchestrator
# ---------------------------------------------------------------------------

_ORCHESTRATOR_SYSTEM_INSTRUCTION = (
    "You are Rio's autonomous task execution engine. "
    "The user has asked you to complete a specific task on their computer. "
    "Execute it fully using the available tools — do NOT stop to ask for "
    "confirmation mid-task. Complete the ENTIRE task autonomously.\n\n"
    "MEMORY — RECALL BEFORE RESPOND:\n"
    "- FIRST: Call search_notes(query) with keywords from the current task to "
    "recall relevant context from previous sessions. Do this BEFORE any actions.\n"
    "- Use save_note(key, value) to persist important information (e.g. file paths, "
    "user preferences, progress, decisions) for future tasks.\n"
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
    "- When done, return ONLY the summary text (no markdown, no bullet points).\n"
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
        genai_client: Any,
        tool_fns: list,
        model: str | None = None,
        memory_store: Any | None = None,
        broadcast_fn: Any | None = None,
    ) -> None:
        self._client = genai_client
        self._tool_fns = tool_fns
        self._tool_map: dict[str, Any] = {fn.__name__: fn for fn in tool_fns}
        self._model = (
            model
            or os.environ.get("ORCHESTRATOR_MODEL", _DEFAULT_ORCHESTRATOR_MODEL)
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

        # Unified memory facade (A5) — wraps vector + notes + chat
        self._unified_memory = None
        try:
            import sys, pathlib
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

        # Loop detection: recent call signatures (C6)
        self._recent_calls: collections.deque = collections.deque(maxlen=10)

        # Approval queue (B2) — gate dangerous tools via dashboard
        self._approval_queue: asyncio.Queue = asyncio.Queue()
        self._approval_timeout = 30  # seconds to wait for user approval

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
            import sys, pathlib as _pl
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
        """Save current session state (notes + task_history) to disk."""
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
            }
            (data_dir / f"{self._session_id}.json").write_text(
                json.dumps(state, indent=2, default=str), encoding="utf-8",
            )
        except Exception:
            self._log.debug("session.persist_skip")

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
            self._log.info("session.loaded", session_id=session_id)
            return True
        except Exception:
            return False

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

    def _register_memory_tools(self) -> None:
        """Add save_note, get_notes, and search_notes as callable tools for the model."""

        async def save_note(key: str, value: str) -> dict:
            """Save a persistent note for this session. Use this to remember
            important information like file paths, user preferences, or task
            progress that may be needed by future tasks."""
            # D1: Save to both agent-specific and global namespace
            agent = self._current_agent
            if agent not in self._agent_notes:
                self._agent_notes[agent] = {}
            self._agent_notes[agent][key] = value
            self._notes[key] = value  # also accessible globally
            self._log.info("memory.save_note", key=key, agent=agent)
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
            }
            if format == "full":
                snapshot["tasks"] = [
                    {"goal": g, "result": r[:300]} for g, r in self._task_history
                ]
                snapshot["notes"] = dict(self._notes)
            else:
                snapshot["recent_tasks"] = [
                    {"goal": g, "result": r[:100]}
                    for g, r in list(self._task_history)[-5:]
                ]
            # Also persist to disk
            self._persist_session()
            return {"success": True, "snapshot": snapshot}

        self._tool_fns.append(export_context)
        self._tool_map[export_context.__name__] = export_context

        # C3: Subagent delegation tool — allows the orchestrator to
        # delegate sub-goals to specialized agents with depth limits.
        _orchestrator_self = self
        _MAX_DELEGATION_DEPTH = 3
        _DELEGATION_TIMEOUT = 120  # seconds

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
            if not agent_name:
                agent_name = _select_agent(sub_goal, _orchestrator_self._agent_configs)
            agent_cfg = _orchestrator_self._agent_configs.get(agent_name, {})
            agent_model = agent_cfg.get("model", _orchestrator_self._model)
            max_iters = min(agent_cfg.get("max_iterations", 15), 15)  # Cap sub-agent iterations

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
                sub_tool_fns = [fn for fn in _orchestrator_self._tool_fns if fn.__name__ in tool_set]
                sub_tool_map = {fn.__name__: fn for fn in sub_tool_fns}
            else:
                sub_tool_fns = _orchestrator_self._tool_fns
                sub_tool_map = dict(_orchestrator_self._tool_map)

            # Apply deny policy
            deny_list = agent_cfg.get("deny_tools", [])
            sub_tool_fns = _orchestrator_self._apply_policy_pipeline(sub_tool_fns, deny_list)
            sub_tool_map = {fn.__name__: fn for fn in sub_tool_fns}

            # Run the sub-agent loop with incremented depth
            old_depth = getattr(_orchestrator_self, "_delegation_depth", 0)
            _orchestrator_self._delegation_depth = old_depth + 1
            try:
                result = await asyncio.wait_for(
                    _orchestrator_self._agent_loop(
                        sub_goal,
                        model=agent_model,
                        tool_fns=sub_tool_fns,
                        tool_map=sub_tool_map,
                        max_iterations=max_iters,
                        instruction_suffix=f"You are a sub-agent (depth {current_depth + 1}). "
                                          f"Complete the sub-goal and return a concise result.",
                    ),
                    timeout=timeout_seconds,
                )
                _orchestrator_self._log.info(
                    "subagent.completed",
                    agent=agent_name,
                    result_len=len(result),
                )
                return {"success": True, "agent": agent_name, "result": result[:3000]}
            except asyncio.TimeoutError:
                return {"success": False, "error": f"Sub-agent timed out after {timeout_seconds}s"}
            except Exception as exc:
                return {"success": False, "error": f"Sub-agent failed: {exc}"}
            finally:
                _orchestrator_self._delegation_depth = old_depth

        self._tool_fns.append(delegate_to_agent)
        self._tool_map["delegate_to_agent"] = delegate_to_agent

    # ------------------------------------------------------------------
    # Approval queue (B2) — gate DANGEROUS/CRITICAL tools
    # ------------------------------------------------------------------

    async def request_approval(
        self,
        tool_name: str,
        tool_args: dict,
        inject_fn: Any | None = None,
    ) -> bool:
        """Request user approval for a dangerous tool call.

        Sends an approval request via dashboard broadcast and optionally
        injects a voice prompt.  Waits up to ``_approval_timeout`` seconds.
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
                    "timeout": self._approval_timeout,
                })
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
            approved = await asyncio.wait_for(
                self._approval_queue.get(),
                timeout=self._approval_timeout,
            )
            return bool(approved)
        except asyncio.TimeoutError:
            self._log.info("approval.timeout", tool=tool_name, request_id=request_id)
            return False  # Auto-deny on timeout

    def resolve_approval(self, approved: bool) -> None:
        """Resolve a pending approval request (called from dashboard/voice)."""
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

        if consecutive >= 4:
            self._log.warning(
                "orchestrator.loop.force_stop",
                tool=tool_name, consecutive=consecutive + 1,
            )
            return "STOP"
        elif consecutive >= 2:
            self._log.warning(
                "orchestrator.loop.warning",
                tool=tool_name, consecutive=consecutive + 1,
            )
            return (
                f"WARNING: You've called {tool_name} with the same arguments "
                f"{consecutive + 1} times consecutively. Try a different approach."
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
        """Strip old tool results and summarize early turns."""
        from google.genai import types as _types

        if len(contents) <= 6:
            return contents

        # Keep first item (user goal) + last 5 turns verbatim
        # Strip tool results from earlier turns
        compacted = [contents[0]]
        for item in contents[1:-5]:
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

    def spawn_task(
        self,
        goal: str,
        inject_context: Callable[[str], Awaitable[None]],
    ) -> asyncio.Task:
        """Launch run_task as a background Task and return it."""
        task = asyncio.create_task(
            self.run_task(goal, inject_context),
            name=f"orchestrator-{goal[:30]}",
        )
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        return task

    def cancel_all(self) -> None:
        """Cancel all running orchestrator tasks (call on session disconnect)."""
        for t in list(self._active_tasks):
            t.cancel()

    async def run_task(
        self,
        goal: str,
        inject_context: Callable[[str], Awaitable[None]],
    ) -> None:
        """Run the agentic loop to completion and inject the result.

        Uses multi-agent routing to select the best agent (model + tools)
        for the task. Falls back to the default task_executor if routing fails.
        """
        # Select the best agent for this task
        agent_name = _select_agent(goal, self._agent_configs)
        agent_cfg = self._agent_configs.get(agent_name, {})
        agent_model = agent_cfg.get("model", self._model)
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

        self._log.info(
            "orchestrator.task.start",
            goal=goal[:120],
            agent=agent_name,
            model=agent_model,
        )
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
                instruction_suffix=instruction_suffix,
            )
            # Record in session memory so future tasks have context
            self._task_history.append((goal, result_text))
            self._log_transcript("task_complete", goal=goal, result=result_text[:500])

            # G4: Auto-persist session state after each task completion
            self._persist_session()

            # C4: Mark task done
            if _current_task is not None:
                try:
                    _current_task.mark_done()
                    self._task_store.save(_current_task)
                except Exception:
                    pass
            completion_msg = (
                f"[SYSTEM: The autonomous task executor has completed the task.\n"
                f"Agent: {agent_name} (model: {agent_model})\n"
                f"Original goal: {goal}\n"
                f"What was done: {result_text}\n"
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

    async def _agent_loop(
        self,
        goal: str,
        model: str | None = None,
        tool_fns: list | None = None,
        tool_map: dict | None = None,
        max_iterations: int = _MAX_ITERATIONS,
        task_obj: Any | None = None,
        instruction_suffix: str = "",
    ) -> str:
        """ReAct loop: think → call tools → observe → repeat until done.

        Includes: loop detection (C6), risk logging (B1),
        context compaction (A6), pre-compaction flush (A3),
        strip old tool results (A7), JSONL transcript (G3).

        Returns the final summary text from the orchestrator model.
        Raises on unrecoverable errors.
        """
        from google.genai import types as _types

        use_model = model or self._model
        use_tool_fns = tool_fns if tool_fns is not None else self._tool_fns
        use_tool_map = tool_map if tool_map is not None else self._tool_map

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
        if context_parts:
            initial_text = "\n".join(context_parts) + "\n\n=== CURRENT TASK ===\n" + goal

        history: list[_types.Content] = [
            _types.Content(
                role="user",
                parts=[_types.Part(text=initial_text)],
            )
        ]

        # Reset loop detection for this task
        self._recent_calls.clear()

        for iteration in range(max_iterations):
            self._log.debug(
                "orchestrator.loop", iteration=iteration, goal=goal[:60],
            )

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

            model_task = asyncio.ensure_future(
                self._client.aio.models.generate_content(
                    model=use_model,
                    contents=history,
                    config=_types.GenerateContentConfig(
                        system_instruction=(
                            _ORCHESTRATOR_SYSTEM_INSTRUCTION + "\n" + instruction_suffix
                            if instruction_suffix else _ORCHESTRATOR_SYSTEM_INSTRUCTION
                        ),
                        tools=use_tool_fns,
                        temperature=0.2,
                        max_output_tokens=4096,
                        automatic_function_calling=_types.AutomaticFunctionCallingConfig(
                            disable=True,
                        ),
                    ),
                )
            )
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
                    if _current_task:
                        try:
                            _current_task.mark_cancelled()
                            self._task_store.save(_current_task)
                        except Exception:
                            pass
                    return "Task cancelled by user."
            except asyncio.CancelledError:
                cancel_task.cancel()
                raise

            # --- Parse response into function calls + text ---
            function_calls: list[Any] = []
            text_parts: list[str] = []

            candidates = response.candidates or []
            if not candidates:
                break  # No response — stop

            model_content = candidates[0].content
            for part in (model_content.parts or []) if model_content else []:
                if getattr(part, "function_call", None):
                    function_calls.append(part.function_call)
                elif getattr(part, "text", None):
                    text_parts.append(part.text)

            # Append model turn to history
            history.append(model_content)

            # No function calls → model gave a final text answer
            if not function_calls:
                final = " ".join(text_parts).strip() or "Task completed successfully."
                self._log_transcript("final_answer", text=final[:500])
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

                # Approval gate (B2) — require approval for DANGEROUS/CRITICAL
                if risk in (ToolRisk.DANGEROUS, ToolRisk.CRITICAL):
                    approved = await self.request_approval(tool_name, tool_args)
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
                    try:
                        exec_result = await fn(**tool_args)
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

                    self._log.debug(
                        "orchestrator.tool_result",
                        name=tool_name,
                        success=exec_result.get("success"),
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

        return f"Task reached the {max_iterations}-step safety limit."
