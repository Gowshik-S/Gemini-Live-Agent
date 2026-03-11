"""
Rio Tool Orchestrator — Multi-Agent Tool Execution Engine.

When the native audio Live model cannot invoke function calls itself
(e.g. ``gemini-2.5-flash-native-audio-preview-*``), this orchestrator
provides full agentic tool execution using a capable model.

Architecture
------------
  [Live Audio Model]  ← voice I/O only  (native-audio Live session)
        ↕  transcription
  [ToolOrchestrator]  ← agentic executor (gemini-2.5-flash, full tool calling)
        ↕  ToolBridge
  [Local Client]      ← actual tool execution on the user's machine
        ↕
  [Computer Use Model]← visual grounding for smart_click
                        (gemini-2.5-computer-use-preview-10-2025)

Flow
----
  1. User speech is transcribed by the live session (input_audio_transcription).
  2. _is_task_request(text) decides whether execution is needed.
  3. ToolOrchestrator.run_task(goal, inject_fn) is spawned as a background Task.
  4. The orchestrator calls gemini-2.5-flash via generate_content with the
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
import os
from typing import Any, Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)

# Default orchestrator model.  Must support function calling + vision.
# gemini-2.5-flash has full tool support and is fast enough for agentic loops.
# Override with ORCHESTRATOR_MODEL env var.
_DEFAULT_ORCHESTRATOR_MODEL = "gemini-2.5-flash"

# Hard safety cap: maximum tool-call iterations per task
_MAX_ITERATIONS = 25

# How many past task summaries to keep in the session context
_MAX_TASK_MEMORY = 20

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
    "MEMORY:\n"
    "- You receive context from PREVIOUS TASKS completed earlier in this session.\n"
    "- Use save_note(key, value) to persist important information (e.g. file paths, "
    "user preferences, progress) that you or future tasks may need.\n"
    "- Use get_notes() to retrieve all saved notes before starting work if the task "
    "seems related to earlier work.\n"
    "- The PREVIOUS TASKS section shows what was already done — avoid repeating work.\n\n"
    "THE LOOP for every computer task:\n"
    "1. PLAN: Silently break the task into steps.\n"
    "2. CAPTURE: Call capture_screen first if you need to see the current state.\n"
    "3. ANALYZE: Look at the screenshot. Identify what needs to be clicked/typed.\n"
    "4. ACT: Call the appropriate tool (smart_click, screen_type, screen_hotkey…)\n"
    "5. VERIFY: After each screen action you receive an auto-captured screenshot. "
    "Analyze it to confirm the action succeeded.\n"
    "6. CONTINUE: Repeat 3-5 until the task is fully complete.\n"
    "7. REPORT: Return a concise 1-2 sentence summary of what you accomplished.\n\n"
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
    ) -> None:
        self._client = genai_client
        self._tool_fns = tool_fns
        self._tool_map: dict[str, Any] = {fn.__name__: fn for fn in tool_fns}
        self._model = (
            model
            or os.environ.get("ORCHESTRATOR_MODEL", _DEFAULT_ORCHESTRATOR_MODEL)
        )
        self._log = logger.bind(component="tool_orchestrator", model=self._model)
        # Track running tasks so they can be cancelled on disconnect
        self._active_tasks: set[asyncio.Task] = set()
        # Session-level task memory: ring buffer of (goal, result) summaries
        # so subsequent tasks in the same session can recall what was done before.
        self._task_history: collections.deque[tuple[str, str]] = collections.deque(
            maxlen=_MAX_TASK_MEMORY,
        )
        # Persistent notes the model can save/retrieve during the session
        self._notes: dict[str, str] = {}

        # Register built-in memory tools so the model can save/recall notes
        self._register_memory_tools()

    def _register_memory_tools(self) -> None:
        """Add save_note and get_notes as callable tools for the model."""

        async def save_note(key: str, value: str) -> dict:
            """Save a persistent note for this session. Use this to remember
            important information like file paths, user preferences, or task
            progress that may be needed by future tasks."""
            self._notes[key] = value
            self._log.info("memory.save_note", key=key)
            return {"success": True, "key": key, "message": f"Note '{key}' saved."}

        async def get_notes(key: str = "") -> dict:
            """Retrieve saved notes. If key is provided, return that specific note.
            If key is empty, return all saved notes."""
            if key:
                val = self._notes.get(key)
                if val is None:
                    return {"success": False, "error": f"No note found for key '{key}'."}
                return {"success": True, "key": key, "value": val}
            return {"success": True, "notes": dict(self._notes)}

        for fn in (save_note, get_notes):
            self._tool_fns.append(fn)
            self._tool_map[fn.__name__] = fn

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

        Args:
            goal: Natural-language task description from the user's transcription.
            inject_context: Async callable that sends a SYSTEM message into the
                            live audio session so the model can speak the result.
        """
        self._log.info("orchestrator.task.start", goal=goal[:120])
        try:
            result_text = await self._agent_loop(goal)
            # Record in session memory so future tasks have context
            self._task_history.append((goal, result_text))
            completion_msg = (
                f"[SYSTEM: The autonomous task executor has completed the task.\n"
                f"Original goal: {goal}\n"
                f"What was done: {result_text}\n"
                f"Acknowledge completion in 1-2 sentences. Be concise and natural.]"
            )
            await inject_context(completion_msg)
            self._log.info("orchestrator.task.complete", goal=goal[:80])

        except asyncio.CancelledError:
            self._log.info("orchestrator.task.cancelled", goal=goal[:60])
            raise  # Re-raise so the Task is properly cancelled

        except Exception as exc:
            self._log.exception("orchestrator.task.error", goal=goal[:60])
            error_msg = (
                f"[SYSTEM: The autonomous task executor encountered an error.\n"
                f"Goal: {goal}\n"
                f"Error: {exc}\n"
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

    async def _agent_loop(self, goal: str) -> str:
        """ReAct loop: think → call tools → observe → repeat until done.

        Returns the final summary text from the orchestrator model.
        Raises on unrecoverable errors.
        """
        from google.genai import types as _types

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

        for iteration in range(_MAX_ITERATIONS):
            self._log.debug(
                "orchestrator.loop", iteration=iteration, goal=goal[:60],
            )

            # Call the orchestrator model with full tool list
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=history,
                config=_types.GenerateContentConfig(
                    system_instruction=_ORCHESTRATOR_SYSTEM_INSTRUCTION,
                    # generate_content auto-converts Python callables → schemas
                    tools=self._tool_fns,
                    temperature=0.2,
                    max_output_tokens=4096,
                    # Disable auto function calling — we execute tools ourselves
                    # so we can route them through the ToolBridge to the local client.
                    automatic_function_calling=_types.AutomaticFunctionCallingConfig(
                        disable=True,
                    ),
                ),
            )

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
                return " ".join(text_parts).strip() or "Task completed successfully."

            # --- Execute all function calls and collect responses ---
            fn_response_parts: list[_types.Part] = []

            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                fn = self._tool_map.get(tool_name)
                if fn is None:
                    self._log.warning(
                        "orchestrator.unknown_tool", name=tool_name,
                    )
                    exec_result: dict = {
                        "success": False,
                        "error": f"Unknown tool: '{tool_name}'. "
                                 f"Available: {sorted(self._tool_map.keys())}",
                    }
                else:
                    self._log.info(
                        "orchestrator.tool_call",
                        name=tool_name,
                        args={k: str(v)[:80] for k, v in tool_args.items()},
                    )
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
                    except Exception as exc:
                        self._log.exception(
                            "orchestrator.tool_error", name=tool_name,
                        )
                        exec_result = {"success": False, "error": str(exc)}

                    self._log.debug(
                        "orchestrator.tool_result",
                        name=tool_name,
                        success=exec_result.get("success"),
                    )

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

        return f"Task reached the {_MAX_ITERATIONS}-step safety limit."
