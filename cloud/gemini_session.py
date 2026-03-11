"""
Gemini session wrapper for Rio.

Dual-mode architecture:
  TEXT mode (L0): Uses standard generate_content API. Fallback when Live fails.
  LIVE mode (L1): Uses Live API (bidiGenerateContent) for bidirectional audio.
  L2: Vision support in both modes.
  L3: Function calling (tools) in both modes.
"""

from __future__ import annotations

import asyncio
import random
from typing import AsyncGenerator, Union

import structlog
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from websockets.exceptions import ConnectionClosed

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Rio system instruction
# ---------------------------------------------------------------------------

_ROLE_DESCRIPTIONS = {
    "assistant": (
        "You are {name}, the user's virtual friend and AI assistant. "
        "You are a LIVE AGENT — you see the user's screen in real-time, hear "
        "their voice, and can execute actions on their computer. You are warm, "
        "helpful, and proactive. You reference past sessions and adapt to the "
        "user's style."
    ),
    "developer": (
        "You are {name}, a proactive AI pair programmer. You are a LIVE AGENT — "
        "you see the developer's screen in real-time, hear their voice, and can "
        "execute code changes. Your key differentiator: you detect when the "
        "developer is struggling and offer help BEFORE they ask. You reference "
        "past sessions. You are not a chatbot — you are a screen-aware, "
        "voice-first, code-executing live agent."
    ),
    "tutor": (
        "You are {name}, an adaptive AI tutor. You are a LIVE AGENT — you see "
        "the student's screen in real-time, hear their voice, and guide them "
        "through learning. You are patient, encouraging, and adjust your "
        "teaching style to the student's level."
    ),
    "customer_care": (
        "You are {name}, an AI customer care agent. You are a LIVE AGENT — "
        "you see the agent's screen in real-time, hear their voice, and help "
        "them resolve customer issues efficiently. Follow the HEAR framework: "
        "Hear, Empathize, Act, Resolve."
    ),
}


def _load_agent_config() -> tuple[str, str]:
    """Load agent_name and agent_role from config.yaml (best-effort)."""
    import os
    from pathlib import Path as _P
    try:
        import yaml  # type: ignore
        for candidate in (
            _P(__file__).resolve().parent.parent / "config.yaml",
            _P(os.getcwd()) / "config.yaml",
        ):
            if candidate.is_file():
                data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                rio = data.get("rio", {})
                name = rio.get("agent_name", "Rio")
                role = rio.get("agent_role", "assistant")
                return name, role
    except Exception:
        pass
    return os.environ.get("RIO_AGENT_NAME", "Rio"), os.environ.get("RIO_AGENT_ROLE", "assistant")


def _build_role_intro(name: str, role: str) -> str:
    template = _ROLE_DESCRIPTIONS.get(role, _ROLE_DESCRIPTIONS["assistant"])
    return template.format(name=name)


RIO_BASE_INSTRUCTION = None  # built lazily


def _build_base_instruction() -> str:
    name, role = _load_agent_config()
    role_intro = _build_role_intro(name, role)
    return (
        f"{role_intro}\n\n"
        "Keep responses concise and conversational. When the user asks about code, "
        "be specific and actionable. You have access to tools that can read files, "
        "write files, patch files, and run shell commands on the user's machine. "
        "Use them when the user asks you to examine, edit, or run code.\n\n"
        "SCREEN CAPTURE: You are NOT always seeing the user's screen. Screen "
        "vision is on-demand by default to save API credits. When the user asks "
        "you to look at their screen, check their code visually, or says anything "
        "like 'capture my screen', 'look at my screen', 'what's on my screen', "
        "'check my screen' — call the capture_screen tool to take a screenshot. "
        "The screenshot will be sent to your vision context. If the user has "
        "enabled autonomous mode, you will receive periodic screen frames "
        "automatically and do NOT need to call capture_screen.\n\n"
        "SCREEN NAVIGATION: You can interact with the user's screen using these "
        "tools. Screenshots are captured at 75% resolution (approximately 1440x810 "
        "pixels for a 1920x1080 display).\n"
        "Available screen actions:\n"
        "- smart_click(target, action) — PREFERRED: click a UI element by describing it "
        "in natural language (e.g. 'the Save button', 'search input field'). Uses Computer "
        "Vision to visually locate the element. ALWAYS use this instead of screen_click.\n"
        "- screen_type(text) — type text at the current cursor position\n"
        "- screen_scroll(x, y, clicks) — scroll at position (+up / -down)\n"
        "- screen_hotkey(keys) — press keyboard shortcut (e.g. 'ctrl+s', 'alt+tab')\n"
        "- screen_move(x, y) — move mouse without clicking (hover)\n"
        "- screen_drag(start_x, start_y, end_x, end_y) — drag from A to B\n"
        "- find_window(title) — search for open windows by title\n"
        "- focus_window(title) — bring a window to the foreground\n"
        "- screen_click(x, y) — FALLBACK ONLY: raw coordinate click. Only use when "
        "smart_click cannot identify the target.\n\n"
        "WINDOWS POWER TOOLS — Full desktop control:\n"
        "- open_application(name_or_path) — launch any app by name (notepad, chrome, "
        "explorer, calc, code, etc.) or path. Also opens files and URLs.\n"
        "- list_all_windows() — list all visible windows with titles, positions, sizes\n"
        "- get_active_window() — info about the current foreground window\n"
        "- minimize_window(title) / maximize_window(title) / close_window(title) — "
        "window state management by title substring\n"
        "- resize_window(title, width, height) — resize a window\n"
        "- move_window(title, x, y) — reposition a window on screen\n"
        "- list_processes(name_filter) — list running processes by memory\n"
        "- kill_process(name_or_pid) — terminate a process (protected processes blocked)\n"
        "- get_clipboard() / set_clipboard(text) — read/write clipboard\n"
        "- get_screen_info() — monitor resolution, DPI, multi-monitor layout\n\n"
        "USE open_application INSTEAD of run_command for launching apps. It handles "
        "name resolution, aliases, and Windows protocols automatically.\n\n"
        "WORKFLOW for clicking UI elements:\n"
        "1. ALWAYS use smart_click(target='description of element') — it takes a fresh "
        "screenshot and uses Computer Vision to find and click the exact element\n"
        "2. Only fall back to screen_click if smart_click explicitly fails\n"
        "3. After EVERY screen action, an auto-captured screenshot is automatically sent "
        "to your vision context. STOP and WAIT to see the result before calling another tool.\n"
        "4. If the click missed, try smart_click again with a different description\n"
        "IMPORTANT: smart_click is far more reliable than guessing pixel coordinates.\n"
        "IMPORTANT: NEVER batch multiple screen actions in one response. Execute ONE "
        "screen action, then WAIT for the screenshot to verify it worked before "
        "deciding the next action.\n\n"
        "AUTONOMOUS TASK EXECUTION (COMPUTER-USE AGENT):\n"
        "When the user gives you a task that involves interacting with their computer, "
        "you MUST execute it autonomously to completion. Do NOT stop after one action "
        "and ask the user what to do next. Complete the ENTIRE task yourself.\n\n"
        "THE LOOP — follow this for EVERY computer task:\n"
        "1. PLAN: Silently break the task into steps.\n"
        "2. CAPTURE: If you don't have a recent screenshot, call capture_screen first.\n"
        "3. ANALYZE: Look at the screenshot. Identify what needs to be clicked/typed/scrolled.\n"
        "4. ACT: Call ONE screen action tool. Only ONE per turn.\n"
        "5. VERIFY: After each action, an auto-captured screenshot is sent to your vision "
        "context automatically. WAIT for it. Look at it to confirm the action worked.\n"
        "6. CONTINUE: If the task isn't complete, go back to step 3. Keep going until done.\n"
        "7. REPORT: When the task is fully complete, tell the user what you accomplished.\n\n"
        "CRITICAL RULES for autonomous execution:\n"
        "- ONE REQUEST = FULL COMPLETION. Never stop mid-task to ask for confirmation.\n"
        "- ONE ACTION PER TURN. Call one screen tool, then wait to see the screenshot "
        "result before deciding the next action. Never batch screen actions.\n"
        "- After each screen action, you receive an auto-captured screenshot. USE IT to "
        "verify your action and plan the next one.\n"
        "- If an action fails (element not found, wrong location), TRY AGAIN with adjusted "
        "coordinates or an alternative approach.\n"
        "- Wait for UI transitions: if the UI hasn't loaded yet after an action, call "
        "capture_screen again after a moment.\n"
        "- For typing: use screen_type for text input, screen_hotkey for keyboard shortcuts.\n"
        "- Give brief voice status updates for long tasks: 'Opening Chrome...', "
        "'Typing the search query...', 'Almost done...' — but don't narrate every micro-step.\n"
        "- Maximum 25 actions per task. If you can't finish in 25 actions, tell the user "
        "what you've done so far and what's remaining.\n\n"
        "RESPONSE VARIATION:\n"
        "Keep your responses natural and varied. You're a coworker, not a script:\n"
        "- Use different sentence structures\n"
        "- Match the user's energy (casual vs focused)\n"
        "- Don't start every response with 'Sure' or 'Alright'\n"
        "- When you see the user's screen, describe what you observe briefly, then help\n"
        "- If you notice errors or issues proactively, mention them\n"
    )


def build_system_instruction() -> str:
    """Build the full system instruction from base + loaded profiles.

    Deterministic: same profile files = same instruction every time.
    Loads profiles from rio_profiles/ directory (adjacent to rio/ project root).
    """
    import sys
    from pathlib import Path as _Path

    parts = [_build_base_instruction()]

    # Attempt to load profiles — gracefully degrade if profiles module not available
    try:
        _local_dir = str(_Path(__file__).resolve().parent.parent / "local")
        if _local_dir not in sys.path:
            sys.path.insert(0, _local_dir)
        from profiles import (
            build_customer_care_instruction,
            build_tutor_instruction,
            load_customer_care_profile,
            load_tutor_profile,
        )

        _profiles_base = str(_Path(__file__).resolve().parent.parent / "rio_profiles")

        cc_profile = load_customer_care_profile(_profiles_base)
        if cc_profile.enabled and cc_profile.business.business_name:
            parts.append("")
            parts.append(build_customer_care_instruction(cc_profile))

        tutor_profile = load_tutor_profile(_profiles_base)
        if tutor_profile.enabled and tutor_profile.student.student_name:
            parts.append("")
            parts.append(build_tutor_instruction(tutor_profile))

    except ImportError:
        # profiles.py not available — use fallback generic instructions
        parts.append(
            "\n\nCUSTOMER CARE MODE: When the user is handling customer support, you can "
            "create and track support tickets using create_ticket and update_ticket. Follow the HEAR "
            "framework: Hear (listen fully), Empathize (validate), Act (resolve), "
            "Resolve (confirm). Use empathetic language. Never blame the customer. "
            "Detect frustration through voice/text signals and escalate when needed. "
            "Reference past interactions via memory."
        )
        parts.append(
            "\n\nTUTOR MODE: When helping a student learn, use the Socratic method — "
            "guide with questions instead of giving direct answers. Use generate_quiz "
            "to create practice problems, track_progress to monitor learning, and "
            "explain_concept for structured explanations. Assess the student's level "
            "(novice/intermediate/advanced) and adapt accordingly. Never do homework "
            "for the student — help them understand so they can do it themselves. "
            "Celebrate progress and use growth mindset language."
        )

    return "\n".join(parts)


# Module-level fallback — safe constant, no filesystem access at import time.
# The real instruction is built fresh per-session inside GeminiSession.connect()
# by calling build_system_instruction() which loads profile files.
RIO_SYSTEM_INSTRUCTION = RIO_BASE_INSTRUCTION

# Model identifiers — defaults; can be overridden via constructor args
DEFAULT_TEXT_MODEL = "gemini-2.5-flash"  # Standard API for L0 text mode
DEFAULT_LIVE_MODEL = "gemini-2.5-flash-native-audio-latest"  # Live API for L1+ audio

# Maximum consecutive stream errors before giving up in receive_live()
MAX_STREAM_RETRIES = 5

# ---------------------------------------------------------------------------
# Tool declarations — exposed to Gemini via function calling
# ---------------------------------------------------------------------------
RIO_TOOL_DECLARATIONS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="read_file",
                description=(
                    "Read the contents of a file on the user's machine. "
                    "Use this to examine code, config files, logs, or any text file."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "path": {
                            "type": "STRING",
                            "description": "Absolute or relative path to the file to read",
                        },
                    },
                    "required": ["path"],
                },
            ),
            types.FunctionDeclaration(
                name="write_file",
                description=(
                    "Write content to a file on the user's machine. "
                    "A .rio.bak backup is created automatically before overwriting. "
                    "Use this to create new files or completely rewrite existing ones."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "path": {
                            "type": "STRING",
                            "description": "Absolute or relative path to the file to write",
                        },
                        "content": {
                            "type": "STRING",
                            "description": "The full content to write to the file",
                        },
                    },
                    "required": ["path", "content"],
                },
            ),
            types.FunctionDeclaration(
                name="patch_file",
                description=(
                    "Apply a find-and-replace edit to a file. Replaces the first "
                    "occurrence of old_text with new_text. A .rio.bak backup is "
                    "created before patching. old_text must match exactly."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "path": {
                            "type": "STRING",
                            "description": "Path to the file to patch",
                        },
                        "old_text": {
                            "type": "STRING",
                            "description": "The exact text to find (must match exactly)",
                        },
                        "new_text": {
                            "type": "STRING",
                            "description": "The replacement text",
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            ),
            types.FunctionDeclaration(
                name="run_command",
                description=(
                    "Execute a shell command on the user's machine. "
                    "Has a 30-second timeout. Dangerous commands are blocked. "
                    "Use this for running tests, builds, git operations, etc."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "command": {
                            "type": "STRING",
                            "description": "The shell command to execute",
                        },
                    },
                    "required": ["command"],
                },
            ),
            types.FunctionDeclaration(
                name="capture_screen",
                description=(
                    "Capture a screenshot of the user's screen right now and "
                    "send it to your vision context. Use this when the user "
                    "asks you to look at their screen, check something visual, "
                    "or when you need to see what they are currently working on. "
                    "The screenshot will appear as an inline image in your next "
                    "context window."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {},
                    "required": [],
                },
            ),
            # --- Customer Care Tools ---
            types.FunctionDeclaration(
                name="create_ticket",
                description=(
                    "Create a customer support ticket to track an issue. "
                    "Use when a customer reports a problem, requests help, or "
                    "needs follow-up. The ticket is saved locally for tracking."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "title": {
                            "type": "STRING",
                            "description": "Brief ticket title summarizing the issue",
                        },
                        "category": {
                            "type": "STRING",
                            "description": "Issue category: bug, feature, how-to, billing, or account",
                        },
                        "priority": {
                            "type": "STRING",
                            "description": "Priority level: low, medium, high, or critical",
                        },
                        "description": {
                            "type": "STRING",
                            "description": "Full description with context, steps to reproduce, and current state",
                        },
                        "customer_id": {
                            "type": "STRING",
                            "description": "Optional customer identifier for tracking",
                        },
                        "tags": {
                            "type": "STRING",
                            "description": "Comma-separated tags for categorization",
                        },
                    },
                    "required": ["title", "category", "priority", "description"],
                },
            ),
            types.FunctionDeclaration(
                name="update_ticket",
                description=(
                    "Update an existing support ticket: change status, priority, "
                    "escalation tier, or add notes. Use to escalate issues, close "
                    "resolved tickets, or log follow-up actions."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "ticket_id": {
                            "type": "STRING",
                            "description": "The ticket ID to update (e.g., 'TKT-abc123')",
                        },
                        "status": {
                            "type": "STRING",
                            "description": "New status: open, in-progress, escalated, resolved, closed",
                        },
                        "priority": {
                            "type": "STRING",
                            "description": "New priority: low, medium, high, critical",
                        },
                        "escalation_tier": {
                            "type": "STRING",
                            "description": "Escalation tier: tier_0, tier_1, tier_2, tier_3",
                        },
                        "notes": {
                            "type": "STRING",
                            "description": "Notes about the update (reason for escalation, resolution details, etc.)",
                        },
                    },
                    "required": ["ticket_id"],
                },
            ),
            # --- Tutor Tools ---
            types.FunctionDeclaration(
                name="generate_quiz",
                description=(
                    "Generate a quiz or practice problems for a student on a "
                    "specific topic. Supports math, programming, and science. "
                    "Use when a student needs practice, wants to test their "
                    "understanding, or after explaining a concept to reinforce "
                    "learning."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "topic": {
                            "type": "STRING",
                            "description": "The subject topic (e.g., 'quadratic equations', 'python loops', 'cell biology')",
                        },
                        "difficulty": {
                            "type": "STRING",
                            "description": "Difficulty level: novice, intermediate, or advanced",
                        },
                        "num_questions": {
                            "type": "INTEGER",
                            "description": "Number of questions to generate (default: 5)",
                        },
                        "question_types": {
                            "type": "STRING",
                            "description": "Comma-separated types: multiple-choice, short-answer, problem-solving, true-false",
                        },
                        "focus_areas": {
                            "type": "STRING",
                            "description": "Comma-separated subtopics to focus on (from student's weak areas)",
                        },
                    },
                    "required": ["topic"],
                },
            ),
            types.FunctionDeclaration(
                name="track_progress",
                description=(
                    "Track or retrieve a student's learning progress. Use to "
                    "record quiz scores, note mastered/struggling topics, "
                    "generate progress reports, or update study plans."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "action": {
                            "type": "STRING",
                            "description": "Action: record (save progress), report (generate report), update (modify plan), or status (current state)",
                        },
                        "subject": {
                            "type": "STRING",
                            "description": "Subject area (e.g., 'math', 'programming', 'science')",
                        },
                        "topic": {
                            "type": "STRING",
                            "description": "Specific topic within the subject",
                        },
                        "score": {
                            "type": "STRING",
                            "description": "Quiz score or performance metric (e.g., '4/5', '80%')",
                        },
                        "notes": {
                            "type": "STRING",
                            "description": "Additional context (struggles, breakthroughs, patterns noticed)",
                        },
                        "student_id": {
                            "type": "STRING",
                            "description": "Optional student identifier for multi-student tracking",
                        },
                    },
                    "required": ["action", "subject"],
                },
            ),
            types.FunctionDeclaration(
                name="explain_concept",
                description=(
                    "Retrieve a structured explanation of a concept at the "
                    "student's level. Includes analogies, examples, and "
                    "practice suggestions. Use when a student asks 'what is' "
                    "or 'explain' or when they're struggling with fundamentals."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "concept": {
                            "type": "STRING",
                            "description": "The concept to explain (e.g., 'recursion', 'photosynthesis', 'derivatives')",
                        },
                        "level": {
                            "type": "STRING",
                            "description": "Student level: novice, intermediate, or advanced",
                        },
                        "context": {
                            "type": "STRING",
                            "description": "What the student is working on that prompted this explanation",
                        },
                    },
                    "required": ["concept"],
                },
            ),
            # --- Screen Navigation Tools ---
            types.FunctionDeclaration(
                name="screen_click",
                description=(
                    "Click at a position on the user's screen. Coordinates are in "
                    "screenshot space (the resized image you see). The system "
                    "automatically maps them to real screen coordinates."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "x": {
                            "type": "INTEGER",
                            "description": "X coordinate in the screenshot image",
                        },
                        "y": {
                            "type": "INTEGER",
                            "description": "Y coordinate in the screenshot image",
                        },
                        "button": {
                            "type": "STRING",
                            "description": "Mouse button: left, right, or middle (default: left)",
                        },
                        "clicks": {
                            "type": "INTEGER",
                            "description": "Number of clicks: 1=single, 2=double (default: 1)",
                        },
                    },
                    "required": ["x", "y"],
                },
            ),
            types.FunctionDeclaration(
                name="screen_type",
                description=(
                    "Type text at the current cursor position on the user's screen. "
                    "Use this after clicking on a text field. For keyboard shortcuts "
                    "like Ctrl+S, use screen_hotkey instead."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "text": {
                            "type": "STRING",
                            "description": "The text to type",
                        },
                        "interval": {
                            "type": "NUMBER",
                            "description": "Delay between keystrokes in seconds (default: 0.02)",
                        },
                    },
                    "required": ["text"],
                },
            ),
            types.FunctionDeclaration(
                name="screen_scroll",
                description=(
                    "Scroll at a position on the user's screen. Positive clicks "
                    "scroll up, negative scroll down. Position the cursor first."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "x": {
                            "type": "INTEGER",
                            "description": "X coordinate in the screenshot image",
                        },
                        "y": {
                            "type": "INTEGER",
                            "description": "Y coordinate in the screenshot image",
                        },
                        "clicks": {
                            "type": "INTEGER",
                            "description": "Scroll amount: positive=up, negative=down (e.g. 3 or -5)",
                        },
                    },
                    "required": ["x", "y", "clicks"],
                },
            ),
            types.FunctionDeclaration(
                name="screen_hotkey",
                description=(
                    "Press a keyboard shortcut on the user's machine. "
                    "Keys are specified as a '+'-separated string. "
                    "Examples: 'ctrl+s', 'alt+tab', 'ctrl+shift+t', 'enter', 'escape'."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "keys": {
                            "type": "STRING",
                            "description": "Key combination, e.g. 'ctrl+s', 'alt+f4', 'enter'",
                        },
                    },
                    "required": ["keys"],
                },
            ),
            types.FunctionDeclaration(
                name="screen_move",
                description=(
                    "Move the mouse cursor to a position on the screen without "
                    "clicking. Useful for hovering over elements to reveal tooltips."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "x": {
                            "type": "INTEGER",
                            "description": "X coordinate in the screenshot image",
                        },
                        "y": {
                            "type": "INTEGER",
                            "description": "Y coordinate in the screenshot image",
                        },
                    },
                    "required": ["x", "y"],
                },
            ),
            types.FunctionDeclaration(
                name="screen_drag",
                description=(
                    "Click-and-drag from one position to another on the screen. "
                    "Useful for selecting text, moving elements, or drawing."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "start_x": {
                            "type": "INTEGER",
                            "description": "Start X coordinate in the screenshot image",
                        },
                        "start_y": {
                            "type": "INTEGER",
                            "description": "Start Y coordinate in the screenshot image",
                        },
                        "end_x": {
                            "type": "INTEGER",
                            "description": "End X coordinate in the screenshot image",
                        },
                        "end_y": {
                            "type": "INTEGER",
                            "description": "End Y coordinate in the screenshot image",
                        },
                        "duration": {
                            "type": "NUMBER",
                            "description": "Duration of drag in seconds (default: 0.5)",
                        },
                    },
                    "required": ["start_x", "start_y", "end_x", "end_y"],
                },
            ),
            types.FunctionDeclaration(
                name="find_window",
                description=(
                    "Search for open windows by title. Returns a list of matching "
                    "windows with their position and size."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "title": {
                            "type": "STRING",
                            "description": "Substring to search for in window titles",
                        },
                    },
                    "required": ["title"],
                },
            ),
            types.FunctionDeclaration(
                name="focus_window",
                description=(
                    "Bring a window to the foreground by its title. "
                    "Restores minimized windows automatically."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "title": {
                            "type": "STRING",
                            "description": "Substring to search for in window titles",
                        },
                    },
                    "required": ["title"],
                },
            ),
        ]
    ),
]


class GeminiSession:
    """Manages a Gemini session for one connected client.

    L0: Uses standard generate_content (text-only, stateful via history).
    L1+: Will use Live API (bidiGenerateContent) for audio/vision streaming.
    L2: Vision support -- screenshots are pending context for next message.
    L3: Tool execution -- Gemini can call read_file, write_file, patch_file,
        run_command. receive() yields tool_call dicts alongside text chunks.
    """

    # Type alias for items yielded by receive()
    # str = text chunk, dict = tool_call descriptor
    ReceiveItem = Union[str, dict]

    def __init__(self, api_key: str, client_id: str, mode: str = "live",
                 text_model: str | None = None, live_model: str | None = None) -> None:
        self._api_key = api_key
        self._client_id = client_id
        self._client: genai.Client | None = None
        self._connected = False
        self._log = logger.bind(client_id=client_id)
        # Model selection — use provided values or fall back to defaults
        self._text_model = text_model or DEFAULT_TEXT_MODEL
        self._live_model = live_model or DEFAULT_LIVE_MODEL
        # Conversation history for stateful text mode (L0)
        self._history: list[types.Content] = []
        # L2: pending screenshot for next interaction
        self._pending_image: tuple[bytes, str] | None = None
        # L3: whether tool calling is enabled
        self._tools_enabled = True
        # L1: Session mode — "live" (bidirectional audio) or "text" (fallback)
        self._requested_mode = mode
        self._mode = mode  # May change to "text" if Live API connect fails
        self._live_session = None  # Live API session object
        self._live_ctx = None      # Async context manager for Live session
        self._live_connect_error: str | None = None  # Error detail when Live API fails
        # System instruction — rebuilt fresh per session from profiles
        self._system_instruction: str = ""
        # Lock to prevent concurrent reconnect calls from sender+receiver racing
        self._reconnect_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialize the Gemini client and (optionally) a Live API session.

        When mode='live', attempts to open a bidirectional audio session
        with the Live API.  Falls back to text mode on any failure so the
        agent remains functional (voice-less but not dead).
        """
        self._log.info("gemini.connect.start", requested_mode=self._requested_mode)
        try:
            self._client = genai.Client(api_key=self._api_key)
            self._history = []
            self._pending_image = None

            # Build system instruction fresh from current profile files
            self._system_instruction = build_system_instruction()
            self._log.info(
                "gemini.connect.instruction_built",
                length=len(self._system_instruction),
            )

            if self._mode == "live":
                try:
                    await self._connect_live()
                    self._connected = True
                    self._log.info(
                        "gemini.connect.ok", mode="live", model=self._live_model,
                    )
                    return
                except Exception as exc:
                    self._live_connect_error = str(exc)
                    self._log.exception(
                        "gemini.connect.live_failed",
                        note="Falling back to text mode",
                        error=self._live_connect_error,
                    )
                    self._mode = "text"
                    self._live_session = None
                    self._live_ctx = None

            # Text mode (L0 fallback)
            self._connected = True
            self._log.info("gemini.connect.ok", mode="text", model=self._text_model)
        except Exception:
            self._connected = False
            self._log.exception("gemini.connect.failed")
            raise

    async def close(self) -> None:
        """Gracefully shut down the session (text or live)."""
        self._log.info("gemini.close", mode=self._mode)
        self._connected = False

        # Close Live API session if active
        if self._live_ctx is not None:
            try:
                await self._live_ctx.__aexit__(None, None, None)
            except Exception:
                self._log.debug("gemini.close.live_ctx_error")
            self._live_ctx = None
            self._live_session = None
        elif self._live_session is not None:
            try:
                await self._live_session.close()
            except Exception:
                self._log.debug("gemini.close.live_session_error")
            self._live_session = None

        self._history = []
        self._pending_image = None
        self._client = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def has_pending_user_message(self) -> bool:
        """True if the last history entry is from the user (needs response)."""
        return bool(self._history) and self._history[-1].role == "user"

    @property
    def mode(self) -> str:
        """Current session mode: ``'live'`` or ``'text'``."""
        return self._mode

    # ------------------------------------------------------------------
    # Send + Receive (dual-mode: text L0 + live L1)
    # ------------------------------------------------------------------

    async def send_text(self, text: str) -> None:
        """Send a text message to Gemini.

        Live mode:  Sends directly into the Live API session with
                    end_of_turn=True so Gemini responds immediately.
        Text mode:  Stores in history for the next receive() call.
        """
        if not self._connected or self._client is None:
            self._log.warning("gemini.send_text.not_connected")
            return
        self._log.debug("gemini.send_text", text_length=len(text), mode=self._mode)

        # ---- Live mode: inject text into the bidirectional stream ----
        if self._mode == "live" and self._live_session is not None:
            try:
                await self._live_session.send(input=text, end_of_turn=True)
            except Exception:
                self._log.exception("gemini.send_text.live_error")
            return

        # ---- Text mode (L0): append to history ----
        parts: list[types.Part] = []

        # Attach pending screenshot if available
        if self._pending_image is not None:
            img_data, img_mime = self._pending_image
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type=img_mime, data=img_data,
                    )
                )
            )
            self._pending_image = None
            self._log.debug("gemini.send_text.attached_image", mime=img_mime)

        parts.append(types.Part(text=text))
        self._history.append(types.Content(role="user", parts=parts))

    async def send_context(self, context_text: str) -> None:
        """Inject proactive context into the Gemini session.

        Used by the struggle detector (L4) to prompt Gemini to offer
        help without the user explicitly asking.

        Live mode:  Sends text into the Live session with end_of_turn=True
                    so Gemini responds proactively with voice.
        Text mode:  Appends to history as a user message.  The relay task
                    will call receive() and get Gemini's proactive response.
        """
        if not self._connected or self._client is None:
            self._log.warning("gemini.send_context.not_connected")
            return

        self._log.info(
            "gemini.send_context",
            text_length=len(context_text),
            mode=self._mode,
        )

        # ---- Live mode: inject into the bidirectional stream ----
        if self._mode == "live" and self._live_session is not None:
            try:
                await self._live_session.send(
                    input=context_text, end_of_turn=True,
                )
            except Exception:
                self._log.exception("gemini.send_context.live_error")
            return

        # ---- Text mode (L0): append to history ----
        parts = [types.Part(text=context_text)]
        self._history.append(types.Content(role="user", parts=parts))

    async def receive(self) -> AsyncGenerator[ReceiveItem, None]:
        """Async generator yielding text chunks or tool_call dicts.

        L0/L3: Calls generate_content with full history + tool declarations.

        Yields:
          - ``str``: text response chunk (stream to client)
          - ``dict``: ``{"type": "tool_call", "name": ..., "args": ...}``
            When a tool_call is yielded, the caller should:
            1. Execute the tool locally
            2. Call ``send_tool_result(name, result)``
            3. Call ``receive()`` again to get Gemini's follow-up response
        """
        if not self._connected or self._client is None:
            self._log.warning("gemini.receive.not_connected")
            return

        if not self._history:
            self._log.debug("gemini.receive.no_pending_message")
            return

        # Only call API if the last message is from the user (text or tool result)
        if self._history[-1].role != "user":
            self._log.debug("gemini.receive.no_user_turn")
            return

        # Build config with tool declarations
        tool_config = RIO_TOOL_DECLARATIONS if self._tools_enabled else None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._text_model,
                    contents=self._history,
                    config=types.GenerateContentConfig(
                        system_instruction=self._system_instruction or RIO_SYSTEM_INSTRUCTION,
                        tools=tool_config,
                        temperature=0.7,
                        max_output_tokens=2048,
                    ),
                )

                if not response.candidates:
                    self._log.warning("gemini.receive.no_candidates")
                    return

                model_content = response.candidates[0].content

                # Check if the response contains function calls
                function_calls = []
                text_parts = []
                for part in model_content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_calls.append(part.function_call)
                    elif part.text:
                        text_parts.append(part.text)

                if function_calls:
                    # Model wants to call tools — add its response to history
                    self._history.append(model_content)

                    self._log.info(
                        "gemini.receive.tool_calls",
                        count=len(function_calls),
                        names=[fc.name for fc in function_calls],
                    )

                    # Yield each function call for the relay to dispatch
                    for fc in function_calls:
                        yield {
                            "type": "tool_call",
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                            "id": getattr(fc, "id", None),
                        }
                else:
                    # Regular text response
                    full_text = "".join(text_parts)

                    if full_text:
                        self._history.append(model_content)
                        self._log.debug(
                            "gemini.receive.text",
                            response_length=len(full_text),
                        )
                        yield full_text

                # Keep history bounded -- but never trim in the middle of
                # a tool-call chain (model function_call followed by user
                # function_response).  Find a safe trim point.
                if len(self._history) > 40:
                    trim_to = 20
                    # Walk backwards from trim_to to find a safe cut point
                    # (not between a tool call and its result)
                    while trim_to > 0:
                        entry = self._history[-(trim_to)]
                        # Check if this entry is a function_response (tool result)
                        is_tool_result = False
                        if entry.role == "user":
                            for p in (entry.parts or []):
                                if hasattr(p, "function_response") and p.function_response:
                                    is_tool_result = True
                                    break
                        if is_tool_result:
                            # Don't cut here -- the preceding model tool_call
                            # would be orphaned.  Include one more entry.
                            trim_to += 1
                        else:
                            break
                    trim_to = min(trim_to, len(self._history))
                    self._history = self._history[-trim_to:]

                self._log.debug("gemini.receive.turn_complete")
                return  # Success — exit retry loop

            except ClientError as exc:
                if exc.code == 429 and attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0.5, 2.0)
                    self._log.warning(
                        "gemini.receive.rate_limited",
                        attempt=attempt + 1,
                        wait_seconds=round(wait, 1),
                    )
                    await asyncio.sleep(wait)
                    continue
                self._log.exception("gemini.receive.api_error")
                raise
            except asyncio.CancelledError:
                self._log.info("gemini.receive.cancelled")
                raise
            except Exception:
                self._log.exception("gemini.receive.error")
                raise

    # ------------------------------------------------------------------
    # L1: Live API audio methods
    # ------------------------------------------------------------------

    async def _connect_live(self) -> None:
        """Open a persistent Live API session for bidirectional streaming."""
        self._log.info(
            "gemini.live.connecting",
            model=self._live_model,
            tools_enabled=self._tools_enabled,
        )
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Puck"
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=self._system_instruction or RIO_SYSTEM_INSTRUCTION)]
            ),
            tools=RIO_TOOL_DECLARATIONS if self._tools_enabled else None,
        )
        self._live_ctx = self._client.aio.live.connect(
            model=self._live_model, config=config,
        )
        self._live_session = await self._live_ctx.__aenter__()
        self._log.info(
            "gemini.live.session_opened",
            model=self._live_model,
            tools_enabled=self._tools_enabled,
        )

    async def _reconnect_live(self) -> None:
        """Tear down the dead Live session and open a fresh one.

        Called when a send or receive detects ConnectionClosed.  Uses a
        lock so that concurrent callers (sender task + receiver task)
        don't double-reconnect: the second caller returns immediately
        when it sees a live session was already restored.

        Nulling _live_session BEFORE acquiring the lock is deliberate:
        asyncio is single-threaded so there is no await between the
        assignment and the lock acquisition.  This way the guard inside
        the lock correctly distinguishes "dead on entry" from "already
        reconnected by a concurrent caller".
        """
        # Mark the dead session as gone before anyone else can observe it
        self._live_session = None

        async with self._reconnect_lock:
            # Another concurrent caller beat us here and already reconnected
            if self._live_session is not None:
                self._log.debug("gemini.live.reconnect_skipped_already_alive")
                return

            self._log.warning("gemini.live.reconnecting")

            # Clean up the stale context if present
            old_ctx = self._live_ctx
            self._live_ctx = None
            if old_ctx is not None:
                try:
                    await old_ctx.__aexit__(None, None, None)
                except Exception:
                    pass

            # Attempt to open a new session
            try:
                await self._connect_live()
                self._log.info("gemini.live.reconnected")
            except Exception:
                self._log.exception("gemini.live.reconnect_failed")
                self._live_session = None
                self._live_ctx = None

    async def send_audio(self, data: bytes) -> None:
        """Send PCM audio bytes to Gemini Live API.

        In live mode, streams audio into the bidirectional session.
        In text mode, audio is silently dropped (no Live session).
        """
        if self._mode != "live" or self._live_session is None:
            return  # Nothing to do in text mode
        try:
            await self._live_session.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(
                            data=data,
                            mime_type="audio/pcm;rate=16000",
                        )
                    ]
                )
            )
        except ConnectionClosed as exc:
            self._log.warning("gemini.send_audio.session_closed", code=exc.code, reason=str(exc.reason)[:100])
            await self._reconnect_live()
        except Exception:
            self._log.exception("gemini.send_audio.error", bytes_len=len(data))

    async def send_end_of_turn(self) -> None:
        """Signal end-of-speech to the Live API session.

        Called when the user releases the push-to-talk key, telling
        Gemini that the user has finished speaking and it should respond.
        """
        if self._mode != "live" or self._live_session is None:
            return
        try:
            await self._live_session.send(input=".", end_of_turn=True)
            self._log.debug("gemini.send_end_of_turn.ok")
        except ConnectionClosed as exc:
            self._log.warning("gemini.send_end_of_turn.session_closed", code=exc.code)
            await self._reconnect_live()
        except Exception:
            self._log.exception("gemini.send_end_of_turn.error")

    async def receive_live(self) -> AsyncGenerator[dict, None]:
        """Persistent async generator for Live API responses.

        Runs continuously while the session is connected, yielding
        parsed response items as dicts:

          ``{"type": "audio", "data": <bytes>}``
          ``{"type": "text",  "text": <str>}``
          ``{"type": "tool_call", "name": <str>, "args": <dict>}``
          ``{"type": "turn_complete"}``
          ``{"type": "setup_complete"}``

        The caller (relay task in main.py) should run this in a long-lived
        asyncio task and dispatch items to the local WebSocket client.
        """
        if self._mode != "live" or self._live_session is None:
            self._log.warning("gemini.receive_live.not_live_mode")
            return

        self._log.info("gemini.receive_live.start")

        consecutive_errors = 0

        try:
            while self._connected and self._live_session is not None:
                try:
                    async for response in self._live_session.receive():
                        # Reset error counter on successful receive
                        consecutive_errors = 0

                        # Track whether we yielded anything from this
                        # response so we don't double-emit via the
                        # shorthand property accessors (response.data /
                        # response.text are aliases for server_content
                        # fields in the Gemini SDK).
                        _yielded = False

                        # -- Server content (model turn / turn complete) --
                        server_content = getattr(response, "server_content", None)
                        if server_content is not None:
                            if getattr(server_content, "turn_complete", False):
                                yield {"type": "turn_complete"}
                                continue

                            model_turn = getattr(server_content, "model_turn", None)
                            if model_turn is not None:
                                parts = getattr(model_turn, "parts", None) or []
                                for part in parts:
                                    # Audio data
                                    inline = getattr(part, "inline_data", None)
                                    if inline is not None:
                                        raw = getattr(inline, "data", None)
                                        if raw:
                                            yield {"type": "audio", "data": raw}
                                            _yielded = True

                                    # Text data
                                    txt = getattr(part, "text", None)
                                    if txt:
                                        yield {"type": "text", "text": txt}
                                        _yielded = True

                                    # Function call within model turn
                                    fc = getattr(part, "function_call", None)
                                    if fc is not None:
                                        yield {
                                            "type": "tool_call",
                                            "name": fc.name,
                                            "args": dict(fc.args) if fc.args else {},
                                            "id": getattr(fc, "id", None),
                                        }
                                        _yielded = True

                            if _yielded:
                                continue  # already processed server_content

                        # -- Shorthand: response.data / response.text ------
                        # Only use these if server_content didn't yield
                        # anything, to avoid duplicate audio/text.
                        if not _yielded:
                            raw_data = getattr(response, "data", None)
                            if raw_data and isinstance(raw_data, (bytes, bytearray)):
                                yield {"type": "audio", "data": bytes(raw_data)}
                                _yielded = True

                            raw_text = getattr(response, "text", None)
                            if raw_text and isinstance(raw_text, str):
                                yield {"type": "text", "text": raw_text}
                                _yielded = True

                        # -- Tool call at response level -------------------
                        tool_call = getattr(response, "tool_call", None)
                        if tool_call is not None:
                            for fc in getattr(tool_call, "function_calls", None) or []:
                                yield {
                                    "type": "tool_call",
                                    "name": fc.name,
                                    "args": dict(fc.args) if fc.args else {},
                                    "id": getattr(fc, "id", None),
                                }

                        # -- Setup complete --------------------------------
                        if getattr(response, "setup_complete", None) is not None:
                            yield {"type": "setup_complete"}

                except asyncio.CancelledError:
                    raise
                except ConnectionClosed as exc:
                    self._log.warning(
                        "gemini.receive_live.session_closed",
                        code=exc.code,
                        reason=str(exc.reason)[:100],
                    )
                    await self._reconnect_live()
                    if self._live_session is not None:
                        continue  # Retry with new session
                    break
                except Exception:
                    consecutive_errors += 1
                    self._log.exception(
                        "gemini.receive_live.stream_error",
                        consecutive_errors=consecutive_errors,
                        max_retries=MAX_STREAM_RETRIES,
                    )
                    if consecutive_errors >= MAX_STREAM_RETRIES:
                        self._log.error(
                            "gemini.receive_live.max_retries_exceeded",
                            consecutive_errors=consecutive_errors,
                        )
                        break
                    if self._connected:
                        await asyncio.sleep(1.0)
                        continue
                    break

        except asyncio.CancelledError:
            self._log.info("gemini.receive_live.cancelled")

        self._log.info("gemini.receive_live.ended")

    # ------------------------------------------------------------------
    # L2: Vision (screenshots)
    # ------------------------------------------------------------------

    async def send_image(
        self,
        data: bytes,
        mime_type: str = "image/jpeg",
        prompt: str | None = None,
    ) -> None:
        """Send a screenshot image to Gemini.

        Live mode:  Injects the image inline into the bidirectional
                    stream.  Gemini sees it as part of the ongoing session.
        Text mode:
          - **Deferred** (prompt=None): stores the image as pending context
            that is attached to the *next* send_text() call.
          - **Immediate** (prompt given): adds the image + prompt as a user
            turn right away.
        """
        if not self._connected or self._client is None:
            self._log.warning("gemini.send_image.not_connected")
            return

        self._log.info(
            "gemini.send_image",
            bytes_len=len(data),
            mime=mime_type,
            mode=self._mode,
        )

        # ---- Live mode: send inline into the bidirectional stream ----
        if self._mode == "live" and self._live_session is not None:
            try:
                await self._live_session.send(
                    input=types.LiveClientRealtimeInput(
                        media_chunks=[
                            types.Blob(data=data, mime_type=mime_type)
                        ]
                    )
                )
            except ConnectionClosed as exc:
                self._log.warning("gemini.send_image.session_closed", code=exc.code)
                await self._reconnect_live()
            except Exception:
                self._log.exception("gemini.send_image.live_error")
            return

        # ---- Text mode (L0) ----
        if prompt is not None:
            # Immediate mode: add image + prompt as a user turn
            parts = [
                types.Part(
                    inline_data=types.Blob(mime_type=mime_type, data=data)
                ),
                types.Part(text=prompt),
            ]
            self._history.append(types.Content(role="user", parts=parts))
        else:
            # Deferred mode: store for the next send_text() call
            self._pending_image = (data, mime_type)

    # ------------------------------------------------------------------
    # L3: Tool result injection
    # ------------------------------------------------------------------

    async def send_tool_result(
        self,
        name: str,
        result: dict,
        call_id: str | None = None,
    ) -> None:
        """Feed a tool execution result back into Gemini.

        Live mode:  Sends via the Live session.  Gemini automatically
                    incorporates the result and continues its response.
        Text mode:  Appends to history.  Caller should call receive()
                    again to get Gemini's follow-up.

        Args:
            name: The tool function name (must match the original call).
            result: The result dict from the tool executor.
        """
        if not self._connected or self._client is None:
            self._log.warning("gemini.send_tool_result.not_connected")
            return

        self._log.info(
            "gemini.send_tool_result",
            name=name,
            success=result.get("success"),
            mode=self._mode,
        )

        # ---- Live mode ----
        if self._mode == "live" and self._live_session is not None:
            try:
                await self._live_session.send(
                    input=types.LiveClientToolResponse(
                        function_responses=[
                            types.FunctionResponse(
                                id=call_id,
                                name=name,
                                response=result,
                            )
                        ]
                    )
                )
            except ConnectionClosed as exc:
                self._log.warning("gemini.send_tool_result.session_closed", code=exc.code)
                await self._reconnect_live()
            except Exception:
                self._log.exception("gemini.send_tool_result.live_error")
            return

        # ---- Text mode ----
        self._history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name,
                            response=result,
                        )
                    ),
                ],
            )
        )
