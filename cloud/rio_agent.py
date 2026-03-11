"""
Rio ADK Agent Definition.

Provides ``create_rio_agent()`` which builds a per-connection ADK Agent
whose tools are closured over a ``ToolBridge`` instance.  This allows tool
calls made by the Gemini model to be transparently proxied to the local
client over WebSocket and resolved when the client sends back results.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Optional

import structlog

from gemini_session import build_system_instruction

# ADK Agent is only needed for the legacy create_rio_agent() factory.
# Optional import so the module works even when google-adk is not installed
# (the direct Live API server only needs ToolBridge + _make_tools).
try:
    from google.adk.agents import Agent
except ImportError:
    Agent = None  # type: ignore[assignment,misc]

logger = structlog.get_logger(__name__)

# Model IDs — override via env vars
DEFAULT_LIVE_MODEL = os.environ.get(
    "LIVE_MODEL", "gemini-2.5-flash-native-audio-latest"
)


# ---------------------------------------------------------------------------
# ToolBridge — per-connection async bridge for remote tool execution
# ---------------------------------------------------------------------------

class ToolBridge:
    """Routes ADK tool calls to the local client via WebSocket.

    Flow:
      1. ADK invokes an async tool function (closured over this bridge).
      2. ``dispatch()`` sends a ``tool_call`` JSON frame to the client.
      3. The upstream reader calls ``resolve()`` when the client sends
         back a ``tool_result`` frame.
      4. The original ``dispatch()`` await completes with the result.
    """

    def __init__(self, websocket: Any, broadcast_fn: Any = None) -> None:
        self._ws = websocket
        self._broadcast = broadcast_fn
        self._pending: dict[str, asyncio.Future[dict]] = {}
        self._log = logger.bind(component="tool_bridge")

    async def dispatch(self, name: str, args: dict) -> dict:
        """Send a tool call to the local client and await the result."""
        call_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self._pending[call_id] = future

        frame = json.dumps({
            "type": "tool_call",
            "name": name,
            "args": args,
            "id": call_id,
        })
        self._log.info("tool_bridge.dispatch", name=name, call_id=call_id)
        await self._ws.send_text(frame)

        if self._broadcast:
            await self._broadcast({
                "type": "dashboard",
                "subtype": "tool_call",
                "name": name,
                "args": args,
            })

        try:
            result = await asyncio.wait_for(future, timeout=60.0)
            self._log.info(
                "tool_bridge.result",
                name=name, call_id=call_id,
                success=result.get("success"),
            )
            return result
        except asyncio.TimeoutError:
            self._log.error("tool_bridge.timeout", name=name, call_id=call_id)
            return {"success": False, "error": f"Tool '{name}' timed out (60s)"}
        finally:
            self._pending.pop(call_id, None)

    def resolve(self, call_id: str, result: dict) -> None:
        """Resolve a pending tool call future with the client's result."""
        future = self._pending.get(call_id)
        if future is not None and not future.done():
            future.set_result(result)
        else:
            self._log.warning(
                "tool_bridge.resolve.no_pending",
                call_id=call_id,
            )

    def resolve_by_name(self, name: str, result: dict) -> None:
        """Resolve the oldest pending call matching *name* (fallback)."""
        for cid, future in list(self._pending.items()):
            if not future.done():
                future.set_result(result)
                self._pending.pop(cid, None)
                return
        self._log.warning("tool_bridge.resolve_by_name.none", name=name)


# ---------------------------------------------------------------------------
# Tool factory — creates closured async tool functions per connection
# ---------------------------------------------------------------------------

def _make_tools(bridge: ToolBridge) -> list:
    """Return a list of async tool functions closured over *bridge*.

    Each function's docstring acts as the tool description for the model.
    """

    # -- Core dev tools --

    async def read_file(path: str) -> dict:
        """Read the contents of a file on the user's machine.
        Use this to examine code, config files, logs, or any text file."""
        return await bridge.dispatch("read_file", {"path": path})

    async def write_file(path: str, content: str) -> dict:
        """Write content to a file on the user's machine. A .rio.bak
        backup is created automatically before overwriting."""
        return await bridge.dispatch("write_file", {"path": path, "content": content})

    async def patch_file(path: str, old_text: str, new_text: str) -> dict:
        """Apply a find-and-replace edit to a file. old_text must match exactly.
        A .rio.bak backup is created before patching."""
        return await bridge.dispatch(
            "patch_file", {"path": path, "old_text": old_text, "new_text": new_text}
        )

    async def run_command(command: str) -> dict:
        """Execute a shell command on the user's machine (30s timeout).
        Dangerous commands are blocked."""
        return await bridge.dispatch("run_command", {"command": command})

    async def capture_screen() -> dict:
        """Capture a screenshot of the user's screen right now."""
        return await bridge.dispatch("capture_screen", {})

    # -- Screen navigation --

    async def screen_click(x: int, y: int, button: str = "left", clicks: int = 1) -> dict:
        """Click at a position on the user's screen. Coordinates are in
        screenshot space (the resized image you see)."""
        return await bridge.dispatch(
            "screen_click", {"x": x, "y": y, "button": button, "clicks": clicks}
        )

    async def screen_type(text: str, interval: float = 0.02) -> dict:
        """Type text at the current cursor position on the user's screen."""
        return await bridge.dispatch(
            "screen_type", {"text": text, "interval": interval}
        )

    async def screen_scroll(x: int, y: int, clicks: int) -> dict:
        """Scroll at a position on the screen. Positive=up, negative=down."""
        return await bridge.dispatch(
            "screen_scroll", {"x": x, "y": y, "clicks": clicks}
        )

    async def screen_hotkey(keys: str) -> dict:
        """Press a keyboard shortcut (e.g. 'ctrl+s', 'alt+tab', 'enter')."""
        return await bridge.dispatch("screen_hotkey", {"keys": keys})

    async def screen_move(x: int, y: int) -> dict:
        """Move the mouse cursor to a position without clicking."""
        return await bridge.dispatch("screen_move", {"x": x, "y": y})

    async def screen_drag(
        start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5,
    ) -> dict:
        """Click-and-drag from one position to another."""
        return await bridge.dispatch(
            "screen_drag", {
                "start_x": start_x, "start_y": start_y,
                "end_x": end_x, "end_y": end_y, "duration": duration,
            }
        )

    async def find_window(title: str) -> dict:
        """Search for open windows by title."""
        return await bridge.dispatch("find_window", {"title": title})

    async def focus_window(title: str) -> dict:
        """Bring a window to the foreground."""
        return await bridge.dispatch("focus_window", {"title": title})

    # -- Customer Care --

    async def create_ticket(
        title: str, category: str, priority: str, description: str,
        customer_id: str = "", tags: str = "",
    ) -> dict:
        """Create a customer support ticket to track an issue."""
        return await bridge.dispatch(
            "create_ticket", {
                "title": title, "category": category,
                "priority": priority, "description": description,
                "customer_id": customer_id, "tags": tags,
            }
        )

    async def update_ticket(
        ticket_id: str, status: str = "", priority: str = "",
        escalation_tier: str = "", notes: str = "",
    ) -> dict:
        """Update an existing support ticket."""
        return await bridge.dispatch(
            "update_ticket", {
                "ticket_id": ticket_id, "status": status,
                "priority": priority, "escalation_tier": escalation_tier,
                "notes": notes,
            }
        )

    # -- Tutor --

    async def generate_quiz(
        topic: str, difficulty: str = "intermediate",
        num_questions: int = 5, question_types: str = "", focus_areas: str = "",
    ) -> dict:
        """Generate a quiz or practice problems for a student on a topic."""
        return await bridge.dispatch(
            "generate_quiz", {
                "topic": topic, "difficulty": difficulty,
                "num_questions": num_questions,
                "question_types": question_types,
                "focus_areas": focus_areas,
            }
        )

    async def track_progress(
        action: str, subject: str, topic: str = "",
        score: str = "", notes: str = "", student_id: str = "",
    ) -> dict:
        """Track or retrieve a student's learning progress."""
        return await bridge.dispatch(
            "track_progress", {
                "action": action, "subject": subject, "topic": topic,
                "score": score, "notes": notes, "student_id": student_id,
            }
        )

    async def explain_concept(
        concept: str, level: str = "intermediate", context: str = "",
    ) -> dict:
        """Retrieve a structured explanation of a concept at the student's level."""
        return await bridge.dispatch(
            "explain_concept", {"concept": concept, "level": level, "context": context}
        )

    # -- GenMedia (Imagen 3 + Veo 2) --

    async def generate_image(
        prompt: str, aspect_ratio: str = "1:1",
        style: str = "", negative_prompt: str = "",
    ) -> dict:
        """Generate an image using Imagen 3. Returns base64-encoded image data.
        aspect_ratio: 1:1, 16:9, 9:16, 4:3, 3:4.
        style: photorealistic, illustration, etc."""
        return await bridge.dispatch(
            "generate_image", {
                "prompt": prompt, "aspect_ratio": aspect_ratio,
                "style": style, "negative_prompt": negative_prompt,
            }
        )

    async def generate_video(
        prompt: str, duration_seconds: int = 5, aspect_ratio: str = "16:9",
    ) -> dict:
        """Generate a short video using Veo 2 (5-10 seconds).
        Returns a download URL or file path."""
        return await bridge.dispatch(
            "generate_video", {
                "prompt": prompt,
                "duration_seconds": duration_seconds,
                "aspect_ratio": aspect_ratio,
            }
        )

    # -- Vision-grounded navigation (Computer Use model) --

    async def smart_click(
        target: str, action: str = "click", clicks: int = 1,
    ) -> dict:
        """Click a UI element by describing it in natural language.

        Uses gemini-2.5-computer-use-preview-10-2025 to take a fresh screenshot
        and visually locate the element before clicking — far more reliable
        than guessing pixel coordinates from an earlier screenshot.

        target: describe what to click, e.g. "the Save button",
                "search input field", "file menu", "OK dialog button"
        action: "click" (default) | "double_click" | "right_click"
        clicks: number of clicks (default 1)

        PREFER this over screen_click whenever you know what you want
        to click but not the exact coordinates."""
        return await bridge.dispatch(
            "smart_click", {"target": target, "action": action, "clicks": clicks}
        )

    return [
        # Core dev tools
        read_file, write_file, patch_file, run_command, capture_screen,
        # Screen navigation (coordinate-based)
        screen_click, screen_type, screen_scroll, screen_hotkey,
        screen_move, screen_drag, find_window, focus_window,
        # Vision-grounded navigation (Computer Use model)
        smart_click,
        # Customer care
        create_ticket, update_ticket,
        # Tutor
        generate_quiz, track_progress, explain_concept,
        # GenMedia
        generate_image, generate_video,
    ]


# ---------------------------------------------------------------------------
# Agent factory — one per WebSocket connection
# ---------------------------------------------------------------------------

def create_rio_agent(bridge: ToolBridge, model: str | None = None) -> Agent:
    """Create an ADK Agent with tools closured over the given ToolBridge.

    Call this once per WebSocket connection so each client gets its own
    tool dispatch channel.
    """
    instruction = build_system_instruction()
    agent_model = model or DEFAULT_LIVE_MODEL
    tools = _make_tools(bridge)

    return Agent(
        name="rio_live_agent",
        model=agent_model,
        instruction=instruction,
        tools=tools,
    )
