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

try:
    from .gemini_session import build_system_instruction
except ImportError:
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

    # -- Windows Power Tools --

    async def open_application(name_or_path: str) -> dict:
        """Open an application by name or path.
        Supports common app names (notepad, chrome, firefox, explorer, calc,
        paint, cmd, powershell, code, edge, word, excel, teams, spotify, etc.),
        full executable paths, file associations, and URLs.
        The application launches asynchronously — it may take a moment to appear."""
        return await bridge.dispatch(
            "open_application", {"name_or_path": name_or_path}
        )

    async def list_all_windows() -> dict:
        """List all visible windows on the desktop with their titles,
        positions, sizes, and states (minimized/maximized/active)."""
        return await bridge.dispatch("list_all_windows", {})

    async def get_active_window() -> dict:
        """Get information about the currently active (foreground) window."""
        return await bridge.dispatch("get_active_window", {})

    async def minimize_window(title: str) -> dict:
        """Minimize a window by title (substring match)."""
        return await bridge.dispatch("minimize_window", {"title": title})

    async def maximize_window(title: str) -> dict:
        """Maximize a window by title (substring match)."""
        return await bridge.dispatch("maximize_window", {"title": title})

    async def close_window(title: str) -> dict:
        """Close a window by title (substring match).
        Use with caution — unsaved work in the window may be lost."""
        return await bridge.dispatch("close_window", {"title": title})

    async def resize_window(title: str, width: int, height: int) -> dict:
        """Resize a window by title to the specified width and height (pixels)."""
        return await bridge.dispatch(
            "resize_window", {"title": title, "width": width, "height": height}
        )

    async def move_window(title: str, x: int, y: int) -> dict:
        """Move a window by title to position (x, y) on screen."""
        return await bridge.dispatch(
            "move_window", {"title": title, "x": x, "y": y}
        )

    async def list_processes(name_filter: str = "") -> dict:
        """List running processes sorted by memory usage.
        Optionally filter by name (case-insensitive substring match).
        Returns top 50 processes with PID, name, and memory usage."""
        return await bridge.dispatch(
            "list_processes", {"name_filter": name_filter}
        )

    async def kill_process(name_or_pid: str) -> dict:
        """Kill a process by name or PID. Multi-match support:
        if name matches multiple processes, all matching are terminated.
        Protected system processes (csrss, lsass, svchost, etc.) cannot be killed."""
        return await bridge.dispatch(
            "kill_process", {"name_or_pid": name_or_pid}
        )

    async def get_clipboard() -> dict:
        """Read the current clipboard text content."""
        return await bridge.dispatch("get_clipboard", {})

    async def set_clipboard(text: str) -> dict:
        """Set the clipboard text content. Useful for sharing text
        between applications."""
        return await bridge.dispatch("set_clipboard", {"text": text})

    async def get_screen_info() -> dict:
        """Get monitor information: resolution, DPI, bounds for all screens.
        Useful for understanding the display layout."""
        return await bridge.dispatch("get_screen_info", {})

    # -- Persistent Memory (enhanced) --

    async def search_notes(query: str, limit: int = 5) -> dict:
        """Search persistent notes by keyword. Returns matching notes ranked
        by relevance. Use this BEFORE starting a task to recall relevant context
        from previous sessions."""
        return await bridge.dispatch(
            "search_notes", {"query": query, "limit": limit}
        )

    async def export_context() -> dict:
        """Export all session memory to a compact context.txt file.
        Useful for creating a persistent reference of everything learned."""
        return await bridge.dispatch("export_context", {})

    async def memory_stats() -> dict:
        """Get memory system statistics: note count, total size,
        and whether compaction is needed."""
        return await bridge.dispatch("memory_stats", {})

    # -- Web tools (E3) --

    async def web_search(query: str, max_results: int = 5) -> dict:
        """Search the web using DuckDuckGo. No API key required.
        Returns top results with title, url, and snippet.
        Use this for quick factual lookups or research without opening a browser."""
        return await bridge.dispatch(
            "web_search", {"query": query, "max_results": max_results}
        )

    async def web_fetch(url: str, max_chars: int = 8000) -> dict:
        """Fetch a web page and return its text content.
        HTML is automatically converted to plain text.
        Use this to read documentation, articles, or API references."""
        return await bridge.dispatch(
            "web_fetch", {"url": url, "max_chars": max_chars}
        )

    async def web_cache_get(url: str) -> dict:
        """Get a cached web page, or fetch and cache it if not cached.
        Cached pages expire after 1 hour."""
        return await bridge.dispatch("web_cache_get", {"url": url})

    # -- Long-running process management (E2) --

    async def start_process(command: str, label: str = "") -> dict:
        """Start a long-running background process (server, watcher, etc.).
        Returns a PID for later status checks or stopping.
        Use this instead of run_command for commands that run indefinitely."""
        return await bridge.dispatch(
            "start_process", {"command": command, "label": label}
        )

    async def check_process(pid: str) -> dict:
        """Check the status of a background process by PID.
        Returns running/exited status with any output."""
        return await bridge.dispatch("check_process", {"pid": pid})

    async def stop_process(pid: str) -> dict:
        """Stop a background process by PID."""
        return await bridge.dispatch("stop_process", {"pid": pid})

    # -- Browser automation (E1: Playwright CDP) --

    async def browser_connect(
        cdp_url: str = "http://127.0.0.1:9222",
        browser: str = "auto",
        profile: str = "",
    ) -> dict:
        """Connect to (or auto-launch) a Chromium browser via Chrome DevTools Protocol.

        The browser is launched automatically with --remote-debugging-port=9222
        if it is not already running — no manual setup needed.

        Args:
            cdp_url:  CDP endpoint. Default: http://localhost:9222
            browser:  Which browser to use/launch — "auto" (tries Chrome, Edge,
                      Chromium, Brave in that order), "chrome", "chromium",
                      "edge", or "brave". Default: "auto"
            profile:  Chrome profile directory name to use, e.g. "Default",
                      "Profile 1", "Profile 2". Leave empty for the default
                      profile. Example: profile="Profile 1"

        Examples:
            browser_connect()                          # auto-detect, default profile
            browser_connect(browser="chrome")          # force Chrome
            browser_connect(browser="edge", profile="Profile 2")  # Edge with a specific profile
        """
        return await bridge.dispatch("browser_connect", {
            "cdp_url": cdp_url, "browser": browser, "profile": profile,
        })

    async def browser_evaluate(javascript: str) -> dict:
        """Execute JavaScript in the browser page and return the result.
        Use for reading DOM values, page state, or running scripts."""
        return await bridge.dispatch("browser_evaluate", {"javascript": javascript})

    async def browser_fill_form(selector: str, value: str) -> dict:
        """Fill a form field identified by CSS selector."""
        return await bridge.dispatch("browser_fill_form", {"selector": selector, "value": value})

    async def browser_click_element(selector: str) -> dict:
        """Click an element identified by CSS selector. More precise than
        screen_click for web pages — uses the actual DOM element."""
        return await bridge.dispatch("browser_click_element", {"selector": selector})

    async def browser_extract_text(selector: str) -> dict:
        """Extract text content from an element by CSS selector."""
        return await bridge.dispatch("browser_extract_text", {"selector": selector})

    async def browser_wait_for(selector: str, timeout: int = 30000) -> dict:
        """Wait for a CSS selector to appear on the page."""
        return await bridge.dispatch("browser_wait_for", {"selector": selector, "timeout": timeout})

    async def browser_navigate(url: str) -> dict:
        """Navigate the browser to a URL."""
        return await bridge.dispatch("browser_navigate", {"url": url})

    return [
        # Core dev tools
        read_file, write_file, patch_file, run_command, capture_screen,
        # Screen navigation (coordinate-based)
        screen_click, screen_type, screen_scroll, screen_hotkey,
        screen_move, screen_drag, find_window, focus_window,
        # Vision-grounded navigation (Computer Use model)
        smart_click,
        # Windows power tools
        open_application, list_all_windows, get_active_window,
        minimize_window, maximize_window, close_window,
        resize_window, move_window,
        list_processes, kill_process,
        get_clipboard, set_clipboard, get_screen_info,
        # Persistent memory (enhanced)
        search_notes, export_context, memory_stats,
        # Web tools (E3)
        web_search, web_fetch, web_cache_get,
        # Long-running processes (E2)
        start_process, check_process, stop_process,
        # Browser automation (E1: Playwright CDP)
        browser_connect, browser_evaluate, browser_fill_form,
        browser_click_element, browser_extract_text,
        browser_wait_for, browser_navigate,
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
