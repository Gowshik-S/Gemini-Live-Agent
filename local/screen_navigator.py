"""
Rio Local -- Screen Navigator

Provides screen interaction capabilities for the AI agent:
click, type, scroll, hotkey, drag, move, and window management.

Uses pyautogui for cross-platform mouse/keyboard control.
CoordinateMapper handles resize_factor mapping and DPI scaling.

Dependencies: pyautogui, pygetwindow (Windows)

Usage::

    nav = ScreenNavigator(resize_factor=0.5)
    result = await nav.click(450, 320)            # screenshot coords
    result = await nav.type_text("hello world")
    result = await nav.hotkey("ctrl+s")
"""

from __future__ import annotations

import asyncio
import platform
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — avoid hard crash if deps missing
# ---------------------------------------------------------------------------

_pyautogui = None
_pygetwindow = None


def _ensure_pyautogui() -> bool:
    global _pyautogui
    if _pyautogui is not None:
        return True
    try:
        import pyautogui as _pag
        # Safety defaults
        _pag.PAUSE = 0.08          # 80ms pause between actions
        _pag.FAILSAFE = True       # move mouse to (0,0) to abort
        _pyautogui = _pag
        return True
    except ImportError:
        return False


def _safe_execute(func, *args, **kwargs):
    """Execute a pyautogui call with FailSafeException handling.

    Instead of crashing the entire agent when mouse hits (0,0),
    catches the exception and returns a structured error dict.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        exc_name = type(exc).__name__
        if exc_name == "FailSafeException" or "FailSafe" in exc_name:
            log.warning("screen_nav.failsafe_triggered",
                        note="Mouse reached corner — action aborted safely")
            raise _FailSafeAbort("PyAutoGUI FAILSAFE triggered — mouse reached screen corner. "
                                 "Action aborted safely. Move mouse away from top-left corner.")
        raise


class _FailSafeAbort(Exception):
    """Raised when PyAutoGUI FAILSAFE is triggered, for clean handling."""
    pass


def _ensure_pygetwindow() -> bool:
    global _pygetwindow
    if _pygetwindow is not None:
        return True
    try:
        import pygetwindow as _pgw
        _pygetwindow = _pgw
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# DPI awareness (Windows)
# ---------------------------------------------------------------------------

def _setup_dpi_awareness() -> bool:
    """Enable per-monitor DPI awareness on Windows.

    Must be called before any coordinate operations.
    Returns True if DPI awareness was set.
    """
    if platform.system() != "Windows":
        return False
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        log.info("screen_nav.dpi_aware", level="per_monitor")
        return True
    except Exception as exc:
        log.debug("screen_nav.dpi_awareness_failed", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Action log entry
# ---------------------------------------------------------------------------

@dataclass
class ActionLogEntry:
    timestamp: float
    action: str
    screenshot_coords: tuple[int, int] | None
    real_coords: tuple[int, int] | None
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Prevents runaway automation (>N actions in M seconds)."""

    def __init__(self, max_actions: int = 15, window_seconds: float = 5.0):
        self._max = max_actions
        self._window = window_seconds
        self._timestamps: list[float] = []

    def check(self) -> bool:
        """Returns True if action is allowed, False if rate-limited."""
        now = time.monotonic()
        cutoff = now - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self._max:
            return False
        self._timestamps.append(now)
        return True


# ---------------------------------------------------------------------------
# ScreenNavigator
# ---------------------------------------------------------------------------

class ScreenNavigator:
    """Screen interaction engine with coordinate mapping.

    Gemini sees screenshots at ``resize_factor`` scale (e.g. 0.5 = 50%).
    All coordinates from Gemini are in **screenshot space**. This class
    maps them to real screen coordinates before executing pyautogui actions.

    Features:
      - CoordinateMapper: screenshot coords → real screen coords
      - DPI awareness on Windows (per-monitor)
      - Rate limiting to prevent runaway automation
      - Action log for audit trail
      - pyautogui.FAILSAFE for emergency abort (mouse to top-left)
    """

    def __init__(
        self,
        resize_factor: float = 0.5,
        monitor_left: int = 0,
        monitor_top: int = 0,
    ) -> None:
        self._resize_factor = resize_factor
        self._monitor_left = monitor_left
        self._monitor_top = monitor_top
        self._dpi_aware = _setup_dpi_awareness()
        self._available = _ensure_pyautogui()
        self._rate_limiter = _RateLimiter(max_actions=15, window_seconds=5.0)
        self._action_log: list[ActionLogEntry] = []
        self._max_log_entries = 200

        if self._available:
            log.info(
                "screen_nav.init",
                resize_factor=resize_factor,
                dpi_aware=self._dpi_aware,
                failsafe=True,
            )
        else:
            log.warning(
                "screen_nav.unavailable",
                note="Install pyautogui: pip install pyautogui",
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    @property
    def resize_factor(self) -> float:
        return self._resize_factor

    @property
    def action_log(self) -> list[ActionLogEntry]:
        return list(self._action_log)

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def update_monitor_offset(self, left: int, top: int) -> None:
        """Update monitor offset from the latest screen capture metadata."""
        self._monitor_left = left
        self._monitor_top = top

    def _map_coords(self, sx: int, sy: int) -> tuple[int, int]:
        """Map screenshot coordinates → real screen coordinates.

        Args:
            sx: X coordinate in screenshot space.
            sy: Y coordinate in screenshot space.

        Returns:
            (real_x, real_y) in actual screen pixels.
        """
        rx = int(sx / self._resize_factor) + self._monitor_left
        ry = int(sy / self._resize_factor) + self._monitor_top
        return rx, ry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_action(
        self,
        action: str,
        screenshot_coords: tuple[int, int] | None = None,
        real_coords: tuple[int, int] | None = None,
        **details: Any,
    ) -> None:
        entry = ActionLogEntry(
            timestamp=time.time(),
            action=action,
            screenshot_coords=screenshot_coords,
            real_coords=real_coords,
            details=details,
        )
        self._action_log.append(entry)
        # Trim old entries
        if len(self._action_log) > self._max_log_entries:
            self._action_log = self._action_log[-self._max_log_entries:]

        log.info(
            "screen_nav.action",
            action=action,
            screenshot_coords=screenshot_coords,
            real_coords=real_coords,
            **details,
        )

    def _check_available(self) -> dict[str, Any] | None:
        """Returns error dict if navigator is not available, else None."""
        if not self._available or _pyautogui is None:
            return {
                "success": False,
                "error": "Screen navigator unavailable — install pyautogui",
            }
        return None

    def _check_rate_limit(self) -> dict[str, Any] | None:
        """Returns error dict if rate-limited, else None."""
        if not self._rate_limiter.check():
            log.warning("screen_nav.rate_limited")
            return {
                "success": False,
                "error": "Rate limited — too many actions in quick succession. "
                         "Wait a moment before the next action.",
            }
        return None

    # ------------------------------------------------------------------
    # Actions — Phase 1: Core Navigation
    # ------------------------------------------------------------------

    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> dict[str, Any]:
        """Click at screenshot coordinates (x, y).

        Args:
            x: X in screenshot space.
            y: Y in screenshot space.
            button: "left", "right", or "middle".
            clicks: 1 for single, 2 for double, 3 for triple.
        """
        if err := self._check_available():
            return err
        if err := self._check_rate_limit():
            return err

        rx, ry = self._map_coords(x, y)

        # Validate button
        if button not in ("left", "right", "middle"):
            button = "left"
        clicks = max(1, min(clicks, 3))

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: _safe_execute(_pyautogui.click, rx, ry, button=button, clicks=clicks)
            )
        except _FailSafeAbort as e:
            self._log_action("click", (x, y), (rx, ry), error="failsafe")
            return {"success": False, "error": str(e)}

        self._log_action("click", (x, y), (rx, ry), button=button, clicks=clicks)
        return {
            "success": True,
            "action": "click",
            "screenshot_coords": [x, y],
            "real_coords": [rx, ry],
            "button": button,
            "clicks": clicks,
        }

    async def type_text(
        self,
        text: str,
        interval: float = 0.02,
    ) -> dict[str, Any]:
        """Type text at the current cursor position.

        Uses pyautogui.write() for ASCII and falls back to
        pyperclip + hotkey paste for unicode characters.

        Args:
            text: The text to type.
            interval: Delay between keystrokes in seconds.
        """
        if err := self._check_available():
            return err
        if err := self._check_rate_limit():
            return err

        loop = asyncio.get_running_loop()

        try:
            # Check if text is pure ASCII — pyautogui.write only handles ASCII
            if all(ord(c) < 128 for c in text):
                await loop.run_in_executor(
                    None, lambda: _safe_execute(_pyautogui.write, text, interval=interval)
                )
            else:
                # For unicode: use clipboard paste
                try:
                    import pyperclip
                    pyperclip.copy(text)
                    if platform.system() == "Darwin":
                        await loop.run_in_executor(
                            None, lambda: _safe_execute(_pyautogui.hotkey, "command", "v")
                        )
                    else:
                        await loop.run_in_executor(
                            None, lambda: _safe_execute(_pyautogui.hotkey, "ctrl", "v")
                        )
                except ImportError:
                    # No pyperclip — try character by character via write
                    # (may lose non-ASCII chars)
                    safe = text.encode("ascii", "replace").decode("ascii")
                    await loop.run_in_executor(
                        None, lambda: _safe_execute(_pyautogui.write, safe, interval=interval)
                    )
        except _FailSafeAbort as e:
            self._log_action("type_text", details={"length": len(text), "error": "failsafe"})
            return {"success": False, "error": str(e)}

        self._log_action("type_text", details={"length": len(text)})
        return {
            "success": True,
            "action": "type_text",
            "characters_typed": len(text),
        }

    async def scroll(
        self,
        x: int,
        y: int,
        clicks: int,
    ) -> dict[str, Any]:
        """Scroll at screenshot coordinates.

        Args:
            x: X in screenshot space (where to position cursor).
            y: Y in screenshot space.
            clicks: Positive = scroll up, negative = scroll down.
        """
        if err := self._check_available():
            return err
        if err := self._check_rate_limit():
            return err

        rx, ry = self._map_coords(x, y)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: _safe_execute(_pyautogui.scroll, clicks, rx, ry)
            )
        except _FailSafeAbort as e:
            self._log_action("scroll", (x, y), (rx, ry), error="failsafe")
            return {"success": False, "error": str(e)}

        direction = "up" if clicks > 0 else "down"
        self._log_action("scroll", (x, y), (rx, ry), clicks=clicks, direction=direction)
        return {
            "success": True,
            "action": "scroll",
            "screenshot_coords": [x, y],
            "real_coords": [rx, ry],
            "clicks": clicks,
            "direction": direction,
        }

    async def hotkey(self, keys: str) -> dict[str, Any]:
        """Press a keyboard shortcut.

        Args:
            keys: Key combination string, e.g. "ctrl+s", "alt+tab", "enter".
                  Keys are separated by "+".
        """
        if err := self._check_available():
            return err
        if err := self._check_rate_limit():
            return err

        key_list = [k.strip().lower() for k in keys.split("+")]

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: _safe_execute(_pyautogui.hotkey, *key_list)
            )
        except _FailSafeAbort as e:
            self._log_action("hotkey", details={"keys": key_list, "error": "failsafe"})
            return {"success": False, "error": str(e)}

        self._log_action("hotkey", details={"keys": key_list})
        return {
            "success": True,
            "action": "hotkey",
            "keys": key_list,
        }

    async def move(self, x: int, y: int) -> dict[str, Any]:
        """Move the mouse to screenshot coordinates without clicking.

        Args:
            x: X in screenshot space.
            y: Y in screenshot space.
        """
        if err := self._check_available():
            return err
        if err := self._check_rate_limit():
            return err

        rx, ry = self._map_coords(x, y)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: _safe_execute(_pyautogui.moveTo, rx, ry)
            )
        except _FailSafeAbort as e:
            self._log_action("move", (x, y), (rx, ry), error="failsafe")
            return {"success": False, "error": str(e)}

        self._log_action("move", (x, y), (rx, ry))
        return {
            "success": True,
            "action": "move",
            "screenshot_coords": [x, y],
            "real_coords": [rx, ry],
        }

    async def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
    ) -> dict[str, Any]:
        """Click-and-drag from start to end in screenshot coordinates.

        Args:
            start_x: Start X in screenshot space.
            start_y: Start Y in screenshot space.
            end_x:   End X in screenshot space.
            end_y:   End Y in screenshot space.
            duration: Time for the drag motion in seconds.
        """
        if err := self._check_available():
            return err
        if err := self._check_rate_limit():
            return err

        rsx, rsy = self._map_coords(start_x, start_y)
        rex, rey = self._map_coords(end_x, end_y)

        loop = asyncio.get_running_loop()

        def _do_drag():
            _safe_execute(_pyautogui.moveTo, rsx, rsy)
            _safe_execute(_pyautogui.drag, rex - rsx, rey - rsy, duration=duration)

        try:
            await loop.run_in_executor(None, _do_drag)
        except _FailSafeAbort as e:
            self._log_action("drag", (start_x, start_y), (rsx, rsy), error="failsafe")
            return {"success": False, "error": str(e)}

        self._log_action(
            "drag",
            (start_x, start_y),
            (rsx, rsy),
            end_screenshot=[end_x, end_y],
            end_real=[rex, rey],
            duration=duration,
        )
        return {
            "success": True,
            "action": "drag",
            "start_screenshot": [start_x, start_y],
            "start_real": [rsx, rsy],
            "end_screenshot": [end_x, end_y],
            "end_real": [rex, rey],
        }

    # ------------------------------------------------------------------
    # Actions — Phase 3: Window Management
    # ------------------------------------------------------------------

    async def find_window(self, title_contains: str) -> dict[str, Any]:
        """Search for windows whose title contains the given text.

        Args:
            title_contains: Substring to search for in window titles.
        """
        if err := self._check_available():
            return err

        if not _ensure_pygetwindow():
            return {
                "success": False,
                "error": "Window management unavailable — install pygetwindow",
            }

        loop = asyncio.get_running_loop()

        def _find():
            windows = _pygetwindow.getWindowsWithTitle(title_contains)
            return [
                {
                    "title": w.title,
                    "position": [w.left, w.top],
                    "size": [w.width, w.height],
                    "visible": w.visible,
                    "minimized": w.isMinimized,
                }
                for w in windows
            ]

        results = await loop.run_in_executor(None, _find)
        self._log_action("find_window", details={"query": title_contains, "found": len(results)})
        return {
            "success": True,
            "action": "find_window",
            "query": title_contains,
            "windows": results,
            "count": len(results),
        }

    async def focus_window(self, title_contains: str) -> dict[str, Any]:
        """Bring a window to the foreground by title.

        Args:
            title_contains: Substring to search for in window titles.
        """
        if err := self._check_available():
            return err

        if not _ensure_pygetwindow():
            return {
                "success": False,
                "error": "Window management unavailable — install pygetwindow",
            }

        loop = asyncio.get_running_loop()

        def _focus():
            windows = _pygetwindow.getWindowsWithTitle(title_contains)
            if not windows:
                return None
            w = windows[0]
            if w.isMinimized:
                w.restore()
            w.activate()
            return w.title

        title = await loop.run_in_executor(None, _focus)
        if title is None:
            return {
                "success": False,
                "error": f"No window found matching '{title_contains}'",
            }

        self._log_action("focus_window", details={"query": title_contains, "focused": title})
        return {
            "success": True,
            "action": "focus_window",
            "title": title,
        }
