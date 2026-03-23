"""
Rio Local -- Screen Navigator

Provides screen interaction capabilities for the AI agent:
click, type, scroll, hotkey, drag, move, and window management.

Uses pyautogui for cross-platform mouse/keyboard control.
CoordinateMapper handles resize_factor mapping and DPI scaling.

Comprehensive Windows control via:
  - pyautogui: mouse/keyboard automation
  - pygetwindow: window enumeration and management
  - subprocess: application launching
  - psutil: process management
  - win32gui / win32com.client: deep Windows integration (optional)

Dependencies: pyautogui, pygetwindow (Windows), psutil

Usage::

    nav = ScreenNavigator(resize_factor=0.5)
    result = await nav.click(450, 320)            # screenshot coords
    result = await nav.type_text("hello world")
    result = await nav.hotkey("ctrl+s")
    result = await nav.open_application("notepad")
    result = await nav.list_all_windows()
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import subprocess
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


# -- psutil (process management) --

_psutil = None


def _ensure_psutil() -> bool:
    global _psutil
    if _psutil is not None:
        return True
    try:
        import psutil as _ps
        _psutil = _ps
        return True
    except ImportError:
        return False


# -- win32gui / win32com (deep Windows integration, optional) --

_win32gui = None
_win32com_client = None
_win32api = None
_win32con = None

def _ensure_win32() -> bool:
    """Attempt to load pywin32 modules. Returns True if win32gui loaded."""
    global _win32gui, _win32com_client, _win32api, _win32con
    if _win32gui is not None:
        return True
    try:
        import win32gui as _wg
        import win32api as _wa
        import win32con as _wccon
        _win32gui = _wg
        _win32api = _wa
        _win32con = _wccon
    except ImportError:
        pass
    try:
        import win32com.client as _wc
        _win32com_client = _wc
    except ImportError:
        pass
    return _win32gui is not None


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
# pywin32 execution helpers
# ---------------------------------------------------------------------------

def _win32_click(x: int, y: int, button: str = "left", clicks: int = 1) -> None:
    if _win32api is None or _win32con is None:
        raise RuntimeError("win32api not available")
    
    _win32api.SetCursorPos((x, y))
    
    if button == "left":
        down = _win32con.MOUSEEVENTF_LEFTDOWN
        up = _win32con.MOUSEEVENTF_LEFTUP
    elif button == "right":
        down = _win32con.MOUSEEVENTF_RIGHTDOWN
        up = _win32con.MOUSEEVENTF_RIGHTUP
    else:
        down = _win32con.MOUSEEVENTF_MIDDLEDOWN
        up = _win32con.MOUSEEVENTF_MIDDLEUP

    for _ in range(clicks):
        _win32api.mouse_event(down, x, y, 0, 0)
        time.sleep(0.02)
        _win32api.mouse_event(up, x, y, 0, 0)
        if clicks > 1:
            time.sleep(0.05)

def _win32_type_text(text: str, interval: float = 0.02) -> None:
    if _win32api is None or _win32con is None:
        raise RuntimeError("win32api not available")
        
    for char in text:
        # Simplistic approach for ASCII. Real unicode injection requires SendInput.
        vk = _win32api.VkKeyScan(char)
        if vk != -1:
            shift = (vk & 0x100) != 0
            code = vk & 0xFF
            if shift:
                _win32api.keybd_event(_win32con.VK_SHIFT, 0, 0, 0)
            _win32api.keybd_event(code, 0, 0, 0)
            _win32api.keybd_event(code, 0, _win32con.KEYEVENTF_KEYUP, 0)
            if shift:
                _win32api.keybd_event(_win32con.VK_SHIFT, 0, _win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(interval)
        else:
            # Fallback to clip/paste if character not found on keyboard
            try:
                import pyperclip
                pyperclip.copy(char)
                _win32api.keybd_event(_win32con.VK_CONTROL, 0, 0, 0)
                _win32api.keybd_event(ord('V'), 0, 0, 0)
                _win32api.keybd_event(ord('V'), 0, _win32con.KEYEVENTF_KEYUP, 0)
                _win32api.keybd_event(_win32con.VK_CONTROL, 0, _win32con.KEYEVENTF_KEYUP, 0)
                time.sleep(interval)
            except ImportError:
                pass


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
        backend: str = "pyautogui",
    ) -> None:
        self._resize_factor = resize_factor
        self._monitor_left = monitor_left
        self._monitor_top = monitor_top
        self._backend = backend.lower()
        self._dpi_aware = _setup_dpi_awareness()
        
        self._available = False
        if self._backend == "pywin32":
            self._available = _ensure_win32()
            if not self._available:
                log.warning("screen_nav.pywin32_unavailable", note="Falling back to pyautogui")
                self._backend = "pyautogui"
        
        if self._backend == "pyautogui":
            self._available = _ensure_pyautogui()
            
        self._rate_limiter = _RateLimiter(max_actions=15, window_seconds=5.0)
        self._action_log: list[ActionLogEntry] = []
        self._max_log_entries = 200

        if self._available:
            log.info(
                "screen_nav.init",
                resize_factor=resize_factor,
                dpi_aware=self._dpi_aware,
                backend=self._backend,
            )
        else:
            log.warning(
                "screen_nav.unavailable",
                note="Install required dependencies (pyautogui or pywin32)",
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
        if not self._available:
            return {
                "success": False,
                "error": f"Screen navigator unavailable — install dependencies for {self._backend}",
            }
        if self._backend == "pyautogui" and _pyautogui is None:
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

    async def click_absolute(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> dict[str, Any]:
        """Click at absolute screen coordinates (no coordinate mapping).

        Used by smart_click when the computer-use model returns coordinates
        based on a full-resolution screenshot that maps 1:1 to the real screen.
        """
        if err := self._check_available():
            return err
        if err := self._check_rate_limit():
            return err

        if button not in ("left", "right", "middle"):
            button = "left"
        clicks = max(1, min(clicks, 3))

        loop = asyncio.get_running_loop()
        try:
            if self._backend == "pywin32":
                await loop.run_in_executor(None, _win32_click, x, y, button, clicks)
            else:
                await loop.run_in_executor(
                    None, lambda: _safe_execute(_pyautogui.click, x, y, button=button, clicks=clicks)
                )
        except _FailSafeAbort as e:
            self._log_action("click_absolute", (x, y), (x, y), error="failsafe")
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

        self._log_action("click_absolute", (x, y), (x, y), button=button, clicks=clicks)
        return {
            "success": True,
            "action": "click",
            "screenshot_coords": [x, y],
            "real_coords": [x, y],
            "button": button,
            "clicks": clicks,
        }

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
            if self._backend == "pywin32":
                await loop.run_in_executor(None, _win32_click, rx, ry, button, clicks)
            else:
                await loop.run_in_executor(
                    None, lambda: _safe_execute(_pyautogui.click, rx, ry, button=button, clicks=clicks)
                )
        except _FailSafeAbort as e:
            self._log_action("click", (x, y), (rx, ry), error="failsafe")
            return {"success": False, "error": str(e)}
        except Exception as e:
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
            if self._backend == "pywin32":
                await loop.run_in_executor(None, _win32_type_text, text, interval)
            else:
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
        except Exception as e:
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

        If this succeeds but the window still isn't visible, use list_all_windows to check its state.

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
            try:
                windows = _pygetwindow.getWindowsWithTitle(title_contains)
                if not windows:
                    return None
                w = windows[0]
                
                try:
                    if w.isMinimized:
                        w.restore()
                except Exception as e:
                    log.debug("navigator.focus_window.restore_failed", error=str(e))
                
                try:
                    w.activate()
                except Exception as e:
                    # Often fails due to 'Focus Stealing Prevention' or PyGetWindow 'Error 0' bug
                    log.debug("navigator.focus_window.activate_failed", title=w.title, error=str(e))
                
                return w.title
            except Exception as e:
                log.error("navigator.focus_window.error", query=title_contains, error=str(e))
                return None

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

    # ------------------------------------------------------------------
    # Actions — Phase 4: Comprehensive Windows Control
    # ------------------------------------------------------------------

    async def open_application(
        self,
        name_or_path: str,
    ) -> dict[str, Any]:
        """Open an application by name or executable path.

        Supports:
          - Common app names: "notepad", "chrome", "firefox", "explorer",
            "cmd", "powershell", "code" (VS Code), "calc", "paint"
          - Full executable paths: "C:\\\\Program Files\\\\...\\\\app.exe"
          - File associations via os.startfile: "document.pdf", "image.png"
          - URLs: "https://google.com" (opens in default browser)

        IMPORTANT: If this returns success but the app isn't visible yet, it may be loading.
        Wait a moment and use `focus_window` or `list_all_windows` to verify.

        Args:
            name_or_path: Application name, path, file, or URL.
        """
        loop = asyncio.get_running_loop()

        # Well-known app aliases (Windows)
        _APP_ALIASES = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "paint": "mspaint.exe",
            "explorer": "explorer.exe",
            "file explorer": "explorer.exe",
            "cmd": "cmd.exe",
            "command prompt": "cmd.exe",
            "terminal": "wt.exe",
            "powershell": "powershell.exe",
            "task manager": "taskmgr.exe",
            "taskmgr": "taskmgr.exe",
            "control panel": "control.exe",
            "settings": "ms-settings:",
            "snipping tool": "snippingtool.exe",
            "snip": "snippingtool.exe",
            "chrome": "chrome.exe",
            "google chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "edge": "msedge.exe",
            "microsoft edge": "msedge.exe",
            "code": "code.exe",
            "vscode": "code.exe",
            "vs code": "code.exe",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "outlook": "outlook.exe",
            "teams": "ms-teams:",
            "discord": "discord.exe",
            "slack": "slack.exe",
            # UWP / Microsoft Store apps — use URI protocol handlers
            "whatsapp": "whatsapp:",
            "spotify": "spotify:",
            "xbox": "xbox:",
            "xbox game bar": "xbox:",
            "store": "ms-windows-store:",
            "microsoft store": "ms-windows-store:",
            "mail": "outlookmail:",
            "calendar": "outlookcal:",
            "maps": "bingmaps:",
            "photos": "ms-photos:",
            "camera": "microsoft.windows.camera:",
            "clock": "ms-clock:",
            "alarms": "ms-clock:",
            "weather": "msnweather:",
            "news": "msn-news:",
            "voice recorder": "ms-callrecording:",
            "feedback hub": "feedback-hub:",
            "tips": "ms-get-started:",
        }

        def _open():
            target = name_or_path.strip()
            target_lower = target.lower()

            # Normalize natural launch phrases from model/user prompts.
            for prefix in ("open ", "launch ", "start ", "run "):
                if target_lower.startswith(prefix):
                    target = target[len(prefix):].strip()
                    target_lower = target.lower()
                    break
            if target_lower.startswith("the "):
                target = target[4:].strip()
                target_lower = target.lower()
            if target_lower.endswith(" app"):
                target = target[:-4].strip()
                target_lower = target.lower()
            elif target_lower.endswith(" application"):
                target = target[:-12].strip()
                target_lower = target.lower()

            resolved = _APP_ALIASES.get(target_lower, target)

            # Check for existing browser windows to reuse if possible
            lower_target = target_lower
            if any(b in lower_target for b in ("chrome", "google chrome", "browser", "edge", "msedge", "brave")):
                if _ensure_pygetwindow():
                    try:
                        all_windows = _pygetwindow.getAllWindows()
                        # Simple heuristic: find window containing target name or browser keywords
                        found = [w for w in all_windows if w.title and w.visible and any(
                            b in w.title.lower() for b in ("chrome", "edge", "brave", "chromium")
                        )]
                        if found:
                            log.info("open_application.reusing_window", title=found[0].title)
                            # We can't easily await focus_window here as we are in a sync _open block
                            # but we can do a best-effort activation.
                            try:
                                if found[0].isMinimized:
                                    found[0].restore()
                                found[0].activate()
                            except Exception:
                                pass
                            return found[0].title, "reuse", None
                    except Exception:
                        pass

            is_http_url = resolved.startswith(("http://", "https://"))
            # Generic URI detection: scheme:... (but not Windows drive path like C:\)
            is_uri = bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", resolved)) and not os.path.isabs(resolved)

            # URLs or custom protocol links: use OS handlers
            if is_http_url or is_uri:
                if platform.system() == "Windows":
                    # Check if we should override the default browser for URLs
                    try:
                        from config import RioConfig
                        cfg = RioConfig.load()
                        preferred = cfg.browser.default_browser
                        if preferred == "chrome" and is_http_url:
                            subprocess.Popen(f'start chrome "{resolved}"', shell=True)
                            return resolved, "url", None
                        elif preferred == "edge" and is_http_url:
                            subprocess.Popen(f'start msedge "{resolved}"', shell=True)
                            return resolved, "url", None
                    except Exception:
                        pass
                    
                    os.startfile(resolved)
                else:
                    subprocess.Popen(["xdg-open", resolved])
                return resolved, ("url" if is_http_url else "protocol"), None

            # If it looks like a file path with extension (not .exe), use startfile
            if (
                os.path.splitext(resolved)[1]
                and os.path.splitext(resolved)[1].lower() != ".exe"
                and os.path.exists(resolved)
            ):
                if platform.system() == "Windows":
                    os.startfile(resolved)
                else:
                    subprocess.Popen(["xdg-open", resolved])
                return resolved, "file", None

            # Launch as executable — capture stderr for error detection
            try:
                proc = subprocess.Popen(
                    resolved,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                # Quick check: did cmd.exe fail immediately?
                import time as _time
                _time.sleep(0.3)
                rc = proc.poll()
                if rc is not None and rc != 0:
                    # Direct exe failed — try Windows Start Menu search
                    if platform.system() == "Windows":
                        proc2 = subprocess.Popen(
                            f'start "" "{resolved}"',
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                        _time.sleep(0.3)
                        rc2 = proc2.poll()
                        if rc2 is None or rc2 == 0:
                            return resolved, "start", proc2
                        # Start also failed — try PowerShell UWP lookup
                        ps_cmd = (
                            f'powershell -NoProfile -Command "'
                            f'$pkg = Get-AppxPackage | Where-Object {{$_.Name -like \"*{target}*\"}} | Select-Object -First 1; '
                            f'if ($pkg) {{ Start-Process \"shell:AppsFolder\\$($pkg.PackageFamilyName)!App\" }}'
                            f' else {{ exit 1 }}"'
                        )
                        proc3 = subprocess.Popen(
                            ps_cmd,
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                        return resolved, "uwp", proc3
                return resolved, "exe", proc
            except FileNotFoundError:
                if platform.system() == "Windows":
                    proc = subprocess.Popen(
                        f'start "" "{resolved}"',
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    return resolved, "start", proc
                raise

        def _verify_launch(proc, resolved, method):
            """Verify the process actually launched successfully."""
            verification = {"verified": False}

            # For URL/file launches via os.startfile, we can't easily verify
            if proc is None:
                verification["method"] = "startfile"
                # Best-effort window verification for known protocol launches.
                if _ensure_pygetwindow():
                    try:
                        time.sleep(0.8)
                        all_windows = _pygetwindow.getAllWindows()
                        titles = [w.title for w in all_windows if w.title and w.visible]
                        normalized = str(resolved).lower()
                        if "whatsapp" in normalized:
                            matched = [t for t in titles if "whatsapp" in t.lower()]
                            if matched:
                                verification["verified"] = True
                                verification["window_title"] = matched[0]
                                verification["note"] = "WhatsApp window detected."
                                return verification
                        if "explorer" in normalized:
                            matched = [
                                t for t in titles
                                if any(k in t.lower() for k in ("file explorer", "this pc", "quick access", "downloads", "documents"))
                            ]
                            if matched:
                                verification["verified"] = True
                                verification["window_title"] = matched[0]
                                verification["note"] = "File Explorer window detected."
                                return verification
                    except Exception:
                        pass
                verification["note"] = "Launched via OS handler — cannot verify process directly."
                return verification

            # Wait briefly for the process to either start or fail
            import time
            time.sleep(1.0)

            exit_code = proc.poll()
            if exit_code is not None and exit_code != 0:
                # Process exited with error
                stderr_output = ""
                try:
                    stderr_output = proc.stderr.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
                verification["failed"] = True
                verification["exit_code"] = exit_code
                if stderr_output:
                    verification["stderr"] = stderr_output[:500]
                return verification

            # Check if a new window appeared (best-effort)
            if _ensure_pygetwindow():
                try:
                    # Look for windows matching the resolved app name
                    app_base = os.path.splitext(os.path.basename(resolved))[0].lower()
                    all_windows = _pygetwindow.getAllWindows()
                    matching = [
                        w for w in all_windows
                        if w.title and w.visible and app_base in w.title.lower()
                    ]
                    if not matching and app_base == "explorer":
                        matching = [
                            w for w in all_windows
                            if w.title and w.visible and any(
                                k in w.title.lower() for k in ("file explorer", "this pc", "quick access", "downloads", "documents")
                            )
                        ]
                    if matching:
                        verification["verified"] = True
                        verification["window_title"] = matching[0].title
                    else:
                        verification["note"] = (
                            "Process started but no matching window found yet. "
                            "App may still be loading."
                        )
                except Exception:
                    pass

            # If process is still running, that's a good sign
            if exit_code is None:
                verification["process_running"] = True
                if not verification.get("verified"):
                    verification["verified"] = True
                    verification["note"] = "Process is running."

            return verification

        try:
            resolved_name, method, proc = await loop.run_in_executor(None, _open)
            verification = await loop.run_in_executor(
                None, _verify_launch, proc, resolved_name, method,
            )

            # Check if launch actually failed
            if verification.get("failed"):
                self._log_action("open_application", details={
                    "target": name_or_path, "resolved": resolved_name,
                    "method": method, "verification": verification,
                })
                error_msg = f"Application '{name_or_path}' failed to launch"
                if verification.get("stderr"):
                    error_msg += f": {verification['stderr']}"
                if verification.get("exit_code") is not None:
                    error_msg += f" (exit code {verification['exit_code']})"
                return {"success": False, "error": error_msg}

            self._log_action("open_application", details={
                "target": name_or_path, "resolved": resolved_name,
                "method": method, "verification": verification,
            })
            return {
                "success": True,
                "action": "open_application",
                "target": name_or_path,
                "resolved": resolved_name,
                "method": method,
                "verified": verification.get("verified", False),
                "window_title": verification.get("window_title"),
                "note": verification.get("note", "Application launched and verified."),
            }
        except Exception as exc:
            return {"success": False, "error": f"Failed to open '{name_or_path}': {exc}"}

    async def list_all_windows(self) -> dict[str, Any]:
        """List all visible windows on the desktop.

        Returns a list of window info dicts with title, position, size,
        and state. Uses pygetwindow with win32gui fallback for more detail.
        """
        if not _ensure_pygetwindow():
            return {"success": False, "error": "Window management unavailable — install pygetwindow"}

        loop = asyncio.get_running_loop()

        def _list():
            all_windows = _pygetwindow.getAllWindows()
            results = []
            for w in all_windows:
                # Skip invisible/untitled windows
                if not w.title or not w.title.strip():
                    continue
                if not w.visible:
                    continue
                results.append({
                    "title": w.title,
                    "position": [w.left, w.top],
                    "size": [w.width, w.height],
                    "minimized": w.isMinimized,
                    "maximized": w.isMaximized,
                    "active": w.isActive,
                })
            return results

        windows = await loop.run_in_executor(None, _list)
        self._log_action("list_all_windows", details={"count": len(windows)})
        return {
            "success": True,
            "action": "list_all_windows",
            "windows": windows,
            "count": len(windows),
        }

    async def get_active_window(self) -> dict[str, Any]:
        """Get information about the currently active (foreground) window."""
        if not _ensure_pygetwindow():
            return {"success": False, "error": "Window management unavailable — install pygetwindow"}

        loop = asyncio.get_running_loop()

        def _get():
            w = _pygetwindow.getActiveWindow()
            if w is None:
                return None
            return {
                "title": w.title,
                "position": [w.left, w.top],
                "size": [w.width, w.height],
                "minimized": w.isMinimized,
                "maximized": w.isMaximized,
            }

        info = await loop.run_in_executor(None, _get)
        if info is None:
            return {"success": False, "error": "No active window detected"}

        self._log_action("get_active_window", details={"title": info["title"]})
        return {"success": True, "action": "get_active_window", **info}

    async def minimize_window(self, title_contains: str) -> dict[str, Any]:
        """Minimize a window by title substring."""
        if not _ensure_pygetwindow():
            return {"success": False, "error": "Window management unavailable — install pygetwindow"}

        loop = asyncio.get_running_loop()

        def _minimize():
            windows = _pygetwindow.getWindowsWithTitle(title_contains)
            if not windows:
                return None
            w = windows[0]
            w.minimize()
            return w.title

        title = await loop.run_in_executor(None, _minimize)
        if title is None:
            return {"success": False, "error": f"No window found matching '{title_contains}'"}

        self._log_action("minimize_window", details={"title": title})
        return {"success": True, "action": "minimize_window", "title": title}

    async def maximize_window(self, title_contains: str) -> dict[str, Any]:
        """Maximize a window by title substring."""
        if not _ensure_pygetwindow():
            return {"success": False, "error": "Window management unavailable — install pygetwindow"}

        loop = asyncio.get_running_loop()

        def _maximize():
            windows = _pygetwindow.getWindowsWithTitle(title_contains)
            if not windows:
                return None
            w = windows[0]
            w.maximize()
            return w.title

        title = await loop.run_in_executor(None, _maximize)
        if title is None:
            return {"success": False, "error": f"No window found matching '{title_contains}'"}

        self._log_action("maximize_window", details={"title": title})
        return {"success": True, "action": "maximize_window", "title": title}

    async def close_window(self, title_contains: str) -> dict[str, Any]:
        """Close a window by title substring."""
        if not _ensure_pygetwindow():
            return {"success": False, "error": "Window management unavailable — install pygetwindow"}

        loop = asyncio.get_running_loop()

        def _close():
            windows = _pygetwindow.getWindowsWithTitle(title_contains)
            if not windows:
                return None
            w = windows[0]
            title = w.title
            w.close()
            return title

        title = await loop.run_in_executor(None, _close)
        if title is None:
            return {"success": False, "error": f"No window found matching '{title_contains}'"}

        self._log_action("close_window", details={"title": title})
        return {"success": True, "action": "close_window", "title": title}

    async def resize_window(
        self, title_contains: str, width: int, height: int,
    ) -> dict[str, Any]:
        """Resize a window by title substring.

        Args:
            title_contains: Substring of the window title.
            width: New width in pixels.
            height: New height in pixels.
        """
        if not _ensure_pygetwindow():
            return {"success": False, "error": "Window management unavailable — install pygetwindow"}

        loop = asyncio.get_running_loop()

        def _resize():
            windows = _pygetwindow.getWindowsWithTitle(title_contains)
            if not windows:
                return None
            w = windows[0]
            w.resizeTo(width, height)
            return w.title

        title = await loop.run_in_executor(None, _resize)
        if title is None:
            return {"success": False, "error": f"No window found matching '{title_contains}'"}

        self._log_action("resize_window", details={"title": title, "width": width, "height": height})
        return {"success": True, "action": "resize_window", "title": title, "width": width, "height": height}

    async def move_window(
        self, title_contains: str, x: int, y: int,
    ) -> dict[str, Any]:
        """Move a window to a specific position on screen.

        Args:
            title_contains: Substring of the window title.
            x: New X position.
            y: New Y position.
        """
        if not _ensure_pygetwindow():
            return {"success": False, "error": "Window management unavailable — install pygetwindow"}

        loop = asyncio.get_running_loop()

        def _move():
            windows = _pygetwindow.getWindowsWithTitle(title_contains)
            if not windows:
                return None
            w = windows[0]
            w.moveTo(x, y)
            return w.title

        title = await loop.run_in_executor(None, _move)
        if title is None:
            return {"success": False, "error": f"No window found matching '{title_contains}'"}

        self._log_action("move_window", details={"title": title, "x": x, "y": y})
        return {"success": True, "action": "move_window", "title": title, "x": x, "y": y}

    # ------------------------------------------------------------------
    # Actions — Phase 5: Process Management (psutil)
    # ------------------------------------------------------------------

    async def list_processes(self, name_filter: str = "") -> dict[str, Any]:
        """List running processes, optionally filtered by name.

        Args:
            name_filter: Only show processes containing this string (case-insensitive).
        """
        if not _ensure_psutil():
            return {"success": False, "error": "Process management unavailable — install psutil"}

        loop = asyncio.get_running_loop()

        def _list():
            results = []
            for proc in _psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                try:
                    info = proc.info
                    pname = info.get("name", "")
                    if name_filter and name_filter.lower() not in pname.lower():
                        continue
                    mem = info.get("memory_info")
                    results.append({
                        "pid": info["pid"],
                        "name": pname,
                        "memory_mb": round(mem.rss / (1024 * 1024), 1) if mem else 0,
                    })
                except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                    continue
            # Sort by memory descending, limit to top 50
            results.sort(key=lambda p: p["memory_mb"], reverse=True)
            return results[:50]

        processes = await loop.run_in_executor(None, _list)
        self._log_action("list_processes", details={"filter": name_filter, "count": len(processes)})
        return {
            "success": True,
            "action": "list_processes",
            "processes": processes,
            "count": len(processes),
        }

    async def kill_process(
        self, name_or_pid: str,
    ) -> dict[str, Any]:
        """Kill a process by name or PID.

        Safety: Will not kill critical system processes (csrss, lsass, svchost, etc.)

        Args:
            name_or_pid: Process name (e.g. "chrome") or PID as string.
        """
        if not _ensure_psutil():
            return {"success": False, "error": "Process management unavailable — install psutil"}

        # Protected system processes
        PROTECTED = frozenset({
            "system", "csrss.exe", "lsass.exe", "smss.exe", "svchost.exe",
            "winlogon.exe", "services.exe", "wininit.exe", "dwm.exe",
            "explorer.exe",  # Don't kill Windows shell
        })

        loop = asyncio.get_running_loop()

        def _kill():
            target = name_or_pid.strip()
            killed = []

            # Try as PID first
            try:
                pid = int(target)
                proc = _psutil.Process(pid)
                pname = proc.name()
                if pname.lower() in PROTECTED:
                    return None, f"Cannot kill protected system process: {pname}"
                proc.terminate()
                return [{"pid": pid, "name": pname}], None
            except (ValueError, _psutil.NoSuchProcess):
                pass

            # Kill by name
            for proc in _psutil.process_iter(["pid", "name"]):
                try:
                    pname = proc.info["name"] or ""
                    if target.lower() in pname.lower():
                        if pname.lower() in PROTECTED:
                            continue
                        proc.terminate()
                        killed.append({"pid": proc.info["pid"], "name": pname})
                except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                    continue

            if not killed:
                return None, f"No process found matching '{target}'"
            return killed, None

        killed, error = await loop.run_in_executor(None, _kill)
        if error:
            return {"success": False, "error": error}

        self._log_action("kill_process", details={"target": name_or_pid, "killed": len(killed)})
        return {
            "success": True,
            "action": "kill_process",
            "killed": killed,
            "count": len(killed),
        }

    # ------------------------------------------------------------------
    # Actions — Phase 6: Clipboard Operations
    # ------------------------------------------------------------------

    async def get_clipboard(self) -> dict[str, Any]:
        """Get the current clipboard text content."""
        loop = asyncio.get_running_loop()

        def _get():
            try:
                import pyperclip
                return pyperclip.paste()
            except ImportError:
                # Fallback: use PowerShell on Windows
                if platform.system() == "Windows":
                    result = subprocess.run(
                        ["powershell", "-command", "Get-Clipboard"],
                        capture_output=True, text=True, timeout=5,
                    )
                    return result.stdout.strip()
                return None

        text = await loop.run_in_executor(None, _get)
        if text is None:
            return {"success": False, "error": "Clipboard access unavailable — install pyperclip"}

        self._log_action("get_clipboard", details={"length": len(text)})
        return {
            "success": True,
            "action": "get_clipboard",
            "text": text[:10000],  # Limit to 10K chars
            "length": len(text),
        }

    async def set_clipboard(self, text: str) -> dict[str, Any]:
        """Set the clipboard text content.

        Args:
            text: Text to copy to the clipboard.
        """
        loop = asyncio.get_running_loop()

        def _set():
            try:
                import pyperclip
                pyperclip.copy(text)
                return True
            except ImportError:
                # Fallback: use PowerShell on Windows
                if platform.system() == "Windows":
                    # Use stdin pipe to avoid shell injection
                    proc = subprocess.run(
                        ["powershell", "-command", "Set-Clipboard -Value $input"],
                        input=text, capture_output=True, text=True, timeout=5,
                    )
                    return proc.returncode == 0
                return False

        success = await loop.run_in_executor(None, _set)
        if not success:
            return {"success": False, "error": "Clipboard access unavailable — install pyperclip"}

        self._log_action("set_clipboard", details={"length": len(text)})
        return {
            "success": True,
            "action": "set_clipboard",
            "length": len(text),
        }

    # ------------------------------------------------------------------
    # Actions — Phase 7: Screen Info
    # ------------------------------------------------------------------

    async def get_screen_info(self) -> dict[str, Any]:
        """Get information about all monitors: resolution, DPI, bounds."""
        loop = asyncio.get_running_loop()

        def _info():
            monitors = []
            try:
                import mss
                with mss.mss() as sct:
                    for i, m in enumerate(sct.monitors):
                        if i == 0:
                            continue  # Skip the "all monitors" entry
                        monitors.append({
                            "index": i,
                            "left": m["left"],
                            "top": m["top"],
                            "width": m["width"],
                            "height": m["height"],
                        })
            except ImportError:
                pass

            # DPI info on Windows
            dpi = None
            if platform.system() == "Windows":
                try:
                    import ctypes
                    hdc = ctypes.windll.user32.GetDC(0)
                    dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
                    ctypes.windll.user32.ReleaseDC(0, hdc)
                except Exception:
                    pass

            return {
                "monitors": monitors,
                "monitor_count": len(monitors),
                "dpi": dpi,
                "platform": platform.system(),
            }

        info = await loop.run_in_executor(None, _info)
        self._log_action("get_screen_info", details=info)
        return {"success": True, "action": "get_screen_info", **info}
