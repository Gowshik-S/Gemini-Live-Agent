"""
Rio Local — Windows Agent (pywinauto)

Provides structured interaction with native Windows applications:
- App lifecycle (launch, focus, minimize, close)
- Element introspection (find buttons, text fields, menus)
- Direct control (click element by name, fill text, select menu)
- Clipboard integration
- Fallback to pyautogui for apps pywinauto can't introspect.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

try:
    import pywinauto
    from pywinauto import Application, Desktop
    from pywinauto.findwindows import ElementNotFoundError, ElementAmbiguousError
    _PYWINAUTO_AVAILABLE = True
except ImportError:
    _PYWINAUTO_AVAILABLE = False

try:
    import pyautogui
    _PYAUTOGUI_AVAILABLE = True
except ImportError:
    _PYAUTOGUI_AVAILABLE = False


class WindowsAgent:
    """Structured Windows app automation using pywinauto.

    Usage::

        agent = WindowsAgent()
        await agent.focus_app("Notepad")
        await agent.type_in_control("Edit", "Hello, world!")
        windows = await agent.list_windows()
    """

    def __init__(self) -> None:
        self._log = log.bind(component="windows_agent")
        self._app: Optional[Application] = None

    @property
    def available(self) -> bool:
        return _PYWINAUTO_AVAILABLE

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    async def list_windows(self, title_filter: str = "") -> list[dict]:
        """List open top-level windows, optionally filtered by title substring."""
        if not _PYWINAUTO_AVAILABLE:
            return []

        def _list():
            desktop = Desktop(backend="uia")
            windows = desktop.windows()
            results = []
            for w in windows:
                title = w.window_text()
                if not title:
                    continue
                if title_filter and title_filter.lower() not in title.lower():
                    continue
                try:
                    rect = w.rectangle()
                    results.append({
                        "title": title,
                        "left": rect.left,
                        "top": rect.top,
                        "width": rect.width(),
                        "height": rect.height(),
                        "visible": w.is_visible(),
                    })
                except Exception:
                    results.append({"title": title, "visible": False})
            return results

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _list)

    async def focus_app(self, title_contains: str) -> dict:
        """Focus (bring to front) a window matching the title."""
        if not _PYWINAUTO_AVAILABLE:
            return {"success": False, "error": "pywinauto not installed"}

        def _focus():
            try:
                app = Application(backend="uia").connect(
                    title_re=f".*{title_contains}.*", timeout=3,
                )
                win = app.top_window()
                if win.is_minimized():
                    win.restore()
                win.set_focus()
                self._app = app
                return {"success": True, "title": win.window_text()}
            except ElementNotFoundError:
                return {"success": False, "error": f"No window matching '{title_contains}'"}
            except ElementAmbiguousError:
                return {"success": False, "error": f"Multiple windows match '{title_contains}'"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _focus)

    async def launch_app(self, path: str) -> dict:
        """Launch an application and wait for its main window."""
        if not _PYWINAUTO_AVAILABLE:
            return {"success": False, "error": "pywinauto not installed"}

        def _launch():
            try:
                app = Application(backend="uia").start(path, timeout=10)
                win = app.top_window()
                win.wait("ready", timeout=10)
                self._app = app
                return {"success": True, "title": win.window_text(), "pid": app.process}
            except Exception as e:
                return {"success": False, "error": str(e)}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _launch)

    async def close_app(self, title_contains: str) -> dict:
        """Close an application window gracefully."""
        if not _PYWINAUTO_AVAILABLE:
            return {"success": False, "error": "pywinauto not installed"}

        def _close():
            try:
                app = Application(backend="uia").connect(
                    title_re=f".*{title_contains}.*", timeout=3,
                )
                app.top_window().close()
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _close)

    # ------------------------------------------------------------------
    # Element interaction
    # ------------------------------------------------------------------

    async def click_control(self, control_name: str, title_contains: str = "") -> dict:
        """Click a UI control by its text or automation name."""
        if not _PYWINAUTO_AVAILABLE:
            return {"success": False, "error": "pywinauto not installed"}

        def _click():
            try:
                app = self._get_app(title_contains)
                if app is None:
                    return {"success": False, "error": "No connected app"}
                win = app.top_window()
                ctrl = win.child_window(title=control_name, found_index=0)
                ctrl.click_input()
                return {"success": True, "control": control_name}
            except ElementNotFoundError:
                return {"success": False, "error": f"Control '{control_name}' not found"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _click)

    async def type_in_control(
        self, control_name: str, text: str, title_contains: str = "",
    ) -> dict:
        """Type text into a named control (edit box, text field, etc.)."""
        if not _PYWINAUTO_AVAILABLE:
            return {"success": False, "error": "pywinauto not installed"}

        def _type():
            try:
                app = self._get_app(title_contains)
                if app is None:
                    return {"success": False, "error": "No connected app"}
                win = app.top_window()
                ctrl = win.child_window(title=control_name, found_index=0)
                ctrl.set_edit_text(text)
                return {"success": True, "control": control_name, "chars": len(text)}
            except ElementNotFoundError:
                return {"success": False, "error": f"Control '{control_name}' not found"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _type)

    async def get_controls(self, title_contains: str = "") -> list[dict]:
        """List all UI controls in the focused window (for introspection)."""
        if not _PYWINAUTO_AVAILABLE:
            return []

        def _list():
            try:
                app = self._get_app(title_contains)
                if app is None:
                    return []
                win = app.top_window()
                controls = []
                for child in win.descendants():
                    ctrl_type = child.friendly_class_name()
                    name = child.window_text()
                    if not name and ctrl_type in ("Static", ""):
                        continue
                    controls.append({
                        "type": ctrl_type,
                        "name": name[:80] if name else "",
                        "enabled": child.is_enabled(),
                        "visible": child.is_visible(),
                    })
                return controls[:50]  # cap to avoid flooding
            except Exception:
                return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _list)

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    async def clipboard_get(self) -> str:
        """Read text from the Windows clipboard."""
        def _get():
            try:
                import pywinauto.clipboard as cb
                return cb.GetData() or ""
            except Exception:
                return ""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _get)

    async def clipboard_set(self, text: str) -> bool:
        """Write text to the Windows clipboard."""
        def _set():
            try:
                import pywinauto.clipboard as cb
                cb.EmptyClipboard()
                cb.SetClipboardText(text)
                return True
            except Exception:
                return False
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _set)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_app(self, title_contains: str = "") -> Optional[Application]:
        """Get the app handle, connect if needed."""
        if title_contains:
            try:
                app = Application(backend="uia").connect(
                    title_re=f".*{title_contains}.*", timeout=3,
                )
                self._app = app
                return app
            except Exception:
                return None
        return self._app
