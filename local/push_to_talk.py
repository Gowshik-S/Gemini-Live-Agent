"""
Rio Local -- Push-to-Talk (F2 Hotkey)

Uses pynput to detect F2 key press/release for push-to-talk.
Thread-safe communication with the asyncio event loop.

Gotcha #7: pynput may fail on Wayland Linux. The module
gracefully returns None from create() in that case.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


class PushToTalk:
    """F2 push-to-talk hotkey listener.

    pynput runs in its own OS thread. This class bridges to asyncio.

    Usage::

        ptt = PushToTalk.create(key_name="f2")
        if ptt is not None:
            ptt.start(asyncio.get_running_loop())
            # In audio loop:
            if ptt.is_active:
                send(chunk)
            ptt.stop()
    """

    def __init__(self, key_name: str = "f2") -> None:
        self._key_name = key_name.lower()
        self._key = None  # resolved pynput Key object
        self._active = False
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._press_event: Optional[asyncio.Event] = None
        self._release_event: Optional[asyncio.Event] = None
        self._listener = None

        # Resolve the key at init time
        self._resolve_key()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, key_name: str = "f2") -> Optional["PushToTalk"]:
        """Factory that returns None if pynput is unavailable.

        This handles Wayland, missing X11, or pynput not installed.
        """
        try:
            from pynput import keyboard as _kb  # noqa: F401 -- test import
            instance = cls(key_name)
            if instance._key is None:
                log.warning("ptt.key_resolve_failed", key=key_name)
                return None
            return instance
        except ImportError:
            log.warning(
                "ptt.pynput_not_installed",
                note="Install pynput for push-to-talk. Using VAD-only mode.",
            )
            return None
        except Exception:
            log.exception(
                "ptt.init_failed",
                note="Push-to-talk unavailable. Using VAD-only mode.",
            )
            return None

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------

    def _resolve_key(self) -> None:
        """Map key name string (e.g. 'f2') to pynput Key enum."""
        try:
            from pynput.keyboard import Key
            # Try Key.f2, Key.f3, etc.
            self._key = getattr(Key, self._key_name, None)
            if self._key is None:
                log.warning(
                    "ptt.unknown_key",
                    key=self._key_name,
                    note="Expected f1-f12, ctrl, alt, shift, etc.",
                )
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Thread-safe check: is the PTT key currently held down?"""
        with self._lock:
            return self._active

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Begin listening for key events.

        Must be called from the event loop thread.
        """
        from pynput.keyboard import Listener

        self._loop = loop
        self._press_event = asyncio.Event()
        self._release_event = asyncio.Event()

        self._listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        log.info("ptt.started", key=self._key_name)

    def stop(self) -> None:
        """Stop the keyboard listener."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        with self._lock:
            self._active = False
        log.info("ptt.stopped")

    # ------------------------------------------------------------------
    # Async edge notifications
    # ------------------------------------------------------------------

    async def wait_for_press(self) -> None:
        """Block until the PTT key is pressed."""
        if self._press_event is not None:
            await self._press_event.wait()
            self._press_event.clear()

    async def wait_for_release(self) -> None:
        """Block until the PTT key is released."""
        if self._release_event is not None:
            await self._release_event.wait()
            self._release_event.clear()

    # ------------------------------------------------------------------
    # pynput callbacks (run in pynput's OS thread)
    # ------------------------------------------------------------------

    def _on_press(self, key) -> None:
        if key != self._key:
            return
        with self._lock:
            if self._active:
                return  # Ignore key-repeat events
            self._active = True
        # Notify the asyncio event loop
        if self._loop is not None and self._press_event is not None:
            self._loop.call_soon_threadsafe(self._press_event.set)

    def _on_release(self, key) -> None:
        if key != self._key:
            return
        with self._lock:
            if not self._active:
                return
            self._active = False
        # Notify the asyncio event loop
        if self._loop is not None and self._release_event is not None:
            self._loop.call_soon_threadsafe(self._release_event.set)
