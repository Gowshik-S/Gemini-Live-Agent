"""
Session lifecycle manager for Rio cloud service.

Manages GeminiSession instances per connected client. Handles creation,
lookup, removal, heartbeat tracking, and auto-reconnect on timeout.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

import structlog

from gemini_session import GeminiSession

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HEARTBEAT_INTERVAL_S = 10
SESSION_TIMEOUT_S = 15 * 60  # 15 minutes -- reconnect before Gemini drops us
SESSION_MAX_LIFETIME_S = 30 * 60  # 30 minutes hard cap


class SessionManager:
    """Thread-safe manager for per-client GeminiSession instances."""

    def __init__(self, api_key: str, session_mode: str = "live",
                 text_model: str | None = None, live_model: str | None = None) -> None:
        self._api_key = api_key
        self._session_mode = session_mode
        self._text_model = text_model
        self._live_model = live_model
        self._sessions: Dict[str, GeminiSession] = {}
        self._heartbeats: Dict[str, float] = {}
        self._created_at: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._log = logger.bind(component="session_manager", session_mode=session_mode)
        # Reconnect notifications — main.py polls this to detect stale sessions
        self._reconnect_needed: Dict[str, bool] = {}
        self._reconnected_sessions: Dict[str, GeminiSession] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_session(
        self,
        client_id: str,
        mode: str | None = None,
    ) -> GeminiSession:
        """Create and connect a new GeminiSession for *client_id*.

        If a session already exists for this client, it is closed first.

        Args:
            client_id: Unique identifier for the client connection.
            mode: Session mode — ``'live'`` for bidirectional audio via
                  Live API, ``'text'`` for standard text-only fallback.
                  Defaults to the manager's configured session_mode.
        """
        if mode is None:
            mode = self._session_mode
        async with self._lock:
            # Tear down any stale session
            if client_id in self._sessions:
                self._log.info("session.replace_existing", client_id=client_id)
                await self._close_session_unlocked(client_id)

            session = GeminiSession(
                api_key=self._api_key,
                client_id=client_id,
                mode=mode,
                text_model=self._text_model,
                live_model=self._live_model,
            )
            await session.connect()

            now = time.monotonic()
            self._sessions[client_id] = session
            self._heartbeats[client_id] = now
            self._created_at[client_id] = now

            self._log.info("session.created", client_id=client_id, mode=mode)

            # Ensure the heartbeat monitor is running
            self._ensure_heartbeat_task()

            return session

    async def get_session(self, client_id: str) -> Optional[GeminiSession]:
        """Return the active session for *client_id*, or None."""
        async with self._lock:
            session = self._sessions.get(client_id)
            if session is not None:
                self._heartbeats[client_id] = time.monotonic()
            return session

    async def remove_session(self, client_id: str) -> None:
        """Close and remove the session for *client_id*."""
        async with self._lock:
            await self._close_session_unlocked(client_id)

    async def touch(self, client_id: str) -> None:
        """Record a heartbeat for *client_id*."""
        async with self._lock:
            if client_id in self._sessions:
                self._heartbeats[client_id] = time.monotonic()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Reconnect logic
    # ------------------------------------------------------------------

    async def reconnect_session(self, client_id: str) -> Optional[GeminiSession]:
        """Tear down an existing session and create a fresh one.

        System instruction is automatically restored because
        ``GeminiSession.connect()`` always injects it.
        Uses the manager's configured session_mode.

        After reconnect, the new session is flagged in
        ``_reconnect_needed`` so that ``main.py`` can pick it up
        and swap the stale reference + restart the relay task.
        """
        self._log.info("session.reconnect", client_id=client_id, mode=self._session_mode)
        async with self._lock:
            await self._close_session_unlocked(client_id)

        # create_session acquires its own lock — uses default session_mode
        new_session = await self.create_session(client_id)

        # Flag for main.py to pick up
        async with self._lock:
            self._reconnect_needed[client_id] = True
            self._reconnected_sessions[client_id] = new_session

        return new_session

    async def check_reconnect(self, client_id: str) -> Optional[GeminiSession]:
        """Check if a reconnect happened for *client_id*.

        Returns the new GeminiSession if a reconnect occurred (and
        clears the flag), or None if no reconnect is pending.
        Called by main.py's WS loop to swap session references.
        """
        async with self._lock:
            if self._reconnect_needed.get(client_id):
                self._reconnect_needed[client_id] = False
                new_session = self._reconnected_sessions.pop(client_id, None)
                if new_session:
                    self._log.info("session.reconnect.consumed", client_id=client_id)
                return new_session
        return None

    # ------------------------------------------------------------------
    # Heartbeat monitor
    # ------------------------------------------------------------------

    def _ensure_heartbeat_task(self) -> None:
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="session-heartbeat"
            )

    async def _heartbeat_loop(self) -> None:
        """Periodically check for timed-out sessions and reconnect them."""
        self._log.info("heartbeat.start")
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
                await self._check_timeouts()
        except asyncio.CancelledError:
            self._log.info("heartbeat.cancelled")

    async def _check_timeouts(self) -> None:
        now = time.monotonic()
        expired_ids: list[str] = []
        idle_ids: list[str] = []

        async with self._lock:
            for cid, last_hb in list(self._heartbeats.items()):
                age = now - self._created_at.get(cid, now)
                idle = now - last_hb

                # Hard lifetime cap -- close without reconnect
                if age >= SESSION_MAX_LIFETIME_S:
                    self._log.warning(
                        "session.max_lifetime_reached",
                        client_id=cid,
                        age_s=round(age),
                    )
                    expired_ids.append(cid)
                    continue

                # Idle timeout -- log warning but do NOT reconnect.
                # Reconnecting server-side orphans the client's relay task
                # and leaves the connection in a broken state.  Instead,
                # rely on the client heartbeat to keep the session alive.
                if idle >= SESSION_TIMEOUT_S:
                    self._log.warning(
                        "session.idle_timeout",
                        client_id=cid,
                        idle_s=round(idle),
                        note="Session idle but not reconnecting (would orphan relay)",
                    )

        # Remove expired sessions (hard lifetime cap -- no reconnect)
        for cid in expired_ids:
            try:
                await self.remove_session(cid)
            except Exception:
                self._log.exception("session.remove_failed", client_id=cid)

        # NOTE: Idle sessions are no longer auto-reconnected server-side.
        # The client is responsible for reconnecting if it detects staleness.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _close_session_unlocked(self, client_id: str) -> None:
        """Close & remove a session. Caller MUST hold ``self._lock``."""
        session = self._sessions.pop(client_id, None)
        self._heartbeats.pop(client_id, None)
        self._created_at.pop(client_id, None)
        if session is not None:
            try:
                await session.close()
            except Exception:
                self._log.exception("session.close_error", client_id=client_id)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully close all sessions and stop the heartbeat task."""
        self._log.info("session_manager.shutdown")
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for cid in list(self._sessions.keys()):
                await self._close_session_unlocked(cid)
