"""
Rio Local — WebSocket Client

Maintains a persistent WebSocket connection to the Rio cloud backend with:
  - Exponential backoff reconnection (1s -> 2s -> 4s -> ... -> 30s cap)
  - Heartbeat pings every 10s to keep the connection alive
  - Connection state tracking
  - Pluggable callbacks: on_connect, on_disconnect, on_message
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum, auto
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import structlog
import websockets
from websockets.asyncio.client import ClientConnection, connect

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------

class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()


# ---------------------------------------------------------------------------
# Type aliases for callbacks
# ---------------------------------------------------------------------------

OnConnect = Callable[[], Awaitable[None] | None]
OnDisconnect = Callable[[], Awaitable[None] | None]
OnMessage = Callable[[dict | bytes], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# WSClient
# ---------------------------------------------------------------------------

class WSClient:
    """Async WebSocket client for the Rio cloud backend.

    Usage::

        client = WSClient("ws://localhost:8080/ws/rio/live")
        await client.connect()

        # Send
        await client.send_text("hello from Rio")

        # Receive (async generator)
        async for msg in client.receive():
            print(msg)

        await client.close()
    """

    # Reconnect parameters
    _INITIAL_BACKOFF: float = 1.0
    _MAX_BACKOFF: float = 30.0
    _HEARTBEAT_INTERVAL: float = 5.0

    def __init__(
        self,
        url: str,
        *,
        on_connect: Optional[OnConnect] = None,
        on_disconnect: Optional[OnDisconnect] = None,
        on_message: Optional[OnMessage] = None,
    ) -> None:
        self._url = url
        self._ws: Optional[ClientConnection] = None
        self._state = ConnectionState.DISCONNECTED
        self._should_reconnect = True
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Pluggable callbacks
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._on_message = on_message

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state is ConnectionState.CONNECTED

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish the WebSocket connection (with reconnect loop)."""
        self._should_reconnect = True
        backoff = self._INITIAL_BACKOFF

        while self._should_reconnect:
            self._state = ConnectionState.CONNECTING
            log.info("ws.connecting", url=self._url)

            try:
                self._ws = await connect(
                    self._url,
                    ping_interval=None,   # we manage our own heartbeat
                    ping_timeout=None,
                    close_timeout=5,
                    max_size=10 * 1024 * 1024,  # 10 MiB — room for screenshots
                )
                self._state = ConnectionState.CONNECTED
                backoff = self._INITIAL_BACKOFF  # reset on success
                log.info("ws.connected", url=self._url)
                self._start_heartbeat()
                await self._fire_callback(self._on_connect)
                return  # connection established — caller takes over

            except (OSError, websockets.exceptions.WebSocketException) as exc:
                self._state = ConnectionState.DISCONNECTED
                log.warning(
                    "ws.connect_failed",
                    error=str(exc),
                    retry_in=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._MAX_BACKOFF)

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send_text(self, text: str) -> None:
        """Send a JSON text frame.  *text* is the raw string payload."""
        self._ensure_connected()
        assert self._ws is not None
        log.debug("ws.send_text", length=len(text))
        try:
            await self._ws.send(text)
        except websockets.exceptions.ConnectionClosed as exc:
            self._state = ConnectionState.DISCONNECTED
            self._stop_heartbeat()
            raise ConnectionError(f"WebSocket closed: {exc}") from exc

    async def send_json(self, obj: dict[str, Any]) -> None:
        """Convenience: serialize *obj* as JSON and send as a text frame."""
        await self.send_text(json.dumps(obj))

    async def send_binary(self, data: bytes) -> None:
        """Send a binary frame (audio chunks, screenshots, etc.)."""
        self._ensure_connected()
        assert self._ws is not None
        log.debug("ws.send_binary", length=len(data))
        try:
            await self._ws.send(data)
        except websockets.exceptions.ConnectionClosed as exc:
            self._state = ConnectionState.DISCONNECTED
            self._stop_heartbeat()
            raise ConnectionError(f"WebSocket closed: {exc}") from exc

    async def send_json_resilient(self, obj: dict[str, Any], retries: int = 2) -> bool:
        """Send JSON with retry on transient failures.

        Returns True if sent successfully, False otherwise.
        Unlike ``send_json``, this never raises — it logs failures.
        """
        for attempt in range(retries + 1):
            try:
                await self.send_json(obj)
                return True
            except ConnectionError:
                if attempt < retries:
                    log.debug("ws.send_json_resilient.retry", attempt=attempt + 1)
                    await asyncio.sleep(0.5)
                else:
                    log.warning("ws.send_json_resilient.failed", attempts=retries + 1)
                    return False
            except Exception:
                log.exception("ws.send_json_resilient.error")
                return False
        return False

    # ------------------------------------------------------------------
    # Receive — async generator
    # ------------------------------------------------------------------

    async def receive(self) -> AsyncIterator[dict | bytes]:
        """Yield messages from the cloud, reconnecting on failure.

        Text frames are JSON-decoded into dicts; binary frames are
        yielded as raw bytes.
        """
        while self._should_reconnect:
            if not self.is_connected:
                await self.connect()

            assert self._ws is not None

            try:
                async for raw in self._ws:
                    if isinstance(raw, bytes):
                        msg: dict | bytes = raw
                    else:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            log.warning("ws.invalid_json", data=raw[:200])
                            continue

                    await self._fire_callback(self._on_message, msg)
                    yield msg

            except websockets.exceptions.ConnectionClosed as exc:
                log.warning("ws.disconnected", code=exc.code, reason=exc.reason)
            except Exception as exc:
                log.error("ws.receive_error", error=str(exc))

            # Connection dropped — clean up and retry
            self._state = ConnectionState.DISCONNECTED
            self._stop_heartbeat()
            await self._fire_callback(self._on_disconnect)

            if self._should_reconnect:
                log.info("ws.will_reconnect")

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully shut down the connection.  Disables auto-reconnect."""
        self._should_reconnect = False
        self._stop_heartbeat()

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass  # best-effort close

        self._state = ConnectionState.DISCONNECTED
        log.info("ws.closed")

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        self._stop_heartbeat()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Send a WebSocket ping every ``_HEARTBEAT_INTERVAL`` seconds.

        Tolerates up to 4 consecutive failures before marking the
        connection dead and triggering a reconnect.
        """
        consecutive_failures = 0
        try:
            while True:
                await asyncio.sleep(self._HEARTBEAT_INTERVAL)
                if self._ws is not None and self.is_connected:
                    try:
                        pong = await self._ws.ping()
                        await asyncio.wait_for(pong, timeout=8.0)
                        log.debug("ws.heartbeat_ok")
                        consecutive_failures = 0
                    except Exception as exc:
                        consecutive_failures += 1
                        log.warning("ws.heartbeat_failed", error=str(exc), failures=consecutive_failures)
                        if consecutive_failures >= 4:
                            log.warning("ws.heartbeat_dead", note="marking connection dead after 4 consecutive failures")
                            self._state = ConnectionState.DISCONNECTED
                            try:
                                await self._ws.close()
                            except Exception:
                                pass
                            await self._fire_callback(self._on_disconnect)
                            return
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if self._ws is None or not self.is_connected:
            raise ConnectionError("WebSocket is not connected")

    @staticmethod
    async def _fire_callback(cb: Optional[Callable], *args: Any) -> None:
        if cb is None:
            return
        result = cb(*args)
        if asyncio.iscoroutine(result):
            await result
