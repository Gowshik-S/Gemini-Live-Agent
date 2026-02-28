"""
Rio Cloud Service -- FastAPI application entry point.

Endpoints:
  WS  /ws/rio/live   -- Main relay between local client and Gemini Live API
  WS  /ws/dashboard   -- Real-time dashboard broadcast
  GET /health         -- Health check
  GET /dashboard/*    -- Static files for the dashboard UI
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Set

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model_router import ModelRouter
from rate_limiter import Priority, RateLimiter
from session_manager import SessionManager

# ---------------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------------
load_dotenv()

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),  # DEBUG+
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SESSION_MODE = os.environ.get("SESSION_MODE", "live")  # "live" or "text"

# ---------------------------------------------------------------------------
# Shared singletons (created during lifespan)
# ---------------------------------------------------------------------------
session_manager: SessionManager | None = None
rate_limiter: RateLimiter = RateLimiter(budget_rpm=30)
model_router: ModelRouter = ModelRouter()

# Connected dashboard WebSocket clients
dashboard_clients: Set[WebSocket] = set()
dashboard_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global session_manager

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set -- Gemini sessions will fail")

    session_manager = SessionManager(api_key=GEMINI_API_KEY, session_mode=SESSION_MODE)
    logger.info("rio.startup", gemini_key_set=bool(GEMINI_API_KEY), session_mode=SESSION_MODE)

    yield  # --- application runs here ---

    logger.info("rio.shutdown")
    await session_manager.shutdown()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Rio Cloud Service",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS -- allow dashboard & local client origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static dashboard files (best-effort: directory may not exist yet)
_dashboard_dir = Path(__file__).resolve().parent.parent / "ui" / "dashboard"
if _dashboard_dir.is_dir():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_dashboard_dir), html=True),
        name="dashboard",
    )
else:
    logger.warning("dashboard.static_dir_missing", path=str(_dashboard_dir))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "rio-cloud",
        "version": "0.1.0",
        "sessions_active": session_manager.active_count if session_manager else 0,
        "rate_limiter": rate_limiter.get_usage(),
        "routing": model_router.get_routing_stats(),
    }


# ---------------------------------------------------------------------------
# Dashboard broadcast helpers
# ---------------------------------------------------------------------------
async def _broadcast_dashboard(payload: dict) -> None:
    """Send a JSON payload to every connected dashboard client."""
    if not dashboard_clients:
        return
    message = json.dumps(payload)
    stale: list[WebSocket] = []
    async with dashboard_lock:
        for ws in dashboard_clients:
            try:
                await ws.send_text(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            dashboard_clients.discard(ws)


# ---------------------------------------------------------------------------
# WS /ws/dashboard -- dashboard feed
# ---------------------------------------------------------------------------
@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket) -> None:
    await websocket.accept()
    async with dashboard_lock:
        dashboard_clients.add(websocket)
    logger.info("dashboard.connected", total=len(dashboard_clients))

    try:
        # Keep alive -- read pings/ignore data
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with dashboard_lock:
            dashboard_clients.discard(websocket)
        logger.info("dashboard.disconnected", total=len(dashboard_clients))


# ---------------------------------------------------------------------------
# WS /ws/rio/live -- main relay
# ---------------------------------------------------------------------------
@app.websocket("/ws/rio/live")
async def ws_rio_live(websocket: WebSocket) -> None:
    await websocket.accept()
    client_id = str(uuid.uuid4())
    log = logger.bind(client_id=client_id)
    log.info("client.connected")

    assert session_manager is not None, "SessionManager not initialised"

    # ---- Create Gemini session ----
    try:
        gemini = await session_manager.create_session(client_id)
        # Notify client that Gemini session is ready
        await websocket.send_text(json.dumps({
            "type": "control",
            "action": "connected",
            "detail": "Gemini Live session ready",
        }))
    except Exception:
        log.exception("client.session_create_failed")
        try:
            await websocket.send_text(json.dumps({
                "type": "control",
                "action": "error",
                "detail": "Failed to create Gemini session",
            }))
        except Exception:
            pass
        await websocket.close(code=1011, reason="Failed to create Gemini session")
        return

    # ---- Background task: read Gemini responses and forward to client ----
    async def _relay_gemini_to_client() -> None:
        """Per-request relay: calls Gemini and streams results to client.

        Handles the tool-calling loop:
          1. Call Gemini → receive text or tool_call(s)
          2. If tool_call: forward to local, wait for result, feed back, repeat
          3. If text: stream to client and finish

        Loops up to MAX_TOOL_ROUNDS times to prevent infinite tool chains.
        """
        MAX_TOOL_ROUNDS = 5

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                pending_calls: list[dict] = []

                async for item in gemini.receive():
                    if isinstance(item, str):
                        # Text response — send transcript to client
                        response_frame = {
                            "type": "transcript",
                            "speaker": "rio",
                            "text": item,
                        }
                        await websocket.send_text(json.dumps(response_frame))

                        # Broadcast to dashboard
                        await _broadcast_dashboard({
                            "type": "transcript",
                            "speaker": "rio",
                            "text": item,
                            "client_id": client_id,
                        })

                    elif isinstance(item, dict) and item.get("type") == "tool_call":
                        # Function call — forward to local client for execution
                        pending_calls.append(item)
                        log.info(
                            "relay.tool_call",
                            name=item.get("name"),
                            args_keys=list(item.get("args", {}).keys()),
                        )
                        await websocket.send_text(json.dumps(item))

                        # Broadcast tool call to dashboard
                        await _broadcast_dashboard({
                            "type": "dashboard",
                            "subtype": "tool_call",
                            "client_id": client_id,
                            "name": item.get("name"),
                            "args": item.get("args", {}),
                        })

                if not pending_calls:
                    # No tool calls — Gemini gave final text response. Done.
                    break

                # Wait for each tool result from the local client
                for call in pending_calls:
                    try:
                        result_frame = await asyncio.wait_for(
                            tool_result_queue.get(), timeout=60.0,
                        )
                        result_data = result_frame.get("result", {})

                        log.info(
                            "relay.tool_result",
                            name=call["name"],
                            success=result_data.get("success"),
                        )

                        # Feed result back to Gemini session
                        await gemini.send_tool_result(
                            call["name"], result_data,
                        )

                        # Broadcast result to dashboard
                        await _broadcast_dashboard({
                            "type": "dashboard",
                            "subtype": "tool_result",
                            "client_id": client_id,
                            "name": call["name"],
                            "success": result_data.get("success"),
                        })

                    except asyncio.TimeoutError:
                        log.error(
                            "relay.tool_result.timeout",
                            name=call["name"],
                        )
                        # Feed an error result so Gemini can continue
                        await gemini.send_tool_result(
                            call["name"],
                            {"success": False, "error": "Tool execution timed out (60s)"},
                        )
                        break

                # Loop back — call receive() again with tool results in history
                # Gemini will produce a follow-up response

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("relay.gemini_to_client.error")
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Gemini error: {exc}",
                }))
            except Exception:
                pass  # Client may already be disconnected

    relay_task: asyncio.Task[None] | None = None

    # Queue for tool results flowing back from local client
    tool_result_queue: asyncio.Queue[dict] = asyncio.Queue()

    # ---- Live mode relay (persistent, bidirectional) ----
    async def _relay_live_to_client() -> None:
        """Persistent relay for Live API mode.

        Continuously streams audio/text/tool_call events from the
        Gemini Live session to the local WebSocket client.  Runs for
        the entire lifetime of the connection (not per-request).
        """
        try:
            async for item in gemini.receive_live():
                item_type = item.get("type")

                if item_type == "audio":
                    # Audio response → binary 0x01 frame to local client
                    await websocket.send_bytes(b"\x01" + item["data"])

                elif item_type == "text":
                    text = item["text"]
                    await websocket.send_text(json.dumps({
                        "type": "transcript",
                        "speaker": "rio",
                        "text": text,
                    }))
                    await _broadcast_dashboard({
                        "type": "transcript",
                        "speaker": "rio",
                        "text": text,
                        "client_id": client_id,
                    })

                elif item_type == "tool_call":
                    log.info(
                        "relay.live.tool_call",
                        name=item.get("name"),
                        args_keys=list(item.get("args", {}).keys()),
                    )
                    await websocket.send_text(json.dumps(item))
                    await _broadcast_dashboard({
                        "type": "dashboard",
                        "subtype": "tool_call",
                        "client_id": client_id,
                        "name": item.get("name"),
                        "args": item.get("args", {}),
                    })

                    # Wait for tool result from local client
                    try:
                        result_frame = await asyncio.wait_for(
                            tool_result_queue.get(), timeout=60.0,
                        )
                        result_data = result_frame.get("result", {})
                        log.info(
                            "relay.live.tool_result",
                            name=item["name"],
                            success=result_data.get("success"),
                        )
                        await gemini.send_tool_result(
                            item["name"], result_data,
                        )
                        await _broadcast_dashboard({
                            "type": "dashboard",
                            "subtype": "tool_result",
                            "client_id": client_id,
                            "name": item["name"],
                            "success": result_data.get("success"),
                        })
                    except asyncio.TimeoutError:
                        log.error(
                            "relay.live.tool_result.timeout",
                            name=item["name"],
                        )
                        await gemini.send_tool_result(
                            item["name"],
                            {"success": False, "error": "Tool execution timed out (60s)"},
                        )

                elif item_type == "turn_complete":
                    await websocket.send_text(json.dumps({
                        "type": "control",
                        "action": "turn_complete",
                    }))

                elif item_type == "setup_complete":
                    log.info("relay.live.setup_complete")
                    await websocket.send_text(json.dumps({
                        "type": "control",
                        "action": "live_ready",
                        "detail": "Live API session fully initialized",
                    }))

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("relay.live.error")
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Live relay error: {exc}",
                }))
            except Exception:
                pass

    # Start persistent relay if in Live mode
    if gemini.mode == "live":
        relay_task = asyncio.create_task(
            _relay_live_to_client(),
            name=f"live-relay-{client_id}",
        )
        log.info("relay.live.started")

    # ---- Main loop: read from client WebSocket (text + binary) ----
    try:
        while True:
            ws_msg = await websocket.receive()

            # Handle disconnect
            if ws_msg["type"] == "websocket.disconnect":
                break

            # ---- Binary frames (audio / image per wire protocol) ----
            raw_bytes = ws_msg.get("bytes")
            if raw_bytes and len(raw_bytes) > 1:
                prefix = raw_bytes[0:1]
                payload = raw_bytes[1:]

                if prefix == b"\x01":
                    # Audio frame — relay PCM to Gemini Live session
                    log.debug("client.audio", bytes_len=len(payload))
                    try:
                        await gemini.send_audio(payload)
                    except Exception:
                        log.exception("client.audio.send_error")

                elif prefix == b"\x02":
                    # Image frame — L2: forward screenshot to Gemini session
                    log.debug("client.image", bytes_len=len(payload))
                    try:
                        await gemini.send_image(payload, mime_type="image/jpeg")

                        # Broadcast to dashboard
                        await _broadcast_dashboard({
                            "type": "dashboard",
                            "subtype": "vision",
                            "client_id": client_id,
                            "detail": f"Screenshot received ({len(payload)} bytes)",
                        })
                    except Exception:
                        log.exception("client.image.send_error")

                else:
                    log.warning("client.unknown_binary_prefix", prefix=prefix.hex())

                continue

            # ---- Text frames (JSON control / text messages) ----
            raw_text = ws_msg.get("text")
            if not raw_text:
                continue

            log.debug("client.message_raw", length=len(raw_text))

            try:
                frame = json.loads(raw_text)
            except json.JSONDecodeError:
                log.warning("client.invalid_json")
                continue

            frame_type = frame.get("type")

            if frame_type == "text":
                content = frame.get("content", "")
                if not content:
                    continue

                # Rate-limit check (text mode only; live session = 0 RPM)
                if gemini.mode == "text":
                    if not rate_limiter.can_call(Priority.USER_ASK):
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Rate limit exceeded. Please wait.",
                        }))
                        continue
                    rate_limiter.record_call(Priority.USER_ASK)

                model_router.record_flash_call(reason="user_text")

                log.info("client.text", content_length=len(content))

                # Broadcast user message to dashboard
                await _broadcast_dashboard({
                    "type": "transcript",
                    "speaker": "user",
                    "text": content,
                    "client_id": client_id,
                })

                await gemini.send_text(content)

                # Text mode: create per-request relay task
                if gemini.mode == "text":
                    if relay_task is not None and not relay_task.done():
                        relay_task.cancel()
                        try:
                            await relay_task
                        except asyncio.CancelledError:
                            pass

                    relay_task = asyncio.create_task(
                        _relay_gemini_to_client(),
                        name=f"relay-{client_id}",
                    )
                # Live mode: persistent relay handles the response

            elif frame_type == "tool_result":
                # L3: Tool execution result from local client
                tool_name = frame.get("name", "unknown")
                tool_result = frame.get("result", {})
                log.info(
                    "client.tool_result",
                    name=tool_name,
                    success=tool_result.get("success"),
                )
                # Put it on the queue for the relay to pick up
                await tool_result_queue.put(frame)

            elif frame_type == "context":
                # L4 stub: struggle detection / memory context injection
                log.debug("client.context.stub", note="Context injection not handled in L0")

            elif frame_type == "control":
                action = frame.get("action", "")
                log.debug("client.control", action=action)

                # Handle push-to-talk release → signal end of speech
                if action == "end_of_speech" and gemini.mode == "live":
                    try:
                        await gemini.send_end_of_turn()
                    except Exception:
                        log.exception("client.end_of_speech.error")

            elif frame_type == "heartbeat":
                await session_manager.touch(client_id)

            else:
                log.warning("client.unknown_frame_type", frame_type=frame_type)

    except WebSocketDisconnect:
        log.info("client.disconnected")
    except Exception:
        log.exception("client.error")
    finally:
        # Clean up
        if relay_task is not None and not relay_task.done():
            relay_task.cancel()
            try:
                await relay_task
            except asyncio.CancelledError:
                pass
        await session_manager.remove_session(client_id)
        log.info("client.cleanup_done")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )
