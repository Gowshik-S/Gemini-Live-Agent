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
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Set

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model_router import ModelRouter
from rate_limiter import DegradationLevel, Priority, RateLimiter
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
TEXT_MODEL = os.environ.get("TEXT_MODEL", None)  # Override via env, else use default
LIVE_MODEL = os.environ.get("LIVE_MODEL", None)  # Override via env, else use default
# Shared secret for WebSocket authentication (set via env for production)
WS_AUTH_TOKEN = os.environ.get("RIO_WS_TOKEN", "")

# ---------------------------------------------------------------------------
# Shared singletons (created during lifespan)
# ---------------------------------------------------------------------------
session_manager: SessionManager | None = None
rate_limiter: RateLimiter = RateLimiter(budget_rpm=30)
model_router: ModelRouter | None = None  # Initialized in lifespan with api_key

# Connected dashboard WebSocket clients
dashboard_clients: Set[WebSocket] = set()
dashboard_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Background: periodic dashboard health broadcast
# ---------------------------------------------------------------------------
async def _dashboard_health_broadcast_loop() -> None:
    """Push rate-limiter and health stats to dashboard clients every 3s."""
    while True:
        await asyncio.sleep(3)
        if not dashboard_clients:
            continue
        try:
            usage = rate_limiter.get_usage()
            routing = model_router.get_routing_stats() if model_router else {}
            await _broadcast_dashboard({
                "type": "dashboard",
                "subtype": "health",
                "rpm": usage["rpm"],
                "budget": usage["budget"],
                "utilization_pct": usage["utilization_pct"],
                "model": routing.get("last_model", "Flash") or "Flash",
                "sessions_active": session_manager.active_count if session_manager else 0,
                "pro_rpm": routing.get("pro_rpm_current", 0),
                "pro_budget": routing.get("pro_rpm_budget", 5),
            })
            await _broadcast_dashboard({
                "type": "dashboard",
                "subtype": "rate_limit",
                "current_rpm": usage["rpm"],
                "status": usage["degradation_level"].lower(),
                "vision_active": usage["degradation_level"] not in ("EMERGENCY", "CRITICAL"),
                "pro_active": usage["degradation_level"] not in ("CAUTION", "EMERGENCY", "CRITICAL"),
            })
        except Exception:
            logger.debug("dashboard.health_broadcast.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global session_manager, model_router

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set -- Gemini sessions will fail")

    session_manager = SessionManager(
        api_key=GEMINI_API_KEY,
        session_mode=SESSION_MODE,
        text_model=TEXT_MODEL,
        live_model=LIVE_MODEL,
    )

    model_router = ModelRouter(
        api_key=GEMINI_API_KEY,
        rate_limiter=rate_limiter,
        pro_rpm_budget=int(os.environ.get("PRO_RPM_BUDGET", "5")),
    )

    logger.info("rio.startup", gemini_key_set=bool(GEMINI_API_KEY), session_mode=SESSION_MODE)

    # Start background health broadcaster
    health_task = asyncio.create_task(
        _dashboard_health_broadcast_loop(),
        name="dashboard-health",
    )

    yield  # --- application runs here ---

    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass

    logger.info("rio.shutdown")
    await session_manager.shutdown()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Rio Cloud Service",
    version="0.6.0",
    lifespan=lifespan,
)

# CORS -- allow dashboard & local client origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
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
        "version": "0.7.0",
        "sessions_active": session_manager.active_count if session_manager else 0,
        "rate_limiter": rate_limiter.get_usage(),
        "routing": model_router.get_routing_stats() if model_router else {},
    }


# ---------------------------------------------------------------------------
# Chat history for dashboard (pulls from transcript buffer)
# ---------------------------------------------------------------------------
# In-memory transcript buffer for dashboard (last 200 messages)
_transcript_buffer: list[dict] = []
_TRANSCRIPT_BUFFER_MAX = 200


def _buffer_transcript(entry: dict) -> None:
    """Add a transcript entry to the in-memory buffer."""
    _transcript_buffer.append(entry)
    if len(_transcript_buffer) > _TRANSCRIPT_BUFFER_MAX:
        _transcript_buffer.pop(0)


@app.get("/api/chat-history")
async def get_chat_history(limit: int = 100) -> dict:
    """Return recent chat messages for the dashboard."""
    messages = _transcript_buffer[-limit:]
    return {
        "messages": messages,
        "total": len(_transcript_buffer),
    }


# ---------------------------------------------------------------------------
# Profile API — fixed-schema config for customer care & tutor skills
# ---------------------------------------------------------------------------
# Lazy import to avoid circular deps at module level
_profiles_mod = None

def _get_profiles_mod():
    global _profiles_mod
    if _profiles_mod is None:
        import importlib
        # profiles.py lives at rio/local/profiles.py — add parent to sys.path
        import sys
        _local_dir = str(Path(__file__).resolve().parent.parent / "local")
        if _local_dir not in sys.path:
            sys.path.insert(0, _local_dir)
        _profiles_mod = importlib.import_module("profiles")
    return _profiles_mod

# Resolve profiles directory: same level as rio/ project root
_profiles_base = str(Path(__file__).resolve().parent.parent / "rio_profiles")


@app.get("/api/profiles/{skill_name}")
async def get_profile(skill_name: str) -> dict:
    """Load a saved profile. Returns defaults if none exists."""
    mod = _get_profiles_mod()
    if skill_name == "customer_care":
        profile = mod.load_customer_care_profile(_profiles_base)
        return {"profile": mod.asdict(profile)}
    elif skill_name == "tutor":
        profile = mod.load_tutor_profile(_profiles_base)
        return {"profile": mod.asdict(profile)}
    else:
        return {"error": f"Unknown skill: {skill_name}"}, 404


@app.post("/api/profiles/{skill_name}")
async def save_profile(skill_name: str, request: Request) -> dict:
    """Save a profile from the setup form. Accepts the full JSON object."""
    mod = _get_profiles_mod()
    body = await request.json()
    try:
        if skill_name == "customer_care":
            profile = mod._dict_to_customer_care(body)
            path = mod.save_profile(profile, _profiles_base)
            return {"status": "saved", "path": str(path)}
        elif skill_name == "tutor":
            profile = mod._dict_to_tutor(body)
            path = mod.save_profile(profile, _profiles_base)
            return {"status": "saved", "path": str(path)}
        else:
            return {"error": f"Unknown skill: {skill_name}"}
    except Exception as exc:
        logger.error("profile.save_error", skill=skill_name, error=str(exc))
        return {"error": str(exc)}


@app.get("/api/profiles/{skill_name}/defaults")
async def get_profile_defaults(skill_name: str) -> dict:
    """Return the default/template profile for a skill."""
    mod = _get_profiles_mod()
    if skill_name == "customer_care":
        return {"profile": mod.get_default_customer_care_json()}
    elif skill_name == "tutor":
        return {"profile": mod.get_default_tutor_json()}
    else:
        return {"error": f"Unknown skill: {skill_name}"}


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
        # Keep alive -- read pings and respond with pong for RTT measurement
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
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

    # ---- Authentication via shared secret (Fix #19) ----
    if WS_AUTH_TOKEN:
        try:
            auth_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            auth_frame = json.loads(auth_msg)
            if auth_frame.get("type") != "auth" or auth_frame.get("token") != WS_AUTH_TOKEN:
                log.warning("client.auth_failed")
                await websocket.close(code=4001, reason="Authentication failed")
                return
        except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
            log.warning("client.auth_timeout_or_error")
            await websocket.close(code=4001, reason="Authentication required")
            return

    log.info("client.connected")

    # Broadcast client connection to dashboard
    await _broadcast_dashboard({
        "type": "dashboard",
        "subtype": "client_event",
        "event": "connected",
        "client_id": client_id,
    })

    assert session_manager is not None, "SessionManager not initialised"

    # ---- Create Gemini session ----
    try:
        gemini = await session_manager.create_session(client_id)
        # Record session creation in rate limiter (costs 1 RPM)
        rate_limiter.try_acquire(Priority.USER_ASK)
        # Notify client that Gemini session is ready
        await websocket.send_text(json.dumps({
            "type": "control",
            "action": "connected",
            "detail": "Gemini Live session ready",
        }))

        # Tell the client which session mode is actually active
        await websocket.send_text(json.dumps({
            "type": "control",
            "action": "session_mode",
            "actual_mode": gemini.mode,
            "requested_mode": SESSION_MODE,
        }))

        # If the session silently fell back from live to text, warn the client
        if SESSION_MODE == "live" and gemini.mode == "text":
            log.warning(
                "client.live_fallback",
                requested=SESSION_MODE,
                actual=gemini.mode,
                error=gemini._live_connect_error,
            )
            await websocket.send_text(json.dumps({
                "type": "control",
                "action": "live_api_unavailable",
                "detail": (
                    f"Live API connection failed — running in text mode. "
                    f"Error: {gemini._live_connect_error or 'unknown'}"
                ),
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

    # ---- Set up Pro injection callback (routes Pro results through active session) ----
    async def _inject_pro_into_session(analysis: str) -> None:
        await gemini.send_context(analysis)
    model_router.set_inject_callback(_inject_pro_into_session)

    # ---- Pro escalation helper ----
    async def _pro_escalation(session, user_text: str, cid: str) -> None:
        """Fire-and-forget: call Pro, inject result, broadcast to dashboard."""
        try:
            analysis = await model_router.call_pro(user_text)
            if analysis:
                await model_router.inject_pro_result(analysis)
                await _broadcast_dashboard({
                    "type": "dashboard",
                    "subtype": "model_switch",
                    "model": "Flash",
                    "reason": "pro_complete",
                })
                # In text mode, kick a relay to send Gemini's synthesized response
                if session.mode == "text":
                    asyncio.create_task(
                        _relay_gemini_to_client(),
                        name=f"relay-pro-{cid}",
                    )
            else:
                log.info("pro_escalation.no_result", client_id=cid)
        except Exception:
            log.exception("pro_escalation.error", client_id=cid)

    # ---- Background task: read Gemini responses and forward to client ----
    async def _relay_gemini_to_client() -> None:
        """Per-request relay: calls Gemini and streams results to client.

        Handles the tool-calling loop:
          1. Call Gemini → receive text or tool_call(s)
          2. If tool_call: forward to local, wait for result, feed back, repeat
          3. If text: stream to client and finish

        Loops up to MAX_TOOL_ROUNDS times to prevent infinite tool chains.
        """
        MAX_TOOL_ROUNDS = 25

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

                        # Buffer + broadcast to dashboard
                        _buffer_transcript({
                            "speaker": "rio",
                            "text": item,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
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
                            call_id=call.get("id"),
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
                            call_id=call.get("id"),
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

        Improvements over the original:
          - Text messages are buffered until turn_complete.  This prevents
            the "preamble text before task" UX issue where Gemini says
            "Sure, I'll do that" and the user sees it before any work
            starts.  Preamble text is discarded when a tool_call follows
            it; only conversational turns (no tools) flush text on
            turn_complete.
          - After every tool result is sent to Gemini, a 12-second
            watchdog timer starts.  If Gemini goes silent (no turn_complete,
            no text, no further tool_calls) we synthesise a turn_complete
            so the local client isn't left hanging forever.
        """
        consecutive_tool_calls = 0
        # Text buffered since the last turn_complete / start (not yet sent)
        pending_text: list[str] = []
        # Whether we should apply a timeout on the next receive_live() item
        awaiting_gemini_reply = False
        # Seconds to wait for Gemini's reply after the last tool result
        GEMINI_REPLY_TIMEOUT = 12.0

        live_gen = gemini.receive_live()

        async def _send_pending_text() -> None:
            """Flush buffered text to the client and dashboard."""
            for txt in pending_text:
                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "speaker": "rio",
                    "text": txt,
                }))
                _buffer_transcript({
                    "speaker": "rio",
                    "text": txt,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                await _broadcast_dashboard({
                    "type": "transcript",
                    "speaker": "rio",
                    "text": txt,
                    "client_id": client_id,
                })
            pending_text.clear()

        async def _end_task_mode() -> None:
            nonlocal consecutive_tool_calls
            if consecutive_tool_calls > 0:
                consecutive_tool_calls = 0
                await websocket.send_text(json.dumps({
                    "type": "control",
                    "action": "task_mode",
                    "active": False,
                }))

        try:
            while True:
                # Fetch next item from the Live stream.
                # Apply a timeout only after we sent a tool result and are
                # waiting for Gemini's follow-up, so silent sessions don't
                # leave the client in a permanent "task active" limbo.
                try:
                    if awaiting_gemini_reply:
                        item = await asyncio.wait_for(
                            live_gen.__anext__(), timeout=GEMINI_REPLY_TIMEOUT,
                        )
                    else:
                        item = await live_gen.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    log.warning(
                        "relay.live.gemini_reply_timeout",
                        seconds=GEMINI_REPLY_TIMEOUT,
                        consecutive_tool_calls=consecutive_tool_calls,
                    )
                    # Gemini went silent after a tool result — clean up state
                    # and unblock the local client.
                    pending_text.clear()
                    await _end_task_mode()
                    await websocket.send_text(json.dumps({
                        "type": "control",
                        "action": "turn_complete",
                    }))
                    awaiting_gemini_reply = False
                    continue

                awaiting_gemini_reply = False
                item_type = item.get("type")

                if item_type == "audio":
                    # Audio response → binary 0x01 frame to local client
                    await websocket.send_bytes(b"\x01" + item["data"])

                elif item_type == "text":
                    text = item["text"]
                    if consecutive_tool_calls > 0:
                        # Text that arrives mid-task (e.g. "Done!") comes
                        # after all tool calls; treat it as a post-task
                        # comment — flush it immediately after resetting
                        # task mode so the client sees it in the right order.
                        await _end_task_mode()
                        await websocket.send_text(json.dumps({
                            "type": "transcript",
                            "speaker": "rio",
                            "text": text,
                        }))
                        _buffer_transcript({
                            "speaker": "rio",
                            "text": text,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        await _broadcast_dashboard({
                            "type": "transcript",
                            "speaker": "rio",
                            "text": text,
                            "client_id": client_id,
                        })
                    else:
                        # Pre-task or conversational text — buffer it.
                        # If a tool_call follows, this preamble is discarded.
                        # If turn_complete follows (pure conversation), it's flushed.
                        pending_text.append(text)

                elif item_type == "tool_call":
                    consecutive_tool_calls += 1
                    log.info(
                        "relay.live.tool_call",
                        name=item.get("name"),
                        args_keys=list(item.get("args", {}).keys()),
                        step=consecutive_tool_calls,
                    )
                    # Discard any preamble text — the task action speaks for itself
                    pending_text.clear()

                    await websocket.send_text(json.dumps(item))
                    await _broadcast_dashboard({
                        "type": "dashboard",
                        "subtype": "tool_call",
                        "client_id": client_id,
                        "name": item.get("name"),
                        "args": item.get("args", {}),
                    })

                    # Signal task mode to local client when 2+ consecutive tool calls
                    if consecutive_tool_calls >= 2:
                        await websocket.send_text(json.dumps({
                            "type": "control",
                            "action": "task_mode",
                            "active": True,
                            "step": consecutive_tool_calls,
                        }))

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
                            call_id=item.get("id"),
                        )
                        await _broadcast_dashboard({
                            "type": "dashboard",
                            "subtype": "tool_result",
                            "client_id": client_id,
                            "name": item["name"],
                            "success": result_data.get("success"),
                        })
                        # Arm the watchdog: Gemini should reply within timeout
                        awaiting_gemini_reply = True
                    except asyncio.TimeoutError:
                        log.error(
                            "relay.live.tool_result.timeout",
                            name=item["name"],
                        )
                        await gemini.send_tool_result(
                            item["name"],
                            {"success": False, "error": "Tool execution timed out (60s)"},
                            call_id=item.get("id"),
                        )
                        awaiting_gemini_reply = True

                elif item_type == "turn_complete":
                    # Turn is done — flush any buffered text then notify client
                    await _send_pending_text()
                    await _end_task_mode()
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

    # ---- Background: poll for session reconnects ----
    reconnect_check_task: asyncio.Task | None = None

    async def _reconnect_checker() -> None:
        """Poll SessionManager for reconnects and swap session + relay."""
        nonlocal gemini, relay_task
        try:
            while True:
                await asyncio.sleep(5)
                try:
                    new_session = await session_manager.check_reconnect(client_id)
                except Exception:
                    log.exception("reconnect.check_error")
                    continue

                if new_session is None:
                    continue

                log.info("reconnect.detected", client_id=client_id, new_mode=new_session.mode)

                # Stop old relay
                if relay_task is not None and not relay_task.done():
                    relay_task.cancel()
                    try:
                        await relay_task
                    except asyncio.CancelledError:
                        pass

                # Swap session reference
                gemini = new_session
                # Update the Pro inject callback to use the new session
                async def _new_inject(analysis: str) -> None:
                    await gemini.send_context(analysis)
                model_router.set_inject_callback(_new_inject)

                # Restart relay if live mode
                if gemini.mode == "live":
                    relay_task = asyncio.create_task(
                        _relay_live_to_client(),
                        name=f"live-relay-{client_id}",
                    )

                # Notify the client
                try:
                    await websocket.send_text(json.dumps({
                        "type": "control",
                        "action": "reconnected",
                        "detail": "Gemini session reconnected (timeout recovery)",
                        "mode": gemini.mode,
                    }))
                except Exception:
                    pass

                await _broadcast_dashboard({
                    "type": "dashboard",
                    "subtype": "client_event",
                    "event": "reconnected",
                    "client_id": client_id,
                })
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("reconnect_checker.fatal")

    reconnect_check_task = asyncio.create_task(
        _reconnect_checker(),
        name=f"reconnect-check-{client_id}",
    )

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
                    # Degradation: drop vision in EMERGENCY+ to conserve RPM
                    deg_level = rate_limiter._degradation_level()
                    if deg_level >= DegradationLevel.EMERGENCY:
                        log.debug(
                            "client.image.dropped",
                            reason=f"degradation={deg_level.name}",
                        )
                        continue

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
                    if not rate_limiter.try_acquire(Priority.USER_ASK):
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Rate limit exceeded. Please wait.",
                        }))
                        continue

                model_router.record_flash_call(reason="user_text")

                # -- Pro escalation check --
                deg = rate_limiter._degradation_level()
                if (model_router.should_use_pro(content)
                        and deg < DegradationLevel.CAUTION):
                    log.info("client.pro_escalation", content_length=len(content))
                    await _broadcast_dashboard({
                        "type": "dashboard",
                        "subtype": "model_switch",
                        "model": "Pro",
                        "reason": "user_request",
                    })
                    # Fire-and-forget Pro call (non-blocking)
                    asyncio.create_task(
                        _pro_escalation(gemini, content, client_id),
                        name=f"pro-{client_id}",
                    )

                log.info("client.text", content_length=len(content))

                # Buffer + broadcast user message to dashboard
                _buffer_transcript({
                    "speaker": "user",
                    "text": content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
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
                # L4: Struggle detection / memory context injection
                subtype = frame.get("subtype", "unknown")
                content = frame.get("content", "")
                confidence = frame.get("confidence", 0.0)
                signals = frame.get("signals", [])

                log.info(
                    "client.context",
                    subtype=subtype,
                    confidence=confidence,
                    signals=signals,
                )

                if subtype == "struggle" and content:
                    # Inject the struggle context into Gemini so it responds proactively
                    try:
                        await gemini.send_context(content)
                    except Exception:
                        log.exception("client.context.send_failed")

                    # In text mode, create a relay task so Gemini's proactive
                    # response is actually sent to the client (send_context only
                    # appends to history; we need to call receive() to get the reply)
                    if gemini.mode == "text":
                        if relay_task is not None and not relay_task.done():
                            relay_task.cancel()
                            try:
                                await relay_task
                            except asyncio.CancelledError:
                                pass
                        relay_task = asyncio.create_task(
                            _relay_gemini_to_client(),
                            name=f"relay-context-{client_id}",
                        )

                    # Broadcast to dashboard
                    await _broadcast_dashboard({
                        "type": "dashboard",
                        "subtype": "struggle",
                        "confidence": confidence,
                        "signals": signals,
                    })

                elif subtype in ("task_abort", "mode_change") and content:
                    # Task abort or live mode change — inject into Gemini context
                    log.info("client.context.injecting", subtype=subtype)
                    try:
                        await gemini.send_context(content)
                    except Exception:
                        log.exception("client.context.send_failed")

                    if gemini.mode == "text":
                        if relay_task is not None and not relay_task.done():
                            relay_task.cancel()
                            try:
                                await relay_task
                            except asyncio.CancelledError:
                                pass
                        relay_task = asyncio.create_task(
                            _relay_gemini_to_client(),
                            name=f"relay-context-{client_id}",
                        )

                else:
                    log.debug("client.context.unhandled", subtype=subtype)

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
        if reconnect_check_task is not None and not reconnect_check_task.done():
            reconnect_check_task.cancel()
            try:
                await reconnect_check_task
            except asyncio.CancelledError:
                pass
        if relay_task is not None and not relay_task.done():
            relay_task.cancel()
            try:
                await relay_task
            except asyncio.CancelledError:
                pass
        await session_manager.remove_session(client_id)
        log.info("client.cleanup_done")

        # Broadcast client disconnection to dashboard
        await _broadcast_dashboard({
            "type": "dashboard",
            "subtype": "client_event",
            "event": "disconnected",
            "client_id": client_id,
        })


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
