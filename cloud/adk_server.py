"""
Rio ADK Bidi-Streaming Server.

Drop-in replacement for the custom Gemini relay in main.py.
Uses Google ADK ``Runner.run_live()`` + ``LiveRequestQueue`` for
full-duplex audio/vision streaming with automatic tool execution,
session resumption, and context window compression.

Run with:
    uvicorn adk_server:app --host 0.0.0.0 --port 8080

Or alongside the classic relay:
    RIO_USE_ADK=1 uvicorn adk_server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import base64
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
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from model_router import ModelRouter
from rate_limiter import DegradationLevel, Priority, RateLimiter
from rio_agent import ToolBridge, create_rio_agent

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
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LIVE_MODEL = os.environ.get("LIVE_MODEL", "gemini-2.5-flash-native-audio-latest")
WS_AUTH_TOKEN = os.environ.get("RIO_WS_TOKEN", "")

# Vertex AI / GCP configuration for $300 free credits
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
USE_VERTEX = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Shared singletons
# ---------------------------------------------------------------------------
session_service: InMemorySessionService | None = None
rate_limiter: RateLimiter = RateLimiter(budget_rpm=30)
model_router: ModelRouter | None = None

APP_NAME = "rio"

# Dashboard clients
dashboard_clients: Set[WebSocket] = set()
dashboard_lock = asyncio.Lock()

# Transcript buffer for dashboard
_transcript_buffer: list[dict] = []
_TRANSCRIPT_BUFFER_MAX = 200


def _buffer_transcript(entry: dict) -> None:
    _transcript_buffer.append(entry)
    if len(_transcript_buffer) > _TRANSCRIPT_BUFFER_MAX:
        _transcript_buffer.pop(0)


# ---------------------------------------------------------------------------
# RunConfig builder
# ---------------------------------------------------------------------------

def _build_run_config() -> "RunConfig":
    """Build the ADK RunConfig for bidi-streaming."""
    from google.adk.agents.run_config import RunConfig, StreamingMode

    # Detect native-audio model → AUDIO only modality
    is_native_audio = any(
        kw in LIVE_MODEL.lower() for kw in ("native-audio", "live")
    )
    response_modalities = ["AUDIO"] if is_native_audio else ["TEXT"]

    return RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=response_modalities,
        # Transcription — enables text display in dashboard
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # Session resumption — auto-reconnect on WS timeout (~10min)
        session_resumption=types.SessionResumptionConfig(),
        # Context window compression — removes 15min audio / 2min video limit
        context_window_compression=types.ContextWindowCompressionConfig(),
        # Proactive audio — Rio can speak unprompted
        proactivity=types.ProactivityConfig(proactive_audio=True),
        # Affective dialog — emotional awareness in voice
        enable_affective_dialog=True,
    )


# ---------------------------------------------------------------------------
# Dashboard broadcast
# ---------------------------------------------------------------------------

async def _broadcast_dashboard(payload: dict) -> None:
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


async def _dashboard_health_broadcast_loop() -> None:
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
                "sessions_active": 0,
                "backend": "adk",
            })
        except Exception:
            logger.debug("dashboard.health_broadcast.error")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global session_service, model_router

    if not GEMINI_API_KEY and not USE_VERTEX:
        logger.error("No GEMINI_API_KEY and Vertex AI not enabled")

    # Configure google-genai client
    if USE_VERTEX:
        logger.info("adk.vertex_mode", project=GCP_PROJECT, location=GCP_LOCATION)
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        if GCP_PROJECT:
            os.environ["GOOGLE_CLOUD_PROJECT"] = GCP_PROJECT
        if GCP_LOCATION:
            os.environ["GOOGLE_CLOUD_LOCATION"] = GCP_LOCATION
    else:
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

    session_service = InMemorySessionService()

    model_router = ModelRouter(
        api_key=GEMINI_API_KEY,
        rate_limiter=rate_limiter,
        pro_rpm_budget=int(os.environ.get("PRO_RPM_BUDGET", "5")),
    )

    logger.info(
        "rio.adk.startup",
        model=LIVE_MODEL,
        vertex=USE_VERTEX,
        project=GCP_PROJECT or "(not set)",
    )

    health_task = asyncio.create_task(
        _dashboard_health_broadcast_loop(), name="dashboard-health",
    )

    yield

    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass
    logger.info("rio.adk.shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Rio ADK Cloud Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:5173", "http://localhost:8080",
        "http://127.0.0.1:3000", "http://127.0.0.1:5173", "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static dashboard
_dashboard_dir = Path(__file__).resolve().parent.parent / "ui" / "dashboard"
if _dashboard_dir.is_dir():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_dashboard_dir), html=True),
        name="dashboard",
    )


# ---------------------------------------------------------------------------
# Health / API endpoints (same contract as main.py)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "rio-cloud-adk",
        "version": "1.0.0",
        "backend": "adk-bidi-streaming",
        "model": LIVE_MODEL,
        "vertex_ai": USE_VERTEX,
        "rate_limiter": rate_limiter.get_usage(),
    }


@app.get("/api/chat-history")
async def get_chat_history(limit: int = 100) -> dict:
    return {"messages": _transcript_buffer[-limit:], "total": len(_transcript_buffer)}


# Config API
_config_yaml_path = Path(__file__).resolve().parent.parent / "config.yaml"


@app.get("/api/config")
async def get_config() -> dict:
    import yaml
    if not _config_yaml_path.exists():
        return {"error": "Config file not found"}
    try:
        raw = yaml.safe_load(_config_yaml_path.read_text(encoding="utf-8")) or {}
        return {"config": raw.get("rio", raw)}
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/doctor")
async def doctor_check() -> dict:
    import sys as _sys
    checks = {
        "config": {"status": "ok" if _config_yaml_path.exists() else "missing"},
        "api_key": {
            "status": "ok" if GEMINI_API_KEY or USE_VERTEX else "missing",
            "vertex_ai": USE_VERTEX,
        },
        "python": {
            "version": f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
        },
        "backend": "adk",
        "model": LIVE_MODEL,
    }
    return {"checks": checks}


@app.get("/api/models/status")
async def models_status() -> dict:
    return {
        "routing": model_router.get_routing_stats() if model_router else {},
        "backend": "adk",
        "models": {
            "live": LIVE_MODEL,
            "flash": "gemini-2.5-flash",
            "pro": "gemini-2.5-pro-preview-03-25",
        },
    }


# ---------------------------------------------------------------------------
# WS /ws/dashboard
# ---------------------------------------------------------------------------

@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket) -> None:
    await websocket.accept()
    async with dashboard_lock:
        dashboard_clients.add(websocket)
    logger.info("dashboard.connected", total=len(dashboard_clients))
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        async with dashboard_lock:
            dashboard_clients.discard(websocket)


# ---------------------------------------------------------------------------
# WS /ws/rio/live — ADK bidi-streaming relay
# ---------------------------------------------------------------------------

@app.websocket("/ws/rio/live")
async def ws_rio_live(websocket: WebSocket) -> None:
    await websocket.accept()
    client_id = str(uuid.uuid4())
    log = logger.bind(client_id=client_id)

    # ---- Authentication ----
    if WS_AUTH_TOKEN:
        try:
            auth_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            auth_frame = json.loads(auth_msg)
            if auth_frame.get("type") != "auth" or auth_frame.get("token") != WS_AUTH_TOKEN:
                log.warning("client.auth_failed")
                await websocket.close(code=4001, reason="Authentication failed")
                return
        except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
            await websocket.close(code=4001, reason="Authentication required")
            return

    log.info("adk.client.connected")
    await _broadcast_dashboard({
        "type": "dashboard", "subtype": "client_event",
        "event": "connected", "client_id": client_id,
    })

    # ---- Per-connection setup ----
    assert session_service is not None

    user_id = f"user-{client_id}"
    session_id = f"session-{client_id}"

    # Create ToolBridge for this connection
    bridge = ToolBridge(websocket, broadcast_fn=_broadcast_dashboard)

    # Create per-connection Agent with closured tools
    agent = create_rio_agent(bridge, model=LIVE_MODEL)

    # Create per-connection Runner
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Create ADK session
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id,
    )

    # Build RunConfig
    run_config = _build_run_config()

    # Create the LiveRequestQueue (decouples WS receiver from ADK consumer)
    from google.adk.agents.live_request_queue import LiveRequestQueue
    queue = LiveRequestQueue()

    # Notify client
    await websocket.send_text(json.dumps({
        "type": "control",
        "action": "connected",
        "detail": "ADK bidi-streaming session ready",
        "backend": "adk",
    }))
    await websocket.send_text(json.dumps({
        "type": "control",
        "action": "session_mode",
        "actual_mode": "live",
        "requested_mode": "live",
    }))

    # ---- Upstream task: client → queue ----
    async def upstream_task() -> None:
        """Read from WebSocket, route audio/image to queue, tool results to bridge."""
        try:
            while True:
                ws_msg = await websocket.receive()

                if ws_msg["type"] == "websocket.disconnect":
                    break

                # Binary frames
                raw_bytes = ws_msg.get("bytes")
                if raw_bytes and len(raw_bytes) > 1:
                    prefix = raw_bytes[0:1]
                    payload = raw_bytes[1:]

                    if prefix == b"\x01":
                        # Audio frame — PCM16 at 16kHz
                        audio_blob = types.Blob(
                            mime_type="audio/pcm;rate=16000", data=payload,
                        )
                        queue.send_realtime(audio_blob)

                    elif prefix == b"\x02":
                        # Image frame — JPEG screenshot
                        deg_level = rate_limiter._degradation_level()
                        if deg_level >= DegradationLevel.EMERGENCY:
                            log.debug("upstream.image.dropped", reason=deg_level.name)
                            continue

                        image_blob = types.Blob(
                            mime_type="image/jpeg", data=payload,
                        )
                        queue.send_realtime(image_blob)

                        await _broadcast_dashboard({
                            "type": "dashboard", "subtype": "vision",
                            "client_id": client_id,
                            "detail": f"Screenshot ({len(payload)} bytes)",
                        })
                    continue

                # Text frames
                raw_text = ws_msg.get("text")
                if not raw_text:
                    continue

                try:
                    frame = json.loads(raw_text)
                except json.JSONDecodeError:
                    log.warning("upstream.invalid_json")
                    continue

                frame_type = frame.get("type")

                if frame_type == "tool_result":
                    # Route tool result to the bridge
                    call_id = frame.get("id", "")
                    result = frame.get("result", {})
                    if call_id:
                        bridge.resolve(call_id, result)
                    else:
                        bridge.resolve_by_name(frame.get("name", ""), result)

                    await _broadcast_dashboard({
                        "type": "dashboard", "subtype": "tool_result",
                        "client_id": client_id,
                        "name": frame.get("name"),
                        "success": result.get("success"),
                    })

                elif frame_type == "text":
                    content = frame.get("content", "")
                    if content:
                        # Send text as Content to the model
                        queue.send_content(
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=content)],
                            )
                        )

                        _buffer_transcript({
                            "speaker": "user", "text": content,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        await _broadcast_dashboard({
                            "type": "transcript", "speaker": "user",
                            "text": content, "client_id": client_id,
                        })

                elif frame_type == "context":
                    # Struggle detection / task lifecycle context
                    content = frame.get("content", "")
                    if content:
                        queue.send_content(
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=f"[CONTEXT] {content}")],
                            )
                        )

                elif frame_type == "control":
                    action = frame.get("action", "")
                    if action == "end_of_speech":
                        # Manual VAD — signal end of user turn
                        queue.send_activity_end()

                elif frame_type == "heartbeat":
                    pass  # ADK handles session keep-alive

        except WebSocketDisconnect:
            log.info("upstream.client_disconnected")
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("upstream.error")

    # ---- Downstream task: ADK events → client ----
    async def downstream_task() -> None:
        """Consume events from runner.run_live() and forward to client."""
        try:
            async for event in runner.run_live(
                user_id=user_id,
                session_id=session_id,
                live_request_queue=queue,
                run_config=run_config,
            ):
                # Audio data — send as binary frame (0x01 prefix)
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts or []:
                        # Inline audio data
                        if hasattr(part, "inline_data") and part.inline_data:
                            audio_data = part.inline_data.data
                            if audio_data:
                                await websocket.send_bytes(b"\x01" + audio_data)

                        # Text content
                        if hasattr(part, "text") and part.text:
                            text = part.text
                            await websocket.send_text(json.dumps({
                                "type": "transcript",
                                "speaker": "rio",
                                "text": text,
                            }))
                            _buffer_transcript({
                                "speaker": "rio", "text": text,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            await _broadcast_dashboard({
                                "type": "transcript", "speaker": "rio",
                                "text": text, "client_id": client_id,
                            })

                # Input transcription (user's speech as text)
                input_tx = getattr(event, "input_transcription", None)
                if input_tx and hasattr(input_tx, "text") and input_tx.text:
                    _buffer_transcript({
                        "speaker": "user", "text": input_tx.text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "transcription",
                    })
                    await _broadcast_dashboard({
                        "type": "transcript", "speaker": "user",
                        "text": input_tx.text, "client_id": client_id,
                        "source": "transcription",
                    })

                # Output transcription (Rio's speech as text)
                output_tx = getattr(event, "output_transcription", None)
                if output_tx and hasattr(output_tx, "text") and output_tx.text:
                    await websocket.send_text(json.dumps({
                        "type": "transcript",
                        "speaker": "rio",
                        "text": output_tx.text,
                        "source": "transcription",
                    }))

                # Interrupted flag — user spoke while agent was responding
                if getattr(event, "interrupted", False):
                    await websocket.send_text(json.dumps({
                        "type": "control",
                        "action": "interrupted",
                    }))

                # Turn complete
                if getattr(event, "turn_complete", False):
                    await websocket.send_text(json.dumps({
                        "type": "control",
                        "action": "turn_complete",
                    }))

                # Partial flag for streaming text
                if getattr(event, "partial", False):
                    pass  # already sent above as transcript

                # Error events
                error_code = getattr(event, "error_code", None)
                if error_code:
                    error_msg = getattr(event, "error_message", "Unknown error")
                    log.error("downstream.error_event", code=error_code, msg=error_msg)
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"ADK error [{error_code}]: {error_msg}",
                    }))

        except WebSocketDisconnect:
            log.info("downstream.client_disconnected")
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("downstream.error")
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "ADK streaming error — session will reconnect",
                }))
            except Exception:
                pass

    # ---- Run upstream + downstream concurrently (true bidi) ----
    try:
        await asyncio.gather(upstream_task(), downstream_task())
    except WebSocketDisconnect:
        log.info("client.disconnected")
    except Exception:
        log.exception("client.error")
    finally:
        queue.close()
        log.info("adk.client.cleanup_done", client_id=client_id)
        await _broadcast_dashboard({
            "type": "dashboard", "subtype": "client_event",
            "event": "disconnected", "client_id": client_id,
        })


# ---------------------------------------------------------------------------
# Profile API (same contract as main.py)
# ---------------------------------------------------------------------------
_profiles_mod = None
_profiles_base = str(Path(__file__).resolve().parent.parent / "rio_profiles")


def _get_profiles_mod():
    global _profiles_mod
    if _profiles_mod is None:
        import importlib, sys
        _local_dir = str(Path(__file__).resolve().parent.parent / "local")
        if _local_dir not in sys.path:
            sys.path.insert(0, _local_dir)
        _profiles_mod = importlib.import_module("profiles")
    return _profiles_mod


@app.get("/api/profiles/{skill_name}")
async def get_profile(skill_name: str) -> dict:
    mod = _get_profiles_mod()
    if skill_name == "customer_care":
        return {"profile": mod.asdict(mod.load_customer_care_profile(_profiles_base))}
    elif skill_name == "tutor":
        return {"profile": mod.asdict(mod.load_tutor_profile(_profiles_base))}
    return {"error": f"Unknown skill: {skill_name}"}


@app.post("/api/profiles/{skill_name}")
async def save_profile(skill_name: str, request: Request) -> dict:
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
        return {"error": f"Unknown skill: {skill_name}"}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "adk_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )
