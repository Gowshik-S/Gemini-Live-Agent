"""
Rio Direct Live API Server.

Connects to Gemini's Live API directly via ``client.aio.live.connect()``
for minimum-latency native audio streaming with server-side VAD.

Tools are dispatched to the local client via ToolBridge over WebSocket
and results sent back to the model via ``session.send()``.

Replaces the ADK Runner-based relay for:
  - Lower latency (no Runner → Agent → LlmFlow → LLM layers)
  - Proper native audio (direct LiveConnectConfig with voice selection)
  - Reliable server-side VAD and interruption
  - Session resumption with automatic reconnect

Run with:
    python adk_server.py
    # or: uvicorn adk_server:app --host 0.0.0.0 --port 8080
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
from google import genai
from google.genai import types

from gemini_session import build_system_instruction
from rio_agent import ToolBridge, _make_tools
from tool_orchestrator import ToolOrchestrator, _is_task_request

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
WS_AUTH_TOKEN = os.environ.get("RIO_WS_TOKEN", "")
VOICE_NAME = os.environ.get("RIO_VOICE", "Puck")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# Vertex AI / GCP configuration ($300 free credits)
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
USE_VERTEX = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in (
    "true", "1", "yes",
)

# Default live model for native audio streaming + function calling.
# gemini-2.0-flash is deprecated — use 2.5 native-audio.
_DEFAULT_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
LIVE_MODEL = os.environ.get("LIVE_MODEL", _DEFAULT_MODEL)
ORCHESTRATOR_MODEL = os.environ.get(
    "ORCHESTRATOR_MODEL", "gemini-2.5-flash"
)
# Whether the live model itself can invoke function calls.
# gemini-2.5-flash-native-audio-preview-12-2025 DOES support function calling
# per https://ai.google.dev/gemini-api/docs/live-api/tools
LIVE_MODEL_TOOLS = _env_flag("RIO_LIVE_MODEL_TOOLS", True)

ENABLE_SERVER_VAD = _env_flag("RIO_LIVE_ENABLE_SERVER_VAD", False)
ENABLE_INPUT_AUDIO_TRANSCRIPTION = _env_flag(
    "RIO_LIVE_ENABLE_INPUT_AUDIO_TRANSCRIPTION", False,
)
ENABLE_OUTPUT_AUDIO_TRANSCRIPTION = _env_flag(
    "RIO_LIVE_ENABLE_OUTPUT_AUDIO_TRANSCRIPTION", False,
)
ENABLE_SESSION_RESUMPTION = _env_flag(
    "RIO_LIVE_ENABLE_SESSION_RESUMPTION", False,
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
genai_client: genai.Client | None = None

# Dashboard clients
dashboard_clients: Set[WebSocket] = set()
dashboard_lock = asyncio.Lock()

# Transcript buffer for dashboard
_transcript_buffer: list[dict] = []
_TRANSCRIPT_BUFFER_MAX = 200

# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------
_conversations_dir = Path(__file__).resolve().parent.parent / "data" / "conversations"
_conversations_dir.mkdir(parents=True, exist_ok=True)

_current_conversation_id: str | None = None
_current_conversation_messages: list[dict] = []


def _start_new_conversation() -> str:
    """Start a new conversation and return its ID."""
    global _current_conversation_id, _current_conversation_messages
    _current_conversation_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    _current_conversation_messages = []
    return _current_conversation_id


def _save_conversation() -> None:
    """Persist the current conversation to disk as JSON."""
    if not _current_conversation_id or not _current_conversation_messages:
        return
    filepath = _conversations_dir / f"{_current_conversation_id}.json"
    data = {
        "id": _current_conversation_id,
        "created_at": _current_conversation_messages[0].get("timestamp", ""),
        "updated_at": _current_conversation_messages[-1].get("timestamp", ""),
        "message_count": len(_current_conversation_messages),
        "preview": _current_conversation_messages[0].get("text", "")[:100],
        "messages": _current_conversation_messages,
    }
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _buffer_transcript(entry: dict) -> None:
    global _current_conversation_id
    _transcript_buffer.append(entry)
    if len(_transcript_buffer) > _TRANSCRIPT_BUFFER_MAX:
        _transcript_buffer.pop(0)
    # Also persist to current conversation
    if not _current_conversation_id:
        _start_new_conversation()
    _current_conversation_messages.append(entry)
    # Auto-save every 5 messages
    if len(_current_conversation_messages) % 5 == 0:
        _save_conversation()


# ---------------------------------------------------------------------------
# LiveConnectConfig builder
# ---------------------------------------------------------------------------

def _callable_to_tool_declarations(
    client: genai.Client,
    tool_fns: list,
) -> list[types.Tool]:
    """Convert a list of Python callables into ``types.Tool`` objects.

    The Live API's ``LiveConnectConfig`` does NOT auto-convert callables
    (unlike ``generate_content``).  We must explicitly build
    ``FunctionDeclaration`` objects so the model actually sees the tools.
    """
    declarations: list[types.FunctionDeclaration] = []
    for fn in tool_fns:
        fd = types.FunctionDeclaration.from_callable(
            client=client,
            callable=fn,
        )
        declarations.append(fd)
    return [types.Tool(function_declarations=declarations)]


def _build_live_config(
    tool_objects: list[types.Tool],
    system_instruction: str,
    session_handle: str | None = None,
) -> types.LiveConnectConfig:
    """Build a LiveConnectConfig for the direct Gemini Live API."""

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=VOICE_NAME,
                ),
            ),
        ),
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=system_instruction)],
        ),
        # Register tools with the live model for native function calling.
        # gemini-2.5-flash-native-audio-preview supports function calling.
        # Set RIO_LIVE_MODEL_TOOLS=False to fall back to the ToolOrchestrator.
        tools=tool_objects if LIVE_MODEL_TOOLS else None,
        # Always enable input transcription so the ToolOrchestrator can detect
        # task requests from the user's speech, even if the env flag is off.
        input_audio_transcription=types.AudioTranscriptionConfig(),
        # Disable thinking/reasoning tokens — native-audio preview models can
        # emit thought=True parts which cause the Live API to close with 1011.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    if ENABLE_SERVER_VAD:
        config.realtime_input_config = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=20,
                silence_duration_ms=300,
            ),
        )
    if ENABLE_INPUT_AUDIO_TRANSCRIPTION:
        config.input_audio_transcription = types.AudioTranscriptionConfig()
    if ENABLE_OUTPUT_AUDIO_TRANSCRIPTION:
        config.output_audio_transcription = types.AudioTranscriptionConfig()
    if ENABLE_SESSION_RESUMPTION:
        config.session_resumption = types.SessionResumptionConfig(
            handle=session_handle,
        )
    return config


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
            await _broadcast_dashboard({
                "type": "dashboard",
                "subtype": "health",
                "model": LIVE_MODEL,
                "sessions_active": 0,
                "backend": "direct-live-api",
            })
        except Exception:
            logger.debug("dashboard.health_broadcast.error")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global genai_client

    if not GEMINI_API_KEY and not USE_VERTEX:
        logger.error("No GEMINI_API_KEY and Vertex AI not enabled")

    # Create the google-genai client
    if USE_VERTEX:
        logger.info("vertex_mode", project=GCP_PROJECT, location=GCP_LOCATION)
        genai_client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
        )
    else:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)

    logger.info(
        "rio.live.startup",
        model=LIVE_MODEL,
        vertex=USE_VERTEX,
        voice=VOICE_NAME,
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
    logger.info("rio.live.shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Rio Live API Server", version="2.0.0", lifespan=lifespan)

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
# Health / API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "rio-live-direct",
        "version": "2.0.0",
        "backend": "direct-live-api",
        "model": LIVE_MODEL,
        "vertex_ai": USE_VERTEX,
        "voice": VOICE_NAME,
    }


@app.get("/api/chat-history")
async def get_chat_history(limit: int = 100) -> dict:
    return {"messages": _transcript_buffer[-limit:], "total": len(_transcript_buffer)}


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
        "backend": "direct-live-api",
        "model": LIVE_MODEL,
    }
    return {"checks": checks}


@app.get("/api/models/status")
async def models_status() -> dict:
    return {
        "backend": "direct-live-api",
        "models": {
            "live": LIVE_MODEL,
            "orchestrator": ORCHESTRATOR_MODEL,
            "voice": VOICE_NAME,
        },
        "api_key_set": bool(GEMINI_API_KEY),
        "vertex_ai": USE_VERTEX,
        "gcp_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    }


@app.get("/api/settings")
async def get_settings() -> dict:
    """Return current agent settings from config.yaml."""
    import yaml
    if not _config_yaml_path.exists():
        return {"settings": {"agent_name": "Rio", "agent_tagline": "Your Virtual Friend & Assistant", "agent_role": "assistant"}}
    try:
        raw = yaml.safe_load(_config_yaml_path.read_text(encoding="utf-8")) or {}
        rio = raw.get("rio", {})
        return {
            "settings": {
                "agent_name": rio.get("agent_name", "Rio"),
                "agent_tagline": rio.get("agent_tagline", "Your Virtual Friend & Assistant"),
                "agent_role": rio.get("agent_role", "assistant"),
                "api_key_set": bool(GEMINI_API_KEY),
                "live_model": rio.get("models", {}).get("live", LIVE_MODEL),
                "orchestrator_model": rio.get("models", {}).get("primary", ORCHESTRATOR_MODEL),
            }
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/api/settings")
async def save_settings(request: Request) -> dict:
    """Update agent settings in config.yaml."""
    import yaml
    body = await request.json()
    try:
        raw = {}
        if _config_yaml_path.exists():
            raw = yaml.safe_load(_config_yaml_path.read_text(encoding="utf-8")) or {}
        rio = raw.setdefault("rio", {})

        # Only update fields that are provided
        for key in ("agent_name", "agent_tagline", "agent_role"):
            if key in body:
                rio[key] = body[key]

        _config_yaml_path.write_text(
            yaml.dump(raw, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        return {"status": "saved"}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Conversation API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/conversations")
async def list_conversations() -> dict:
    """List all saved conversations, newest first."""
    conversations = []
    for f in sorted(_conversations_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            conversations.append({
                "id": data.get("id", f.stem),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": data.get("message_count", 0),
                "preview": data.get("preview", ""),
            })
        except Exception:
            continue
    # Include current active conversation if it has messages
    if _current_conversation_id and _current_conversation_messages:
        active = {
            "id": _current_conversation_id,
            "created_at": _current_conversation_messages[0].get("timestamp", ""),
            "updated_at": _current_conversation_messages[-1].get("timestamp", ""),
            "message_count": len(_current_conversation_messages),
            "preview": _current_conversation_messages[0].get("text", "")[:100],
            "active": True,
        }
        # Replace if already in list, else prepend
        existing_ids = {c["id"] for c in conversations}
        if _current_conversation_id not in existing_ids:
            conversations.insert(0, active)
        else:
            for i, c in enumerate(conversations):
                if c["id"] == _current_conversation_id:
                    conversations[i] = active
                    break
    return {"conversations": conversations}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str) -> dict:
    """Get all messages for a specific conversation."""
    # Sanitize path to prevent directory traversal
    safe_id = Path(conv_id).name
    if safe_id != conv_id:
        return {"error": "Invalid conversation ID"}
    # Check if it's the current active conversation
    if conv_id == _current_conversation_id:
        return {
            "id": conv_id,
            "messages": _current_conversation_messages,
            "active": True,
        }
    filepath = _conversations_dir / f"{safe_id}.json"
    if not filepath.exists():
        return {"error": "Conversation not found"}
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/api/conversations/new")
async def new_conversation() -> dict:
    """End the current conversation and start a new one."""
    _save_conversation()
    cid = _start_new_conversation()
    return {"id": cid}


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
# WS /ws/rio/live — Direct Gemini Live bidi-streaming
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
            if (
                auth_frame.get("type") != "auth"
                or auth_frame.get("token") != WS_AUTH_TOKEN
            ):
                log.warning("client.auth_failed")
                await websocket.close(code=4001, reason="Authentication failed")
                return
        except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
            await websocket.close(code=4001, reason="Authentication required")
            return

    log.info("live.client.connected")
    await _broadcast_dashboard({
        "type": "dashboard", "subtype": "client_event",
        "event": "connected", "client_id": client_id,
    })

    assert genai_client is not None

    # ---- Create ToolBridge + tool functions ----
    bridge = ToolBridge(websocket, broadcast_fn=_broadcast_dashboard)
    tool_fns = _make_tools(bridge)
    tool_map = {fn.__name__: fn for fn in tool_fns}

    # Convert callables → proper FunctionDeclaration objects.
    # Used ONLY when LIVE_MODEL_TOOLS=True (i.e. the live model supports tools).
    # generate_content (used by the orchestrator) auto-converts callables itself.
    tool_objects = _callable_to_tool_declarations(genai_client, tool_fns)
    log.info(
        "tools.declared",
        count=sum(len(t.function_declarations or []) for t in tool_objects),
        live_model_tools=LIVE_MODEL_TOOLS,
    )

    # ---- Tool Orchestrator — parallel agentic executor ----
    # Uses ORCHESTRATOR_MODEL (gemini-2.5-flash by default) with full
    # function calling via generate_content while the live session handles
    # voice I/O only.
    orchestrator = ToolOrchestrator(
        genai_client=genai_client,
        tool_fns=tool_fns,
        model=ORCHESTRATOR_MODEL,
    )
    log.info(
        "orchestrator.ready",
        model=ORCHESTRATOR_MODEL,
        tools=len(tool_fns),
    )

    system_instruction = build_system_instruction()

    # ---- Notify client ----
    await websocket.send_text(json.dumps({
        "type": "control",
        "action": "connected",
        "detail": "Direct Live API session ready",
        "backend": "direct-live-api",
    }))

    # ---- Session with auto-reconnect ----
    session_handle: str | None = None
    active = True
    # Silent 20ms chunk for heartbeat (320 samples × 2 bytes = 640 bytes)
    silent_chunk = b"\x00" * 640

    while active:
        config = _build_live_config(tool_objects, system_instruction, session_handle)
        log.info(
            "live.config",
            server_vad=ENABLE_SERVER_VAD,
            input_audio_transcription=ENABLE_INPUT_AUDIO_TRANSCRIPTION,
            output_audio_transcription=ENABLE_OUTPUT_AUDIO_TRANSCRIPTION,
            session_resumption=ENABLE_SESSION_RESUMPTION,
        )

        try:
            async with genai_client.aio.live.connect(
                model=LIVE_MODEL, config=config,
            ) as session:

                if session_handle:
                    log.info("live.reconnected", handle=session_handle[:16])
                    await websocket.send_text(json.dumps({
                        "type": "control",
                        "action": "reconnected",
                        "detail": "Session resumed",
                    }))
                else:
                    log.info("live.session.started")

                await websocket.send_text(json.dumps({
                    "type": "control",
                    "action": "live_ready",
                }))

                # ---- Upstream: client WebSocket → Gemini session ----
                async def upstream_task() -> None:
                    nonlocal active
                    try:
                        while active:
                            ws_msg = await websocket.receive()

                            if ws_msg["type"] == "websocket.disconnect":
                                active = False
                                break

                            # Binary frames (audio / image)
                            raw_bytes = ws_msg.get("bytes")
                            if raw_bytes and len(raw_bytes) > 1:
                                prefix = raw_bytes[0:1]
                                payload = raw_bytes[1:]

                                if prefix == b"\x01":
                                    # Audio: PCM16 @ 16kHz
                                    await session.send_realtime_input(
                                        audio=types.Blob(
                                            data=payload,
                                            mime_type="audio/pcm;rate=16000",
                                        ),
                                    )
                                elif prefix == b"\x02":
                                    # Image: JPEG screenshot
                                    await session.send_realtime_input(
                                        video=types.Blob(
                                            data=payload,
                                            mime_type="image/jpeg",
                                        ),
                                    )
                                    await _broadcast_dashboard({
                                        "type": "dashboard",
                                        "subtype": "vision",
                                        "client_id": client_id,
                                        "detail": f"Screenshot ({len(payload)} bytes)",
                                    })
                                continue

                            # Text frames (JSON)
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
                                call_id = frame.get("id", "")
                                result = frame.get("result", {})
                                if call_id:
                                    bridge.resolve(call_id, result)
                                else:
                                    bridge.resolve_by_name(
                                        frame.get("name", ""), result,
                                    )
                                await _broadcast_dashboard({
                                    "type": "dashboard",
                                    "subtype": "tool_result",
                                    "client_id": client_id,
                                    "name": frame.get("name"),
                                    "success": result.get("success"),
                                })

                            elif frame_type == "text":
                                content = frame.get("content", "")
                                if content:
                                    await session.send_client_content(
                                        turns=types.Content(
                                            role="user",
                                            parts=[types.Part(text=content)],
                                        ),
                                        turn_complete=True,
                                    )
                                    _buffer_transcript({
                                        "speaker": "user",
                                        "text": content,
                                        "timestamp": datetime.now(
                                            timezone.utc,
                                        ).isoformat(),
                                    })
                                    await _broadcast_dashboard({
                                        "type": "transcript",
                                        "speaker": "user",
                                        "text": content,
                                        "client_id": client_id,
                                    })

                            elif frame_type == "context":
                                content = frame.get("content", "")
                                if content:
                                    await session.send_client_content(
                                        turns=types.Content(
                                            role="user",
                                            parts=[types.Part(text=f"[CONTEXT] {content}")],
                                        ),
                                        turn_complete=False,
                                    )

                            elif frame_type == "heartbeat":
                                pass  # Handled by heartbeat task

                    except WebSocketDisconnect:
                        log.info("upstream.client_disconnected")
                        active = False
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        log.exception("upstream.error")
                        active = False

                # ---- Heartbeat: keep Gemini session alive ----
                async def heartbeat_task() -> None:
                    while active:
                        try:
                            await asyncio.sleep(5)
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=silent_chunk,
                                    mime_type="audio/pcm;rate=16000",
                                ),
                            )
                        except asyncio.CancelledError:
                            break
                        except Exception:
                            log.debug("heartbeat.error")
                            break

                # ---- inject_context: send SYSTEM message to live session ----
                async def inject_context(msg: str) -> None:
                    """Inject an orchestrator result into the live audio session."""
                    try:
                        await session.send_client_content(
                            turns=types.Content(
                                role="user",
                                parts=[types.Part(text=msg)],
                            ),
                            turn_complete=True,
                        )
                    except Exception:
                        log.debug("orchestrator.inject.failed")

                # ---- Downstream: Gemini session → client WebSocket ----
                async def downstream_task() -> None:
                    nonlocal session_handle
                    # Buffer partial transcription fragments until turn complete
                    _utterance_buf: list[str] = []
                    try:
                        async for response in session.receive():
                            sc = response.server_content

                            # ---- Audio / text from model ----
                            if sc and sc.model_turn:
                                for part in sc.model_turn.parts or []:
                                    # Skip internal reasoning tokens — they
                                    # cause 1011 crashes if the model leaks
                                    # them and should never reach the client.
                                    if getattr(part, "thought", False):
                                        continue
                                    if part.inline_data and part.inline_data.data:
                                        await websocket.send_bytes(
                                            b"\x01" + part.inline_data.data,
                                        )
                                    if part.text:
                                        await websocket.send_text(json.dumps({
                                            "type": "transcript",
                                            "speaker": "rio",
                                            "text": part.text,
                                        }))
                                        _buffer_transcript({
                                            "speaker": "rio",
                                            "text": part.text,
                                            "timestamp": datetime.now(
                                                timezone.utc,
                                            ).isoformat(),
                                        })
                                        await _broadcast_dashboard({
                                            "type": "transcript",
                                            "speaker": "rio",
                                            "text": part.text,
                                            "client_id": client_id,
                                        })

                            # ---- Input transcription (user speech) ----
                            if sc and getattr(sc, "input_transcription", None):
                                tx = sc.input_transcription
                                text = getattr(tx, "text", "")
                                if text:
                                    _utterance_buf.append(text)
                                    _buffer_transcript({
                                        "speaker": "user",
                                        "text": text,
                                        "timestamp": datetime.now(
                                            timezone.utc,
                                        ).isoformat(),
                                        "source": "transcription",
                                    })
                                    await websocket.send_text(json.dumps({
                                        "type": "transcript",
                                        "speaker": "user",
                                        "text": text,
                                        "source": "transcription",
                                    }))
                                    await _broadcast_dashboard({
                                        "type": "transcript",
                                        "speaker": "user",
                                        "text": text,
                                        "client_id": client_id,
                                        "source": "transcription",
                                    })

                            # ---- Output transcription (model speech) ----
                            if sc and getattr(sc, "output_transcription", None):
                                tx = sc.output_transcription
                                text = getattr(tx, "text", "")
                                if text:
                                    await websocket.send_text(json.dumps({
                                        "type": "transcript",
                                        "speaker": "rio",
                                        "text": text,
                                        "source": "transcription",
                                    }))

                            # ---- Interrupted by user ----
                            if sc and getattr(sc, "interrupted", False):
                                log.info("live.interrupted")
                                _utterance_buf.clear()  # Discard partial utterance
                                await websocket.send_text(json.dumps({
                                    "type": "control",
                                    "action": "interrupted",
                                }))

                            # ---- Turn complete ----
                            # The model finished responding → the user's full
                            # utterance is now finalised in _utterance_buf.
                            # Check if it's an executable task and defer to
                            # the ToolOrchestrator if so.
                            if sc and getattr(sc, "turn_complete", False):
                                await websocket.send_text(json.dumps({
                                    "type": "control",
                                    "action": "turn_complete",
                                }))

                                # Flush buffered transcription
                                full_utterance = " ".join(_utterance_buf).strip()
                                _utterance_buf.clear()

                                if full_utterance and _is_task_request(full_utterance):
                                    log.info(
                                        "orchestrator.task_detected",
                                        utterance=full_utterance[:100],
                                    )
                                    # Notify client the orchestrator is running
                                    await websocket.send_text(json.dumps({
                                        "type": "control",
                                        "action": "task_mode",
                                        "active": True,
                                        "goal": full_utterance[:100],
                                    }))
                                    # Spawn the orchestrator as a background task.
                                    # Tool results flow through ToolBridge → client
                                    # transparently, just like before.
                                    orchestrator.spawn_task(
                                        full_utterance, inject_context,
                                    )
                                    await _broadcast_dashboard({
                                        "type": "dashboard",
                                        "subtype": "orchestrator",
                                        "event": "task_started",
                                        "goal": full_utterance[:100],
                                        "model": ORCHESTRATOR_MODEL,
                                        "client_id": client_id,
                                    })

                            # ---- Session resumption update ----
                            sru = getattr(response, "session_resumption_update", None)
                            if sru:
                                new_handle = getattr(sru, "new_handle", None)
                                if new_handle:
                                    session_handle = new_handle
                                    log.info(
                                        "live.session_handle_saved",
                                        handle=session_handle[:16],
                                    )

                            # ---- Tool calls from model ----
                            tc = getattr(response, "tool_call", None)
                            if tc and tc.function_calls:
                                # Screen-action tools that change what's on screen.
                                # When the model batches multiple screen actions in
                                # one turn, execute ONLY the first one and return
                                # its result so the auto-captured screenshot can be
                                # processed before the model decides the next step.
                                # This forces a verify-then-act loop instead of
                                # blind fire-and-forget sequences.
                                _SCREEN_ACTIONS = frozenset({
                                    "smart_click", "screen_click", "screen_type",
                                    "screen_scroll", "screen_hotkey", "screen_move",
                                    "screen_drag", "focus_window",
                                })

                                calls_to_run = list(tc.function_calls)
                                # If there are multiple calls and any is a screen
                                # action, only run up to and including the first
                                # screen action — drop the rest so the model
                                # re-evaluates after seeing the screenshot.
                                if len(calls_to_run) > 1:
                                    trimmed = []
                                    for fc in calls_to_run:
                                        trimmed.append(fc)
                                        if fc.name in _SCREEN_ACTIONS:
                                            if len(trimmed) < len(calls_to_run):
                                                log.info(
                                                    "tool_call.batch_trimmed",
                                                    executed=fc.name,
                                                    dropped=[c.name for c in calls_to_run[len(trimmed):]],
                                                    reason="verify screen action before continuing",
                                                )
                                            break
                                    calls_to_run = trimmed

                                fn_responses = []
                                for fc in calls_to_run:
                                    fn = tool_map.get(fc.name)
                                    if fn:
                                        log.info(
                                            "tool_call",
                                            name=fc.name,
                                            args=fc.args,
                                        )
                                        try:
                                            result = await fn(**fc.args)
                                        except Exception as exc:
                                            log.exception(
                                                "tool_call.error", name=fc.name,
                                            )
                                            result = {
                                                "success": False,
                                                "error": str(exc),
                                            }
                                    else:
                                        log.warning(
                                            "tool_call.unknown", name=fc.name,
                                        )
                                        result = {
                                            "success": False,
                                            "error": f"Unknown tool: {fc.name}",
                                        }

                                    fn_responses.append(
                                        types.FunctionResponse(
                                            name=fc.name,
                                            response=result,
                                            id=fc.id,
                                        ),
                                    )

                                    # After a screen action, pause briefly so the
                                    # auto-capture screenshot can arrive and be fed
                                    # into the session before the model continues.
                                    if fc.name in _SCREEN_ACTIONS:
                                        await asyncio.sleep(0.5)

                                # Send tool responses back to model
                                await session.send_tool_response(
                                    function_responses=fn_responses,
                                )

                    except WebSocketDisconnect:
                        log.info("downstream.client_disconnected")
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        log.exception("downstream.error")

                # ---- Run upstream + heartbeat + downstream concurrently ----
                tasks = [
                    asyncio.create_task(upstream_task(), name="upstream"),
                    asyncio.create_task(heartbeat_task(), name="heartbeat"),
                    asyncio.create_task(downstream_task(), name="downstream"),
                ]
                try:
                    done, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    # Propagate exceptions from completed tasks
                    for t in done:
                        if t.exception():
                            log.error("task.error", task=t.get_name(), exc=t.exception())
                except Exception:
                    log.exception("live.session.error")
                    for t in tasks:
                        t.cancel()

        except WebSocketDisconnect:
            log.info("client.disconnected")
            active = False
        except Exception as exc:
            log.error("live.connection.error", error=str(exc))
            if active:
                log.info("live.reconnecting", delay=2)
                try:
                    await websocket.send_text(json.dumps({
                        "type": "control",
                        "action": "reconnecting",
                    }))
                except Exception:
                    # Client already disconnected — stop reconnect loop
                    active = False
                    break
                await asyncio.sleep(2)

    # Cancel any still-running orchestrator tasks for this session
    orchestrator.cancel_all()
    # Persist conversation on disconnect
    _save_conversation()
    log.info("live.client.cleanup_done", client_id=client_id)
    await _broadcast_dashboard({
        "type": "dashboard", "subtype": "client_event",
        "event": "disconnected", "client_id": client_id,
    })


# ---------------------------------------------------------------------------
# Profile API (same contract as before)
# ---------------------------------------------------------------------------

_profiles_mod = None
_profiles_base = str(Path(__file__).resolve().parent.parent / "rio_profiles")


def _get_profiles_mod():
    global _profiles_mod
    if _profiles_mod is None:
        import importlib
        import sys
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
