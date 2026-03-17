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
import importlib
import importlib.util
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional, Set

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from starlette.websockets import WebSocketState
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

try:
    from .gemini_session import build_system_instruction
    from .rio_agent import ToolBridge, _make_tools
    from .tool_orchestrator import ToolOrchestrator, _is_task_request
    from .voice_plugin import apply_voice_plugin, load_voice_runtime
except ImportError:
    from gemini_session import build_system_instruction
    from rio_agent import ToolBridge, _make_tools
    from tool_orchestrator import ToolOrchestrator, _is_task_request
    from voice_plugin import apply_voice_plugin, load_voice_runtime

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
VOICE_RUNTIME = load_voice_runtime(VOICE_NAME)
ACTIVE_VOICE_NAME = VOICE_RUNTIME.active_voice


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

# Model resolution: env → yaml → hardcoded default
_DEFAULT_MODEL = "gemini-2.5-flash-native-audio-latest"


async def _classify_intent_via_llm(client: genai.Client, text: str) -> bool:
    """Fallback classifier: use Gemini 3 Flash to decide if utterance is a task."""
    try:
        prompt = (
            f"User Utterance: \"{text}\"\n\n"
            "Analyze if the user is asking the AI to perform a specific action on their computer "
            "(e.g., opening an app, sending a message, researching a topic, writing code, "
            "modifying files, or managing their calendar/email).\n"
            "If it is a task request, reply exactly with: TASK\n"
            "If it is just a greeting, a simple question, or casual conversation, reply exactly with: CONV"
        )
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=5,
            )
        )
        ans = response.text.strip().upper()
        return "TASK" in ans
    except Exception:
        return False


def _resolve_models() -> tuple[str, str]:
    """Resolve live/orchestrator models from config in a robust way.

    Running adk_server.py directly from rio/cloud can break package imports like
    ``from rio.local.config import get_model``. This loader tries:
      1) normal package import
      2) direct load of rio/local/config.py via importlib
      3) env/default fallback
    """
    try:
        from rio.local.config import get_model as _get_model
        return _get_model("live"), _get_model("orchestrator")
    except Exception:
        pass

    try:
        config_path = Path(__file__).resolve().parent.parent / "local" / "config.py"
        spec = importlib.util.spec_from_file_location("rio_local_config", str(config_path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_model = getattr(module, "get_model", None)
            if callable(get_model):
                return get_model("live"), get_model("orchestrator")
    except Exception:
        pass

    return (
        os.environ.get("LIVE_MODEL", _DEFAULT_MODEL),
        os.environ.get("ORCHESTRATOR_MODEL", "gemini-3-flash-preview"),
    )


LIVE_MODEL, ORCHESTRATOR_MODEL = _resolve_models()
# Whether the live model itself can invoke function calls.
# gemini-2.5-flash-native-audio-preview-12-2025 DOES support function calling
# per https://ai.google.dev/gemini-api/docs/live-api/tools
# Use orchestrator-mediated tools by default to avoid spontaneous direct
# tool execution from residual model context during startup.
LIVE_MODEL_TOOLS = _env_flag("RIO_LIVE_MODEL_TOOLS", False)

ENABLE_SERVER_VAD = _env_flag("RIO_LIVE_ENABLE_SERVER_VAD", False)
ENABLE_INPUT_AUDIO_TRANSCRIPTION = _env_flag(
    "RIO_LIVE_ENABLE_INPUT_AUDIO_TRANSCRIPTION", False,
)
ENABLE_OUTPUT_AUDIO_TRANSCRIPTION = _env_flag(
    "RIO_LIVE_ENABLE_OUTPUT_AUDIO_TRANSCRIPTION", False,
)
ENABLE_SESSION_RESUMPTION = _env_flag(
    "RIO_LIVE_ENABLE_SESSION_RESUMPTION", True,
)
ENABLE_HISTORY_RESTORE = _env_flag(
    "RIO_LIVE_ENABLE_HISTORY_RESTORE", False,
)
ENABLE_PERSISTENT_CLIENT_SESSION = _env_flag(
    "RIO_LIVE_ENABLE_PERSISTENT_CLIENT_SESSION", False,
)
ENABLE_RESUME_INTERRUPTED_TASKS = _env_flag(
    "RIO_LIVE_ENABLE_RESUME_INTERRUPTED_TASKS", False,
)

# ---------------------------------------------------------------------------
# Shared memory store singleton — loaded once at module level
# ---------------------------------------------------------------------------
_shared_memory_store = None
_memory_store_loaded = False


def _get_shared_memory_store(genai_client: Optional[genai.Client] = None):
    """Return a shared MemoryStore singleton (lazy-init on first call)."""
    global _shared_memory_store, _memory_store_loaded
    if _memory_store_loaded:
        if _shared_memory_store and genai_client:
            _shared_memory_store.set_client(genai_client)
        return _shared_memory_store
    _memory_store_loaded = True
    try:
        import sys, pathlib
        _local_dir = str(pathlib.Path(__file__).resolve().parent.parent / "local")
        if _local_dir not in sys.path:
            sys.path.insert(0, _local_dir)
        from memory import MemoryStore
        _shared_memory_store = MemoryStore(genai_client=genai_client)
        logger.info("memory_store.singleton_loaded")
    except Exception as exc:
        logger.warning("memory_store.singleton_unavailable", error=str(exc))
    return _shared_memory_store


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
genai_client: genai.Client | None = None

# Dashboard clients
dashboard_clients: Set[WebSocket] = set()
dashboard_lock = asyncio.Lock()

# Priority 4.4: basic user-scoped runtime registry
_trigger_engines_by_user: dict[str, Any] = {}

# Session continuity registry: user_id -> (ToolBridge, ToolOrchestrator)
_active_sessions_by_user: dict[str, tuple[Any, Any]] = {}

# Transcript buffer for dashboard
_transcript_buffer: list[dict] = []
_TRANSCRIPT_BUFFER_MAX = 200

# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------
_conversations_dir = Path(__file__).resolve().parent.parent / "data" / "conversations"
_conversations_dir.mkdir(parents=True, exist_ok=True)
_user_configs_dir = Path(__file__).resolve().parent.parent / "data" / "users"
_user_configs_dir.mkdir(parents=True, exist_ok=True)
_workspaces_dir = Path(__file__).resolve().parent.parent / "data" / "workspaces"
_workspaces_dir.mkdir(parents=True, exist_ok=True)

_current_conversation_id: str | None = None
_current_conversation_messages: list[dict] = []
_CONVERSATION_MESSAGES_MAX = 500  # Cap to prevent unbounded RAM growth


def _start_new_conversation() -> str:
    """Start a new conversation and return its ID."""
    global _current_conversation_id, _current_conversation_messages
    _current_conversation_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    _current_conversation_messages = []
    return _current_conversation_id


def _load_latest_conversation() -> None:
    """Load the most recent conversation from disk on startup."""
    global _current_conversation_id, _current_conversation_messages
    try:
        files = sorted(_conversations_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        if files:
            latest = files[0]
            data = json.loads(latest.read_text(encoding="utf-8"))
            _current_conversation_id = data.get("id")
            _current_conversation_messages = data.get("messages", [])
            logger.info("conversation.loaded", id=_current_conversation_id, messages=len(_current_conversation_messages))
        else:
            _start_new_conversation()
    except Exception as exc:
        logger.warning("conversation.load_failed", error=str(exc))
        _start_new_conversation()


def _get_history_contents() -> list[types.Content]:
    """Convert stored transcript messages into SDK Content objects for restoration."""
    contents = []
    for msg in _current_conversation_messages:
        speaker = msg.get("speaker", "user")
        text = msg.get("text", "")
        if not text:
            continue
        
        # Map Rio internal speakers to Gemini roles
        role = "model" if speaker == "rio" else "user"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=text)]
        ))
    
    # Filter to ensure alternating roles (required by some versions of the API)
    # and that it doesn't end with a model turn if we're about to add a user turn.
    cleaned = []
    for c in contents:
        if not cleaned or cleaned[-1].role != c.role:
            cleaned.append(c)
        else:
            # Merge consecutive same-role turns
            cleaned[-1].parts.extend(c.parts)
            
    return cleaned


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
    # Trim oldest messages if over limit
    if len(_current_conversation_messages) > _CONVERSATION_MESSAGES_MAX:
        # Save before trimming
        _save_conversation()
        _current_conversation_messages[:] = _current_conversation_messages[-(_CONVERSATION_MESSAGES_MAX // 2):]
    # Auto-save every 5 messages
    if len(_current_conversation_messages) % 5 == 0:
        _save_conversation()


def _sanitize_user_id(value: str | None) -> str:
    raw = (value or "default").strip().lower()
    if not raw:
        return "default"
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in raw)
    return (safe[:64] or "default")


def _default_user_config(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "workspace_policy": {
            "shared_workspace_enabled": True,
            "allowed_shared_users": [],
            "read_paths": ["."],
            "write_paths": ["."],
        },
        "preferences": {
            "preferred_channel": "telegram",
        },
    }


def _user_config_path(user_id: str) -> Path:
    return _user_configs_dir / f"{_sanitize_user_id(user_id)}.json"


def _load_user_config(user_id: str) -> dict[str, Any]:
    safe_user = _sanitize_user_id(user_id)
    path = _user_config_path(safe_user)
    default_cfg = _default_user_config(safe_user)
    if not path.exists():
        return default_cfg
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return default_cfg
    except Exception:
        return default_cfg

    cfg = dict(default_cfg)
    cfg.update(raw)
    wp = cfg.get("workspace_policy", {})
    if not isinstance(wp, dict):
        wp = {}
    merged_wp = dict(default_cfg["workspace_policy"])
    merged_wp.update(wp)
    merged_wp["read_paths"] = list(merged_wp.get("read_paths") or ["."])
    merged_wp["write_paths"] = list(merged_wp.get("write_paths") or ["."])
    cfg["workspace_policy"] = merged_wp
    cfg["user_id"] = safe_user
    return cfg


def _save_user_config(user_id: str, config: dict[str, Any]) -> dict[str, Any]:
    safe_user = _sanitize_user_id(user_id)
    merged = _load_user_config(safe_user)
    merged.update(config or {})
    merged["user_id"] = safe_user

    wp = merged.get("workspace_policy", {})
    if not isinstance(wp, dict):
        wp = {}
    merged_wp = dict(_default_user_config(safe_user)["workspace_policy"])
    merged_wp.update(wp)
    merged_wp["read_paths"] = list(merged_wp.get("read_paths") or ["."])
    merged_wp["write_paths"] = list(merged_wp.get("write_paths") or ["."])
    merged["workspace_policy"] = merged_wp

    path = _user_config_path(safe_user)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


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

    def _audio_transcription_config() -> types.AudioTranscriptionConfig:
        """Build transcription config across SDK versions.

        Some google-genai versions expose AudioTranscriptionConfig without
        accepting language_code. Newer variants may include that field.
        """
        fields = getattr(types.AudioTranscriptionConfig, "model_fields", {}) or {}
        if "language_code" in fields:
            try:
                return types.AudioTranscriptionConfig(language_code="en")
            except Exception:
                pass
        return types.AudioTranscriptionConfig()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=ACTIVE_VOICE_NAME,
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
        input_audio_transcription=_audio_transcription_config(),
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
        config.input_audio_transcription = _audio_transcription_config()
    if ENABLE_OUTPUT_AUDIO_TRANSCRIPTION:
        config.output_audio_transcription = _audio_transcription_config()
    if ENABLE_SESSION_RESUMPTION:
        config.session_resumption = types.SessionResumptionConfig(
            handle=session_handle,
        )
    config = apply_voice_plugin(config, VOICE_RUNTIME)
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
        voice=ACTIVE_VOICE_NAME,
        voice_plugin=VOICE_RUNTIME.plugin_name or "none",
        project=GCP_PROJECT or "(not set)",
    )

    # Pre-warm memory store so first WS connection doesn't block 4-6s
    _get_shared_memory_store(genai_client=genai_client)

    # Load latest conversation from disk to resume state
    _load_latest_conversation()

    # A4+G2: Run startup maintenance (prune old conversations, vacuum DBs)
    try:
        import sys as _sys, pathlib as _pl
        _local_dir = str(_pl.Path(__file__).resolve().parent.parent / "local")
        if _local_dir not in _sys.path:
            _sys.path.insert(0, _local_dir)
        from maintenance import run_maintenance
        _data_dir = _pl.Path(__file__).resolve().parent.parent / "data"
        asyncio.get_event_loop().run_in_executor(
            None, lambda: run_maintenance(data_dir=_data_dir)
        )
    except Exception as _maint_err:
        logger.warning("maintenance.startup_skip", error=str(_maint_err))

    # F2: Connect MCP client to configured external tool servers
    try:
        from mcp_client import McpClient
        import yaml as _yaml
        _cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        _mcp_config = []
        if _cfg_path.is_file():
            _cfg_data = _yaml.safe_load(_cfg_path.read_text(encoding="utf-8")) or {}
            _mcp_config = _cfg_data.get("rio", {}).get("mcp_servers", [])
        if _mcp_config:
            _mcp_client = McpClient()
            asyncio.create_task(_mcp_client.connect_from_config(_mcp_config))
            app.state.mcp_client = _mcp_client
            logger.info("mcp_client.startup", servers=len(_mcp_config))
        else:
            app.state.mcp_client = None
    except Exception as _mcp_err:
        logger.debug("mcp_client.startup_skip", error=str(_mcp_err))
        app.state.mcp_client = None

    # F5: Initialize notifier (optional Telegram push)
    try:
        import sys as _sys2, pathlib as _pl2
        _local_dir2 = str(_pl2.Path(__file__).resolve().parent.parent / "local")
        if _local_dir2 not in _sys2.path:
            _sys2.path.insert(0, _local_dir2)
        from notifier import Notifier
        app.state.notifier = Notifier()
        if app.state.notifier.is_enabled:
            logger.info("notifier.initialized")
    except Exception as _notif_err:
        logger.debug("notifier.startup_skip", error=str(_notif_err))
        app.state.notifier = None

    # F6: Initialize bidirectional Telegram bot (long-polling)
    try:
        from telegram_bot import TelegramBot
        app.state.telegram_bot = TelegramBot()
        if app.state.telegram_bot.enabled:
            app.state.telegram_bot.start()
            logger.info("telegram_bot.initialized")
    except Exception as _tg_err:
        logger.debug("telegram_bot.startup_skip", error=str(_tg_err))
        app.state.telegram_bot = None

    # Priority 4.2: ChannelManager (Telegram + WhatsApp only)
    try:
        import sys as _sys3, pathlib as _pl3
        _local_dir3 = str(_pl3.Path(__file__).resolve().parent.parent / "local")
        if _local_dir3 not in _sys3.path:
            _sys3.path.insert(0, _local_dir3)
        from channel_manager import ChannelManager, TelegramChannel
        from whatsapp_channel import WhatsAppChannel

        _tg_channel = TelegramChannel(getattr(app.state, "telegram_bot", None))
        _wa_channel = WhatsAppChannel()
        app.state.channel_manager = ChannelManager([_tg_channel, _wa_channel])
        logger.info(
            "channel_manager.initialized",
            channels=app.state.channel_manager.enabled_channels(),
        )
    except Exception as _ch_err:
        logger.debug("channel_manager.startup_skip", error=str(_ch_err))
        app.state.channel_manager = None

    health_task = asyncio.create_task(
        _dashboard_health_broadcast_loop(), name="dashboard-health",
    )

    yield

    # Shutdown
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass
    # Stop Telegram bot polling
    _tg_bot = getattr(app.state, "telegram_bot", None)
    if _tg_bot:
        _tg_bot.stop()
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
        "voice": ACTIVE_VOICE_NAME,
        "voice_plugin": VOICE_RUNTIME.plugin_name,
        "voice_plugin_metadata": VOICE_RUNTIME.plugin_metadata,
    }


@app.get("/api/chat-history")
async def get_chat_history(limit: int = 100) -> dict:
    return {"messages": _transcript_buffer[-limit:], "total": len(_transcript_buffer)}


@app.get("/api/users/{user_id}/config")
async def get_user_config(user_id: str) -> dict:
    safe_user = _sanitize_user_id(user_id)
    return {"config": _load_user_config(safe_user)}


@app.post("/api/users/{user_id}/config")
async def save_user_config(user_id: str, request: Request) -> dict:
    body = await request.json()
    if not isinstance(body, dict):
        return {"success": False, "error": "JSON object body required"}
    saved = _save_user_config(_sanitize_user_id(user_id), body)
    return {"success": True, "config": saved}


@app.get("/api/triggers")
async def list_triggers(user_id: str = "default") -> dict:
    safe_user = _sanitize_user_id(user_id)
    engine = _trigger_engines_by_user.get(safe_user)
    if engine is None:
        return {"user_id": safe_user, "triggers": [], "active": False}
    return {"user_id": safe_user, "triggers": engine.list_triggers(), "active": True}


@app.post("/api/triggers/schedule")
async def add_schedule_trigger(request: Request) -> dict:
    body = await request.json()
    user_id = _sanitize_user_id(str(body.get("user_id", "default")))
    engine = _trigger_engines_by_user.get(user_id)
    if engine is None:
        return {"success": False, "error": f"No active session for user '{user_id}'"}

    name = str(body.get("name", "")).strip() or f"schedule_{int(datetime.now(timezone.utc).timestamp())}"
    goal = str(body.get("goal", "")).strip()
    interval_seconds = int(body.get("interval_seconds", 0) or 0)
    cooldown_seconds = int(body.get("cooldown_seconds", min(max(interval_seconds, 30), 60)) or 60)
    if not goal or interval_seconds <= 0:
        return {"success": False, "error": "goal and interval_seconds are required"}

    trigger = engine.add_schedule(
        name=name,
        interval_seconds=interval_seconds,
        goal=goal,
        cooldown_seconds=cooldown_seconds,
    )
    return {
        "success": True,
        "user_id": user_id,
        "trigger": {
            "name": trigger.name,
            "type": trigger.trigger_type,
            "goal": trigger.goal,
            "interval_seconds": trigger.interval_seconds,
            "run_once": trigger.run_once,
        },
    }


@app.delete("/api/triggers/{name}")
async def remove_trigger(name: str, user_id: str = "default") -> dict:
    safe_user = _sanitize_user_id(user_id)
    engine = _trigger_engines_by_user.get(safe_user)
    if engine is None:
        return {"success": False, "error": f"No active session for user '{safe_user}'"}
    ok = engine.remove(name)
    return {"success": ok, "user_id": safe_user, "name": name}


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
            "voice": ACTIVE_VOICE_NAME,
            "voice_plugin": VOICE_RUNTIME.plugin_name,
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
# Multi-Agent configuration API
# ---------------------------------------------------------------------------

@app.get("/api/agents")
async def get_agents() -> dict:
    """Return multi-agent configurations from config.yaml."""
    import yaml
    if not _config_yaml_path.exists():
        return {"agents": {}}
    try:
        raw = yaml.safe_load(_config_yaml_path.read_text(encoding="utf-8")) or {}
        agents = raw.get("rio", {}).get("agents", {})
        return {"agents": agents}
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/api/agents")
async def save_agents(request: Request) -> dict:
    """Update multi-agent configurations in config.yaml."""
    import yaml
    body = await request.json()
    agents_data = body.get("agents", {})
    try:
        raw = {}
        if _config_yaml_path.exists():
            raw = yaml.safe_load(_config_yaml_path.read_text(encoding="utf-8")) or {}
        rio = raw.setdefault("rio", {})
        rio["agents"] = agents_data

        _config_yaml_path.write_text(
            yaml.dump(raw, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        return {"status": "saved", "agents": agents_data}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Evaluation API endpoints (Agent Factory Podcast framework)
# ---------------------------------------------------------------------------

@app.get("/api/evaluation/stats")
async def evaluation_stats() -> dict:
    """Return aggregate evaluation statistics.

    Provides data on task success rates, per-agent scores,
    trajectory metrics, and LLM-as-judge dimension scores.
    Powered by the EvaluationStore in ToolOrchestrator.
    """
    # The orchestrator is per-session, so we return stats from the
    # module-level evaluation store if available.
    try:
        from evaluation import EvaluationStore
        # Access via the orchestrator attached to the current session
        # For now, return a placeholder that gets populated when sessions are active
        return {
            "status": "evaluation_available",
            "note": "Evaluation data is per-session. Connect via WebSocket to populate.",
            "framework": "Agent Factory Podcast (Google Cloud)",
            "dimensions": [
                "task_completion", "efficiency", "safety",
                "output_quality", "reasoning_quality",
                "hallucination_risk", "memory_relevance",
            ],
            "enable_hint": "Set RIO_EVAL_ENABLED=1 environment variable to enable LLM-as-judge scoring.",
        }
    except ImportError:
        return {"error": "evaluation module not available"}


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
# WS /ui/stream -- Dashboard & Mission Control feed
# ---------------------------------------------------------------------------

async def _ws_dashboard_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    async with dashboard_lock:
        dashboard_clients.add(websocket)
    logger.info("ui_stream.connected", total=len(dashboard_clients))
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
            else:
                # B2: UI interaction responses
                try:
                    frame = json.loads(msg)
                    if frame.get("type") == "approval_response":
                        # Match OpenClaw UI naming: decision=approve/deny
                        approved = frame.get("decision") == "approve" or frame.get("approved", False)
                        await _broadcast_dashboard({
                            "type": "dashboard",
                            "subtype": "approval_resolved",
                            "approved": approved,
                            "id": frame.get("id"),
                        })
                    elif frame.get("type") == "task_stop":
                        # For single-user desktop, we stop all active tasks
                        # In adk_server, orchestrator is session-scoped, so we broadcast a control event
                        await _broadcast_dashboard({
                            "type": "control",
                            "action": "task_stop_requested",
                        })
                except (json.JSONDecodeError, Exception):
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        async with dashboard_lock:
            dashboard_clients.discard(websocket)
        logger.info("ui_stream.disconnected", total=len(dashboard_clients))


@app.websocket("/ui/stream")
async def ws_ui_stream(websocket: WebSocket) -> None:
    await _ws_dashboard_stream(websocket)


@app.websocket("/ws/dashboard")
async def ws_dashboard_compat(websocket: WebSocket) -> None:
    """Backward-compatible dashboard stream endpoint.

    Some clients still connect to /ws/dashboard. Route them to the
    same dashboard stream handler used by /ui/stream.
    """
    await _ws_dashboard_stream(websocket)


# ---------------------------------------------------------------------------
# WS /ws/rio/live — Direct Gemini Live bidi-streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/rio/live")
async def ws_rio_live(websocket: WebSocket) -> None:
    await websocket.accept()
    client_id = str(uuid.uuid4())
    user_id = _sanitize_user_id(
        (
        websocket.query_params.get("user_id")
        or websocket.headers.get("x-rio-user-id")
        or "default"
        )
    )
    log = logger.bind(client_id=client_id, user_id=user_id)

    # Session isolation key:
    # - If client provides a stable session_id, it can be used for explicit
    #   continuity (when persistent sessions are enabled).
    # - Otherwise, default to this websocket's unique client_id so every new
    #   connection starts with clean in-memory context.
    raw_session_id = (
        websocket.query_params.get("session_id")
        or websocket.headers.get("x-rio-session-id")
        or ""
    )
    client_session_id = _sanitize_user_id(raw_session_id) if raw_session_id else client_id
    session_key = f"{user_id}:{client_session_id}"

    # Per-live-session conversation state (kept during this WS session and
    # across transient Live API reconnects inside the same WS lifecycle).
    session_conversation_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    session_conversation_messages: list[dict] = []

    def _save_session_conversation() -> None:
        if not session_conversation_messages:
            return
        filepath = _conversations_dir / f"{session_conversation_id}.json"
        data = {
            "id": session_conversation_id,
            "session_key": session_key,
            "user_id": user_id,
            "created_at": session_conversation_messages[0].get("timestamp", ""),
            "updated_at": session_conversation_messages[-1].get("timestamp", ""),
            "message_count": len(session_conversation_messages),
            "preview": session_conversation_messages[0].get("text", "")[:100],
            "messages": session_conversation_messages,
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _buffer_session_transcript(entry: dict) -> None:
        _transcript_buffer.append(entry)
        if len(_transcript_buffer) > _TRANSCRIPT_BUFFER_MAX:
            _transcript_buffer.pop(0)

        session_conversation_messages.append(entry)
        if len(session_conversation_messages) > _CONVERSATION_MESSAGES_MAX:
            _save_session_conversation()
            session_conversation_messages[:] = session_conversation_messages[-(_CONVERSATION_MESSAGES_MAX // 2):]
        if len(session_conversation_messages) % 5 == 0:
            _save_session_conversation()

    def _get_session_history_contents() -> list[types.Content]:
        contents: list[types.Content] = []
        for msg in session_conversation_messages:
            speaker = msg.get("speaker", "user")
            text = msg.get("text", "")
            if not text:
                continue
            role = "model" if speaker == "rio" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

        cleaned: list[types.Content] = []
        for c in contents:
            if not cleaned or cleaned[-1].role != c.role:
                cleaned.append(c)
            else:
                cleaned[-1].parts.extend(c.parts)
        return cleaned

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
        "event": "connected", "client_id": client_id, "user_id": user_id,
    })

    assert genai_client is not None
    user_cfg = _load_user_config(user_id)
    workspace_policy = user_cfg.get("workspace_policy", {}) if isinstance(user_cfg, dict) else {}
    if not isinstance(workspace_policy, dict):
        workspace_policy = {}

    shared_enabled = bool(workspace_policy.get("shared_workspace_enabled", True))
    allowed_shared_users = {
        _sanitize_user_id(v)
        for v in (workspace_policy.get("allowed_shared_users", []) or [])
        if str(v).strip()
    }
    if shared_enabled and allowed_shared_users and user_id not in allowed_shared_users:
        await websocket.send_text(json.dumps({
            "type": "control",
            "action": "workspace_access_denied",
            "detail": "User not allowed for shared workspace policy",
        }))
        await websocket.close(code=4003, reason="shared workspace access denied")
        return

    # ---- Create ToolBridge + tool functions ----
    # Check for existing session to facilitate auto-resumption (Persistent Lane)
    
    # We define a deferred inject proxy because the actual inject_context
    # is defined further down, inside the active session block.
    async def _deferred_inject(msg: str) -> None:
        pass # Will be swapped later
    
    if ENABLE_PERSISTENT_CLIENT_SESSION and session_key in _active_sessions_by_user:
        log.info("live.client.resuming_session", user_id=user_id, session_key=session_key)
        bridge, orchestrator = _active_sessions_by_user[session_key]
        bridge.rebind(websocket, _broadcast_dashboard)
        # Update inject_context for running tasks
        orchestrator.rebind(_deferred_inject)
        tool_fns = _make_tools(bridge)
        tool_map = {fn.__name__: fn for fn in tool_fns}
    else:
        bridge = ToolBridge(websocket, broadcast_fn=_broadcast_dashboard)
        tool_fns = _make_tools(bridge)
        tool_map = {fn.__name__: fn for fn in tool_fns}

        # ---- Tool Orchestrator — parallel agentic executor ----
        # Uses ORCHESTRATOR_MODEL (gemini-2.5-flash by default) with full
        # function calling via generate_content while the live session handles
        # voice I/O only.

        # Wire long-term memory store (ChromaDB) for semantic search (A1)
        # Use module-level singleton so the BGE model is loaded once at startup
        # instead of blocking each WS connection for 4-6 seconds.
        _mem_store = _get_shared_memory_store()

        orchestrator = ToolOrchestrator(
            genai_client=genai_client,
            tool_fns=tool_fns,
            model=ORCHESTRATOR_MODEL,
            memory_store=_mem_store,
            broadcast_fn=_broadcast_dashboard,
        )
        # Register in session registry only when explicit persistence is enabled.
        if ENABLE_PERSISTENT_CLIENT_SESSION:
            _active_sessions_by_user[session_key] = (bridge, orchestrator)

    # Convert callables → proper FunctionDeclaration objects.
    # Used ONLY when LIVE_MODEL_TOOLS=True (i.e. the live model supports tools).
    # generate_content (used by the orchestrator) auto-converts callables itself.
    tool_objects = _callable_to_tool_declarations(genai_client, tool_fns)
    log.info(
        "tools.declared",
        count=sum(len(t.function_declarations or []) for t in tool_objects),
        live_model_tools=LIVE_MODEL_TOOLS,
    )
    if LIVE_MODEL_TOOLS:
        log.warning(
            "live_model_tools.enabled",
            note="Direct Live tool execution can duplicate orchestrator actions and increase latency",
        )

    # Priority 4.4: per-user workspace/policy overlay
    if shared_enabled:
        workspace_root = Path(__file__).resolve().parent.parent
    else:
        workspace_root = _workspaces_dir / user_id
        workspace_root.mkdir(parents=True, exist_ok=True)

    read_paths = list(workspace_policy.get("read_paths", ["."]) or ["."])
    write_paths = list(workspace_policy.get("write_paths", ["."]) or ["."])
    orchestrator._workspace_root = workspace_root
    orchestrator._filesystem_policy = {
        "enabled": True,
        "read_paths": read_paths,
        "write_paths": write_paths,
    }
    # F5: Wire notifier as an after_tool hook for task completion notifications
    _notifier = getattr(app.state, "notifier", None)
    if _notifier and _notifier.is_enabled:
        async def _notify_on_task_complete(tool_name: str, args: dict, result: dict) -> dict:
            """After-tool hook: send a Telegram notification on task completion."""
            # Only notify when the orchestrator finishes (not on every tool call)
            return result
        orchestrator.register_hook("after_tool", _notify_on_task_complete)

    # P1.1: Attach telegram bot reference so approval gate can notify via Telegram
    _tg_bot_ref = getattr(app.state, "telegram_bot", None)
    if _tg_bot_ref and _tg_bot_ref.enabled:
        orchestrator._telegram_bot = _tg_bot_ref

    log.info(
        "orchestrator.ready",
        model=ORCHESTRATOR_MODEL,
        tools=len(tool_fns),
        workspace_root=str(workspace_root),
        shared_workspace=shared_enabled,
    )

    # C5: TriggerEngine — event-driven automation (keyword + schedule triggers)
    _trigger_engine = None
    try:
        from trigger_engine import TriggerEngine
        # inject_context is defined later inside the live session scope,
        # so create the engine with a noop callback, then swap in inject_context.
        async def _noop_inject(_: str) -> None:
            return
        _trigger_engine = TriggerEngine(orchestrator, _noop_inject)
        _trigger_engines_by_user[user_id] = _trigger_engine
        log.info("trigger_engine.created")
    except Exception as _te_err:
        log.debug("trigger_engine.skip", error=str(_te_err))

    system_instruction = build_system_instruction()

    # ---- Notify client ----
    try:
        await websocket.send_text(json.dumps({
            "type": "control",
            "action": "connected",
            "detail": "Direct Live API session ready",
            "backend": "direct-live-api",
        }))
    except (WebSocketDisconnect, Exception):
        log.info("client.disconnected_before_session")
        orchestrator.cancel_all()
        return

    # ---- Session with auto-reconnect ----
    session_handle: str | None = None
    active = True
    rapid_reconnects = 0

    def _ws_is_open() -> bool:
        return (
            active
            and websocket.client_state == WebSocketState.CONNECTED
            and websocket.application_state == WebSocketState.CONNECTED
        )

    def _normalize_user_command(text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _strip_wake_prefix(cleaned: str) -> str:
        prefixes = (
            "rio ",
            "hey rio ",
            "ok rio ",
            "okay rio ",
        )
        for p in prefixes:
            if cleaned.startswith(p):
                return cleaned[len(p):].strip()
        return cleaned

    def _is_explicit_stop_task_command(text: str) -> bool:
        """Only explicit stop/cancel task commands may interrupt active tasks."""
        cleaned = _strip_wake_prefix(_normalize_user_command(text))
        if not cleaned:
            return False

        verbs = ("stop", "cancel", "abort", "quit", "end")
        task_nouns = ("task", "job", "work")
        return any(v in cleaned for v in verbs) and any(n in cleaned for n in task_nouns)

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
            voice=ACTIVE_VOICE_NAME,
            voice_plugin=VOICE_RUNTIME.plugin_name or "none",
        )

        try:
            async with genai_client.aio.live.connect(
                model=LIVE_MODEL, config=config,
            ) as session:

                if session_handle:
                    log.info("live.reconnected", handle=session_handle[:16])
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "control",
                            "action": "reconnected",
                            "detail": "Session resumed",
                        }))
                    except (WebSocketDisconnect, Exception):
                        log.info("client.disconnected_during_reconnect")
                        orchestrator.cancel_all()
                        return
                else:
                    log.info("live.session.started")
                    
                    # Restore history on fresh connection only when explicitly enabled.
                    if ENABLE_HISTORY_RESTORE and session_conversation_messages:
                        history = _get_session_history_contents()
                        if history:
                            log.info("live.restoring_history", turns=len(history))
                            try:
                                await session.send_client_content(turns=history, turn_complete=False)
                            except Exception as h_exc:
                                log.warning("live.history_restore_failed", error=str(h_exc))

                try:
                    await websocket.send_text(json.dumps({
                        "type": "control",
                        "action": "live_ready",
                    }))
                except (WebSocketDisconnect, Exception):
                    log.info("client.disconnected_before_live_ready")
                    orchestrator.cancel_all()
                    return

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

                            elif frame_type == "tool_event_update":
                                name = frame.get("name", "unknown")
                                result = frame.get("result", {})
                                log.info("client.tool_event_update", name=name, success=result.get("success"))
                                
                                # Broadcast to dashboard
                                await _broadcast_dashboard({
                                    "type": "dashboard",
                                    "subtype": "tool_event_update",
                                    "client_id": client_id,
                                    "name": name,
                                    "success": result.get("success"),
                                    "result": result
                                })
                                
                                # Inject back to orchestrator if running, or live session
                                msg = f"[SYSTEM: Async background tool '{name}' has completed its operation. Result: {json.dumps(result)}]"
                                if orchestrator._active_tasks:
                                    orchestrator.inject_user_message(msg)
                                elif session:
                                    # Inform Live Voice session so it can comment on it if needed
                                    try:
                                        await session.send_client_content(
                                            turns=types.Content(
                                                role="user",
                                                parts=[types.Part.from_text(text=msg)]
                                            ),
                                            turn_complete=True
                                        )
                                    except Exception as e:
                                        log.warning("tool_event_update.live_inject_failed", error=str(e))

                            elif frame_type == "text":
                                content = frame.get("content", "")
                                if content:
                                    _buffer_session_transcript({
                                        "speaker": "user",
                                        "text": content,
                                        "timestamp": datetime.now(
                                            timezone.utc,
                                        ).isoformat(),
                                        "source": "text",
                                    })
                                    await _broadcast_dashboard({
                                        "type": "transcript",
                                        "speaker": "user",
                                        "text": content,
                                        "client_id": client_id,
                                        "source": "text",
                                    })

                                    # Route typed chat with the same orchestration logic
                                    # used for transcribed speech turns.
                                    if orchestrator._active_tasks:
                                        if _is_explicit_stop_task_command(content):
                                            orchestrator.inject_user_message("cancel")
                                            log.info(
                                                "orchestrator.user_message_injected",
                                                source="text_active_task_stop",
                                                utterance=content[:60],
                                            )
                                        elif session:
                                            try:
                                                await session.send_client_content(
                                                    turns=types.Content(
                                                        role="user",
                                                        parts=[types.Part.from_text(text=(
                                                            "[SYSTEM: A background task is still running. "
                                                            "Reply briefly: 'I'm still working on it. What do you need?' "
                                                            "Do not stop the background task unless the user explicitly says stop/cancel task.]"
                                                        ))],
                                                    ),
                                                    turn_complete=True,
                                                )
                                            except Exception:
                                                pass
                                            log.info(
                                                "orchestrator.user_message_ignored_active_task",
                                                source="text",
                                                utterance=content[:60],
                                            )
                                    elif _is_task_request(content):
                                        log.info(
                                            "orchestrator.task_detected",
                                            source="text",
                                            utterance=content[:100],
                                        )
                                        await websocket.send_text(json.dumps({
                                            "type": "control",
                                            "action": "task_mode",
                                            "active": True,
                                            "goal": content[:100],
                                        }))
                                        await orchestrator.spawn_task(content, inject_context)
                                        await _broadcast_dashboard({
                                            "type": "dashboard",
                                            "subtype": "orchestrator",
                                            "event": "task_started",
                                            "goal": content[:100],
                                            "model": ORCHESTRATOR_MODEL,
                                            "client_id": client_id,
                                            "source": "text",
                                        })
                                    else:
                                        await session.send_realtime_input(
                                            text=content,
                                        )

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

                            elif frame_type == "approval_response":
                                # B2: User approved/denied a tool call
                                approved = frame.get("approved", False)
                                orchestrator.resolve_approval(approved)

                            elif frame_type == "heartbeat":
                                pass  # Handled by heartbeat task

                    except WebSocketDisconnect:
                        log.info("upstream.client_disconnected")
                        active = False
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        # Live API session may close transiently (e.g., 1011).
                        # Treat as recoverable and let the outer reconnect loop
                        # handle it, instead of forcing a full client disconnect.
                        err = str(exc)
                        if "1011" in err or "ConnectionClosed" in err:
                            log.warning("upstream.session_closed", error=err[:200])
                            return
                        log.exception("upstream.error")
                        active = False

                # ---- Heartbeat: keep Gemini session alive ----
                async def heartbeat_task() -> None:
                    _hb_errors = 0
                    while active:
                        try:
                            await asyncio.sleep(3)
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=silent_chunk,
                                    mime_type="audio/pcm;rate=16000",
                                ),
                            )
                            _hb_errors = 0  # Reset on success
                        except asyncio.CancelledError:
                            break
                        except Exception:
                            _hb_errors += 1
                            if _hb_errors >= 3:
                                log.warning("heartbeat.failed_repeatedly")
                                break
                            log.debug("heartbeat.error", consecutive=_hb_errors)
                            await asyncio.sleep(1)

                # ---- inject_context: send SYSTEM message to live session ----
                async def inject_context(msg: str) -> None:
                    """Inject an orchestrator result into the live audio session.

                    Called when an orchestrator task completes or sends
                    progress updates.  Heartbeats and progress messages
                    use turn_complete=False so the voice model is NOT
                    interrupted.  Only final results (task complete, error,
                    cancellation) use turn_complete=True to trigger a
                    spoken response.
                    """
                    import re

                    # Strip machine-facing report schema so the voice model never
                    # reads lines like "GROUNDED: true" or "INTERRUPT: false".
                    voice_msg = re.sub(
                        r"(?im)^\s*(SCREEN|USER_INPUT|INTERRUPT|CONTEXT|GROUNDED|AUTH_REQUIRED|AUTH_RESOLVED)\s*:\s*.*$",
                        "",
                        msg,
                    )
                    voice_msg = re.sub(
                        r"(?i)\b(?:grounded|interrupt|auth_required|auth_resolved)\s*[:=]?\s*(?:true|false)\b",
                        "",
                        voice_msg,
                    )
                    voice_msg = re.sub(r"\n{3,}", "\n\n", voice_msg).strip()
                    if not voice_msg:
                        voice_msg = "[SYSTEM: I'm on it.]"

                    # Classify message: heartbeats are non-final context
                    # that should NOT interrupt the voice model.
                    is_heartbeat = (
                        "[SYSTEM: Still working" in voice_msg
                        or "[SYSTEM: Approval required" in voice_msg
                        or "[REASONING]" in voice_msg
                    )

                    if _ws_is_open():
                        # Only send task_mode:false for final results,
                        # not for progress heartbeats (prevents UI flicker).
                        if not is_heartbeat:
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "control",
                                    "action": "task_mode",
                                    "active": False,
                                }))
                            except Exception:
                                pass
                        try:
                            await session.send_client_content(
                                turns=types.Content(
                                    role="user",
                                    parts=[types.Part(text=voice_msg)],
                                ),
                                # Heartbeats: turn_complete=False → voice
                                # model treats this as background context
                                # and does NOT interrupt ongoing speech.
                                # Final results: turn_complete=True → voice
                                # model generates a spoken response.
                                turn_complete=not is_heartbeat,
                            )
                        except ConnectionClosedOK:
                            # Normal close during reconnect/shutdown race.
                            log.info("orchestrator.inject.skipped_closed_live_session")
                        except ConnectionClosed:
                            log.info("orchestrator.inject.skipped_closed_live_session")
                        except Exception:
                            log.warning("orchestrator.inject.failed", exc_info=True)
                    else:
                        log.info("orchestrator.inject.skipped_closed_session")

                    # F5: Send push notification if configured
                    if _notifier and _notifier.is_enabled:
                        try:
                            # Only notify on actual task completions, not system messages
                            if "task executor has completed" in voice_msg.lower() or "task complete" in voice_msg.lower():
                                short_msg = voice_msg[:200].replace("[SYSTEM:", "").strip().rstrip("]")
                                await _notifier.send(f"🤖 Rio: {short_msg}")
                        except Exception:
                            pass

                    _channel_mgr = getattr(app.state, "channel_manager", None)
                    if _channel_mgr:
                        try:
                            if "task executor has completed" in voice_msg.lower() or "task complete" in voice_msg.lower():
                                short_msg = voice_msg[:200].replace("[SYSTEM:", "").strip().rstrip("]")
                                await _channel_mgr.send_all(f"🤖 Rio: {short_msg}")
                        except Exception:
                            pass

                # C5: Wire TriggerEngine with inject_context now that it's defined
                if _trigger_engine is not None:
                    _trigger_engine._inject_fn = inject_context
                    await _trigger_engine.start()

                # Resume interrupted tasks only when explicitly enabled.
                if ENABLE_RESUME_INTERRUPTED_TASKS:
                    try:
                        resumed = orchestrator.resume_interrupted_tasks(inject_context)
                        if resumed:
                            log.info("session.tasks_resumed", count=len(resumed))
                    except Exception:
                        log.debug("session.resume_skip")

                # F6: Wire Telegram bot to route incoming messages as PC workspace tasks
                _telegram_bot = getattr(app.state, "telegram_bot", None)
                _channel_manager = getattr(app.state, "channel_manager", None)
                if _telegram_bot and _telegram_bot.enabled:
                    app.state.telegram_active_client_id = client_id

                    # Status function: returns human-readable list of active tasks
                    def _telegram_status() -> str:
                        owner = getattr(app.state, "telegram_active_client_id", None)
                        if owner != client_id or not _ws_is_open():
                            return "Desktop session is offline. Reconnect your local Rio client."
                        active = [
                            t.get_name().replace("orchestrator-", "• ", 1)
                            for t in orchestrator._active_tasks
                            if not t.done()
                        ]
                        if not active:
                            return "No tasks currently running."
                        return "\n".join(active)

                    _telegram_bot.set_status_fn(_telegram_status)

                    # Approval handler: forward yes/no to orchestrator
                    async def _telegram_approve(approved: bool) -> None:
                        owner = getattr(app.state, "telegram_active_client_id", None)
                        if owner != client_id or not _ws_is_open():
                            return
                        orchestrator.resolve_approval(approved)

                    _telegram_bot.set_approval_handler(_telegram_approve)

                    async def _telegram_on_command(cmd: str, args: list[str]) -> None:
                        """Handle slash commands from Telegram (Next Level)."""
                        owner = getattr(app.state, "telegram_active_client_id", None)
                        if owner != client_id:
                            return
                        logger.info("telegram_bot.command_received", cmd=cmd, args=args)

                        if cmd == "reset":
                            orchestrator.reset()
                            await _telegram_bot.send("🔄 *Session Reset:* History and notes have been cleared.")

                        elif cmd == "model":
                            if not args:
                                current = orchestrator.model
                                await _telegram_bot.send(
                                    f"🤖 *Current Orchestrator Model:* `{current}`\n\n"
                                    f"To change: `/model [name]`\n"
                                    f"Example: `/model gemini-3-pro-preview`"
                                )
                                return
                            new_model = args[0]
                            orchestrator.set_model(new_model)
                            await _telegram_bot.send(f"✅ *Model Switched:* Now using `{new_model}`")

                        elif cmd == "models":
                            # List models from config
                            from config import RioConfig
                            try:
                                cfg = RioConfig.load()
                                models = cfg.get("rio", {}).get("models", {})
                                lines = ["📊 *Available Models (from config):*"]
                                for k, v in models.items():
                                    if isinstance(v, str):
                                        lines.append(f"• `{k}`: {v}")
                                await _telegram_bot.send("\n".join(lines))
                            except Exception:
                                await _telegram_bot.send("❌ Error loading model list.")

                        elif cmd == "screenshot":
                            if not _ws_is_open():
                                await _telegram_bot.send("⚠️ Desktop session is offline. Reconnect your local Rio client first.")
                                return
                            # Request a manual screenshot from the local client
                            await _telegram_bot.send("📸 *Capturing PC screen...*")
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "tool_call",
                                    "name": "capture_screen",
                                    "args": {"force": True},
                                    "id": f"tg_screenshot_{int(time.time())}"
                                }))
                            except Exception as e:
                                await _telegram_bot.send(f"❌ Failed to request screenshot: {e}")

                        elif cmd == "memory":
                            if not args:
                                await _telegram_bot.send("🧠 *Memory Search*\n\nUsage: `/memory [query]`")
                                return
                            query = " ".join(args)
                            _mem_store = getattr(orchestrator, "_memory_store", None)
                            if _mem_store:
                                try:
                                    results = _mem_store.query(query, top_k=3)
                                    if not results:
                                        await _telegram_bot.send(f"🔍 No memories found for: `{query}`")
                                    else:
                                        lines = [f"🧠 *Top 3 Memories for:* `{query}`"]
                                        for r in results:
                                            # Truncate content for Telegram
                                            content = r.content[:300] + ("..." if len(r.content) > 300 else "")
                                            lines.append(f"• {content} _(dist: {r.distance:.2f})_")
                                        await _telegram_bot.send("\n".join(lines))
                                except Exception as e:
                                    await _telegram_bot.send(f"❌ Memory search failed: {e}")
                            else:
                                await _telegram_bot.send("❌ Memory store not initialized on this session.")

                        elif cmd == "agents":
                            try:
                                agents = orchestrator.get_agents()
                                lines = ["🕵️ *Specialist Agents:*"]
                                for name, data in agents.items():
                                    if data.get("enabled"):
                                        desc = data.get("description", "No description")
                                        model = data.get("model", "default")
                                        lines.append(f"• *{name}* ({model}): {desc}")
                                await _telegram_bot.send("\n".join(lines))
                            except Exception:
                                await _telegram_bot.send("❌ Error loading agent list.")

                        elif cmd == "voice":
                            await _telegram_bot.send(
                                "🎤 *Voice Control*\n\n"
                                "To mute Rio's voice, type: `stop speaking` or `be quiet`.\n"
                                "To enable/disable fully, use the F6 key on your PC."
                            )

                        else:
                            await _telegram_bot.send(f"❓ Unknown command: `/{cmd}`. Type /help for a list.")

                    _telegram_bot.set_command_handler(_telegram_on_command)

                    async def _telegram_to_orchestrator(text: str) -> None:
                        """Route incoming Telegram message to the PC workspace orchestrator.

                        Creates a Telegram-aware inject_context wrapper so that:
                        - Progress heartbeats are forwarded to Telegram
                        - Task results are sent back to Telegram when done
                        """
                        owner = getattr(app.state, "telegram_active_client_id", None)
                        if owner != client_id:
                            return
                        if not _ws_is_open():
                            await _telegram_bot.send("⚠️ Desktop session is offline. Reconnect your local Rio client first.")
                            return
                        log.info("telegram_bot.task_received", text=text[:80])
                        draft_handle = None
                        draft_finished = False

                        async def _draft_update(text_payload: str, final: bool = False) -> None:
                            nonlocal draft_handle, draft_finished
                            if _channel_manager is None:
                                return
                            if final and draft_finished:
                                return
                            if draft_handle is None:
                                draft_handle = await _channel_manager.start_draft(text_payload, channel="telegram")
                                if draft_handle is None:
                                    await _channel_manager.send(text_payload, channel="telegram")
                                    if final:
                                        draft_finished = True
                                return
                            if final:
                                await _channel_manager.finish_draft(draft_handle, text_payload)
                                draft_finished = True
                            else:
                                await _channel_manager.update_draft(draft_handle, text_payload)

                        # Telegram-aware inject_context: forwards results AND heartbeats
                        async def _tg_inject(msg: str) -> None:
                            # Always forward to voice model
                            await inject_context(msg)
                            # Also send to Telegram, filtering out raw system noise
                            clean = msg.strip()
                            if not clean:
                                return
                            # Heartbeat in progress
                            if "[SYSTEM: Still working" in clean:
                                try:
                                    elapsed_hint = clean.split("elapsed")[0].split("(")[-1].strip() if "elapsed" in clean else ""
                                    tool_hint = clean.split("Still working on")[-1].split("...")[0].strip() if "Still working on" in clean else "task"
                                    await _draft_update(
                                        f"⏳ Working on: {tool_hint}{' ('+elapsed_hint+')' if elapsed_hint else ''}...",
                                        final=False,
                                    )
                                except Exception:
                                    pass
                            # Task was cancelled
                            elif "[SYSTEM: The task was cancelled" in clean or "task was stopped" in clean.lower():
                                try:
                                    await _draft_update("🛑 Task was cancelled.", final=True)
                                except Exception:
                                    pass
                            # Approval required
                            elif "Approval required" in clean:
                                try:
                                    await _draft_update(f"⚠️ {clean[:500]}", final=False)
                                except Exception:
                                    pass
                            # Normal result — strip [SYSTEM: ...] wrapper and send
                            elif "[SYSTEM:" in clean:
                                inner = clean.replace("[SYSTEM:", "").rstrip("]").strip()
                                if len(inner) > 5:
                                    try:
                                        await _draft_update(f"✅ {inner[:1000]}", final=True)
                                    except Exception:
                                        pass

                        await orchestrator.spawn_task(
                            f"[From Telegram] {text}", _tg_inject,
                        )
                        await _draft_update(f"▶️ Starting task: _{text[:120]}_", final=False)

                    _telegram_bot.set_message_handler(_telegram_to_orchestrator)

                    # Wire chat handler: conversational (non-task) messages
                    # get injected into the live Gemini session as context,
                    # so the model responds naturally without spawning a task.
                    async def _telegram_chat_handler(text: str) -> None:
                        """Handle conversational Telegram messages (non-tasks).

                        Injects the message into the live model session so Gemini
                        can respond conversationally. The response comes back through
                        the normal downstream audio/text pipeline.
                        """
                        owner = getattr(app.state, "telegram_active_client_id", None)
                        if owner != client_id:
                            return
                        if not _ws_is_open():
                            await _telegram_bot.send("⚠️ Desktop session is offline.")
                            return
                        log.info("telegram_bot.chat_message", text=text[:80])
                        try:
                            # Inject as a user turn so the live model responds
                            await session.send_client_content(
                                turns=types.Content(
                                    role="user",
                                    parts=[types.Part(text=f"[Message from Telegram] {text}")],
                                ),
                                turn_complete=True,
                            )
                            await _telegram_bot.send(f"💬 _{text[:100]}_ — response coming via voice...")
                        except Exception as exc:
                            log.warning("telegram_bot.chat_inject_error", error=str(exc))
                            await _telegram_bot.send(f"❌ Could not process: {exc}")

                    _telegram_bot.set_chat_handler(_telegram_chat_handler)

                # ---- Downstream: Gemini session → client WebSocket ----
                async def downstream_task() -> None:
                    nonlocal session_handle
                    # Buffer partial transcription fragments until turn complete
                    _user_utterance_buf: list[str] = []
                    _rio_utterance_buf: list[str] = []
                    
                    try:
                        async for response in session.receive():
                            sc = response.server_content

                            # ---- Audio / text from model ----
                            if sc and sc.model_turn:
                                for part in sc.model_turn.parts or []:
                                    if getattr(part, "thought", False):
                                        continue
                                    if part.inline_data and part.inline_data.data:
                                        await websocket.send_bytes(
                                            b"\x01" + part.inline_data.data,
                                        )
                                    if part.text:
                                        _rio_utterance_buf.append(part.text)
                                        await websocket.send_text(json.dumps({
                                            "type": "transcript",
                                            "speaker": "rio",
                                            "text": part.text,
                                        }))
                                        # Broadcast partial to dashboard
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
                                    _user_utterance_buf.append(text)
                                    # Send partial to client UI
                                    await websocket.send_text(json.dumps({
                                        "type": "transcript",
                                        "speaker": "user",
                                        "text": text,
                                        "source": "transcription",
                                    }))
                                    # Broadcast partial to dashboard
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
                                _user_utterance_buf.clear()  # Discard partial
                                _rio_utterance_buf.clear()
                                await websocket.send_text(json.dumps({
                                    "type": "control",
                                    "action": "interrupted",
                                }))

                            # ---- Turn complete ----
                            if sc and getattr(sc, "turn_complete", False):
                                await websocket.send_text(json.dumps({
                                    "type": "control",
                                    "action": "turn_complete",
                                }))

                                # 1. Finalise User Turn
                                full_user_text = " ".join(_user_utterance_buf).strip()
                                if full_user_text:
                                    _buffer_session_transcript({
                                        "speaker": "user",
                                        "text": full_user_text,
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "source": "transcription",
                                    })
                                _user_utterance_buf.clear()

                                # 2. Finalise Rio Turn
                                full_rio_text = "".join(_rio_utterance_buf).strip()
                                if full_rio_text:
                                    _buffer_session_transcript({
                                        "speaker": "rio",
                                        "text": full_rio_text,
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                    })
                                _rio_utterance_buf.clear()

                                # 3. Check for Task Execution
                                if full_user_text and orchestrator._active_tasks:
                                    if _is_explicit_stop_task_command(full_user_text):
                                        orchestrator.inject_user_message("cancel")
                                        log.info(
                                            "orchestrator.user_message_injected",
                                            source="transcription_active_task_stop",
                                            utterance=full_user_text[:80],
                                        )
                                    elif session:
                                        try:
                                            await session.send_client_content(
                                                turns=types.Content(
                                                    role="user",
                                                    parts=[types.Part.from_text(text=(
                                                        "[SYSTEM: A background task is still running. "
                                                        "Reply briefly: 'I'm still working on it. What do you need?' "
                                                        "Do not stop the background task unless the user explicitly says stop/cancel task.]"
                                                    ))],
                                                ),
                                                turn_complete=True,
                                            )
                                        except Exception:
                                            pass
                                        log.info(
                                            "orchestrator.user_message_ignored_active_task",
                                            source="transcription",
                                            utterance=full_user_text[:80],
                                        )
                                    else:
                                        # Ignore ambient/noisy transcriptions while a task is running.
                                        log.info(
                                            "orchestrator.user_message_ignored_active_task",
                                            source="transcription_active_task",
                                            utterance=full_user_text[:80],
                                        )
                                elif full_user_text:
                                    is_task = _is_task_request(full_user_text)
                                    if not is_task and len(full_user_text.split()) > 5:
                                        is_task = await _classify_intent_via_llm(genai_client, full_user_text)
                                    
                                    if is_task:
                                        log.info("orchestrator.task_detected", utterance=full_user_text[:100])
                                        await websocket.send_text(json.dumps({
                                            "type": "control",
                                            "action": "task_mode",
                                            "active": True,
                                            "goal": full_user_text[:100],
                                        }))
                                        await orchestrator.spawn_task(full_user_text, inject_context)
                                        await _broadcast_dashboard({
                                            "type": "dashboard",
                                            "subtype": "orchestrator",
                                            "event": "task_started",
                                            "goal": full_user_text[:100],
                                            "model": ORCHESTRATOR_MODEL,
                                            "client_id": client_id,
                                        })
                                # else: normal voice conversation — no action needed

                                # C5: Check keyword triggers on every utterance
                                utterance_for_triggers = full_user_text
                                if utterance_for_triggers and _trigger_engine is not None:
                                    _trigger_engine.check_utterance(utterance_for_triggers)
                                    scheduled = _trigger_engine.try_schedule_from_utterance(utterance_for_triggers)
                                    if scheduled:
                                        mode = scheduled.get("mode", "schedule")
                                        goal = scheduled.get("goal", "")
                                        interval = int(scheduled.get("interval_seconds", 0) or 0)
                                        human = (
                                            f"[SYSTEM: Scheduled {mode} task '{goal}' every {interval} seconds.]"
                                            if not scheduled.get("run_once")
                                            else f"[SYSTEM: Scheduled one-time reminder '{goal}' in {interval} seconds.]"
                                        )
                                        await inject_context(human)

                            # ---- Go-away (session about to expire) ----
                            ga = getattr(response, "go_away", None)
                            if ga:
                                time_left = getattr(ga, "time_left", "?")
                                log.warning(
                                    "live.go_away",
                                    time_left=time_left,
                                )
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "control",
                                        "action": "go_away",
                                        "detail": f"Session expiring in {time_left}",
                                    }))
                                except Exception:
                                    pass
                                # Break out of the receive loop — the outer
                                # while-active loop will reconnect automatically
                                # using session_handle for resumption.
                                break

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
                                if orchestrator._active_tasks:
                                    log.warning(
                                        "live.tool_call.ignored_during_orchestrator_task",
                                        names=[fc.name for fc in tc.function_calls],
                                    )
                                    await session.send_tool_response(
                                        function_responses=[
                                            types.FunctionResponse(
                                                name=fc.name,
                                                response={
                                                    "success": False,
                                                    "error": "Live model tool call skipped while orchestrator task is active",
                                                },
                                                id=fc.id,
                                            )
                                            for fc in tc.function_calls
                                        ],
                                    )
                                    continue

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
                                    "open_application", "minimize_window",
                                    "maximize_window", "close_window",
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
                                    # Intercept start_orchestrator_task
                                    if fc.name == "start_orchestrator_task":
                                        log.info("live.start_orchestrator_task_triggered", args=fc.args)
                                        goal = fc.args.get("goal", "unspecified task")
                                        
                                        # Spawn the task
                                        await orchestrator.spawn_task(goal, inject_context)
                                        
                                        # Notify client
                                        await websocket.send_text(json.dumps({
                                            "type": "control",
                                            "action": "task_mode",
                                            "active": True,
                                            "goal": goal[:100],
                                        }))
                                        
                                        fn_responses.append(
                                            types.FunctionResponse(
                                                name=fc.name,
                                                response={"status": "TASK_SPAWNED", "goal": goal},
                                                id=fc.id,
                                            ),
                                        )
                                        continue

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
                                        await asyncio.sleep(0.15)

                                # Send tool responses back to model
                                await session.send_tool_response(
                                    function_responses=fn_responses,
                                )

                    except WebSocketDisconnect:
                        log.info("downstream.client_disconnected")
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        # The Live API sometimes returns transient 1011 internal
                        # errors. Log compactly and bubble up so outer loop
                        # reconnects cleanly.
                        err = str(exc)
                        if "1011" in err or "Internal error occurred" in err:
                            log.warning("downstream.live_api_internal_error", error=err[:200])
                        else:
                            log.exception("downstream.error")
                        raise

                # ---- Run session tasks ----
                upstream_handle = asyncio.create_task(upstream_task(), name="upstream")
                downstream_handle = asyncio.create_task(downstream_task(), name="downstream")
                heartbeat_handle = asyncio.create_task(heartbeat_task(), name="heartbeat")
                tasks = [upstream_handle, heartbeat_handle, downstream_handle]
                session_started_monotonic = time.monotonic()
                try:
                    done, pending = await asyncio.wait(
                        {upstream_handle, downstream_handle},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # The heartbeat task is best-effort; it should not force
                    # a full Live session reconnect when it exits.
                    if not heartbeat_handle.done():
                        pending.add(heartbeat_handle)

                    for t in pending:
                        t.cancel()
                    # Propagate exceptions from completed tasks
                    for t in done:
                        if t.exception():
                            log.error("task.error", task=t.get_name(), exc=t.exception())

                    # If one stream ends cleanly while WS is still open, this
                    # is usually a transient Live API close; reconnect, but back
                    # off when it happens repeatedly in a short window.
                    completed = [t.get_name() for t in done]
                    elapsed = time.monotonic() - session_started_monotonic
                    log.info(
                        "live.session.stream_ended",
                        completed=completed,
                        elapsed_seconds=round(elapsed, 2),
                    )
                    if elapsed < 20:
                        rapid_reconnects += 1
                    else:
                        rapid_reconnects = 0

                    if rapid_reconnects > 0 and active and _ws_is_open():
                        delay = min(2 * rapid_reconnects, 10)
                        log.warning(
                            "live.session.churn_backoff",
                            rapid_reconnects=rapid_reconnects,
                            delay_seconds=delay,
                        )
                        await asyncio.sleep(delay)
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
    if not ENABLE_PERSISTENT_CLIENT_SESSION:
        _active_sessions_by_user.pop(session_key, None)
    # C5: Stop the trigger engine for this session
    if _trigger_engine is not None:
        try:
            await _trigger_engine.stop()
        except Exception:
            pass
        _trigger_engines_by_user.pop(user_id, None)

    _telegram_bot = getattr(app.state, "telegram_bot", None)
    if _telegram_bot and _telegram_bot.enabled:
        owner = getattr(app.state, "telegram_active_client_id", None)
        if owner == client_id:
            async def _telegram_offline_message(_: str) -> None:
                await _telegram_bot.send("⚠️ Desktop session is offline. Reconnect your local Rio client first.")

            async def _telegram_offline_command(cmd: str, _: list[str]) -> None:
                if cmd in {"help", "status", "tasks"}:
                    await _telegram_bot.send("⚠️ Desktop session is offline. Reconnect your local Rio client first.")

            async def _telegram_offline_approval(_: bool) -> None:
                return

            _telegram_bot.set_message_handler(_telegram_offline_message)
            _telegram_bot.set_command_handler(_telegram_offline_command)
            _telegram_bot.set_approval_handler(_telegram_offline_approval)
            _telegram_bot.set_status_fn(lambda: "Desktop session is offline. Reconnect your local Rio client.")
            app.state.telegram_active_client_id = None

    # Persist conversation on disconnect
    _save_session_conversation()
    log.info("live.client.cleanup_done", client_id=client_id)
    await _broadcast_dashboard({
        "type": "dashboard", "subtype": "client_event",
        "event": "disconnected", "client_id": client_id, "user_id": user_id,
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
# F3: OpenAI-Compatible HTTP API
# ---------------------------------------------------------------------------
# Enables IDEs (Cursor, Continue) and any OpenAI SDK client to use Rio
# as a drop-in backend via /v1/chat/completions and /v1/models.
# ---------------------------------------------------------------------------

from fastapi.responses import StreamingResponse


@app.get("/v1/models")
async def openai_list_models():
    """Return available models in OpenAI format."""
    models = [
        {
            "id": LIVE_MODEL,
            "object": "model",
            "created": 1700000000,
            "owned_by": "rio",
        },
        {
            "id": ORCHESTRATOR_MODEL,
            "object": "model",
            "created": 1700000000,
            "owned_by": "rio",
        },
    ]
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint.

    Accepts the standard OpenAI request format and routes through
    the orchestrator model for tool-augmented responses.
    """
    if genai_client is None:
        return {"error": {"message": "Server not ready", "type": "server_error"}}

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", ORCHESTRATOR_MODEL)
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 0.7)

    if not messages:
        return {"error": {"message": "messages is required", "type": "invalid_request_error"}}

    # Convert OpenAI messages to Gemini contents
    contents = []
    system_instruction = None
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "assistant":
            contents.append(types.Content(
                role="model",
                parts=[types.Part(text=content)],
            ))
        else:
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=content)],
            ))

    if not contents:
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text="Hello")],
        ))

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(datetime.now(timezone.utc).timestamp())

    config = types.GenerateContentConfig(
        system_instruction=system_instruction or build_system_instruction(),
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    if stream:
        async def _stream_response():
            try:
                async for chunk in genai_client.aio.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                ):
                    if chunk.text:
                        data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk.text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                # Final chunk
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error("openai_compat.stream_error", error=str(exc))
                yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"

        return StreamingResponse(
            _stream_response(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        try:
            response = await genai_client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            text = response.text or ""
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        except Exception as exc:
            logger.error("openai_compat.error", error=str(exc))
            return {"error": {"message": str(exc), "type": "server_error"}}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", "8080")),
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("server.shutdown_requested")
    except asyncio.CancelledError:
        logger.info("server.cancelled")
    except SystemExit:
        pass
    finally:
        logger.info("server.stopped")
