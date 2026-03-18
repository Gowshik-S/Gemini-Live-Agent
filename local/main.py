"""
Rio Local — Entry Point & Asyncio Orchestrator

Layer 3 (L3): voice + vision + tools client.
  - Loads configuration from config.yaml
  - Connects to the Rio cloud backend over WebSocket
  - Captures microphone audio (PCM 16-bit 16kHz mono) and streams to cloud
  - Plays back Gemini audio responses (PCM 24kHz) through the speaker
  - Push-to-Talk (F2) gates audio capture when available
    - Silero VAD filters silence when available
    - Screen capture: periodic + on-demand via voice/tool requests
    - F3 toggle: mute/unmute microphone input
    - F8: announce current ongoing task status
  - Tool execution: read_file, write_file, patch_file, run_command
  - Graceful degradation: ptt+vad / ptt-only / vad-only / always-on
  - Reads user text input from stdin as fallback
  - Prints Gemini text transcripts from the cloud
  - Graceful shutdown on Ctrl-C
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path


def _configure_openmp_runtime() -> None:
    """Set conservative OpenMP defaults to reduce startup OMP failures.

    On Windows, mixed native stacks (torch/sklearn/vosk/onnx) can spin up large
    thread pools and occasionally fail with OMP memory allocation errors during
    initialization. We only set defaults when values are not already provided by
    the user.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_BLOCKTIME", "0")


_configure_openmp_runtime()

import structlog

from config import RioConfig
from ws_client import WSClient

try:
    from audio_io import (
        AudioCapture,
        AudioPlayback,
        AUDIO_PREFIX,
        list_audio_devices,
        get_default_input_device,
        get_default_output_device,
    )
except ImportError:
    AudioCapture = None  # Audio deps not installed — text-only mode
    AudioPlayback = None
    AUDIO_PREFIX = b"\x01"
    list_audio_devices = None
    get_default_input_device = None
    get_default_output_device = None

try:
    from vad import SileroVAD
except ImportError:
    SileroVAD = None

try:
    from push_to_talk import PushToTalk
except ImportError:
    PushToTalk = None

try:
    from screen_capture import ScreenCapture, IMAGE_PREFIX
except ImportError:
    ScreenCapture = None
    IMAGE_PREFIX = b"\x02"

try:
    from tools import ToolExecutor
except ImportError:
    ToolExecutor = None

try:
    from screen_navigator import ScreenNavigator
except ImportError:
    ScreenNavigator = None

try:
    from ui_navigator import UINavigator
except ImportError:
    UINavigator = None

try:
    from struggle_detector import StruggleDetector
except ImportError:
    StruggleDetector = None

try:
    from memory import MemoryStore
except ImportError:
    MemoryStore = None

try:
    from ocr import OCREngine
except ImportError:
    OCREngine = None

try:
    from google import genai as _genai
except ImportError:
    _genai = None

try:
    from wake_word import WakeWordDetector, WakeWordState
except ImportError:
    WakeWordDetector = None
    WakeWordState = None

try:
    from chat_store import ChatStore
except ImportError:
    ChatStore = None

try:
    from task_state import TaskStore, SessionMemory
except ImportError:
    TaskStore = None
    SessionMemory = None

try:
    from browser_agent import BrowserAgent
except ImportError:
    BrowserAgent = None

try:
    from windows_agent import WindowsAgent
except ImportError:
    WindowsAgent = None

try:
    from orchestrator import Orchestrator
except ImportError:
    Orchestrator = None

try:
    from creative_agent import CreativeAgent
except ImportError:
    CreativeAgent = None

try:
    from user_pattern_model import UserPatternModel
except ImportError:
    UserPatternModel = None

# ML Pipeline (ensemble learning)
try:
    import sys as _sys
    _ml_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ml_parent not in _sys.path:
        _sys.path.insert(0, _ml_parent)
    from ml.user_model_manager import UserModelManager
    from ml.ensemble_model import SKLEARN_AVAILABLE as _ML_READY
except ImportError:
    UserModelManager = None
    _ML_READY = False

# Rio structured logging + platform detection
try:
    from rio_logging import setup_logging, get_logger as rio_get_logger
    _RIO_LOGGING = True
except ImportError:
    _RIO_LOGGING = False

try:
    from platform_utils import get_platform, print_platform_summary, get_missing_dependencies
    _PLATFORM_UTILS = True
except ImportError:
    _PLATFORM_UTILS = False

# ---------------------------------------------------------------------------
# Structlog configuration
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),  # show DEBUG+
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger("rio.main")


def _load_cloud_env() -> Path | None:
    """Load rio/cloud/.env into process env without overriding existing vars."""
    candidates: list[Path] = []
    override = os.environ.get("RIO_ENV_FILE", "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(Path(__file__).resolve().parent.parent / "cloud" / ".env")
    candidates.append(Path.cwd() / "cloud" / ".env")

    seen: set[str] = set()
    for env_path in candidates:
        try:
            path_key = str(env_path.resolve())
        except OSError:
            path_key = str(env_path)
        if path_key in seen:
            continue
        seen.add(path_key)

        if not env_path.is_file():
            continue

        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                os.environ[key] = value.strip().strip('"').strip("'")
        except OSError:
            return None
        return env_path

    return None


_loaded_env_path = _load_cloud_env()
if _loaded_env_path is not None:
    log.info(
        "env.loaded",
        path=str(_loaded_env_path),
        gemini_key_set=bool(os.environ.get("GEMINI_API_KEY")),
    )

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
  ____  _
 |  _ \(_) ___
 | |_) | |/ _ \
 |  _ <| | (_) |
 |_| \_\_|\___/   v0.9.0 — Low-Latency Audio Pipeline

 Proactive AI Pair Programmer
 Voice + screen vision + tools + struggle detection + ML ensemble + Pro routing
 WASAPI low-latency audio, 20ms capture chunks, 40ms jitter buffer
 Wake word: say "Rio" or "Hey Rio" to activate
 F2=Push-to-Talk  F3=Mute Toggle  F4=Force-trigger(demo)  F5=Screen-mode  F6=Live-Mode  F7=Live-Translation  F8=Task Status
 Text input via stdin.  Ctrl-C to quit.
"""


# Decline phrases that indicate user doesn't want proactive help
DECLINE_PHRASES = frozenset({
    "no thanks", "no thank you", "i'm fine", "im fine", "i'm good",
    "im good", "not now", "go away", "leave me alone", "stop",
    "don't help", "dont help", "i got it", "i've got it",
    "no need", "nevermind", "never mind", "nah",
})


def _is_decline(text: str) -> bool:
    """Check if user text indicates declining proactive help.

    Uses substring matching so phrases like 'no thanks, I'm fine'
    or 'nah I got it' are also caught.
    """
    text_lower = text.strip().lower().rstrip(".!?")
    # Exact match first
    if text_lower in DECLINE_PHRASES:
        return True
    # Substring match: if any decline phrase appears within the text
    for phrase in DECLINE_PHRASES:
        if phrase in text_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Task detection — determines if user input needs autonomous execution
# ---------------------------------------------------------------------------

_TASK_ACTION_VERBS = (
    "open", "go to", "goto", "navigate", "search", "create", "download",
    "install", "click", "close", "launch", "start", "browse", "find",
    "visit", "play", "stop", "delete", "move", "copy", "paste", "save",
    "upload", "send", "book", "order", "sign in", "sign out", "log in",
    "log out", "register", "fill", "submit", "scroll", "type", "write",
    "run", "execute", "build", "deploy", "setup", "configure", "update",
    "uninstall", "rename", "drag",
)

_TASK_PHRASES = (
    "for me", "please do", "can you do", "i need you to", "i want you to",
    "go ahead and", "take over", "handle this", "do this task",
    "complete this", "finish this",
)

_NON_TASK_STARTS = (
    "what", "why", "how", "when", "where", "who", "which", "is ", "are ",
    "was ", "were ", "do you", "does ", "did ", "can you explain",
    "tell me about", "explain", "describe", "hey rio", "hello", "hi ",
    "thanks", "thank you", "good", "great", "nice", "cool",
)


def _is_task_request(text: str) -> bool:
    """Detect if user input is a task that needs autonomous plan+execute.

    Returns True for actionable tasks like 'open Chrome and search for X'.
    Returns False for questions like 'what is Python?' or casual chat.
    """
    text_lower = text.strip().lower()

    # Explicit task prefixes (user explicitly flags a task)
    if text_lower.startswith(("task:", "do:", "execute:", "automate:")):
        return True

    # Resume/continue existing task
    if text_lower in ("resume", "continue", "go on", "keep going"):
        return True

    # Skip greetings and questions
    for start in _NON_TASK_STARTS:
        if text_lower.startswith(start):
            return False

    # Very short messages are unlikely tasks
    if len(text_lower.split()) < 3:
        return False

    # Check for action verbs at start of message
    for verb in _TASK_ACTION_VERBS:
        if text_lower.startswith(verb + " ") or text_lower.startswith(verb + ":"):
            return True

    # Check for task-indicating phrases anywhere
    for phrase in _TASK_PHRASES:
        if phrase in text_lower:
            return True

    # URL detection with action context
    if ("http://" in text_lower or "https://" in text_lower or "www." in text_lower):
        return True

    return False


async def _run_autonomous_task(
    goal: str,
    orchestrator,
    client: WSClient,
    autonomous_mode: asyncio.Event,
    task_active: asyncio.Event,
) -> None:
    """Run a task through the local orchestrator autonomously.

    The orchestrator plans steps via Gemini Pro, then executes each step
    using BrowserAgent / ScreenNavigator / ToolExecutor — all locally.
    Progress updates are sent through the WebSocket to the cloud so the
    user gets voice/text feedback via Gemini.
    """
    # Activate autonomous screen mode for visual feedback
    was_autonomous = autonomous_mode.is_set()
    autonomous_mode.set()
    task_active.set()

    try:
        # Handle resume of paused/active task
        if goal == "resume":
            print("\n  [Orchestrator] Resuming previous task...")
            task = await orchestrator.resume()
            if task is None:
                print("  [Orchestrator] No task to resume.")
                await client.send_json({
                    "type": "text",
                    "content": "There's no paused task to resume. Give me a new task!",
                })
                return
        else:
            print(f"\n  [Orchestrator] Planning: {goal[:100]}...")
            task = await orchestrator.plan_task(goal)

        if not task.steps:
            print("  [Orchestrator] Could not create a plan.")
            await client.send_json({
                "type": "text",
                "content": (
                    f"I tried to plan this task but couldn't break it into steps: "
                    f"{goal}. Can you rephrase or give me more specific instructions?"
                ),
            })
            return

        # Notify user of the plan
        step_list = ", ".join(s.action for s in task.steps[:5])
        plan_msg = (
            f"[SYSTEM: Autonomous task started. Goal: {goal}. "
            f"Plan ({len(task.steps)} steps): {step_list}. "
            f"I am executing this now — you'll get updates as I go. "
            f"Briefly acknowledge the plan to the user.]"
        )
        try:
            await client.send_json({
                "type": "context",
                "subtype": "task_plan",
                "content": plan_msg,
            })
        except ConnectionError:
            pass

        print(f"  [Orchestrator] Executing {len(task.steps)} steps...")
        task = await orchestrator.execute_task(task)

        # Report completion
        report = orchestrator.get_progress_report()
        status = task.status.value
        completion_msg = (
            f"[SYSTEM: Autonomous task {status}. Goal: {goal}. "
            f"Progress: {task.progress}. Report:\n{report}\n"
            f"Summarize what was accomplished to the user. "
            f"If anything failed, explain what went wrong.]"
        )
        try:
            await client.send_json({
                "type": "context",
                "subtype": "task_complete",
                "content": completion_msg,
            })
        except ConnectionError:
            pass

        print(f"  [Orchestrator] Task {status}: {task.goal}")
        print(f"  {report}")

    except Exception as exc:
        log.exception("autonomous_task.error")
        print(f"\n  [Orchestrator] Task failed: {exc}")
        try:
            await client.send_json({
                "type": "context",
                "subtype": "task_error",
                "content": (
                    f"[SYSTEM: Autonomous task failed with error: {exc}. "
                    f"Goal was: {goal}. Explain the error to the user and "
                    f"suggest how they might accomplish it manually.]"
                ),
            })
        except ConnectionError:
            pass
    finally:
        task_active.clear()
        if not was_autonomous:
            autonomous_mode.clear()
        print("  You: ", end="", flush=True)


# Tool names that modify the filesystem / execute commands and need approval
_DANGEROUS_TOOLS = frozenset({"write_file", "patch_file", "run_command"})

_CONFIRM_TIMEOUT = float(os.environ.get("RIO_CONFIRM_TIMEOUT_SECONDS", "0") or 0)


async def _confirm_tool_call(tool_name: str, tool_args: dict) -> bool:
    """Prompt user for confirmation before executing a dangerous tool.

    Returns True if approved, False otherwise.

    If ``RIO_CONFIRM_TIMEOUT_SECONDS`` is <= 0 (default), waits indefinitely
    for explicit user approval/denial.
    """
    summary = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
    print(f"\n    Rio wants to run: {tool_name}({summary})")
    if _CONFIRM_TIMEOUT <= 0:
        print("  Approve? [Y/n] (waiting for your response): ", end="", flush=True)
    else:
        print(f"  Approve? [Y/n] (auto-approve in {int(_CONFIRM_TIMEOUT)}s): ", end="", flush=True)

    loop = asyncio.get_running_loop()
    try:
        if _CONFIRM_TIMEOUT <= 0:
            answer = await loop.run_in_executor(None, sys.stdin.readline)
        else:
            answer = await asyncio.wait_for(
                loop.run_in_executor(None, sys.stdin.readline),
                timeout=_CONFIRM_TIMEOUT,
            )
        return answer.strip().lower() not in ("n", "no")
    except asyncio.TimeoutError:
        print("  (timed out — auto-approved)")
        return True


# ---------------------------------------------------------------------------
# Audio mode helper
# ---------------------------------------------------------------------------

def _audio_mode(ptt, vad) -> str:
    """Return a human-readable string describing the audio capture mode."""
    has_ptt = ptt is not None
    has_vad = vad is not None and vad.available
    if has_ptt and has_vad:
        return "ptt+vad"
    elif has_ptt:
        return "ptt-only"
    elif has_vad:
        return "vad-only"
    else:
        return "always-on"


# ---------------------------------------------------------------------------
# Receive loop — runs as a background task
# ---------------------------------------------------------------------------

async def receive_loop(client: WSClient, playback=None, tool_executor=None,
                       struggle_detector=None, screen=None, memory_store=None,
                       session_ready: asyncio.Event | None = None,
                       chat_store=None, session_id: str = "",
                       pattern_model=None, ml_manager=None,
                       screen_navigator=None,
                       autonomous_mode: asyncio.Event | None = None,
                       task_active: asyncio.Event | None = None,
                       wake_word=None,
                       translation_state: dict[str, object] | None = None) -> None:
    """Continuously read messages from the cloud and print/play them."""
    audio_frames_received = 0
    task_in_progress = False  # Track autonomous task execution
    async for msg in client.receive():
        if isinstance(msg, dict):
            msg_type = msg.get("type", "")

            if msg_type == "transcript":
                # Text response from Gemini (relayed through cloud)
                speaker = msg.get("speaker", "rio")
                text = msg.get("text", "")
                if speaker == "rio" and text:
                    # Keep wake word alive while Rio is responding
                    if wake_word is not None and wake_word.is_listening:
                        wake_word.keep_alive()
                    print(f"\n  Rio: {text}")
                    print("  You: ", end="", flush=True)
                    # Save to chat store
                    if chat_store is not None and session_id:
                        try:
                            chat_store.add_message(session_id, "rio", text)
                        except Exception:
                            log.debug("chat_store.save_failed")
                    # L4: Feed response to struggle detector for error keyword tracking
                    if struggle_detector is not None:
                        struggle_detector.feed_gemini_response(text)
                    # ML: Record language patterns from Rio's response
                    if pattern_model is not None:
                        try:
                            pattern_model.record_language(text, source="rio_response")
                        except Exception:
                            pass
                    # ML Pipeline: Record Rio's response
                    if ml_manager is not None:
                        try:
                            ml_manager.record_interaction("rio", text, time.time())
                        except Exception:
                            pass
                elif speaker == "user":
                    # Check if user said an exit phrase to deactivate listening
                    if wake_word is not None and wake_word.is_listening and text:
                        if wake_word.check_exit_phrase(text):
                            print("\n  [Rio deactivated — say 'Rio' to wake]")
                            print("  You: ", end="", flush=True)
                    # Otherwise: Echo of our own message — ignore

            elif msg_type == "translation":
                translated_text = msg.get("translated_text", "")
                if translated_text:
                    source_language = msg.get("source_language", "auto")
                    target_language = msg.get("target_language", "en")
                    direction = msg.get("direction", "translation")
                    print(
                        f"\n  Translator [{direction} {source_language}->{target_language}]: {translated_text}",
                    )
                    print("  You: ", end="", flush=True)

            elif msg_type == "text":
                # Direct text response (fallback / future use)
                content = msg.get("content", msg.get("text", ""))
                if content:
                    print(f"\n  Rio: {content}")
                    print("  You: ", end="", flush=True)

            elif msg_type == "error":
                error = msg.get("message", msg.get("content", msg.get("error", "unknown error")))
                log.error("cloud.error", detail=error)
                print(f"\n  [error] {error}")
                print("  You: ", end="", flush=True)

            elif msg_type == "status":
                status = msg.get("content", msg.get("status", ""))
                log.info("cloud.status", status=status)

            elif msg_type == "control":
                action = msg.get("action", "")
                log.info("cloud.control", action=action)
                if action == "connected":
                    print("  [connected to Gemini session]")
                    if session_ready is not None:
                        session_ready.set()
                elif action == "error":
                    print(f"  [connection error: {msg.get('detail', '')}]")
                elif action == "reconnecting":
                    print("  [reconnecting...]")
                elif action == "reconnected":
                    detail = msg.get("detail", "session reconnected")
                    print(f"  [{detail}]")
                    print("  You: ", end="", flush=True)
                elif action == "turn_complete":
                    log.debug("cloud.turn_complete")
                    if task_in_progress:
                        task_in_progress = False
                        if task_active is not None:
                            task_active.clear()
                        log.info("task_mode.ended")
                elif action == "task_mode":
                    active = msg.get("active", False)
                    step = msg.get("step", 0)
                    task_in_progress = active
                    if task_active is not None:
                        if active:
                            task_active.set()
                        else:
                            task_active.clear()
                    if active:
                        log.info("task_mode.active", step=step)
                    else:
                        log.info("task_mode.ended")
                elif action == "session_mode":
                    actual = msg.get("actual_mode", "???")
                    requested = msg.get("requested_mode", "???")
                    print(f"  [session mode: {actual} (requested: {requested})]")
                elif action == "live_api_unavailable":
                    detail = msg.get("detail", "")
                    print(f"  [warning] {detail}")
                    print("  [Rio is running in text-only mode — no voice]")
                elif action == "live_ready":
                    print("  [Live API session fully initialized — voice active]")
                elif action == "interrupted":
                    # ADK: user spoke while agent was responding — stop playback
                    log.info("cloud.interrupted")
                    if playback is not None and hasattr(playback, 'interrupt'):
                        playback.interrupt()
                    elif playback is not None and hasattr(playback, 'stop'):
                        playback.stop()
                    print("\n  [interrupted — listening...]")
                    print("  You: ", end="", flush=True)
                elif action == "translator_mode":
                    enabled = bool(msg.get("enabled", False))
                    source_language = str(msg.get("source_language", "auto") or "auto")
                    target_language = str(msg.get("target_language", "en") or "en")
                    bidirectional = bool(msg.get("bidirectional", False))
                    if translation_state is not None:
                        translation_state["enabled"] = enabled
                        translation_state["source_language"] = source_language
                        translation_state["target_language"] = target_language
                        translation_state["bidirectional"] = bidirectional
                    mode = "bidirectional" if bidirectional else "single-direction"
                    status = "ON" if enabled else "OFF"
                    print(f"  [live translation: {status} | {source_language} -> {target_language} | {mode}]")
                    print("  You: ", end="", flush=True)

            elif msg_type == "dashboard":
                # Dashboard-only messages — ignore in CLI
                log.debug("cloud.dashboard", subtype=msg.get("subtype"))

            elif msg_type == "tool_call":
                # L3: Gemini wants us to execute a tool
                tool_name = msg.get("name", "unknown")
                tool_args = msg.get("args", {})
                tool_call_id = msg.get("id", "")  # ADK ToolBridge correlation ID
                log.info("tool_call.received", name=tool_name, args=tool_args, id=tool_call_id)
                print(f"\n  [tool] {tool_name}({_format_tool_args(tool_args)})")

                # Special handling: capture_screen tool
                if tool_name == "capture_screen" and screen is not None:
                    # Text mode status — let user know Rio is analyzing their screen
                    if not playback or not hasattr(playback, 'is_playing') or not playback.is_playing:
                        print("\n  [Rio is looking at your screen...]")
                    try:
                        jpeg = await screen.capture_async(force=True)
                        if jpeg is not None:
                            # Sync monitor offset to screen navigator
                            if screen_navigator is not None:
                                cr = screen.get_last_capture_result()
                                if cr is not None:
                                    screen_navigator.update_monitor_offset(
                                        cr.monitor_left, cr.monitor_top,
                                    )
                            await client.send_binary(IMAGE_PREFIX + jpeg)
                            result = {
                                "success": True,
                                "message": "Screenshot captured and sent to your vision context.",
                                "size_kb": round(len(jpeg) / 1024, 1),
                            }
                            log.info("capture_screen.sent", size_kb=result["size_kb"])
                            print(f"  [screenshot captured and sent — {result['size_kb']} KB]")
                        else:
                            result = {
                                "success": False,
                                "error": "Screen capture returned empty frame.",
                            }
                            print("  [screenshot failed — empty frame]")
                    except Exception as exc:
                        log.exception("capture_screen.error")
                        result = {"success": False, "error": str(exc)}
                        print(f"  [screenshot error: {exc}]")
                    try:
                        result_frame = {
                            "type": "tool_result",
                            "name": "capture_screen",
                            "result": result,
                        }
                        if tool_call_id:
                            result_frame["id"] = tool_call_id
                        await client.send_json(result_frame)
                    except ConnectionError:
                        log.warning("capture_screen.send_failed")
                    print("  You: ", end="", flush=True)
                    continue

                if tool_executor is not None:
                    # Confirmation gate: auto-approve read_file,
                    # prompt for write_file / patch_file / run_command
                    if tool_name in ("write_file", "patch_file", "run_command"):
                        approved = await _confirm_tool_call(tool_name, tool_args)
                        if not approved:
                            result = {
                                "success": False,
                                "error": f"User declined {tool_name} execution.",
                            }
                            log.info("tool_call.declined", name=tool_name)
                            print(f"  [tool] {tool_name} — declined by user")
                            try:
                                result_frame = {
                                    "type": "tool_result",
                                    "name": tool_name,
                                    "result": result,
                                }
                                if tool_call_id:
                                    result_frame["id"] = tool_call_id
                                await client.send_json(result_frame)
                            except ConnectionError:
                                pass
                            print("  You: ", end="", flush=True)
                            continue

                    # Auto-capture after ALL screen actions (not just autonomous mode)
                    # so the model always gets visual feedback after clicks/scrolls/etc.
                    is_screen_action = (
                        hasattr(tool_executor, 'SCREEN_ACTION_TOOLS')
                        and tool_name in tool_executor.SCREEN_ACTION_TOOLS
                    )

                    if is_screen_action:
                        # Mark task active if in autonomous mode
                        if autonomous_mode is not None and autonomous_mode.is_set():
                            task_in_progress = True
                            if task_active is not None:
                                task_active.set()
                        print(f"\n  🔄 Screen: {tool_name}({_format_tool_args(tool_args)})")
                        result = await tool_executor.execute_with_auto_capture(
                            tool_name, tool_args,
                        )
                        if result.get("auto_capture"):
                            print(f"  [auto-capture sent — verifying action]")
                    else:
                        result = await tool_executor.execute(tool_name, tool_args)
                    success = result.get("success", False)
                    log.info("tool_call.executed", name=tool_name, success=success)

                    # Show result summary to user
                    if success:
                        _print_tool_success(tool_name, result)
                    else:
                        print(f"  [tool] FAILED: {result.get('error', 'unknown error')}")

                    # Send result back to cloud (include id for ADK ToolBridge)
                    try:
                        result_frame = {
                            "type": "tool_result",
                            "name": tool_name,
                            "result": result,
                        }
                        if tool_call_id:
                            result_frame["id"] = tool_call_id
                        await client.send_json(result_frame)
                    except ConnectionError:
                        log.warning("tool_call.send_failed", name=tool_name)

                    # L5: Store tool execution in memory
                    if memory_store is not None and success:
                        try:
                            summary = f"Tool {tool_name} executed: {_format_tool_args(tool_args)}"
                            if tool_name == "run_command":
                                output = result.get("output", "")[:200]
                                summary += f" → {output}"
                            memory_store.add(
                                summary,
                                entry_type="tool_use",
                                metadata={"tool": tool_name},
                            )
                        except Exception:
                            log.debug("memory.store_tool_failed")
                else:
                    # No tool executor — send error result
                    print("  [tool] tools not available")
                    try:
                        result_frame = {
                            "type": "tool_result",
                            "name": tool_name,
                            "result": {
                                "success": False,
                                "error": "Tool execution not available on this client",
                            },
                        }
                        if tool_call_id:
                            result_frame["id"] = tool_call_id
                        await client.send_json(result_frame)
                    except ConnectionError:
                        pass

                print("  You: ", end="", flush=True)

            else:
                # Unknown message type — log and display raw
                log.debug("cloud.unknown_type", msg=msg)
                print(f"\n  [cloud] {json.dumps(msg, indent=2)}")
                print("  You: ", end="", flush=True)

        elif isinstance(msg, bytes):
            # Binary frame from cloud — check wire protocol prefix
            if len(msg) > 1:
                prefix = msg[0:1]
                payload = msg[1:]

                if prefix == b"\x01":
                    # Audio frame from Gemini — enqueue for playback
                    audio_frames_received += 1
                    if audio_frames_received == 1:
                        log.info("audio.first_frame_received", bytes=len(payload))
                    # Keep wake word alive while Rio is speaking
                    if wake_word is not None and wake_word.is_listening:
                        wake_word.keep_alive()
                    if playback is not None:
                        playback.enqueue(payload)
                    elif audio_frames_received == 1:
                        # Log only once — repeated per-frame logs flood the console
                        log.warning("playback.skipped", reason="no playback device")
                else:
                    log.debug("cloud.unknown_binary_prefix", prefix=prefix.hex())
            else:
                log.debug("cloud.binary.too_short", length=len(msg))
        else:
            log.warning("cloud.unexpected", msg=msg)


def _format_tool_args(args: dict) -> str:
    """Format tool args for display — truncate long values."""
    parts = []
    for k, v in args.items():
        sv = str(v)
        if len(sv) > 60:
            sv = sv[:57] + "..."
        parts.append(f"{k}={sv!r}")
    return ", ".join(parts)


def _print_tool_success(name: str, result: dict) -> None:
    """Print a brief success summary for each tool type."""
    if name == "read_file":
        lines = result.get("lines", "?")
        path = result.get("path", "?")
        trunc = " (truncated)" if result.get("truncated") else ""
        print(f"  [tool] read {path} — {lines} lines{trunc}")
    elif name == "write_file":
        bw = result.get("bytes_written", 0)
        path = result.get("path", "?")
        print(f"  [tool] wrote {path} — {bw} bytes")
    elif name == "patch_file":
        path = result.get("path", "?")
        print(f"  [tool] patched {path}")
    elif name == "run_command":
        ec = result.get("exit_code", "?")
        output = result.get("output", "")
        # Show first/last few lines of output
        lines = output.strip().split("\n")
        if len(lines) <= 5:
            for line in lines:
                print(f"  [cmd] {line}")
        else:
            for line in lines[:3]:
                print(f"  [cmd] {line}")
            print(f"  [cmd] ... ({len(lines) - 5} more lines)")
            for line in lines[-2:]:
                print(f"  [cmd] {line}")
        print(f"  [tool] exit code: {ec}")
    else:
        print(f"  [tool] {name} OK")


# ---------------------------------------------------------------------------
# Application-level heartbeat — keeps Gemini session alive on the cloud
# ---------------------------------------------------------------------------

async def heartbeat_loop(client: WSClient) -> None:
    """Send periodic heartbeat messages to the cloud so the
    SessionManager doesn't consider us idle and drops the Gemini session.
    """
    while True:
        await asyncio.sleep(10)
        if client.is_connected:
            try:
                await client.send_json({"type": "heartbeat"})
                log.debug("heartbeat.sent")
            except ConnectionError:
                log.debug("heartbeat.skipped", reason="not connected")


# ---------------------------------------------------------------------------
# Audio capture loop — streams microphone PCM to cloud
# ---------------------------------------------------------------------------

async def audio_capture_loop(
    client: WSClient,
    capture,
    ptt=None,
    vad=None,
    playback=None,
    wake_word=None,
    task_active: asyncio.Event | None = None,
    orchestrator=None,
    mic_pause_until: list[float] | None = None,
) -> None:
    """Read audio chunks from the microphone and send as binary frames.

    Supports four degradation modes:
      ptt+vad   — F2 gates capture, VAD filters silence
      ptt-only  — F2 gates capture, all audio sent while held
      vad-only  — always captures, VAD filters silence
      always-on — streams everything (Day 3 behaviour)

    Chunks are 20ms (320 samples @ 16kHz) for low-latency capture.

    Wake word priority: When wake word is active (LISTENING state), audio
    flows to the cloud regardless of PTT state. This lets "Hey Rio"
    override the need to hold F2.

    Task interrupt: If the user presses F2 during an autonomous task
    (task_active is set), the task is paused cleanly with a progress
    report. Can be resumed later with "continue" or "resume".
    """
    has_ptt = ptt is not None
    has_vad = vad is not None and vad.available

    chunks_sent = 0
    ptt_was_active = False  # Track PTT edge transitions
    vad_was_speaking = False  # Track VAD speech-start edge for interrupt

    async for chunk in capture.chunks():
        if mic_pause_until is not None and time.monotonic() < mic_pause_until[0]:
            continue

        if not client.is_connected:
            log.debug("audio_loop.waiting_for_reconnect")
            while not client.is_connected:
                await asyncio.sleep(1.0)
            log.debug("audio_loop.reconnected")
            continue

        # -- Wake word processing (ALWAYS runs, even with PTT) -----------------
        ww_active = False
        ww_just_activated = False
        ww_just_deactivated = False
        if wake_word is not None and wake_word.available:
            ww_result = wake_word.process(chunk)
            ww_active = ww_result.should_send_audio
            ww_just_activated = ww_result.activated
            ww_just_deactivated = ww_result.deactivated

            if ww_just_activated and playback is not None:
                playback.clear()  # Interrupt Rio mid-sentence

            if ww_just_deactivated:
                # Wake word session ended — signal end of speech
                try:
                    await client.send_json({
                        "type": "control", "action": "end_of_speech",
                    })
                except ConnectionError:
                    pass
                continue

        # -- Determine audio gate -----------------------------------------------
        # Wake word LISTENING state overrides PTT entirely.
        if ww_active:
            # Wake word mode: audio flows without needing F2
            pass  # Fall through to VAD gate and send
        elif has_ptt:
            # Normal PTT mode (wake word sleeping or unavailable)
            ptt_now = ptt.is_active
            ptt_just_pressed = ptt_now and not ptt_was_active
            ptt_just_released = not ptt_now and ptt_was_active
            ptt_was_active = ptt_now

            if ptt_just_pressed:
                # Task interrupt: F2 during autonomous task pauses cleanly
                if task_active is not None and task_active.is_set():
                    task_active.clear()
                    # Get progress report from orchestrator if available
                    progress_report = ""
                    if orchestrator is not None:
                        progress_report = orchestrator.pause()
                    log.info("task_pause.ptt", note="User pressed F2 during task")
                    print("\n  ⏸ Task paused by user")
                    if progress_report:
                        print(f"  {progress_report[:200]}")
                    try:
                        abort_content = (
                            "[SYSTEM: User pressed the interrupt key. PAUSE the current "
                            "task gracefully. Here is the progress so far:\n"
                            f"{progress_report}\n"
                            "Tell the user what you've accomplished and what remains. "
                            "They can say 'resume' or 'continue' to pick up where we left off.]"
                        )
                        await client.send_json({
                            "type": "context",
                            "subtype": "task_pause",
                            "content": abort_content,
                        })
                    except ConnectionError:
                        pass

                if has_vad:
                    vad.reset()
                # Interrupt: clear playback buffer so Rio stops mid-sentence
                if playback is not None:
                    playback.clear()
                log.info("ptt.pressed")
                print("\n  [listening...]")
                print("  You: ", end="", flush=True)

            if ptt_just_released:
                log.info("ptt.released")
                print("\n  [stopped listening]")
                print("  You: ", end="", flush=True)
                try:
                    await client.send_json({
                        "type": "control",
                        "action": "end_of_speech",
                    })
                except ConnectionError:
                    pass

            # Gate: skip audio if PTT key is not held
            if not ptt_now:
                continue
        else:
            # No PTT and no wake word active — check if wake word is sleeping
            if wake_word is not None and wake_word.available and not ww_active:
                continue  # Wake word is sleeping, don't send audio

        # -- VAD gate (async to avoid blocking audio I/O) --------------------
        if has_vad:
            try:
                result = await vad.process_async(chunk)
            except Exception as exc:
                # Keep voice loop alive on intermittent model/runtime failures.
                log.warning("vad.process_failed", error=str(exc))
                continue
            if not result.is_speech:
                vad_was_speaking = False
                continue

            # Speech-start edge: clear playback buffer (interrupt)
            if not vad_was_speaking:
                vad_was_speaking = True
                if playback is not None:
                    playback.clear()
                    log.debug("vad.interrupt", note="cleared playback on speech start")

        # -- Send audio --------------------------------------------------------
        try:
            await client.send_binary(AUDIO_PREFIX + chunk)
            chunks_sent += 1
            if chunks_sent % 500 == 0:  # Log every ~10 seconds (500 * 20ms)
                log.debug("audio_loop.progress", chunks_sent=chunks_sent)
        except ConnectionError:
            log.warning("audio_loop.send_failed", reason="disconnected")
            await asyncio.sleep(1)
            # Wait for reconnection before resuming
            while not client.is_connected:
                await asyncio.sleep(0.5)
        except KeyboardInterrupt:
            log.info("audio_loop.interrupted")
            return
        except Exception:
            log.exception("audio_loop.error")
            await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Periodic screen capture loop — sends screenshots at configured fps
# ---------------------------------------------------------------------------

async def screen_capture_loop(
    client: WSClient,
    screen: "ScreenCapture",
    autonomous_mode: asyncio.Event,
    screen_navigator=None,
    task_active: asyncio.Event | None = None,
) -> None:
    """Periodically capture the screen and send as binary vision frames.

    Only runs when autonomous_mode is set.  In on-demand mode, this loop
    sleeps until the mode is toggled (via F5 or voice command), saving
    Gemini API credits.

    When task_active is set (autonomous task in progress), periodic captures
    are suppressed — auto-capture after each screen action handles it instead.

    Uses delta detection to skip unchanged frames. The interval is
    derived from ``vision.fps`` in config.yaml (default: 1 frame / 3s).

    Backpressure: Uses a bounded queue (maxsize=2) so that if Gemini is
    slow processing frames, old frames are dropped instead of queuing
    unboundedly and leaking memory.
    """
    interval = screen.interval
    frames_sent = 0
    # Bounded queue prevents memory leak when processing is slower than capture
    pending_frames: asyncio.Queue = asyncio.Queue(maxsize=2)

    async def _sender():
        """Drain the bounded queue and send frames to cloud."""
        nonlocal frames_sent
        while True:
            try:
                jpeg = await pending_frames.get()
            except asyncio.CancelledError:
                return
            try:
                await client.send_binary(IMAGE_PREFIX + jpeg)
                frames_sent += 1
                if frames_sent % 10 == 0:
                    log.debug(
                        "screen_loop.progress",
                        frames_sent=frames_sent,
                        frame_kb=round(len(jpeg) / 1024, 1),
                    )
            except ConnectionError:
                log.debug("screen_loop.send_failed", reason="disconnected")
            except Exception:
                log.exception("screen_loop.send_error")

    # Start sender task
    sender_task = asyncio.create_task(_sender(), name="screen_capture_sender")

    try:
        while True:
            # Block here until autonomous mode is activated
            await autonomous_mode.wait()
            await asyncio.sleep(interval)

            # Skip periodic capture during autonomous task — auto-capture handles it
            if task_active is not None and task_active.is_set():
                continue

            if not client.is_connected:
                log.debug("screen_loop.waiting_reconnect")
                await asyncio.sleep(2)
                continue

            try:
                jpeg = await screen.capture_async()
                if jpeg is None:
                    continue  # unchanged frame — delta detected

                # Sync monitor offset to screen navigator for coordinate mapping
                if screen_navigator is not None:
                    cr = screen.get_last_capture_result()
                    if cr is not None:
                        screen_navigator.update_monitor_offset(cr.monitor_left, cr.monitor_top)

                # Backpressure: drop frame if queue is full (processing too slow)
                try:
                    pending_frames.put_nowait(jpeg)
                except asyncio.QueueFull:
                    log.debug("screen_loop.frame_dropped", reason="backpressure")

            except ConnectionError:
                log.debug("screen_loop.send_failed", reason="disconnected")
                await asyncio.sleep(2)
            except Exception:
                log.exception("screen_loop.error")
                await asyncio.sleep(1)
    finally:
        sender_task.cancel()
        await asyncio.gather(sender_task, return_exceptions=True)


async def ui_navigator_loop(
    screen: "ScreenCapture",
    autonomous_mode: asyncio.Event,
    ui_navigator,
    screen_navigator=None,
) -> None:
    """Feed the UI Navigator with continuous 10fps JPEG frames in autonomous mode."""
    interval = 1.0 / max(0.1, float(getattr(ui_navigator, "fps", 10.0)))

    while True:
        await autonomous_mode.wait()
        tick_start = time.monotonic()
        try:
            jpeg = await screen.capture_async(force=True)
            if jpeg is not None:
                if screen_navigator is not None:
                    cr = screen.get_last_capture_result()
                    if cr is not None:
                        screen_navigator.update_monitor_offset(cr.monitor_left, cr.monitor_top)
                await ui_navigator.enqueue_frame(jpeg)
        except Exception:
            log.exception("ui_navigator.frame_error")

        elapsed = time.monotonic() - tick_start
        await asyncio.sleep(max(0.0, interval - elapsed))


# ---------------------------------------------------------------------------
# F3 toggle loop — mute/unmute mic input
# ---------------------------------------------------------------------------

async def mute_pause_toggle_loop(
    trigger: "PushToTalk",
    mic_pause_until: list[float],
    mic_muted_state: list[bool],
) -> None:
    """Toggle microphone mute state with a single hotkey (F3)."""

    while True:
        await trigger.wait_for_press()

        if not mic_muted_state[0]:
            mic_muted_state[0] = True
            mic_pause_until[0] = float("inf")

            log.info("f3.toggle", muted=True)
            print("\n  [F3: mic muted]")
        else:
            mic_muted_state[0] = False
            mic_pause_until[0] = 0.0

            log.info("f3.toggle", muted=False)
            print("\n  [F3: mic unmuted]")

        print("  You: ", end="", flush=True)
        await trigger.wait_for_release()


async def task_status_hotkey_loop(
    trigger: "PushToTalk",
    client: WSClient,
    orchestrator=None,
) -> None:
    """F8: Report ongoing task status in plain language (no task IDs)."""
    while True:
        await trigger.wait_for_press()

        if orchestrator is None:
            status_text = "Task status is unavailable right now because the local orchestrator is not initialized."
        elif orchestrator.is_busy:
            report = orchestrator.get_progress_report()
            status_text = f"Current ongoing task status:\n{report}"
        else:
            status_text = "There is no ongoing task right now."

        print(f"\n  [F8 task status]\n{status_text}")

        if client.is_connected:
            try:
                await client.send_json({
                    "type": "context",
                    "subtype": "task_status",
                    "content": (
                        "[SYSTEM: User requested the current task status via F8. "
                        "Respond clearly in plain language and do not mention task IDs.]\n"
                        f"{status_text}"
                    ),
                })
            except ConnectionError:
                pass

        print("  You: ", end="", flush=True)
        await trigger.wait_for_release()


# ---------------------------------------------------------------------------
# Screen mode toggle loop — F5 switches between on-demand and autonomous
# ---------------------------------------------------------------------------

async def screen_mode_toggle_loop(
    trigger: "PushToTalk",
    autonomous_mode: asyncio.Event,
) -> None:
    """Wait for the screen-mode hotkey (F5) and toggle autonomous mode.

        - on_demand (default): Frames sent via voice/capture_screen tool.
      Saves Gemini API credits.
    - autonomous: Periodic frames sent every ~3s (original behaviour).
    """
    while True:
        await trigger.wait_for_press()

        if autonomous_mode.is_set():
            autonomous_mode.clear()
            mode_label = "on-demand"
            log.info("screen_mode.toggled", mode="on_demand")
        else:
            autonomous_mode.set()
            mode_label = "autonomous"
            log.info("screen_mode.toggled", mode="autonomous")

        print(f"\n  [screen mode: {mode_label}]")
        print("  You: ", end="", flush=True)
        await trigger.wait_for_release()


# ---------------------------------------------------------------------------
# Live Mode toggle loop — F6 toggles full autonomous agentic mode
# ---------------------------------------------------------------------------

async def live_mode_toggle_loop(
    trigger: "PushToTalk",
    autonomous_mode: asyncio.Event,
    wake_word=None,
    client: WSClient | None = None,
) -> None:
    """F6: Toggle Live Mode (full autonomous agentic behavior).

    When ON:
      - autonomous_mode.set() (continuous screen capture)
      - Wake word forced to LISTENING (always accepts audio)
      - Sends notification to cloud about mode change

    When OFF:
      - autonomous_mode.clear() (on-demand screen only)
      - Wake word returns to normal SLEEPING cycle
      - Sends notification to cloud
    """
    live_active = False

    while True:
        await trigger.wait_for_press()
        live_active = not live_active

        if live_active:
            autonomous_mode.set()
            if wake_word is not None and wake_word.available:
                wake_word.force_activate()

            print("\n  ⚡ LIVE MODE ON — autonomous screen + voice + navigation")
            print("  [Press F6 to exit Live Mode]")
            log.info("live_mode.activated")

            # Notify cloud so Gemini knows to be proactive
            if client is not None:
                try:
                    await client.send_json({
                        "type": "context",
                        "subtype": "mode_change",
                        "content": (
                            "[SYSTEM: Live Mode activated. You now receive continuous "
                            "screen frames every 3 seconds. Be proactive — describe what "
                            "you see, point out errors, suggest improvements. The user "
                            "wants you actively watching and helping. Don't wait to be "
                            "asked. If you see something wrong, say it.]"
                        ),
                    })
                except ConnectionError:
                    pass
        else:
            autonomous_mode.clear()
            if wake_word is not None and wake_word.available:
                wake_word.force_deactivate()

            print("\n  💤 LIVE MODE OFF — on-demand mode, say 'Rio' to activate")
            log.info("live_mode.deactivated")

            if client is not None:
                try:
                    await client.send_json({
                        "type": "context",
                        "subtype": "mode_change",
                        "content": (
                            "[SYSTEM: Live Mode deactivated. Screen capture is now "
                            "on-demand only. Wait for the user to ask before analyzing "
                            "their screen. Respond only when spoken to.]"
                        ),
                    })
                except ConnectionError:
                    pass

        print("  You: ", end="", flush=True)
        await trigger.wait_for_release()


# ---------------------------------------------------------------------------
# Live translation toggle loop — F7 toggles translator mode
# ---------------------------------------------------------------------------

async def live_translation_toggle_loop(
    trigger: "PushToTalk",
    client: WSClient,
) -> None:
    """F7: Toggle live translator mode in cloud session config."""
    while True:
        await trigger.wait_for_press()
        try:
            await client.send_json({
                "type": "control",
                "action": "translator_toggle",
            })
            print("\n  [live translation toggle requested]")
        except ConnectionError:
            print("\n  [live translation toggle failed — not connected]")

        print("  You: ", end="", flush=True)
        await trigger.wait_for_release()


# ---------------------------------------------------------------------------
# Input loop — reads from stdin in a thread
# ---------------------------------------------------------------------------

async def input_loop(client: WSClient, struggle_detector=None,
                     wake_word=None, chat_store=None, session_id: str = "",
                     pattern_model=None, ml_manager=None,
                     orchestrator=None,
                     autonomous_mode: asyncio.Event | None = None,
                     task_active: asyncio.Event | None = None) -> None:
    """Read user text from stdin and send to the cloud.

    Task requests are detected and routed through the local orchestrator
    for autonomous plan+execute. Questions and chat go to Gemini via cloud.

    Because ``input()`` is blocking, we run it in the default executor
    so it doesn't block the event loop.
    """
    loop = asyncio.get_running_loop()
    _active_task: asyncio.Task | None = None

    while True:
        try:
            text = await loop.run_in_executor(None, _read_input)
        except (EOFError, KeyboardInterrupt):
            break

        text = text.strip()
        if not text:
            continue

        # L4: Note user activity for struggle detection Signal 4
        if struggle_detector is not None:
            struggle_detector.note_user_activity()

        if text.lower() in {"exit", "quit", "/quit", "/exit"}:
            break

        # Cancel running autonomous task
        if text.lower() in {"stop", "cancel", "abort"} and _active_task is not None:
            if orchestrator is not None and orchestrator.is_busy:
                orchestrator.cancel()
                print("\n  [Task cancelled]")
                print("  You: ", end="", flush=True)
                continue

        # Resume existing task
        if text.lower() in {"resume", "continue", "go on", "keep going"}:
            if orchestrator is not None:
                print("\n  [Resuming task...]")
                _active_task = asyncio.create_task(
                    _run_autonomous_task(
                        "resume", orchestrator, client,
                        autonomous_mode or asyncio.Event(),
                        task_active or asyncio.Event(),
                    ),
                    name="autonomous_task",
                )
                continue

        # Wake word: text input always activates listening
        if wake_word is not None and wake_word.available:
            wake_word.force_activate()

        # Save to chat store
        if chat_store is not None and session_id:
            try:
                chat_store.add_message(session_id, "user", text)
            except Exception:
                log.debug("chat_store.save_failed")

        # ML: Record activity and language patterns
        if pattern_model is not None:
            try:
                pattern_model.record_activity("message", {"source": "text_input"})
                pattern_model.record_language(text, source="user_input")
            except Exception:
                pass

        # ML Pipeline: Record user interaction for ensemble learning
        if ml_manager is not None:
            try:
                ml_manager.record_interaction("user", text, time.time())
            except Exception:
                pass

        # L4: Detect decline of proactive help
        if struggle_detector is not None and _is_decline(text):
            struggle_detector.record_decline()
            log.info("input.decline_detected", text=text)
            if pattern_model is not None:
                try:
                    pattern_model.record_help_response(accepted=False, context=text)
                except Exception:
                    pass

        # --- Task detection: route tasks through orchestrator ---
        if (orchestrator is not None
                and autonomous_mode is not None
                and task_active is not None
                and _is_task_request(text)):
            # Don't start a new task if one is already running
            if orchestrator.is_busy:
                print("\n  [A task is already running. Say 'stop' to cancel or wait for it to finish.]")
                print("  You: ", end="", flush=True)
                continue

            goal = text
            # Strip explicit prefix if present
            for prefix in ("task:", "do:", "execute:", "automate:"):
                if goal.lower().startswith(prefix):
                    goal = goal[len(prefix):].strip()
                    break

            log.info("input.task_detected", goal=goal[:100])
            _active_task = asyncio.create_task(
                _run_autonomous_task(
                    goal, orchestrator, client,
                    autonomous_mode, task_active,
                ),
                name="autonomous_task",
            )
            continue  # Don't also send to cloud — orchestrator handles it

        # --- Normal message: send to Gemini via cloud ---
        payload = {"type": "text", "content": text}
        try:
            await client.send_json(payload)
        except ConnectionError:
            log.warning("input.not_connected", note="message dropped — reconnecting")


def _read_input() -> str:
    """Blocking stdin read, meant to run in an executor."""
    return input("  You: ")


# ---------------------------------------------------------------------------
# Struggle detection loop (L4) — evaluates signals every ~2s
# ---------------------------------------------------------------------------

async def struggle_detection_loop(
    client: WSClient,
    screen,
    detector: "StruggleDetector",
    ocr_engine=None,
    memory_store=None,
    orchestrator=None,
) -> None:
    """Periodically evaluate struggle signals and trigger proactive help.

    Runs every 2 seconds.  Captures a screen frame (reusing the existing
    ScreenCapture), feeds it to the detector, and sends a context frame
    to the cloud if the detector fires.

    Priority 11: Auto-takeover — if the user doesn't respond within 15s
    after being offered help, automatically start a diagnostic task via
    the orchestrator.

    When ``ocr_engine`` is provided, OCR text is extracted from each
    frame and fed to the detector so Signal 1 hashes text content
    instead of raw pixels (immune to cursor blink, clock ticks, etc.).
    """
    EVAL_INTERVAL = 2.0  # seconds between evaluations

    while True:
        await asyncio.sleep(EVAL_INTERVAL)

        if not client.is_connected:
            continue

        # -- Priority 11: Check auto-takeover timeout --
        if (orchestrator is not None
                and detector.should_auto_takeover()
                and not orchestrator.is_busy):
            log.info("struggle_loop.auto_takeover")
            print("\n  [Rio is taking over — auto-diagnosing the issue...]")
            try:
                # Build a diagnostic goal from the last trigger context
                diagnostic_goal = (
                    "The developer appears to be stuck. Analyze the current screen, "
                    "identify the error or problem visible, and suggest or implement a fix. "
                    "If you can see a specific error message, address it directly."
                )
                asyncio.create_task(orchestrator.run(diagnostic_goal))
            except Exception:
                log.exception("struggle_loop.auto_takeover_failed")

        # Feed a fresh screen frame (+ OCR text) into the detector
        if screen is not None:
            try:
                if ocr_engine is not None and ocr_engine.available:
                    jpeg, ocr_text = await screen.capture_with_text(
                        ocr_engine, force=True,
                    )
                    detector.feed_frame(jpeg, ocr_text=ocr_text)
                else:
                    jpeg = await screen.capture_async(force=True)
                    detector.feed_frame(jpeg)
            except Exception:
                log.debug("struggle_loop.capture_error")

        # Evaluate all signals
        result = detector.evaluate()

        if not result.should_trigger:
            continue

        # -- Trigger: send proactive context to cloud --
        log.info(
            "struggle_loop.triggering",
            confidence=result.confidence,
            signals=result.active_signals,
        )
        print(f"\n  [Rio notices you might be stuck — asking if you need help...]")
        print("  You: ", end="", flush=True)

        # L5: Query memory for similar past struggles
        memory_context = ""
        if memory_store is not None:
            try:
                query_text = f"struggle: {', '.join(result.active_signals)}. {result.reason}"
                memories = memory_store.query(query_text, top_k=3)
                if memories:
                    memory_context = "\n" + memory_store.format_context(memories) + "\n"
            except Exception:
                log.debug("struggle_loop.memory_query_failed")

        context_payload = {
            "type": "context",
            "subtype": "struggle",
            "confidence": result.confidence,
            "signals": result.active_signals,
            "content": (
                f"[PROACTIVE CONTEXT — Rio's struggle detector fired] "
                f"The developer appears to be struggling. Signals: {', '.join(result.active_signals)}. "
                f"Detail: {result.reason}. "
                f"Confidence: {result.confidence:.2f}. "
                f"{memory_context}"
                f"Ask them a brief, specific question about what they're working on "
                f"and offer concrete help. Don't be generic — reference what you can "
                f"see on their screen if possible."
            ),
        }

        try:
            await client.send_json(context_payload)
            detector.record_trigger()
            # Priority 11: start auto-takeover timer
            detector.mark_offer_sent()
            # L5: Store struggle event in memory
            if memory_store is not None:
                try:
                    memory_store.add(
                        f"Struggle detected: {', '.join(result.active_signals)}. {result.reason}",
                        entry_type="struggle",
                        metadata={"confidence": result.confidence},
                    )
                except Exception:
                    log.debug("struggle_loop.memory_store_failed")
        except ConnectionError:
            log.warning("struggle_loop.send_failed")
        except Exception:
            log.exception("struggle_loop.error")


# ---------------------------------------------------------------------------
# F4 proactive trigger loop (demo mode only)
# ---------------------------------------------------------------------------

async def proactive_trigger_loop(
    client: WSClient,
    detector: "StruggleDetector",
    trigger: "PushToTalk",
) -> None:
    """Wait for F4 press and force a struggle trigger.

    Only active in demo mode.  Bypasses signal evaluation entirely
    for reliable hackathon demonstrations.
    """
    while True:
        await trigger.wait_for_press()
        log.info("proactive_trigger.f4_pressed")
        print("\n  [demo] Forced struggle trigger via F4")

        if not client.is_connected:
            print("  [not connected — trigger dropped]")
            await trigger.wait_for_release()
            continue

        result = detector.force_trigger()

        context_payload = {
            "type": "context",
            "subtype": "struggle",
            "confidence": result.confidence,
            "signals": result.active_signals,
            "content": (
                f"[PROACTIVE CONTEXT — Manual demo trigger] "
                f"The developer may need help. This is a demonstration of Rio's "
                f"proactive capability. Look at the developer's screen and offer "
                f"specific, helpful assistance with whatever they're working on. "
                f"Ask them a brief question about what you see."
            ),
        }

        try:
            await client.send_json(context_payload)
            detector.record_trigger()
        except ConnectionError:
            print("  [trigger send failed — not connected]")
        except Exception:
            log.exception("proactive_trigger.error")

        print("  You: ", end="", flush=True)
        await trigger.wait_for_release()


# ---------------------------------------------------------------------------
# Shutdown helpers
# ---------------------------------------------------------------------------

def _task_allows_goodbye(task: asyncio.Task) -> bool:
    """Return True when a completed task ended cleanly."""
    if task.cancelled():
        return False
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return False

    if exc is None:
        return True
    if isinstance(exc, KeyboardInterrupt):
        log.info("task.interrupted", task=task.get_name())
        return False

    log.warning("task.failed", task=task.get_name(), error=str(exc))
    return False


async def _shutdown_runtime(
    *,
    client: WSClient,
    tasks: set[asyncio.Task],
    session_id: str,
    send_goodbye: bool,
    ptt=None,
    screenshot_trigger=None,
    task_status_trigger=None,
    screen_mode_trigger=None,
    proactive_trigger=None,
    live_mode_trigger=None,
    live_translation_trigger=None,
    capture=None,
    playback=None,
    chat_store=None,
    pattern_model=None,
    ml_manager=None,
    orchestrator=None,
    browser_agent=None,
    session_memory=None,
    ui_navigator=None,
) -> None:
    """Stop background work and persist session state."""
    log.info("rio.shutting_down", goodbye=send_goodbye)

    if send_goodbye:
        try:
            if client.is_connected:
                print("\n  [sending goodbye to Rio...]")
                await client.send_json({
                    "type": "text",
                    "content": "The user is closing the app. Say a brief, warm goodbye!",
                })
                await asyncio.sleep(3.0)
        except Exception:
            log.debug("shutdown.goodbye_failed")

    if chat_store is not None:
        try:
            chat_store.add_message(session_id, "system", "Session ended")
            chat_store.end_session(session_id)
        except Exception:
            log.debug("chat_store.end_session_failed")

    if pattern_model is not None:
        try:
            pattern_model.record_activity("session_end", {"session_id": session_id})
            pattern_model.close()
        except Exception:
            log.debug("user_pattern.close_failed")

    if ml_manager is not None:
        try:
            ml_manager.close()
            log.info("ml_pipeline.saved")
        except Exception:
            log.debug("ml_pipeline.close_failed")

    for trigger in (
        ptt,
        screenshot_trigger,
        task_status_trigger,
        screen_mode_trigger,
        proactive_trigger,
        live_mode_trigger,
        live_translation_trigger,
    ):
        if trigger is not None:
            try:
                trigger.stop()
            except Exception:
                log.debug("trigger.stop_failed")

    if capture is not None:
        capture.stop()
    if playback is not None:
        playback.stop()

    if ui_navigator is not None:
        try:
            await ui_navigator.stop()
        except Exception:
            log.debug("ui_navigator.stop_failed")

    task_list = [task for task in tasks if task is not asyncio.current_task()]

    try:
        await client.close()
    except Exception:
        log.debug("ws.close_failed_during_shutdown")

    for task in task_list:
        if not task.done():
            task.cancel()

    if task_list:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*task_list, return_exceptions=True),
                timeout=8.0,
            )
        except asyncio.TimeoutError:
            log.warning(
                "shutdown.task_cancel_timeout",
                pending=[t.get_name() for t in task_list if not t.done()],
            )
            results = []
        for task, result in zip(task_list, results):
            if isinstance(result, asyncio.CancelledError):
                continue
            if isinstance(result, KeyboardInterrupt):
                log.info("task.interrupted", task=task.get_name())
                continue
            if isinstance(result, Exception):
                log.warning("task.failed", task=task.get_name(), error=str(result))

    if browser_agent is not None and browser_agent.is_running:
        try:
            await asyncio.wait_for(browser_agent.stop(), timeout=5.0)
        except Exception:
            log.debug("browser_agent.stop_failed")

    # Tools may have created a shared Playwright context even when BrowserAgent
    # wasn't actively running. Ensure it is closed before loop teardown.
    try:
        from browser_tools import cleanup as _browser_tools_cleanup
    except Exception:
        _browser_tools_cleanup = None
    if _browser_tools_cleanup is not None:
        try:
            await asyncio.wait_for(_browser_tools_cleanup(), timeout=5.0)
        except Exception:
            log.debug("browser_tools.cleanup_failed")

    if orchestrator is not None:
        try:
            orchestrator.close()
        except Exception:
            pass

    if chat_store is not None:
        try:
            chat_store.close()
        except Exception:
            pass

    if session_memory is not None:
        try:
            session_memory.close()
        except Exception:
            pass

    print("  [Rio session saved. Goodbye!]")
    log.info("rio.stopped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    # -- Graceful shutdown support (Windows-safe) --------------------------
    shutdown_event = asyncio.Event()

    def _signal_shutdown(signum, frame):
        """Handle SIGINT/SIGTERM to trigger clean shutdown."""
        print("\n  [Ctrl+C detected — shutting down gracefully...]")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_shutdown)
    signal.signal(signal.SIGTERM, _signal_shutdown)

    # -- Load config -------------------------------------------------------
    config = RioConfig.load()
    config.validate()
    log.info(
        "config.loaded",
        cloud_url=config.cloud_url,
        model=config.models.primary,
    )

    # -- Print banner ------------------------------------------------------
    print(BANNER)

    # -- Initialize Rio structured logging ---------------------------------
    if _RIO_LOGGING:
        rio_log_dir = Path(__file__).resolve().parent.parent / config.logging.log_dir
        setup_logging(
            log_dir=str(rio_log_dir),
            verbose=config.logging.verbose,
            max_files=config.logging.max_files,
        )
        log.info("rio_logging.initialized", log_dir=str(rio_log_dir))

    # -- Platform detection ------------------------------------------------
    if _PLATFORM_UTILS:
        print_platform_summary()
        missing = get_missing_dependencies()
        if missing:
            print(f"\n  [!] {len(missing)} missing optional dep(s):")
            for dep in missing:
                print(f"      - {dep['name']}: {dep['purpose']}")
                print(f"        Install: {dep['install_cmd']}")
            print()

    # -- Initialize audio capture ------------------------------------------
    capture: AudioCapture | None = None
    if AudioCapture is not None:
        try:
            dev_info = get_default_input_device()
            if dev_info:
                log.info("audio.default_device", **dev_info)
                print(f"  Mic: {dev_info['name']}")
            else:
                log.warning("audio.no_default_device")
                print("  Mic: (no default device found — audio disabled)")

            capture = AudioCapture(
                sample_rate=config.audio.sample_rate,
                block_size=config.audio.block_size,
                input_device=config.audio.input_device,
                use_wasapi=config.audio.use_wasapi,
            )
        except Exception:
            log.exception("audio.init_failed")
            print("  [warning] Audio capture unavailable — text-only mode")

    # -- Initialize audio playback -----------------------------------------
    playback: AudioPlayback | None = None
    if AudioPlayback is not None:
        try:
            out_info = get_default_output_device()
            if out_info:
                log.info("audio.default_output_device", **out_info)
                print(f"  Speaker: {out_info['name']}")
            else:
                log.warning("audio.no_default_output_device")
                print("  Speaker: (no default device found — audio playback disabled)")

            playback = AudioPlayback(
                sample_rate=24_000,  # Gemini output is 24kHz
                output_device=config.audio.output_device,
                use_wasapi=config.audio.use_wasapi,
            )
        except Exception:
            log.exception("playback.init_failed")
            print("  [warning] Audio playback unavailable — text-only responses")

    # -- Initialize VAD --------------------------------------------------------
    vad_instance = None
    if SileroVAD is not None and config.vad.enabled:
        try:
            vad_instance = SileroVAD(threshold=config.vad.threshold)
            if vad_instance.available:
                log.info("vad.ready", threshold=config.vad.threshold)
                print(f"  VAD: Silero (threshold={config.vad.threshold})")
            else:
                log.warning("vad.not_available")
                print("  VAD: unavailable (torch not loaded)")
        except Exception:
            log.exception("vad.init_failed")
            print("  VAD: init failed — no speech filtering")
    else:
        print("  VAD: disabled")

    # -- Initialize PTT --------------------------------------------------------
    ptt = None
    if PushToTalk is not None:
        ptt = PushToTalk.create(key_name=config.hotkeys.push_to_talk)
        if ptt is not None:
            log.info("ptt.ready", key=config.hotkeys.push_to_talk)
            print(f"  PTT: {config.hotkeys.push_to_talk.upper()} key")
        else:
            log.warning("ptt.not_available")
            print("  PTT: unavailable (pynput failed — Wayland?)")
    else:
        print("  PTT: disabled (pynput not installed)")

    # -- Print audio mode ------------------------------------------------------
    mode = _audio_mode(ptt, vad_instance)
    print(f"  Mode: {mode}")
    log.info("audio.mode", mode=mode)

    # -- Initialize screen capture (L2) ----------------------------------------
    screen: ScreenCapture | None = None
    if ScreenCapture is not None:
        try:
            screen = ScreenCapture(
                fps=config.vision.fps,
                quality=config.vision.quality,
                resize_factor=config.vision.resize_factor,
            )
            if screen.available:
                log.info(
                    "vision.ready",
                    fps=config.vision.fps,
                    quality=config.vision.quality,
                    resize=config.vision.resize_factor,
                )
                print(f"  Vision: {config.vision.fps} fps, JPEG q{config.vision.quality}, "
                      f"{int(config.vision.resize_factor*100)}% scale")
            else:
                log.warning("vision.deps_missing")
                print("  Vision: unavailable (install mss + Pillow)")
                screen = None
        except Exception:
            log.exception("vision.init_failed")
            print("  Vision: init failed")
            screen = None
    else:
        print("  Vision: disabled (screen_capture module not found)")

    # -- Initialize F3 mute toggle hotkey --------------------------------------
    screenshot_trigger = None
    if PushToTalk is not None:
        screenshot_trigger = PushToTalk.create(key_name=config.hotkeys.screenshot)
        if screenshot_trigger is not None:
            log.info("f3.hotkey_ready", key=config.hotkeys.screenshot)
            print(f"  Mute Toggle: {config.hotkeys.screenshot.upper()} key")
        else:
            log.warning("f3.hotkey_unavailable")
            print("  Mute Toggle: hotkey unavailable")
    else:
        print("  Mute Toggle: disabled (pynput not installed)")

    # -- Initialize F8 task status hotkey -------------------------------------
    task_status_trigger = None
    if PushToTalk is not None:
        task_status_trigger = PushToTalk.create(key_name=config.hotkeys.task_status)
        if task_status_trigger is not None:
            log.info("task_status.hotkey_ready", key=config.hotkeys.task_status)
            print(f"  Task Status: {config.hotkeys.task_status.upper()} key")
        else:
            log.warning("task_status.hotkey_unavailable")
            print("  Task Status: hotkey unavailable")
    else:
        print("  Task Status: disabled (pynput not installed)")

    # -- Initialize screen mode (on-demand vs autonomous) ----------------------
    autonomous_mode = asyncio.Event()
    task_active = asyncio.Event()  # Set during autonomous task execution
    if config.vision.default_mode == "autonomous":
        autonomous_mode.set()
        screen_mode_label = "autonomous"
    else:
        # on_demand is the default — event stays cleared
        screen_mode_label = "on-demand"
    if screen is not None:
        print(f"  Screen mode: {screen_mode_label} (F5 to toggle)")
        log.info("screen_mode.init", mode=config.vision.default_mode)

    # -- Initialize F5 screen mode toggle hotkey --------------------------------
    screen_mode_trigger = None
    if screen is not None and PushToTalk is not None:
        screen_mode_trigger = PushToTalk.create(
            key_name=config.hotkeys.screen_mode,
        )
        if screen_mode_trigger is not None:
            log.info("screen_mode.hotkey_ready", key=config.hotkeys.screen_mode)
        else:
            log.warning("screen_mode.hotkey_unavailable")
    
    # -- Initialize tool executor (L3) -----------------------------------------
    tool_executor = None
    if ToolExecutor is not None:
        tool_executor = ToolExecutor(working_dir=os.getcwd())
        log.info("tools.ready", working_dir=tool_executor.working_dir)
        print(f"  Tools: read_file, write_file, patch_file, run_command")
    else:
        print("  Tools: disabled (tools module not found)")

    # -- Initialize screen navigator ------------------------------------------
    screen_navigator = None
    if ScreenNavigator is not None:
        try:
            screen_navigator = ScreenNavigator(
                resize_factor=config.vision.resize_factor,
            )
            if screen_navigator.available:
                log.info("screen_nav.ready", resize_factor=config.vision.resize_factor)
                print("  Screen Nav: click, type, scroll, hotkey, drag, windows")
                # Attach to tool executor so tool calls get routed
                if tool_executor is not None:
                    tool_executor.set_screen_navigator(screen_navigator)
                    # Attach screen capture for auto-capture after screen actions
                    if screen is not None:
                        tool_executor.set_screen_capture(screen)
            else:
                log.warning("screen_nav.unavailable")
                print("  Screen Nav: unavailable (install pyautogui)")
                screen_navigator = None
        except Exception:
            log.exception("screen_nav.init_failed")
            print("  Screen Nav: init failed")
            screen_navigator = None
    else:
        print("  Screen Nav: disabled (screen_navigator module not found)")

    # -- Initialize struggle detector (L4) ------------------------------------
    struggle_detector = None
    if StruggleDetector is not None and config.struggle.enabled:
        try:
            struggle_detector = StruggleDetector(config.struggle)
            if struggle_detector.available:
                mode_str = "DEMO" if config.struggle.demo_mode else "normal"
                log.info(
                    "struggle.ready",
                    threshold=config.struggle.threshold,
                    demo_mode=config.struggle.demo_mode,
                )
                print(f"  Struggle: enabled ({mode_str}, threshold={config.struggle.threshold})")
            else:
                log.warning("struggle.not_available")
                print("  Struggle: unavailable")
                struggle_detector = None
        except Exception:
            log.exception("struggle.init_failed")
            print("  Struggle: init failed")
            struggle_detector = None
    elif StruggleDetector is not None and not config.struggle.enabled:
        print("  Struggle: disabled (config)")
    else:
        print("  Struggle: disabled (module not found)")

    # -- Initialize F4 proactive trigger (demo mode) --------------------------
    proactive_trigger = None
    if (struggle_detector is not None
            and config.struggle.demo_mode
            and PushToTalk is not None):
        proactive_trigger = PushToTalk.create(
            key_name=config.hotkeys.toggle_proactive,
        )
        if proactive_trigger is not None:
            log.info("proactive_trigger.ready", key=config.hotkeys.toggle_proactive)
            print(f"  Demo trigger: {config.hotkeys.toggle_proactive.upper()} key")
        else:
            log.warning("proactive_trigger.unavailable")
            print("  Demo trigger: unavailable (pynput failed)")
    elif struggle_detector is not None and config.struggle.demo_mode:
        print("  Demo trigger: unavailable (pynput not installed)")

    # -- Initialize shared genai client (L4+) ---------------------------------
    shared_genai_client = None
    if _genai is not None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if gcp_project:
            try:
                shared_genai_client = _genai.Client(
                    vertexai=True,
                    project=gcp_project,
                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
                )
            except Exception:
                log.warning("genai.client_init_failed_vertex")
        
        if shared_genai_client is None and api_key:
            try:
                shared_genai_client = _genai.Client(api_key=api_key)
            except Exception:
                log.warning("genai.client_init_failed_api_key")

    # -- Initialize memory store (L5) ------------------------------------------
    memory_store = None
    if MemoryStore is not None:
        try:
            memory_store = MemoryStore(
                db_path=config.memory.db_path,
                max_recall=config.memory.max_recall,
                genai_client=shared_genai_client,
            )
            log.info("memory.ready", db_path=config.memory.db_path, entries=memory_store.count())
            print(f"  Memory: {memory_store.count()} entries in {config.memory.db_path}")
        except RuntimeError as e:
            log.warning("memory.deps_missing", detail=str(e))
            print("  Memory: disabled (chromadb/sentence-transformers not installed)")
            memory_store = None
        except Exception:
            log.exception("memory.init_failed")
            print("  Memory: init failed")
            memory_store = None
    else:
        print("  Memory: disabled (module not available)")

    # -- Initialize wake word detector (L8) ------------------------------------
    wake_word = None
    if WakeWordDetector is not None:
        try:
            wake_word = WakeWordDetector(
                sample_rate=config.audio.sample_rate,
                enabled=True,
            )
            if wake_word.available:
                log.info("wake_word.ready")
                print("  Wake word: enabled (say 'Rio' or 'Hey Rio' to activate)")
            else:
                wake_word = None
                print("  Wake word: disabled")
        except Exception:
            log.exception("wake_word.init_failed")
            print("  Wake word: init failed")
            wake_word = None
    else:
        print("  Wake word: disabled (module not found)")

    # -- Initialize chat store (L8) --------------------------------------------
    chat_store = None
    import uuid as _uuid
    session_id = str(_uuid.uuid4())[:8]
    if ChatStore is not None:
        try:
            chat_store = ChatStore()
            chat_store.start_session(session_id)
            log.info("chat_store.ready", total_messages=chat_store.count())
            print(f"  Chat DB: {chat_store.count()} messages stored ({chat_store.db_path})")
        except Exception:
            log.exception("chat_store.init_failed")
            print("  Chat DB: init failed")
            chat_store = None
    else:
        print("  Chat DB: disabled (module not found)")

    # -- Initialize user pattern model (L8/ML) ---------------------------------
    pattern_model = None
    if UserPatternModel is not None:
        try:
            pattern_model = UserPatternModel()
            pattern_model.record_activity("session_start", {"session_id": session_id})
            ctx = pattern_model.get_context_string()
            if ctx:
                log.info("user_pattern.context", context=ctx[:200])
                print(f"  ML Model: active — {len(pattern_model._language_counter)} languages tracked")
            else:
                print("  ML Model: active (no patterns yet)")
        except Exception:
            log.exception("user_pattern.init_failed")
            print("  ML Model: init failed")
            pattern_model = None
    else:
        print("  ML Model: disabled (module not found)")

    # -- Initialize ML ensemble pipeline (v0.8.0) -----------------------------
    ml_manager = None
    if _ML_READY:
        try:
            ml_manager = UserModelManager(user_id="default")
            # Warm-start from historical DB data if available
            try:
                hist = ml_manager.train_on_history(days=30)
                if hist > 0:
                    log.info("ml_pipeline.warmstart", samples=hist)
                    print(f"  ML Pipeline: warm-started on {hist} historical samples")
                else:
                    print("  ML Pipeline: active (no history yet — cold start)")
            except Exception:
                print("  ML Pipeline: active (cold start)")
            # Get initial prediction context
            pred = ml_manager.predict_current()
            if pred:
                log.info("ml_pipeline.initial_prediction",
                         style=pred.get("chat_style", {}).get("label", "unknown"),
                         engagement=pred.get("engagement", {}).get("label", "unknown"))
        except Exception:
            log.exception("ml_pipeline.init_failed")
            print("  ML Pipeline: init failed (falling back to basic model)")
            ml_manager = None
    else:
        print("  ML Pipeline: disabled (scikit-learn not found)")

    # -- Build client ------------------------------------------------------
    client = WSClient(
        config.cloud_url,
        on_connect=lambda: log.info("event.connected"),
        on_disconnect=lambda: log.warning("event.disconnected"),
    )

    # -- Initialize UI Navigator (Live API at 10fps) -------------------------
    ui_navigator = None
    if (
        UINavigator is not None
        and tool_executor is not None
        and screen_navigator is not None
        and screen is not None
        and config.ui_navigator.enabled
    ):
        try:
            ui_navigator = UINavigator(
                tool_executor=tool_executor,
                screen_navigator=screen_navigator,
                model=config.ui_navigator.model,
                fps=config.ui_navigator.fps,
                confidence_threshold=config.ui_navigator.confidence_threshold,
                analyze_every_n_frames=config.ui_navigator.analyze_every_n_frames,
                emit_action=lambda payload: client.send_json_resilient(payload),
                genai_client=shared_genai_client,
                click_tool=config.ui_navigator.click_tool,
            )
            await ui_navigator.start()
            print(
                "  UI Navigator: enabled "
                f"({config.ui_navigator.fps:.1f}fps, model={config.ui_navigator.model}, "
                f"threshold={config.ui_navigator.confidence_threshold:.2f})"
            )
        except Exception:
            log.exception("ui_navigator.init_failed")
            print("  UI Navigator: init failed")
            ui_navigator = None
    elif config.ui_navigator.enabled:
        print("  UI Navigator: unavailable (missing screen/tools/navigator modules)")
    else:
        print("  UI Navigator: disabled (config)")

    # -- Initialize sub-agents & orchestrator (L4+) ----------------------------
    browser_agent = None
    if BrowserAgent is not None:
        browser_agent = BrowserAgent(api_key=os.environ.get("GEMINI_API_KEY", ""))
        if browser_agent.available:
            print("  Browser Agent: ready (Playwright + Gemini)")
        else:
            print("  Browser Agent: unavailable (install playwright)")
            browser_agent = None
    else:
        print("  Browser Agent: disabled (module not found)")

    windows_agent = None
    if WindowsAgent is not None:
        windows_agent = WindowsAgent()
        if windows_agent.available:
            print("  Windows Agent: ready (pywinauto)")
        else:
            print("  Windows Agent: unavailable (install pywinauto)")
            windows_agent = None
    else:
        print("  Windows Agent: disabled (module not found)")

    orchestrator = None
    if Orchestrator is not None:
        try:
            orchestrator = Orchestrator(
                api_key=os.environ.get("GEMINI_API_KEY", ""),
                tool_executor=tool_executor,
                screen_navigator=screen_navigator,
                screen_capture=screen,
                browser_agent=browser_agent,
                windows_agent=windows_agent,
            )
            print("  Orchestrator: ready (autonomous task execution)")
        except Exception:
            log.exception("orchestrator.init_failed")
            print("  Orchestrator: init failed")
            orchestrator = None
    else:
        print("  Orchestrator: disabled (module not found)")

    # -- Attach WebSocket sender to tool executor for auto-capture ---------
    if tool_executor is not None:
        tool_executor.set_ws_sender(client.send_binary)
        tool_executor.set_ws_json_sender(client.send_json)

    # -- Attach WebSocket client to orchestrator --------------------------------
    if orchestrator is not None:
        orchestrator.set_ws_client(client)

    # -- Initialize Creative Agent (Priority 9) --------------------------------
    creative_agent = None
    if CreativeAgent is not None:
        try:
            creative_agent = CreativeAgent(api_key=os.environ.get("GEMINI_API_KEY", ""))
            print("  Creative Agent: ready (Imagen 3 + Gemini)")
            # Attach to orchestrator for creative step dispatch
            if orchestrator is not None:
                orchestrator.set_creative_agent(creative_agent)
        except Exception:
            log.exception("creative_agent.init_failed")
            print("  Creative Agent: init failed")
    else:
        print("  Creative Agent: disabled (module not found)")

    # -- Initialize Session Memory (persistent notes) --------------------------
    session_memory = None
    if SessionMemory is not None:
        try:
            session_memory = SessionMemory()
            log.info("session_memory.ready")
            print(f"  Session Memory: ready ({session_memory.db_path})")
        except Exception:
            log.exception("session_memory.init_failed")
            print("  Session Memory: init failed")
    else:
        print("  Session Memory: disabled (module not found)")

    # -- Attach task store + session memory to tool executor --------------------
    if tool_executor is not None:
        if orchestrator is not None:
            tool_executor.set_task_store(orchestrator._task_store)
        if session_memory is not None:
            tool_executor.set_session_memory(session_memory)

    # -- Initialise OCR engine (L4 enhancement) ----------------------------
    ocr_engine = None
    if OCREngine is not None and struggle_detector is not None:
        ocr_engine = OCREngine()
        if ocr_engine.available:
            print("  [OCR engine ready — text-based struggle detection enabled]")
        else:
            ocr_engine = None
            print("  [OCR unavailable — falling back to pixel-hash detection]")

    # -- Connect -----------------------------------------------------------
    log.info("rio.starting", cloud_url=config.cloud_url)
    await client.connect()

    # -- Start audio capture -----------------------------------------------
    if capture is not None:
        try:
            capture.start()
            if wake_word is not None and wake_word.available:
                print("  [audio capture started — say 'Rio' or 'Hey Rio' to activate]")
            else:
                print("  [audio capture started — speak into your mic]")
            # Start PTT listener after audio capture
            if ptt is not None:
                ptt.start(asyncio.get_running_loop())
        except Exception:
            log.exception("audio.start_failed")
            print("  [warning] Could not start audio capture — text-only mode")
            capture = None

    # -- Start audio playback ----------------------------------------------
    if playback is not None:
        try:
            playback.start()
            print("  [audio playback started — you will hear Rio's voice]")
        except Exception:
            log.exception("playback.start_failed")
            print("  [warning] Could not start audio playback — text-only responses")
            playback = None

    # -- Start F3 mute toggle listener -------------------------------------
    if screenshot_trigger is not None:
        screenshot_trigger.start(asyncio.get_running_loop())
        print(f"  [press {config.hotkeys.screenshot.upper()} to mute/unmute voice input]")

    # -- Start F8 task status listener -------------------------------------
    if task_status_trigger is not None:
        task_status_trigger.start(asyncio.get_running_loop())
        print(f"  [press {config.hotkeys.task_status.upper()} to hear ongoing task status]")

    # -- Run all loops concurrently ----------------------------------------
    tasks: set[asyncio.Task] = set()
    mic_pause_until = [0.0]
    mic_muted_state = [False]
    translation_state: dict[str, object] = {
        "enabled": False,
        "source_language": "auto",
        "target_language": "en",
        "bidirectional": False,
    }

    session_ready = asyncio.Event()
    tasks.add(asyncio.create_task(
        receive_loop(client, playback, tool_executor, struggle_detector,
                     screen, memory_store, session_ready=session_ready,
                     chat_store=chat_store, session_id=session_id,
                     pattern_model=pattern_model, ml_manager=ml_manager,
                     screen_navigator=screen_navigator,
                     autonomous_mode=autonomous_mode,
                     task_active=task_active,
                     wake_word=wake_word,
                     translation_state=translation_state),
        name="recv",
    ))
    tasks.add(asyncio.create_task(
        input_loop(client, struggle_detector, wake_word=wake_word,
                   chat_store=chat_store, session_id=session_id,
                   pattern_model=pattern_model, ml_manager=ml_manager,
                   orchestrator=orchestrator,
                   autonomous_mode=autonomous_mode,
                   task_active=task_active),
        name="input",
    ))
    tasks.add(asyncio.create_task(heartbeat_loop(client), name="heartbeat"))

    if capture is not None:
        tasks.add(asyncio.create_task(
            audio_capture_loop(client, capture, ptt, vad_instance, playback,
                               wake_word=wake_word, task_active=task_active,
                               orchestrator=orchestrator,
                               mic_pause_until=mic_pause_until),
            name="audio",
        ))

    # PyAudio playback drain loop — writes audio to speaker via blocking write
    if playback is not None:
        tasks.add(asyncio.create_task(
            playback.drain_loop(),
            name="playback_drain",
        ))

    if screen is not None:
        tasks.add(asyncio.create_task(
            screen_capture_loop(client, screen, autonomous_mode,
                                screen_navigator=screen_navigator,
                                task_active=task_active),
            name="screen",
        ))

    if screen is not None and ui_navigator is not None:
        tasks.add(asyncio.create_task(
            ui_navigator_loop(
                screen,
                autonomous_mode,
                ui_navigator,
                screen_navigator=screen_navigator,
            ),
            name="ui_navigator",
        ))

    if screenshot_trigger is not None:
        tasks.add(asyncio.create_task(
            mute_pause_toggle_loop(
                screenshot_trigger,
                mic_pause_until,
                mic_muted_state,
            ),
            name="mute_pause_toggle",
        ))

    if task_status_trigger is not None:
        tasks.add(asyncio.create_task(
            task_status_hotkey_loop(task_status_trigger, client, orchestrator=orchestrator),
            name="task_status",
        ))

    # L4: Struggle detection loop
    if struggle_detector is not None and screen is not None:
        tasks.add(asyncio.create_task(
            struggle_detection_loop(client, screen, struggle_detector, ocr_engine,
                                    memory_store, orchestrator=orchestrator),
            name="struggle",
        ))

    # L4: F4 proactive trigger loop (demo mode only)
    if proactive_trigger is not None and struggle_detector is not None:
        proactive_trigger.start(asyncio.get_running_loop())
        print(f"  [press {config.hotkeys.toggle_proactive.upper()} to force struggle trigger (demo)]")
        tasks.add(asyncio.create_task(
            proactive_trigger_loop(client, struggle_detector, proactive_trigger),
            name="proactive_trigger",
        ))

    # Screen mode toggle loop (F5)
    if screen_mode_trigger is not None and screen is not None:
        screen_mode_trigger.start(asyncio.get_running_loop())
        print(f"  [press {config.hotkeys.screen_mode.upper()} to toggle screen mode]")
        tasks.add(asyncio.create_task(
            screen_mode_toggle_loop(screen_mode_trigger, autonomous_mode),
            name="screen_mode",
        ))

    # Live Mode toggle loop (F6)
    live_mode_trigger = None
    if PushToTalk is not None and screen is not None:
        live_mode_trigger = PushToTalk.create(key_name=config.hotkeys.live_mode)
        if live_mode_trigger is not None:
            live_mode_trigger.start(asyncio.get_running_loop())
            print(f"  [press {config.hotkeys.live_mode.upper()} for Live Mode (autonomous agentic)]")
            tasks.add(asyncio.create_task(
                live_mode_toggle_loop(
                    live_mode_trigger, autonomous_mode,
                    wake_word=wake_word, client=client,
                ),
                name="live_mode",
            ))

    # Live translation toggle loop (F7)
    live_translation_trigger = None
    if PushToTalk is not None:
        live_translation_trigger = PushToTalk.create(key_name=config.hotkeys.live_translation)
        if live_translation_trigger is not None:
            live_translation_trigger.start(asyncio.get_running_loop())
            print(f"  [press {config.hotkeys.live_translation.upper()} to toggle live translation]")
            tasks.add(asyncio.create_task(
                live_translation_toggle_loop(live_translation_trigger, client),
                name="live_translation",
            ))

    send_goodbye = False
    send_startup_greeting = os.environ.get("RIO_SEND_STARTUP_GREETING", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    try:
        # -- Send startup greeting ---------------------------------------------
        if send_startup_greeting:
            try:
                await asyncio.wait_for(session_ready.wait(), timeout=30.0)
                log.info("startup.session_ready")
                # Brief pause to let the Live session fully stabilise
                await asyncio.sleep(2.0)
                # Prevent speaker loopback from instantly interrupting startup speech.
                startup_mic_suppress = float(
                    os.environ.get("RIO_STARTUP_MIC_SUPPRESS_SECONDS", "12") or 12,
                )
                mic_pause_until[0] = time.monotonic() + max(0.0, startup_mic_suppress)
                await client.send_json({
                    "type": "text",
                    "content": "Hey Rio, say hello and introduce yourself briefly!",
                })
                log.info("startup.greeting_sent")
            except asyncio.TimeoutError:
                log.warning("startup.session_not_ready", note="Gemini session did not connect within 30s")
            except ConnectionError:
                log.warning("startup.greeting_failed", reason="not connected")
        else:
            log.info("startup.greeting_skipped", reason="disabled_by_default")

        # Wait until any task exits, user presses Ctrl+C, or shutdown_event is set
        while not shutdown_event.is_set():
            done, _ = await asyncio.wait(
                tasks,
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if done:
                break
        send_goodbye = not shutdown_event.is_set() and bool(done) and all(_task_allows_goodbye(task) for task in done)
    finally:
        await asyncio.shield(_shutdown_runtime(
            client=client,
            tasks=tasks,
            session_id=session_id,
            send_goodbye=send_goodbye,
            ptt=ptt,
            screenshot_trigger=screenshot_trigger,
            task_status_trigger=task_status_trigger,
            screen_mode_trigger=screen_mode_trigger,
            proactive_trigger=proactive_trigger,
            live_mode_trigger=live_mode_trigger,
            live_translation_trigger=live_translation_trigger,
            capture=capture,
            playback=playback,
            chat_store=chat_store,
            pattern_model=pattern_model,
            ml_manager=ml_manager,
            orchestrator=orchestrator,
            browser_agent=browser_agent,
            session_memory=session_memory,
            ui_navigator=ui_navigator,
        ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Goodbye!")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  [fatal error] {exc}")
        log.exception("rio.fatal")
        sys.exit(1)
