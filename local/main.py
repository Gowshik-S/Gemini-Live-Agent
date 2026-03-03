"""
Rio Local — Entry Point & Asyncio Orchestrator

Layer 3 (L3): voice + vision + tools client.
  - Loads configuration from config.yaml
  - Connects to the Rio cloud backend over WebSocket
  - Captures microphone audio (PCM 16-bit 16kHz mono) and streams to cloud
  - Plays back Gemini audio responses (PCM 24kHz) through the speaker
  - Push-to-Talk (F2) gates audio capture when available
  - Silero VAD filters silence when available
  - Screen capture: periodic (every 3s) + on-demand (F3 key)
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
import sys
from pathlib import Path

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
    from wake_word import WakeWordDetector, WakeWordState
except ImportError:
    WakeWordDetector = None
    WakeWordState = None

try:
    from chat_store import ChatStore
except ImportError:
    ChatStore = None

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
 F2=Push-to-Talk  F3=Screenshot  F4=Force-trigger(demo)  F5=Screen-mode
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


# Tool names that modify the filesystem / execute commands and need approval
_DANGEROUS_TOOLS = frozenset({"write_file", "patch_file", "run_command"})

_CONFIRM_TIMEOUT = 15  # seconds to wait before auto-declining


async def _confirm_tool_call(tool_name: str, tool_args: dict) -> bool:
    """Prompt user for confirmation before executing a dangerous tool.

    Returns True if approved, False otherwise.
    Auto-**approves** after ``_CONFIRM_TIMEOUT`` seconds of silence so
    tool chains aren't stalled while the user is away or in voice mode.
    """
    summary = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
    print(f"\n    Rio wants to run: {tool_name}({summary})")
    print(f"  Approve? [Y/n] (auto-approve in {_CONFIRM_TIMEOUT}s): ", end="", flush=True)

    loop = asyncio.get_running_loop()
    try:
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
                       pattern_model=None, ml_manager=None) -> None:
    """Continuously read messages from the cloud and print/play them."""
    audio_frames_received = 0
    async for msg in client.receive():
        if isinstance(msg, dict):
            msg_type = msg.get("type", "")

            if msg_type == "transcript":
                # Text response from Gemini (relayed through cloud)
                speaker = msg.get("speaker", "rio")
                text = msg.get("text", "")
                if speaker == "rio" and text:
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
                    # Echo of our own message — ignore
                    pass

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

            elif msg_type == "dashboard":
                # Dashboard-only messages — ignore in CLI
                log.debug("cloud.dashboard", subtype=msg.get("subtype"))

            elif msg_type == "tool_call":
                # L3: Gemini wants us to execute a tool
                tool_name = msg.get("name", "unknown")
                tool_args = msg.get("args", {})
                log.info("tool_call.received", name=tool_name, args=tool_args)
                print(f"\n  [tool] {tool_name}({_format_tool_args(tool_args)})")

                # Special handling: capture_screen tool
                if tool_name == "capture_screen" and screen is not None:
                    try:
                        jpeg = await screen.capture_async(force=True)
                        if jpeg is not None:
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
                        await client.send_json({
                            "type": "tool_result",
                            "name": "capture_screen",
                            "result": result,
                        })
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
                                await client.send_json({
                                    "type": "tool_result",
                                    "name": tool_name,
                                    "result": result,
                                })
                            except ConnectionError:
                                pass
                            print("  You: ", end="", flush=True)
                            continue

                    result = await tool_executor.execute(tool_name, tool_args)
                    success = result.get("success", False)
                    log.info("tool_call.executed", name=tool_name, success=success)

                    # Show result summary to user
                    if success:
                        _print_tool_success(tool_name, result)
                    else:
                        print(f"  [tool] FAILED: {result.get('error', 'unknown error')}")

                    # Send result back to cloud
                    try:
                        await client.send_json({
                            "type": "tool_result",
                            "name": tool_name,
                            "result": result,
                        })
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
                        await client.send_json({
                            "type": "tool_result",
                            "name": tool_name,
                            "result": {
                                "success": False,
                                "error": "Tool execution not available on this client",
                            },
                        })
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
                    if playback is not None:
                        playback.enqueue(payload)
                    else:
                        log.debug("playback.skipped", reason="no playback device")
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
) -> None:
    """Read audio chunks from the microphone and send as binary frames.

    Supports four degradation modes:
      ptt+vad   — F2 gates capture, VAD filters silence
      ptt-only  — F2 gates capture, all audio sent while held
      vad-only  — always captures, VAD filters silence
      always-on — streams everything (Day 3 behaviour)
    
    Chunks are 20ms (320 samples @ 16kHz) for low-latency capture.
    When wake_word is provided, audio is only sent when the wake word
    has been detected (Alexa-style activation).
    """
    has_ptt = ptt is not None
    has_vad = vad is not None and vad.available

    chunks_sent = 0
    ptt_was_active = False  # Track PTT edge transitions
    vad_was_speaking = False  # Track VAD speech-start edge for interrupt

    async for chunk in capture.chunks():
        if not client.is_connected:
            log.debug("audio_loop.skipping", reason="not connected")
            await asyncio.sleep(0.5)
            continue

        # -- Wake word gate (Alexa-style) -----------------------------------
        if wake_word is not None and wake_word.available:
            ww_result = wake_word.process(chunk)
            if not ww_result.should_send_audio:
                continue  # Still sleeping — don't send audio
            # If just activated, clear playback so Rio stops mid-sentence
            if ww_result.activated and playback is not None:
                playback.clear()

        # -- PTT edge detection ------------------------------------------------
        if has_ptt:
            ptt_now = ptt.is_active
            ptt_just_pressed = ptt_now and not ptt_was_active
            ptt_just_released = not ptt_now and ptt_was_active
            ptt_was_active = ptt_now

            if ptt_just_pressed:
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

        # -- VAD gate ----------------------------------------------------------
        if has_vad:
            result = vad.process(chunk)
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
            await asyncio.sleep(0.5)
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
) -> None:
    """Periodically capture the screen and send as binary vision frames.

    Only runs when autonomous_mode is set.  In on-demand mode, this loop
    sleeps until the mode is toggled (via F5 or voice command), saving
    Gemini API credits.

    Uses delta detection to skip unchanged frames. The interval is
    derived from ``vision.fps`` in config.yaml (default: 1 frame / 3s).
    """
    interval = screen.interval
    frames_sent = 0

    while True:
        # Block here until autonomous mode is activated
        await autonomous_mode.wait()
        await asyncio.sleep(interval)

        if not client.is_connected:
            continue

        try:
            jpeg = await screen.capture_async()
            if jpeg is None:
                continue  # unchanged frame — delta detected

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
            log.exception("screen_loop.error")
            await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Screenshot hotkey loop — captures on F3 press
# ---------------------------------------------------------------------------

async def screenshot_hotkey_loop(
    client: WSClient,
    screen: "ScreenCapture",
    trigger: "PushToTalk",
) -> None:
    """Wait for the screenshot hotkey (F3) and send a forced capture.

    Unlike periodic capture, this always sends (no delta skip) so the
    user gets immediate feedback that the screenshot was taken.
    """
    while True:
        await trigger.wait_for_press()
        log.info("screenshot.triggered")
        print("\n  [screenshot captured]")

        if not client.is_connected:
            print("  [not connected — screenshot dropped]")
            await trigger.wait_for_release()
            continue

        try:
            jpeg = await screen.capture_async(force=True)
            if jpeg is not None:
                await client.send_binary(IMAGE_PREFIX + jpeg)
                log.info(
                    "screenshot.sent",
                    size_kb=round(len(jpeg) / 1024, 1),
                )
                print(f"  [sent {round(len(jpeg)/1024, 1)} KB — ask Rio about your screen]")
            else:
                print("  [screenshot failed — capture returned nothing]")
        except ConnectionError:
            print("  [screenshot send failed — not connected]")
        except Exception:
            log.exception("screenshot.error")
            print("  [screenshot error]")

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

    - on_demand (default): Frames only sent via F3, voice, or capture_screen tool.
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
# Input loop — reads from stdin in a thread
# ---------------------------------------------------------------------------

async def input_loop(client: WSClient, struggle_detector=None,
                     wake_word=None, chat_store=None, session_id: str = "",
                     pattern_model=None, ml_manager=None) -> None:
    """Read user text from stdin and send to the cloud.

    Because ``input()`` is blocking, we run it in the default executor
    so it doesn't block the event loop.
    """
    loop = asyncio.get_running_loop()

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
) -> None:
    """Periodically evaluate struggle signals and trigger proactive help.

    Runs every 2 seconds.  Captures a screen frame (reusing the existing
    ScreenCapture), feeds it to the detector, and sends a context frame
    to the cloud if the detector fires.

    When ``ocr_engine`` is provided, OCR text is extracted from each
    frame and fed to the detector so Signal 1 hashes text content
    instead of raw pixels (immune to cursor blink, clock ticks, etc.).
    """
    EVAL_INTERVAL = 2.0  # seconds between evaluations

    while True:
        await asyncio.sleep(EVAL_INTERVAL)

        if not client.is_connected:
            continue

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
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
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

    # -- Initialize screenshot hotkey (F3) -------------------------------------
    screenshot_trigger = None
    if screen is not None and PushToTalk is not None:
        screenshot_trigger = PushToTalk.create(key_name=config.hotkeys.screenshot)
        if screenshot_trigger is not None:
            log.info("screenshot.hotkey_ready", key=config.hotkeys.screenshot)
            print(f"  Screenshot: {config.hotkeys.screenshot.upper()} key")
        else:
            log.warning("screenshot.hotkey_unavailable")
            print("  Screenshot: hotkey unavailable (periodic only)")
    elif screen is not None:
        print("  Screenshot: periodic only (pynput not available)")

    # -- Initialize screen mode (on-demand vs autonomous) ----------------------
    autonomous_mode = asyncio.Event()
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

    # -- Initialize memory store (L5) ------------------------------------------
    memory_store = None
    if MemoryStore is not None:
        try:
            memory_store = MemoryStore(
                db_path=config.memory.db_path,
                max_recall=config.memory.max_recall,
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

    # -- Start screenshot hotkey listener ----------------------------------
    if screenshot_trigger is not None:
        screenshot_trigger.start(asyncio.get_running_loop())
        print(f"  [press {config.hotkeys.screenshot.upper()} to capture your screen]")

    # -- Run all loops concurrently ----------------------------------------
    tasks: set[asyncio.Task] = set()

    session_ready = asyncio.Event()
    tasks.add(asyncio.create_task(
        receive_loop(client, playback, tool_executor, struggle_detector,
                     screen, memory_store, session_ready=session_ready,
                     chat_store=chat_store, session_id=session_id,
                     pattern_model=pattern_model, ml_manager=ml_manager),
        name="recv",
    ))
    tasks.add(asyncio.create_task(
        input_loop(client, struggle_detector, wake_word=wake_word,
                   chat_store=chat_store, session_id=session_id,
                   pattern_model=pattern_model, ml_manager=ml_manager),
        name="input",
    ))
    tasks.add(asyncio.create_task(heartbeat_loop(client), name="heartbeat"))

    if capture is not None:
        tasks.add(asyncio.create_task(
            audio_capture_loop(client, capture, ptt, vad_instance, playback,
                               wake_word=wake_word),
            name="audio",
        ))

    if screen is not None:
        tasks.add(asyncio.create_task(
            screen_capture_loop(client, screen, autonomous_mode), name="screen"
        ))

    if screenshot_trigger is not None and screen is not None:
        tasks.add(asyncio.create_task(
            screenshot_hotkey_loop(client, screen, screenshot_trigger),
            name="screenshot",
        ))

    # L4: Struggle detection loop
    if struggle_detector is not None and screen is not None:
        tasks.add(asyncio.create_task(
            struggle_detection_loop(client, screen, struggle_detector, ocr_engine, memory_store),
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

    # -- Send startup greeting -------------------------------------------------
    try:
        await asyncio.wait_for(session_ready.wait(), timeout=30.0)
        log.info("startup.session_ready")
        # Brief pause to let the Live session fully stabilise
        await asyncio.sleep(2.0)
        await client.send_json({
            "type": "text",
            "content": "Hey Rio, say hello and introduce yourself briefly!",
        })
        log.info("startup.greeting_sent")
    except asyncio.TimeoutError:
        log.warning("startup.session_not_ready", note="Gemini session did not connect within 30s")
    except ConnectionError:
        log.warning("startup.greeting_failed", reason="not connected")

    # Wait until any task exits (user typed /quit, Ctrl-C, or fatal error)
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # -- Shutdown ----------------------------------------------------------
    log.info("rio.shutting_down")

    # Send goodbye message to Rio before closing
    try:
        if client.is_connected:
            print("\n  [sending goodbye to Rio...]")
            await client.send_json({
                "type": "text",
                "content": "The user is closing the app. Say a brief, warm goodbye!",
            })
            # Wait briefly to receive goodbye response
            await asyncio.sleep(3.0)
    except Exception:
        log.debug("shutdown.goodbye_failed")

    # Save session end to chat store
    if chat_store is not None:
        try:
            chat_store.add_message(session_id, "system", "Session ended")
            chat_store.end_session(session_id)
        except Exception:
            log.debug("chat_store.end_session_failed")

    # Save session end to pattern model
    if pattern_model is not None:
        try:
            pattern_model.record_activity("session_end", {"session_id": session_id})
            pattern_model.close()
        except Exception:
            log.debug("user_pattern.close_failed")

    # Train & save ML ensemble model on session data
    if ml_manager is not None:
        try:
            ml_manager.close()  # trains on session + saves pkl
            log.info("ml_pipeline.saved")
        except Exception:
            log.debug("ml_pipeline.close_failed")

    # Stop PTT + screenshot + audio to avoid sending on a closing connection
    if ptt is not None:
        ptt.stop()
    if screenshot_trigger is not None:
        screenshot_trigger.stop()
    if screen_mode_trigger is not None:
        screen_mode_trigger.stop()
    if proactive_trigger is not None:
        proactive_trigger.stop()
    if capture is not None:
        capture.stop()
    if playback is not None:
        playback.stop()

    await client.close()

    # Close stores
    if chat_store is not None:
        try:
            chat_store.close()
        except Exception:
            pass

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    print("  [Rio session saved. Goodbye!]")
    log.info("rio.stopped")


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
