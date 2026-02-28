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
import signal
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
 |_| \_\_|\___/   v0.5.0 — Layer 4 (struggle detection)

 Proactive AI Pair Programmer
 Voice + screen vision + tools + struggle detection with Gemini.
 F2=Push-to-Talk  F3=Screenshot  F4=Force-trigger(demo)  Text input via stdin.
 Ctrl-C to quit.
"""


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
                       struggle_detector=None) -> None:
    """Continuously read messages from the cloud and print/play them."""
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
                    # L4: Feed response to struggle detector for error keyword tracking
                    if struggle_detector is not None:
                        struggle_detector.feed_gemini_response(text)
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
                elif action == "error":
                    print(f"  [connection error: {msg.get('detail', '')}]")
                elif action == "reconnecting":
                    print("  [reconnecting...]")
                elif action == "turn_complete":
                    log.debug("cloud.turn_complete")

            elif msg_type == "dashboard":
                # Dashboard-only messages — ignore in CLI
                log.debug("cloud.dashboard", subtype=msg.get("subtype"))

            elif msg_type == "tool_call":
                # L3: Gemini wants us to execute a tool
                tool_name = msg.get("name", "unknown")
                tool_args = msg.get("args", {})
                log.info("tool_call.received", name=tool_name, args=tool_args)
                print(f"\n  [tool] {tool_name}({_format_tool_args(tool_args)})")

                if tool_executor is not None:
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
) -> None:
    """Read audio chunks from the microphone and send as binary frames.

    Supports four degradation modes:
      ptt+vad   — F2 gates capture, VAD filters silence
      ptt-only  — F2 gates capture, all audio sent while held
      vad-only  — always captures, VAD filters silence
      always-on — streams everything (Day 3 behaviour)
    """
    has_ptt = ptt is not None
    has_vad = vad is not None and vad.available

    chunks_sent = 0
    ptt_was_active = False  # Track PTT edge transitions

    async for chunk in capture.chunks():
        if not client.is_connected:
            log.debug("audio_loop.skipping", reason="not connected")
            await asyncio.sleep(0.5)
            continue

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
                continue

        # -- Send audio --------------------------------------------------------
        try:
            await client.send_binary(AUDIO_PREFIX + chunk)
            chunks_sent += 1
            if chunks_sent % 100 == 0:  # Log every ~10 seconds
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
) -> None:
    """Periodically capture the screen and send as binary vision frames.

    Uses delta detection to skip unchanged frames. The interval is
    derived from ``vision.fps`` in config.yaml (default: 1 frame / 3s).
    """
    interval = screen.interval
    frames_sent = 0

    while True:
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
# Input loop — reads from stdin in a thread
# ---------------------------------------------------------------------------

async def input_loop(client: WSClient, struggle_detector=None) -> None:
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
) -> None:
    """Periodically evaluate struggle signals and trigger proactive help.

    Runs every 2 seconds.  Captures a screen frame (reusing the existing
    ScreenCapture), feeds it to the detector, and sends a context frame
    to the cloud if the detector fires.
    """
    EVAL_INTERVAL = 2.0  # seconds between evaluations

    while True:
        await asyncio.sleep(EVAL_INTERVAL)

        if not client.is_connected:
            continue

        # Feed a fresh screen frame into the detector
        if screen is not None:
            try:
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
                f"Ask them a brief, specific question about what they're working on "
                f"and offer concrete help. Don't be generic — reference what you can "
                f"see on their screen if possible."
            ),
        }

        try:
            await client.send_json(context_payload)
            detector.record_trigger()
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

    # -- Build client ------------------------------------------------------
    client = WSClient(
        config.cloud_url,
        on_connect=lambda: log.info("event.connected"),
        on_disconnect=lambda: log.warning("event.disconnected"),
    )

    # -- Connect -----------------------------------------------------------
    log.info("rio.starting", cloud_url=config.cloud_url)
    await client.connect()

    # -- Start audio capture -----------------------------------------------
    if capture is not None:
        try:
            capture.start()
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

    tasks.add(asyncio.create_task(receive_loop(client, playback, tool_executor, struggle_detector), name="recv"))
    tasks.add(asyncio.create_task(input_loop(client, struggle_detector), name="input"))
    tasks.add(asyncio.create_task(heartbeat_loop(client), name="heartbeat"))

    if capture is not None:
        tasks.add(asyncio.create_task(
            audio_capture_loop(client, capture, ptt, vad_instance, playback), name="audio"
        ))

    if screen is not None:
        tasks.add(asyncio.create_task(
            screen_capture_loop(client, screen), name="screen"
        ))

    if screenshot_trigger is not None and screen is not None:
        tasks.add(asyncio.create_task(
            screenshot_hotkey_loop(client, screen, screenshot_trigger),
            name="screenshot",
        ))

    # L4: Struggle detection loop
    if struggle_detector is not None and screen is not None:
        tasks.add(asyncio.create_task(
            struggle_detection_loop(client, screen, struggle_detector),
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

    # Wait until any task exits (user typed /quit, Ctrl-C, or fatal error)
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # -- Shutdown ----------------------------------------------------------
    log.info("rio.shutting_down")

    # Stop PTT + screenshot + audio to avoid sending on a closing connection
    if ptt is not None:
        ptt.stop()
    if screenshot_trigger is not None:
        screenshot_trigger.stop()
    if proactive_trigger is not None:
        proactive_trigger.stop()
    if capture is not None:
        capture.stop()
    if playback is not None:
        playback.stop()

    await client.close()

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    log.info("rio.stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _handle_sigint() -> None:
    """Allow Ctrl-C to cleanly cancel the asyncio event loop."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Goodbye!")
        sys.exit(0)
