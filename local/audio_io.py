"""
Rio Local — Audio I/O Module (Low-Latency v3)

Layer 1 (L1): Audio capture and playback.
  - Capture: PCM 16-bit, 16kHz, mono audio in 20ms chunks via sounddevice
  - Playback: PCM 16-bit, 24kHz, mono via PyAudio blocking write
    (matches Google's own Gemini Live sample and Pipecat framework)
  - WASAPI fallback on Windows for capture
  - scipy.signal.resample_poly for high-quality resampling
  - Async-friendly: asyncio.to_thread for blocking writes
  - Device enumeration and selection
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

import platform

import numpy as np
import sounddevice as sd
import structlog

log = structlog.get_logger(__name__)

# Wire protocol prefix for audio frames
AUDIO_PREFIX = b"\x01"

# Platform detection for WASAPI
_IS_WINDOWS = platform.system() == "Windows"


def _get_wasapi_settings(exclusive: bool = False) -> object | None:
    """Return WASAPI-specific settings for sounddevice on Windows.

    Uses sd.WasapiSettings for low-latency audio on Windows.
    Returns None on non-Windows platforms.
    """
    if not _IS_WINDOWS:
        return None
    try:
        return sd.WasapiSettings(exclusive=exclusive)
    except Exception:
        log.debug("wasapi.settings_unavailable")
        return None


def _get_device_info(
    device: Optional[str | int],
    kind: str,
) -> dict | None:
    """Resolve a sounddevice device selector to a concrete device record."""
    try:
        return sd.query_devices(device=device, kind=kind)
    except Exception as exc:
        log.debug(
            "audio.device_lookup_failed",
            device=device,
            kind=kind,
            error=str(exc),
        )
        return None


def _get_hostapi_name(device_info: dict | None) -> str | None:
    """Return the PortAudio host API name for a device record."""
    if not device_info:
        return None

    try:
        hostapi_index = int(device_info["hostapi"])
        hostapi = sd.query_hostapis(hostapi_index)
        return str(hostapi.get("name", ""))
    except Exception as exc:
        log.debug("audio.hostapi_lookup_failed", error=str(exc))
        return None


class AudioCapture:
    """Captures audio from the microphone as PCM 16-bit 16kHz mono chunks.

    Uses sounddevice's callback-based InputStream. Audio chunks are placed
    into an asyncio Queue for consumption by the async event loop.

    Usage::

        capture = AudioCapture(sample_rate=16000, block_size=1600)
        capture.start()
        async for chunk in capture.chunks():
            # chunk is bytes (PCM 16-bit LE, 100ms)
            await ws.send_binary(AUDIO_PREFIX + chunk)
        capture.stop()
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        block_size: int = 320,
        input_device: Optional[str | int] = None,
        max_queue_size: int = 100,
        use_wasapi: bool = True,
    ) -> None:
        self._sample_rate = sample_rate
        self._block_size = block_size  # 320 samples = 20ms @ 16kHz
        self._input_device = input_device
        self._use_wasapi = use_wasapi
        self._stream: Optional[sd.InputStream] = None
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_queue_size)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin capturing audio from the microphone."""
        if self._running:
            log.warning("audio.already_running")
            return

        self._loop = asyncio.get_running_loop()

        # WASAPI exclusive mode for lowest latency on Windows
        extra_settings = None
        device_info = _get_device_info(self._input_device, kind="input")
        hostapi_name = _get_hostapi_name(device_info)
        if self._use_wasapi and _IS_WINDOWS:
            if hostapi_name == "Windows WASAPI":
                extra_settings = _get_wasapi_settings(exclusive=False)
                if extra_settings:
                    log.info(
                        "audio.wasapi_enabled",
                        exclusive=False,
                        device=self._input_device,
                        hostapi=hostapi_name,
                    )
            elif hostapi_name:
                log.info(
                    "audio.wasapi_skipped",
                    reason="non_wasapi_device",
                    device=self._input_device,
                    hostapi=hostapi_name,
                )

        log.info(
            "audio.starting",
            sample_rate=self._sample_rate,
            block_size=self._block_size,
            device=self._input_device,
            latency="low",
            wasapi=extra_settings is not None,
            hostapi=hostapi_name,
        )

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                dtype="int16",
                channels=1,
                device=self._input_device,
                latency="low",
                extra_settings=extra_settings,
                callback=self._audio_callback,
            )
            self._stream.start()
        except sd.PortAudioError as _pa_err:
            if extra_settings is not None:
                # WASAPI settings incompatible with this device (e.g. error -9984).
                # Retry without host-API-specific settings.
                log.warning(
                    "audio.wasapi_failed_retrying",
                    error=str(_pa_err),
                    device=self._input_device,
                )
                self._stream = sd.InputStream(
                    samplerate=self._sample_rate,
                    blocksize=self._block_size,
                    dtype="int16",
                    channels=1,
                    device=self._input_device,
                    latency="low",
                    callback=self._audio_callback,
                )
                self._stream.start()
            else:
                raise
        self._running = True
        log.info("audio.started", latency=self._stream.latency)

    def stop(self) -> None:
        """Stop capturing audio."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.exception("audio.stop_error")
            finally:
                self._stream = None
        log.info("audio.stopped")

    # ------------------------------------------------------------------
    # Async chunk generator
    # ------------------------------------------------------------------

    async def chunks(self) -> AsyncGenerator[bytes, None]:
        """Async generator yielding PCM audio chunks.

        Each chunk is raw PCM 16-bit LE, mono, at the configured sample rate.
        Chunk size = block_size * 2 bytes (int16 = 2 bytes per sample).
        At 16kHz with block_size=320, each chunk is 20ms / 640 bytes.
        """
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield chunk
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # sounddevice callback (runs in audio thread)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """sounddevice callback -- runs in a separate thread.

        Converts the numpy array to raw PCM bytes and schedules it
        onto the asyncio event loop's queue.
        """
        if status:
            log.warning("audio.callback_status", status=str(status))

        if not self._running or self._loop is None:
            return

        # Convert numpy int16 array to raw bytes (PCM 16-bit LE on x86/x64)
        pcm_bytes = indata.tobytes()

        # Schedule the put onto the async loop (thread-safe)
        try:
            self._loop.call_soon_threadsafe(self._enqueue, pcm_bytes)
        except RuntimeError:
            # Event loop closed
            pass

    def _enqueue(self, data: bytes) -> None:
        """Put audio data into the async queue (called from the event loop thread)."""
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            # Drop oldest chunk to prevent backpressure
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(data)
            except asyncio.QueueFull:
                pass


class AudioPlayback:
    """Plays PCM 24kHz 16-bit mono audio received from Gemini.

    Uses PyAudio's blocking ``stream.write()`` — the same approach used by:
      - Google's official Gemini Live API starter code
      - Pipecat (Daily.co's voice AI framework, 10k+ stars)
      - LiveKit Python agents

    The blocking write lets the OS audio driver manage its own internal
    buffer and timing.  No application-level callback threading, no
    jitter buffer, no silence padding.  Audio data is enqueued from the
    async event loop via ``enqueue()``, and a dedicated writer coroutine
    drains the queue with ``asyncio.to_thread(stream.write, chunk)``.

    Usage::

        playback = AudioPlayback(input_sample_rate=24000)
        playback.start()
        # In the receive loop:
        playback.enqueue(pcm_bytes)
        # Start the async writer task:
        asyncio.create_task(playback.drain_loop())
        ...
        playback.stop()
    """

    def __init__(
        self,
        sample_rate: int = 24_000,
        channels: int = 1,
        output_device: Optional[str | int] = None,
        max_queue_size: int = 300,
        use_wasapi: bool = True,  # kept for API compat, not used by PyAudio path
    ) -> None:
        self._input_sample_rate = sample_rate   # rate of incoming audio (Gemini)
        self._output_sample_rate: int = 0       # actual device rate (set in start())
        self._channels = channels
        self._output_device = output_device
        self._use_wasapi = use_wasapi
        self._running = False
        self._interrupted = False  # set during interrupt to skip queued chunks
        self._resample_ratio: float = 1.0

        # Thread-safe queue: enqueue() writes, drain_loop() reads
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_queue_size)

        # PyAudio objects
        self._pa = None   # pyaudio.PyAudio instance
        self._stream = None  # pyaudio.Stream

    @property
    def is_playing(self) -> bool:
        return self._running and not self._queue.empty()

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the PyAudio output stream."""
        if self._running:
            log.warning("playback.already_running")
            return

        import pyaudio

        self._pa = pyaudio.PyAudio()

        # Determine output device and native sample rate
        if self._output_device is not None:
            dev_info = self._pa.get_device_info_by_index(int(self._output_device))
        else:
            dev_info = self._pa.get_default_output_device_info()

        self._output_sample_rate = int(dev_info["defaultSampleRate"])
        self._resample_ratio = self._output_sample_rate / self._input_sample_rate

        log.info(
            "playback.starting",
            device=dev_info["name"],
            input_rate=self._input_sample_rate,
            output_rate=self._output_sample_rate,
            resample_ratio=round(self._resample_ratio, 4),
            engine="pyaudio_blocking_write",
        )

        # Open the stream at the device's native rate.
        # frames_per_buffer=1024 is a good default — the OS driver
        # handles the rest.  No callback needed.
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._output_sample_rate,
            output=True,
            output_device_index=(
                int(self._output_device) if self._output_device is not None else None
            ),
            frames_per_buffer=1024,
        )

        self._running = True
        log.info("playback.started", engine="pyaudio_blocking_write",
                 device=dev_info["name"], rate=self._output_sample_rate)

    def stop(self) -> None:
        """Stop playback and release PyAudio resources."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                log.exception("playback.stop_error")
            finally:
                self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        self.clear()
        log.info("playback.stopped")

    def interrupt(self) -> None:
        """Immediately silence playback but keep the stream open for future audio.

        Clears the queue so the drain_loop stops feeding the hardware buffer.
        The stream itself is intentionally left running — calling stop_stream()
        while drain_loop's stream.write() is blocking in an asyncio.to_thread
        task raises an OSError on Windows (the write and the stop race).
        Leaving the stream open is safe: it drains the OS buffer in ~20 ms,
        and the next model response plays immediately with no re-open overhead.
        """
        self._interrupted = True
        self.clear()  # Flush all queued audio — drain_loop skips any in-flight chunk
        # Reset AFTER clearing so drain_loop doesn't skip new audio
        # that arrives after the interrupt is fully handled.
        self._interrupted = False
        log.info("playback.interrupted")

    def _recreate_stream(self) -> None:
        """Close the broken stream and open a fresh one with the same settings."""
        import pyaudio

        old = self._stream
        self._stream = None
        if old is not None:
            try:
                old.stop_stream()
            except Exception:
                pass
            try:
                old.close()
            except Exception:
                pass

        if self._pa is None:
            return

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._output_sample_rate,
                output=True,
                output_device_index=(
                    int(self._output_device) if self._output_device is not None else None
                ),
                frames_per_buffer=1024,
            )
            log.info("playback.stream_recreated")
        except Exception:
            log.exception("playback.stream_recreate_failed")

    # ------------------------------------------------------------------
    # Resampling (scipy polyphase filter)
    # ------------------------------------------------------------------

    def _resample(self, data: bytes) -> bytes:
        """Resample PCM int16 audio from input rate to output rate.

        Uses scipy.signal.resample_poly with a proper FIR anti-aliasing
        filter for high-quality audio resampling.  Falls back to numpy
        linear interpolation only when scipy is unavailable.

        For the common 24 kHz → 44.1 kHz case the polyphase filter
        uses up=147, down=80 (GCD of rates = 300).
        """
        if self._resample_ratio == 1.0:
            return data

        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return data

        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(self._output_sample_rate, self._input_sample_rate)
            up = self._output_sample_rate // g
            down = self._input_sample_rate // g
            resampled = resample_poly(samples, up, down).astype(np.int16)
            return resampled.tobytes()
        except ImportError:
            # Fallback: linear interpolation (lower quality)
            log.warning("playback.resample.scipy_unavailable",
                        note="Install scipy for high-quality audio resampling")
            old_len = len(samples)
            new_len = int(old_len * self._resample_ratio)
            old_indices = np.arange(old_len)
            new_indices = np.linspace(0, old_len - 1, new_len)
            resampled = np.interp(new_indices, old_indices, samples).astype(np.int16)
            return resampled.tobytes()

    # ------------------------------------------------------------------
    # Enqueue / Clear / Drain
    # ------------------------------------------------------------------

    def enqueue(self, data: bytes) -> None:
        """Add PCM audio data to the playback queue.

        Resamples from Gemini's native rate to the output device rate
        if they differ.  Thread-safe.
        """
        if not self._running:
            return
        data = self._resample(data)
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            # Drop oldest to prevent backpressure / growing latency
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

    def clear(self) -> None:
        """Flush the playback queue so the speaker goes silent immediately."""
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def drain_loop(self) -> None:
        """Async coroutine that drains audio from the queue to the speaker.

        Must be run as an asyncio task.  Uses ``asyncio.to_thread()``
        to call the blocking ``stream.write()`` — this is exactly the
        pattern used by Google's Gemini Live API sample code and the
        Pipecat framework.

        The OS audio driver manages internal buffering and timing.
        ``stream.write()`` blocks until the hardware consumes the data,
        which provides natural back-pressure without needing an
        application-level jitter buffer.
        """
        log.info("playback.drain_loop.started")
        while self._running:
            try:
                # Wait for audio data with a timeout so we can check _running
                try:
                    chunk = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                # Skip chunk if interrupted (queue was cleared but this
                # chunk was already dequeued)
                if self._interrupted:
                    continue

                if self._stream is not None and chunk:
                    try:
                        await asyncio.to_thread(self._stream.write, chunk)
                    except OSError:
                        # stream.write() can fail if stream was reset by interrupt()
                        if self._interrupted:
                            continue
                        raise

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("playback.drain_loop.error")
                # Brief pause to avoid tight error loops
                await asyncio.sleep(0.1)

        log.info("playback.drain_loop.ended")


# ---------------------------------------------------------------------------
# Device enumeration utilities
# ---------------------------------------------------------------------------

def list_audio_devices() -> list[dict]:
    """Return a list of available audio input devices."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            hostapi_name = None
            try:
                hostapi_name = hostapis[int(dev["hostapi"])]["name"]
            except Exception:
                pass
            result.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "default_samplerate": dev["default_samplerate"],
                "hostapi": hostapi_name,
                "is_default": i == sd.default.device[0],
            })
    return result


def get_default_input_device() -> dict | None:
    """Return info about the default input device, or None."""
    try:
        dev = _get_device_info(None, kind="input")
        idx = sd.default.device[0]
        if dev is None or idx is None or idx < 0:
            return None
        return {
            "index": idx,
            "name": dev["name"],
            "channels": dev["max_input_channels"],
            "default_samplerate": dev["default_samplerate"],
            "hostapi": _get_hostapi_name(dev),
        }
    except Exception:
        return None


def list_output_devices() -> list[dict]:
    """Return a list of available audio output devices."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            hostapi_name = None
            try:
                hostapi_name = hostapis[int(dev["hostapi"])]["name"]
            except Exception:
                pass
            result.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
                "hostapi": hostapi_name,
                "is_default": i == sd.default.device[1],
            })
    return result


def get_default_output_device() -> dict | None:
    """Return info about the default output device, or None."""
    try:
        dev = _get_device_info(None, kind="output")
        idx = sd.default.device[1]  # [1] is output, [0] is input
        if dev is None or idx is None or idx < 0:
            return None
        return {
            "index": idx,
            "name": dev["name"],
            "channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
            "hostapi": _get_hostapi_name(dev),
        }
    except Exception:
        return None
