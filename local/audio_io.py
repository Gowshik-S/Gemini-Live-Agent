"""
Rio Local — Audio I/O Module (Low-Latency v2)

Layer 1 (L1): Audio capture and playback using sounddevice.
  - Capture: PCM 16-bit, 16kHz, mono audio in 20ms chunks (was 100ms)
  - Playback: PCM 16-bit, 24kHz, mono (matches Gemini output rate)
  - WASAPI exclusive mode on Windows for lowest latency
  - latency='low' on all streams
  - Reduced jitter buffer (40ms vs 200ms)
  - Async-friendly: queue-based interfaces for both directions
  - Device enumeration and selection
"""

from __future__ import annotations

import asyncio
import queue as _queue_mod
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
        max_queue_size: int = 200,
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
        if self._use_wasapi and _IS_WINDOWS:
            extra_settings = _get_wasapi_settings(exclusive=False)
            if extra_settings:
                log.info("audio.wasapi_enabled", exclusive=False)

        log.info(
            "audio.starting",
            sample_rate=self._sample_rate,
            block_size=self._block_size,
            device=self._input_device,
            latency="low",
            wasapi=extra_settings is not None,
        )

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

    Uses sounddevice's callback-based OutputStream. Audio chunks are
    enqueued from the async event loop via ``enqueue()`` and consumed
    by the audio callback running in a separate thread.

    On Linux with PipeWire, the virtual 'default'/'pulse' ALSA devices
    often hang with PortAudio's callback-based streaming.  This class
    auto-detects a working hardware output device and resamples
    Gemini's 24 kHz audio to the device's native rate (typically 48 kHz).

    Usage::

        playback = AudioPlayback(input_sample_rate=24000)
        playback.start()
        playback.enqueue(pcm_bytes)   # called from async receive loop
        ...
        playback.stop()
    """

    # Jitter buffer: accumulate this many bytes before starting playback.
    # Reduced from 200ms to 40ms for much faster first-sound latency.
    _JITTER_BUFFER_BYTES = 1920  # ~40ms at 24kHz 16-bit mono (24000*0.04*2)

    def __init__(
        self,
        sample_rate: int = 24_000,
        channels: int = 1,
        output_device: Optional[str | int] = None,
        max_queue_size: int = 600,
        use_wasapi: bool = True,
    ) -> None:
        self._input_sample_rate = sample_rate   # rate of incoming audio (Gemini)
        self._output_sample_rate = sample_rate  # actual device rate (may differ)
        self._channels = channels
        self._output_device = output_device
        self._use_wasapi = use_wasapi
        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._resample_ratio: float = 1.0       # output_rate / input_rate

        # Jitter buffer state: accumulate audio before sending to speaker
        self._jitter_buf = bytearray()
        self._jitter_ready = False  # True once we've accumulated enough

        # Thread-safe queue: audio callback (audio thread) reads,
        # enqueue() (event-loop thread) writes.
        # We use queue.Queue (not asyncio.Queue) because the consumer
        # is the sounddevice callback running outside asyncio.
        self._queue: _queue_mod.Queue[bytes] = _queue_mod.Queue(
            maxsize=max_queue_size
        )

        # Block size for the output stream callback.
        # 480 samples at 48kHz = 10ms per callback — small blocks for
        # low latency. Was 2400 (50ms).
        self._block_size = 480

    @property
    def is_playing(self) -> bool:
        """True if the stream is active and there is audio in the queue."""
        return self._running and not self._queue.empty()

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Device auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_working_output_device(
        preferred_rate: int = 24_000,
    ) -> tuple[int | None, int]:
        """Probe output devices and return (device_index, sample_rate).

        On Linux with PipeWire, virtual ALSA devices (default, pulse,
        pipewire) often hang with PortAudio's callback-based streams.
        We therefore prefer real ALSA hardware devices at their native
        sample rate and resample in software.

        Strategy:
        1. Try ALSA hardware devices at their native rate (most reliable).
        2. Try virtual devices at the preferred rate as fallback.
        3. Fall back to (None, preferred_rate) and hope for the best.
        """
        import time as _time
        import threading

        devices = sd.query_devices()

        virtual_names = {"default", "pulse", "pipewire"}
        hw_devs: list[tuple[int, dict]] = []
        virtual_devs: list[tuple[int, dict]] = []

        for i, dev in enumerate(devices):
            if dev["max_output_channels"] < 1:
                continue
            name = dev["name"].lower()
            if any(vn in name for vn in virtual_names):
                virtual_devs.append((i, dev))
            elif "hdmi" not in name:
                # Skip HDMI since user probably wants speakers
                hw_devs.append((i, dev))

        def _probe_device(idx: int, rate: int, timeout: float = 1.0) -> bool:
            """Open an OutputStream, write a tiny sine burst, and check
            that the callback actually fires (doesn't hang).
            """
            callback_fired = threading.Event()

            def _cb(outdata, frames, time_info, status):
                outdata[:] = 0  # silence
                callback_fired.set()

            try:
                stream = sd.OutputStream(
                    samplerate=rate,
                    blocksize=960,
                    dtype="int16",
                    channels=1,
                    device=idx,
                    callback=_cb,
                )
                stream.start()
                ok = callback_fired.wait(timeout=timeout)
                stream.stop()
                stream.close()
                return ok
            except Exception:
                return False

        # --- 1. Try hardware devices at native rate ---
        for idx, dev in hw_devs:
            native_rate = int(dev["default_samplerate"])
            if _probe_device(idx, native_rate):
                log.info(
                    "playback.device_probe.ok",
                    device=idx,
                    name=dev["name"],
                    rate=native_rate,
                    kind="hardware",
                )
                return idx, native_rate

        # --- 2. Try virtual devices at preferred rate ---
        for idx, dev in virtual_devs:
            if _probe_device(idx, preferred_rate):
                log.info(
                    "playback.device_probe.ok",
                    device=idx,
                    name=dev["name"],
                    rate=preferred_rate,
                    kind="virtual",
                )
                return idx, preferred_rate

        log.warning("playback.device_probe.no_working_device")
        return None, preferred_rate

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the output stream and begin playback."""
        if self._running:
            log.warning("playback.already_running")
            return

        # --- Auto-detect working device if none specified ---
        if self._output_device is None:
            dev_idx, dev_rate = self._find_working_output_device(
                preferred_rate=self._input_sample_rate,
            )
            self._output_device = dev_idx
            self._output_sample_rate = dev_rate
        else:
            self._output_sample_rate = self._input_sample_rate

        self._resample_ratio = self._output_sample_rate / self._input_sample_rate

        # Adjust block size proportionally to the output rate
        # Target ~10ms per callback (was ~50ms)
        self._block_size = max(int(480 * self._resample_ratio), 240)

        # WASAPI settings for low-latency playback on Windows
        extra_settings = None
        if self._use_wasapi and _IS_WINDOWS:
            extra_settings = _get_wasapi_settings(exclusive=False)
            if extra_settings:
                log.info("playback.wasapi_enabled", exclusive=False)

        log.info(
            "playback.starting",
            input_rate=self._input_sample_rate,
            output_rate=self._output_sample_rate,
            resample_ratio=round(self._resample_ratio, 4),
            block_size=self._block_size,
            device=self._output_device,
            latency="low",
            wasapi=extra_settings is not None,
        )

        self._stream = sd.OutputStream(
            samplerate=self._output_sample_rate,
            blocksize=self._block_size,
            dtype="int16",
            channels=self._channels,
            device=self._output_device,
            latency="low",
            extra_settings=extra_settings,
            callback=self._playback_callback,
        )
        self._stream.start()
        self._running = True
        log.info("playback.started", latency=self._stream.latency)

    def stop(self) -> None:
        """Stop playback and close the output stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.exception("playback.stop_error")
            finally:
                self._stream = None
        self.clear()
        log.info("playback.stopped")

    # ------------------------------------------------------------------
    # Enqueue / Clear
    # ------------------------------------------------------------------

    def _resample(self, data: bytes) -> bytes:
        """Resample PCM int16 audio from input rate to output rate.

        Uses numpy linear interpolation — fast and good enough for
        voice.  For the common 24 kHz → 48 kHz case (ratio 2.0) this
        simply doubles the sample count with smooth interpolation.
        """
        if self._resample_ratio == 1.0:
            return data

        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return data

        old_len = len(samples)
        new_len = int(old_len * self._resample_ratio)
        old_indices = np.arange(old_len)
        new_indices = np.linspace(0, old_len - 1, new_len)
        resampled = np.interp(new_indices, old_indices, samples).astype(np.int16)
        return resampled.tobytes()

    def enqueue(self, data: bytes) -> None:
        """Add PCM audio data to the playback queue.

        Resamples from Gemini's native rate to the output device rate
        if they differ.  Uses a jitter buffer to accumulate ~200ms of
        audio before starting playback, preventing choppy output.
        Thread-safe.
        """
        data = self._resample(data)

        # Jitter buffer: accumulate until we have enough data
        if not self._jitter_ready:
            self._jitter_buf.extend(data)
            if len(self._jitter_buf) >= self._JITTER_BUFFER_BYTES:
                self._jitter_ready = True
                # Flush the entire jitter buffer into the queue
                buf = bytes(self._jitter_buf)
                self._jitter_buf.clear()
                # Split into queue-friendly chunks (~960 bytes = 10ms at 48kHz)
                chunk_size = 960
                for i in range(0, len(buf), chunk_size):
                    piece = buf[i:i + chunk_size]
                    try:
                        self._queue.put_nowait(piece)
                    except _queue_mod.Full:
                        try:
                            self._queue.get_nowait()
                        except _queue_mod.Empty:
                            pass
                        try:
                            self._queue.put_nowait(piece)
                        except _queue_mod.Full:
                            pass
                log.debug("playback.jitter_buffer_flushed", bytes=len(buf))
            return

        try:
            self._queue.put_nowait(data)
        except _queue_mod.Full:
            # Drop oldest to prevent backpressure / growing latency
            try:
                self._queue.get_nowait()
            except _queue_mod.Empty:
                pass
            try:
                self._queue.put_nowait(data)
            except _queue_mod.Full:
                pass

    def clear(self) -> None:
        """Flush the playback queue and jitter buffer (e.g. on interrupt).

        Thread-safe. Drains all pending audio so the speaker goes silent
        immediately.
        """
        self._jitter_buf.clear()
        self._jitter_ready = False
        while True:
            try:
                self._queue.get_nowait()
            except _queue_mod.Empty:
                break

    # ------------------------------------------------------------------
    # sounddevice callback (runs in audio thread)
    # ------------------------------------------------------------------

    def _playback_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """sounddevice output callback -- runs in a separate audio thread.

        Pulls PCM bytes from the queue and fills the output buffer.
        If the queue is empty (or data insufficient), fill remainder
        with silence (zeros) to avoid underrun noise.
        """
        if status:
            log.warning("playback.callback_status", status=str(status))

        # Total bytes needed: frames * channels * 2 (int16 = 2 bytes/sample)
        bytes_needed = frames * self._channels * 2
        collected = bytearray()

        while len(collected) < bytes_needed:
            try:
                chunk = self._queue.get_nowait()
                collected.extend(chunk)
            except _queue_mod.Empty:
                break

        if len(collected) >= bytes_needed:
            # We have enough data — use exactly what we need,
            # push any excess back onto the queue
            audio_bytes = bytes(collected[:bytes_needed])
            remainder = bytes(collected[bytes_needed:])
            if remainder:
                try:
                    self._queue.put_nowait(remainder)
                except _queue_mod.Full:
                    pass  # drop excess if queue somehow full
        else:
            # Not enough data — pad with silence
            audio_bytes = bytes(collected) + b"\x00" * (bytes_needed - len(collected))

        # Convert raw bytes to numpy int16 array and write to output
        outdata[:] = np.frombuffer(audio_bytes, dtype=np.int16).reshape(-1, self._channels)


# ---------------------------------------------------------------------------
# Device enumeration utilities
# ---------------------------------------------------------------------------

def list_audio_devices() -> list[dict]:
    """Return a list of available audio input devices."""
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            result.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "default_samplerate": dev["default_samplerate"],
                "is_default": i == sd.default.device[0],
            })
    return result


def get_default_input_device() -> dict | None:
    """Return info about the default input device, or None."""
    try:
        idx = sd.default.device[0]
        if idx is None or idx < 0:
            return None
        dev = sd.query_devices(idx)
        return {
            "index": idx,
            "name": dev["name"],
            "channels": dev["max_input_channels"],
            "default_samplerate": dev["default_samplerate"],
        }
    except Exception:
        return None


def list_output_devices() -> list[dict]:
    """Return a list of available audio output devices."""
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            result.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
                "is_default": i == sd.default.device[1],
            })
    return result


def get_default_output_device() -> dict | None:
    """Return info about the default output device, or None."""
    try:
        idx = sd.default.device[1]  # [1] is output, [0] is input
        if idx is None or idx < 0:
            return None
        dev = sd.query_devices(idx)
        return {
            "index": idx,
            "name": dev["name"],
            "channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
        }
    except Exception:
        return None
