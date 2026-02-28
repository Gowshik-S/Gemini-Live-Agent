"""
Rio Local — Audio I/O Module

Layer 1 (L1): Audio capture and playback using sounddevice.
  - Capture: PCM 16-bit, 16kHz, mono audio in 100ms chunks
  - Playback: PCM 16-bit, 24kHz, mono (matches Gemini output rate)
  - Async-friendly: queue-based interfaces for both directions
  - Device enumeration and selection
"""

from __future__ import annotations

import asyncio
import queue as _queue_mod
from typing import AsyncGenerator, Optional

import numpy as np
import sounddevice as sd
import structlog

log = structlog.get_logger(__name__)

# Wire protocol prefix for audio frames
AUDIO_PREFIX = b"\x01"


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
        block_size: int = 1_600,
        input_device: Optional[str | int] = None,
        max_queue_size: int = 100,
    ) -> None:
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._input_device = input_device
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

        log.info(
            "audio.starting",
            sample_rate=self._sample_rate,
            block_size=self._block_size,
            device=self._input_device,
        )

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            dtype="int16",
            channels=1,
            device=self._input_device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True
        log.info("audio.started")

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
        At 16kHz with block_size=1600, each chunk is 100ms / 3200 bytes.
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

    Usage::

        playback = AudioPlayback(sample_rate=24000)
        playback.start()
        playback.enqueue(pcm_bytes)   # called from async receive loop
        ...
        playback.stop()
    """

    def __init__(
        self,
        sample_rate: int = 24_000,
        channels: int = 1,
        output_device: Optional[str | int] = None,
        max_queue_size: int = 200,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._output_device = output_device
        self._stream: Optional[sd.OutputStream] = None
        self._running = False

        # Thread-safe queue: audio callback (audio thread) reads,
        # enqueue() (event-loop thread) writes.
        # We use queue.Queue (not asyncio.Queue) because the consumer
        # is the sounddevice callback running outside asyncio.
        self._queue: _queue_mod.Queue[bytes] = _queue_mod.Queue(
            maxsize=max_queue_size
        )

        # Block size for the output stream callback.
        # 960 samples at 24kHz = 40ms per callback — good balance
        # between latency and efficiency.
        self._block_size = 960

    @property
    def is_playing(self) -> bool:
        """True if the stream is active and there is audio in the queue."""
        return self._running and not self._queue.empty()

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the output stream and begin playback."""
        if self._running:
            log.warning("playback.already_running")
            return

        log.info(
            "playback.starting",
            sample_rate=self._sample_rate,
            block_size=self._block_size,
            device=self._output_device,
        )

        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            dtype="int16",
            channels=self._channels,
            device=self._output_device,
            callback=self._playback_callback,
        )
        self._stream.start()
        self._running = True
        log.info("playback.started")

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

    def enqueue(self, data: bytes) -> None:
        """Add PCM audio data to the playback queue.

        Thread-safe. Can be called from the asyncio event loop thread.
        If the queue is full, the oldest chunk is dropped to prevent
        unbounded latency growth.
        """
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
        """Flush the playback queue (e.g. on interrupt).

        Thread-safe. Drains all pending audio so the speaker goes silent
        immediately. Day 5 interrupt handling will call this.
        """
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
