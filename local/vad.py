"""
Rio Local -- Silero VAD Wrapper

Wraps Silero VAD (torch.hub) for per-chunk speech detection.
Accepts PCM 16-bit LE bytes, returns boolean speech decision.

Silero VAD valid window sizes at 16kHz: 256, 512, 768, 1024, 1280, 1536.
With 20ms chunks (320 samples), we use exactly 256 samples for inference
and accumulate the rest. For larger chunks we use the nearest valid window.

If torch is not installed, the module degrades gracefully: process()
returns is_speech=True (pass-through), enabling PTT-only mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Silero expects one of these window sizes at 16kHz
_SILERO_VALID_WINDOWS = [256, 512, 768, 1024, 1280, 1536]
_SILERO_WINDOW_16K = 512  # Default: 32ms -- good balance of speed & accuracy


def _best_window_size(num_samples: int) -> int:
    """Find the largest valid Silero window that fits in num_samples."""
    for ws in reversed(_SILERO_VALID_WINDOWS):
        if ws <= num_samples:
            return ws
    return _SILERO_VALID_WINDOWS[0]  # 256 minimum


@dataclass
class VadResult:
    """Result from a single VAD inference."""
    probability: float
    is_speech: bool


class SileroVAD:
    """Silero VAD wrapper for real-time speech detection.

    Usage::

        vad = SileroVAD(threshold=0.5)
        if vad.available:
            vad.reset()  # call at start of each PTT session
            result = vad.process(pcm_bytes)
            if result.is_speech:
                send(pcm_bytes)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16_000,
    ) -> None:
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._model = None
        self._torch = None  # cached torch module reference
        self._available = False
        self._warned_runtime_failure = False
        self._load_model()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if the VAD model loaded successfully."""
        return self._available

    @property
    def threshold(self) -> float:
        return self._threshold

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load Silero VAD from torch.hub. Graceful failure if unavailable."""
        try:
            import torch
            self._torch = torch

            log.info("vad.loading", source="torch.hub")
            model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                force_reload=False,
            )
            self._model = model
            self._available = True
            log.info("vad.loaded", threshold=self._threshold)

        except ImportError:
            log.warning(
                "vad.torch_not_installed",
                note="Install torch for VAD. Falling back to no-VAD mode.",
            )
        except Exception:
            log.exception(
                "vad.load_failed",
                note="VAD unavailable. Falling back to no-VAD mode.",
            )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset model hidden states. Call at start of each PTT session."""
        if self._model is not None:
            self._model.reset_states()

    def process(self, pcm_bytes: bytes) -> VadResult:
        """Run VAD on a PCM 16-bit LE chunk.

        Args:
            pcm_bytes: Raw PCM 16-bit LE mono audio.
                       Supports variable chunk sizes (320 samples/20ms,
                       640 samples/40ms, 1600 samples/100ms, etc.).

        Returns:
            VadResult with probability and is_speech boolean.
            If VAD is unavailable, returns is_speech=True (pass-through).
        """
        if not self._available or self._model is None or self._torch is None:
            return VadResult(probability=1.0, is_speech=True)

        # Convert PCM bytes -> int16 numpy array
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Find the best valid Silero window size for this chunk
        window_size = _best_window_size(len(audio_int16))

        # Trim to valid Silero window size
        if len(audio_int16) > window_size:
            audio_int16 = audio_int16[:window_size]
        elif len(audio_int16) < _SILERO_VALID_WINDOWS[0]:
            # Chunk too small for any valid window — pass through
            return VadResult(probability=0.5, is_speech=True)

        # Normalize to float32 [-1.0, 1.0]
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Convert to torch tensor
        tensor = self._torch.from_numpy(audio_float)

        # Run inference (single forward pass, <5ms)
        try:
            prob = self._model(tensor, self._sample_rate).item()
        except Exception as exc:
            # Runtime TorchScript errors can occur on malformed/tiny chunks
            # from specific devices. Do not crash voice flow; pass through.
            if not self._warned_runtime_failure:
                self._warned_runtime_failure = True
                log.warning(
                    "vad.runtime_error",
                    error=str(exc),
                    note="Falling back to pass-through for this chunk",
                )
            return VadResult(probability=0.5, is_speech=True)

        return VadResult(
            probability=prob,
            is_speech=prob >= self._threshold,
        )

    async def process_async(self, pcm_bytes: bytes) -> VadResult:
        """Async version of process() — runs inference in executor.

        Use this from the event loop to avoid blocking audio I/O.
        Falls back to synchronous process() if unavailable.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.process, pcm_bytes)
