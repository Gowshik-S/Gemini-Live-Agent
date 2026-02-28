"""
Rio Local -- Silero VAD Wrapper

Wraps Silero VAD (torch.hub) for per-chunk speech detection.
Accepts PCM 16-bit LE bytes, returns boolean speech decision.

Silero VAD valid window sizes at 16kHz: 256, 512, 768, 1024, 1280, 1536.
Our 1600-sample chunks (100ms) are trimmed to 1536 (96ms) for inference.
The 64 discarded samples (4ms) are negligible.

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
_SILERO_WINDOW_16K = 1536  # 96ms -- largest valid size fitting in 1600


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
                       Expected 1600 samples (3200 bytes) from AudioCapture.

        Returns:
            VadResult with probability and is_speech boolean.
            If VAD is unavailable, returns is_speech=True (pass-through).
        """
        if not self._available or self._model is None or self._torch is None:
            return VadResult(probability=1.0, is_speech=True)

        # Convert PCM bytes -> int16 numpy array
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Trim to valid Silero window size (1536 samples from 1600)
        if len(audio_int16) > _SILERO_WINDOW_16K:
            audio_int16 = audio_int16[:_SILERO_WINDOW_16K]

        # Normalize to float32 [-1.0, 1.0]
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Convert to torch tensor
        tensor = self._torch.from_numpy(audio_float)

        # Run inference (single forward pass, <5ms)
        prob = self._model(tensor, self._sample_rate).item()

        return VadResult(
            probability=prob,
            is_speech=prob >= self._threshold,
        )
