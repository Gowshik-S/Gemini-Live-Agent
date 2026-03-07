"""
Rio Local — Wake Word Detector

Detects wake words ("Rio", "Hey Rio", "Hey Rio") from audio input
using a lightweight energy-based voice activity + keyword spotting approach.

When no heavy ML dependency is available, falls back to simple energy-based
detection combined with a short audio buffer that gets sent to Gemini for
actual keyword recognition.

Architecture:
  1. Energy gate: only process chunks above a volume threshold
  2. Buffer short audio segments (~2s rolling window)
  3. When energy spike detected, mark as "potential wake word"
  4. Send buffered audio to Gemini Live session for processing
  5. Gemini will recognize "Rio" / "Hey Rio" and respond

This enables Alexa-style "always listening for wake word" behavior
without requiring a local speech recognition model.

Alternatively, if the `vosk` or `whisper` package is available, 
we use offline keyword spotting for true local wake word detection.
"""

from __future__ import annotations

import asyncio
import collections
import time
from enum import Enum, auto
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Wake word phrases to detect (lowercase)
WAKE_PHRASES = {"rio", "hey rio", "hi rio", "ok rio", "yo rio"}

# Phrases that deactivate listening (lowercase substrings)
EXIT_PHRASES = {
    "exit live mode", "stop listening", "goodbye rio", "bye rio",
    "go to sleep", "goodnight rio", "exit listening",
    "stop live mode", "deactivate",
}

# Try to import lightweight speech recognition for local wake word
_vosk_available = False
_vosk_model = None
_vosk_recognizer = None

try:
    import vosk
    import json as _json
    _vosk_available = True
    log.info("wake_word.vosk_available")
except ImportError:
    log.debug("wake_word.vosk_not_installed", note="Using energy-based wake word detection")


class WakeWordState(Enum):
    """State machine for wake word detection."""
    SLEEPING = auto()       # Waiting for wake word
    LISTENING = auto()      # Wake word detected, actively listening
    COOLDOWN = auto()       # Brief cooldown after session ends


class WakeWordDetector:
    """Detects 'Rio' / 'Hey Rio' wake word from PCM audio.
    
    Two modes:
      - vosk mode: Uses Vosk offline speech recognition for keyword spotting
      - energy mode: Detects speech energy spikes and passes audio through
        to Gemini for server-side wake word recognition
    
    Usage::
    
        detector = WakeWordDetector(sample_rate=16000)
        detector.start()
        
        # In audio loop:
        result = detector.process(pcm_chunk)
        if result.activated:
            # Start streaming to Gemini
            ...
        if result.deactivated:
            # Stop streaming
            ...
    """
    
    # Configuration
    ENERGY_THRESHOLD = 800          # RMS energy threshold for voice activity
    ACTIVATION_WINDOW = 1.5         # seconds of audio to buffer for wake word
    SILENCE_TIMEOUT = 30.0          # seconds of silence before deactivating (fallback)
    COOLDOWN_DURATION = 1.0         # seconds before listening for wake word again
    WAKE_ENERGY_MULTIPLIER = 3.0    # Energy must be 3x ambient to trigger
    
    def __init__(
        self,
        sample_rate: int = 16_000,
        enabled: bool = True,
    ) -> None:
        self._sample_rate = sample_rate
        self._enabled = enabled
        self._state = WakeWordState.SLEEPING
        self._last_speech_time: float = 0.0
        self._activation_time: float = 0.0
        self._cooldown_until: float = 0.0
        
        # Ambient noise tracking (rolling average)
        self._ambient_energy: float = 200.0
        self._energy_alpha = 0.02  # Slow adaptation to ambient noise
        
        # Audio buffer for wake word detection window
        self._audio_buffer: collections.deque = collections.deque(
            maxlen=int(self.ACTIVATION_WINDOW * sample_rate / 1600) 
        )
        
        # Vosk recognizer for offline keyword spotting
        self._vosk_rec = None
        self._use_vosk = False
        
        if _vosk_available and enabled:
            try:
                self._init_vosk()
            except Exception:
                log.debug("wake_word.vosk_init_failed", note="Falling back to energy mode")
        
        self._available = enabled
        log.info(
            "wake_word.init",
            enabled=enabled,
            mode="vosk" if self._use_vosk else "energy",
            energy_threshold=self.ENERGY_THRESHOLD,
            silence_timeout=self.SILENCE_TIMEOUT,
        )
    
    def _init_vosk(self) -> None:
        """Initialize Vosk for offline keyword spotting."""
        import vosk
        import os
        
        # Look for a small Vosk model (user must download one)
        model_paths = [
            os.path.expanduser("~/.rio/vosk-model-small-en-us"),
            os.path.join(os.path.dirname(__file__), "vosk-model"),
            "/usr/share/vosk/model",
        ]
        
        model_path = None
        for p in model_paths:
            if os.path.isdir(p):
                model_path = p
                break
        
        if model_path is None:
            log.info("wake_word.vosk_no_model", 
                     note="Download a model: vosk.org/models. Place in ~/.rio/vosk-model-small-en-us")
            return
        
        vosk.SetLogLevel(-1)  # Suppress vosk logging
        model = vosk.Model(model_path)
        self._vosk_rec = vosk.KaldiRecognizer(model, self._sample_rate)
        self._vosk_rec.SetWords(True)
        self._use_vosk = True
        log.info("wake_word.vosk_ready", model=model_path)
    
    @property
    def available(self) -> bool:
        return self._available
    
    @property
    def state(self) -> WakeWordState:
        return self._state
    
    @property
    def is_listening(self) -> bool:
        """True if wake word was detected and we're in active listening mode."""
        return self._state == WakeWordState.LISTENING
    
    @property
    def is_sleeping(self) -> bool:
        return self._state == WakeWordState.SLEEPING
    
    def process(self, pcm_bytes: bytes) -> "WakeWordResult":
        """Process a PCM audio chunk and return wake word detection result.
        
        Args:
            pcm_bytes: Raw PCM 16-bit LE mono audio chunk
            
        Returns:
            WakeWordResult with state transition information
        """
        if not self._enabled:
            # When disabled, always pass audio through (like always-on mode)
            return WakeWordResult(
                state=WakeWordState.LISTENING,
                activated=False,
                deactivated=False,
                should_send_audio=True,
            )
        
        now = time.monotonic()
        
        # Calculate RMS energy
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        energy = np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 0.0
        
        # Update ambient noise estimate (only when sleeping)
        if self._state == WakeWordState.SLEEPING:
            self._ambient_energy = (
                self._ambient_energy * (1 - self._energy_alpha) +
                energy * self._energy_alpha
            )
        
        # State machine
        if self._state == WakeWordState.COOLDOWN:
            if now >= self._cooldown_until:
                self._state = WakeWordState.SLEEPING
                log.debug("wake_word.cooldown_ended")
            return WakeWordResult(
                state=self._state,
                activated=False,
                deactivated=False,
                should_send_audio=False,
            )
        
        if self._state == WakeWordState.SLEEPING:
            # Check for wake word
            detected = False
            
            if self._use_vosk:
                detected = self._check_vosk(pcm_bytes)
            else:
                # Energy-based detection: speech spike above ambient
                threshold = max(
                    self.ENERGY_THRESHOLD,
                    self._ambient_energy * self.WAKE_ENERGY_MULTIPLIER,
                )
                if energy > threshold:
                    # Buffer this chunk
                    self._audio_buffer.append(pcm_bytes)
                    # After accumulating enough speech frames, assume wake word
                    # (Gemini will do the actual recognition)
                    if len(self._audio_buffer) >= 5:  # ~500ms of sustained speech
                        detected = True
                        self._audio_buffer.clear()
                else:
                    self._audio_buffer.clear()
            
            if detected:
                self._state = WakeWordState.LISTENING
                self._activation_time = now
                self._last_speech_time = now
                log.info("wake_word.activated", energy=round(energy, 1))
                print("\n  [Rio activated — listening continuously. Say 'stop listening' or 'goodbye Rio' to deactivate]")
                print("  You: ", end="", flush=True)
                return WakeWordResult(
                    state=WakeWordState.LISTENING,
                    activated=True,
                    deactivated=False,
                    should_send_audio=True,
                )
            
            return WakeWordResult(
                state=WakeWordState.SLEEPING,
                activated=False,
                deactivated=False,
                should_send_audio=False,
            )
        
        if self._state == WakeWordState.LISTENING:
            # Track speech activity
            if energy > self.ENERGY_THRESHOLD:
                self._last_speech_time = now
            
            # Check for silence timeout → deactivate
            silence_duration = now - self._last_speech_time
            if silence_duration > self.SILENCE_TIMEOUT:
                self._state = WakeWordState.COOLDOWN
                self._cooldown_until = now + self.COOLDOWN_DURATION
                log.info(
                    "wake_word.deactivated",
                    reason="silence_timeout",
                    listened_for=round(now - self._activation_time, 1),
                )
                print("\n  [Rio deactivated — say 'Rio' to wake]")
                print("  You: ", end="", flush=True)
                return WakeWordResult(
                    state=WakeWordState.COOLDOWN,
                    activated=False,
                    deactivated=True,
                    should_send_audio=False,
                )
            
            return WakeWordResult(
                state=WakeWordState.LISTENING,
                activated=False,
                deactivated=False,
                should_send_audio=True,
            )
        
        # Fallback
        return WakeWordResult(
            state=self._state,
            activated=False,
            deactivated=False,
            should_send_audio=False,
        )
    
    def _check_vosk(self, pcm_bytes: bytes) -> bool:
        """Check for wake word using Vosk offline recognizer."""
        if self._vosk_rec is None:
            return False
        
        if self._vosk_rec.AcceptWaveform(pcm_bytes):
            result = _json.loads(self._vosk_rec.Result())
            text = result.get("text", "").lower().strip()
            if text and any(phrase in text for phrase in WAKE_PHRASES):
                log.info("wake_word.vosk_detected", text=text)
                return True
        else:
            partial = _json.loads(self._vosk_rec.PartialResult())
            text = partial.get("partial", "").lower().strip()
            if text and any(phrase in text for phrase in WAKE_PHRASES):
                log.info("wake_word.vosk_partial_detected", text=text)
                self._vosk_rec.Reset()
                return True
        
        return False
    
    def force_activate(self) -> None:
        """Force activation (for text input or manual trigger)."""
        self._state = WakeWordState.LISTENING
        self._activation_time = time.monotonic()
        self._last_speech_time = time.monotonic()
    
    def force_deactivate(self) -> None:
        """Force deactivation."""
        self._state = WakeWordState.COOLDOWN
        self._cooldown_until = time.monotonic() + self.COOLDOWN_DURATION
    
    def keep_alive(self) -> None:
        """Reset the silence timer (call when audio activity occurs — user speech, Gemini playback, etc.)."""
        self._last_speech_time = time.monotonic()

    def check_exit_phrase(self, text: str) -> bool:
        """Check if text contains an exit phrase. If so, deactivate and return True."""
        lower = text.lower().strip()
        if any(phrase in lower for phrase in EXIT_PHRASES):
            log.info("wake_word.exit_phrase_detected", text=lower[:80])
            self.force_deactivate()
            return True
        return False


class WakeWordResult:
    """Result from wake word processing."""
    __slots__ = ("state", "activated", "deactivated", "should_send_audio")
    
    def __init__(
        self,
        state: WakeWordState,
        activated: bool,
        deactivated: bool,
        should_send_audio: bool,
    ) -> None:
        self.state = state
        self.activated = activated
        self.deactivated = deactivated
        self.should_send_audio = should_send_audio
