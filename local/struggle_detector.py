"""
Rio Local — Struggle Detector (Day 9-10 / L4)

Detects developer struggle from screen analysis and timers, then
provides context for Gemini to offer proactive help.

4 screen-based signals (no extra pynput keystroke monitoring):
  1. Repeated error  — same screen hash 3+ times in 2 min
  2. Long pause on error — screen unchanged >30s after error keywords
  3. Rapid small screen changes — 5+ distinct hashes in 60s
  4. Stale screen with activity — unchanged >45s but user is active

No external dependencies beyond stdlib + structlog.

Usage::

    detector = StruggleDetector(config.struggle)
    detector.feed_frame(jpeg_bytes)        # from screen capture loop
    detector.feed_gemini_response(text)    # from receive loop
    detector.note_user_activity()          # from input / audio loops

    result = detector.evaluate()
    if result.should_trigger:
        send_struggle_context(result)
        detector.record_trigger()
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Error keyword patterns — used to detect error-related Gemini responses
# ---------------------------------------------------------------------------

ERROR_KEYWORDS = frozenset({
    "error", "exception", "traceback", "failed", "failure",
    "typeerror", "syntaxerror", "nameerror", "valueerror",
    "keyerror", "indexerror", "attributeerror", "importerror",
    "runtimeerror", "oserror", "ioerror", "zerodivisionerror",
    "filenotfounderror", "permissionerror", "connectionerror",
    "timeout", "segfault", "segmentation fault", "core dumped",
    "panic", "fatal", "undefined", "null pointer", "nullptr",
    "stack overflow", "out of memory", "cannot find module",
    "module not found", "no such file", "command not found",
    "compilation error", "build failed", "lint error",
    "assertion", "assertionerror", "deprecationwarning",
})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StruggleResult:
    """Result of a struggle evaluation pass."""

    confidence: float = 0.0
    active_signals: List[str] = field(default_factory=list)
    should_trigger: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "confidence": round(self.confidence, 3),
            "active_signals": self.active_signals,
            "should_trigger": self.should_trigger,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Frame record — stored in the rolling deque
# ---------------------------------------------------------------------------

@dataclass
class _FrameRecord:
    hash: str           # MD5 of raw JPEG bytes
    timestamp: float
    text_hash: Optional[str] = None   # MD5 of OCR-extracted text (if available)


# ---------------------------------------------------------------------------
# StruggleDetector
# ---------------------------------------------------------------------------

class StruggleDetector:
    """Screen-based developer struggle detection engine.

    Uses 4 signals derived from screen frame hashes, timing, and
    Gemini response analysis.  No heavy dependencies — just stdlib.

    Designed for ~0% CPU: called every ~2s from the struggle loop,
    only hashes and compares short deques.

    Constructor accepts a ``StruggleConfig`` dataclass (from config.py)
    with fields: enabled, threshold, cooldown_seconds, decline_cooldown,
    demo_mode.
    """

    # Signal weights (sum to 1.0)
    W_REPEATED_ERROR = 0.35
    W_LONG_PAUSE = 0.25
    W_RAPID_CHANGES = 0.20
    W_STALE_ACTIVE = 0.20

    # Signal thresholds
    REPEATED_ERROR_COUNT = 3       # same hash N+ times in window
    REPEATED_ERROR_WINDOW = 120.0  # seconds (2 minutes)
    LONG_PAUSE_SECONDS = 30.0      # screen unchanged after error
    RAPID_CHANGE_COUNT = 5         # distinct hashes in window
    RAPID_CHANGE_WINDOW = 60.0     # seconds (1 minute)
    STALE_ACTIVE_SECONDS = 45.0    # screen unchanged but user active

    # Demo mode overrides
    DEMO_THRESHOLD = 0.4
    DEMO_COOLDOWN = 30              # seconds
    DEMO_DECLINE_COOLDOWN = 60      # seconds
    DEMO_MIN_SIGNALS = 1            # only 1 signal needed in demo mode

    def __init__(self, config) -> None:
        """Initialize the struggle detector.

        Args:
            config: A StruggleConfig dataclass with fields:
                    enabled, threshold, cooldown_seconds,
                    decline_cooldown, demo_mode
        """
        self._enabled = config.enabled
        self._demo_mode = config.demo_mode

        # Thresholds — adjusted for demo mode
        if self._demo_mode:
            self._threshold = self.DEMO_THRESHOLD
            self._cooldown = self.DEMO_COOLDOWN
            self._decline_cooldown = self.DEMO_DECLINE_COOLDOWN
            self._min_signals = self.DEMO_MIN_SIGNALS
        else:
            self._threshold = config.threshold
            self._cooldown = config.cooldown_seconds
            self._decline_cooldown = config.decline_cooldown
            self._min_signals = 2  # require 2+ signals in normal mode

        # Rolling frame history — (hash, timestamp) tuples
        # Keep up to 2 minutes of frames at ~0.5 fps = ~60 entries max
        self._frame_history: deque[_FrameRecord] = deque(maxlen=120)

        # Current screen hash (latest frame)
        self._current_hash: Optional[str] = None
        self._last_change_time: float = time.monotonic()

        # Error state — set when Gemini response contains error keywords
        self._error_detected: bool = False
        self._error_detected_time: float = 0.0

        # User activity tracking
        self._last_activity_time: float = time.monotonic()

        # Cooldown timers
        self._last_trigger_time: float = 0.0
        self._last_decline_time: float = 0.0

        # Stats
        self._total_evaluations: int = 0
        self._total_triggers: int = 0

        log.info(
            "struggle.init",
            enabled=self._enabled,
            demo_mode=self._demo_mode,
            threshold=self._threshold,
            cooldown=self._cooldown,
            min_signals=self._min_signals,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Always True — no heavy deps needed."""
        return True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def demo_mode(self) -> bool:
        return self._demo_mode

    @property
    def stats(self) -> dict:
        """Diagnostic stats for dashboard / logging."""
        return {
            "total_evaluations": self._total_evaluations,
            "total_triggers": self._total_triggers,
            "frame_history_size": len(self._frame_history),
            "error_detected": self._error_detected,
            "current_hash": self._current_hash[:8] if self._current_hash else None,
        }

    # ------------------------------------------------------------------
    # Feed methods — called from main.py loops
    # ------------------------------------------------------------------

    def feed_frame(
        self,
        jpeg_bytes: Optional[bytes],
        ocr_text: Optional[str] = None,
    ) -> None:
        """Process a screen capture frame.

        Called by the struggle detection loop with JPEG bytes from
        ScreenCapture.  Stores the MD5 hash + timestamp for signal
        analysis.

        When ``ocr_text`` is provided (from the OCR engine), its hash
        is stored alongside the JPEG hash.  Signal 1 prefers the text
        hash because it is immune to pixel-level noise (cursor blink,
        clock tick).

        Args:
            jpeg_bytes: Raw JPEG bytes, or None if capture failed /
                        returned delta-skip.
            ocr_text: OCR-extracted text from the frame, or None.
        """
        if jpeg_bytes is None:
            return

        now = time.monotonic()
        frame_hash = hashlib.md5(jpeg_bytes).hexdigest()

        # Compute text hash if OCR text available
        text_hash: Optional[str] = None
        if ocr_text and ocr_text.strip():
            # Normalise whitespace so trivial spacing changes don't
            # produce different hashes
            normalised = " ".join(ocr_text.split())
            text_hash = hashlib.md5(normalised.encode("utf-8", errors="replace")).hexdigest()

        # Track screen changes
        if frame_hash != self._current_hash:
            self._last_change_time = now
            self._current_hash = frame_hash

        # Add to rolling history
        self._frame_history.append(
            _FrameRecord(hash=frame_hash, timestamp=now, text_hash=text_hash)
        )

        # Prune old entries beyond the max window (2 minutes)
        cutoff = now - self.REPEATED_ERROR_WINDOW
        while self._frame_history and self._frame_history[0].timestamp < cutoff:
            self._frame_history.popleft()

    def feed_gemini_response(self, text: str) -> None:
        """Analyze a Gemini text response for error-related content.

        Called by receive_loop when Gemini sends a transcript.
        If error keywords are found, primes Signal 2 (long pause on error).

        Args:
            text: The Gemini response text.
        """
        if not text:
            return

        text_lower = text.lower()
        for keyword in ERROR_KEYWORDS:
            if keyword in text_lower:
                if not self._error_detected:
                    log.debug(
                        "struggle.error_detected_in_response",
                        keyword=keyword,
                    )
                self._error_detected = True
                self._error_detected_time = time.monotonic()
                return

        # Don't immediately clear error state on a single non-error response.
        # The error may still be relevant (e.g., Gemini giving fix instructions).
        # Instead, use a time-based decay: error state clears after 2 minutes
        # of no new error keywords in Gemini responses.
        # (Clearing is handled in _signal_long_pause_on_error via timestamp check)

    def note_user_activity(self) -> None:
        """Record that the user did something (typed, spoke, etc).

        Called from input_loop (text) and audio_capture_loop (speech).
        Used by Signal 4 to detect "stale screen but active user".
        """
        self._last_activity_time = time.monotonic()

    # ------------------------------------------------------------------
    # Evaluation — core logic
    # ------------------------------------------------------------------

    def evaluate(self) -> StruggleResult:
        """Evaluate all 4 signals and compute struggle confidence.

        Returns a StruggleResult with confidence, active signals,
        and whether a proactive trigger should fire.

        Safe to call frequently (~every 2s). Very cheap: only
        iterates short deques and compares timestamps.
        """
        if not self._enabled:
            return StruggleResult()

        self._total_evaluations += 1
        now = time.monotonic()

        signals: List[tuple[str, float]] = []  # (name, weight)

        # -- Signal 1: Repeated Error (same screen 3+ times in 2 min) --
        s1_active, s1_detail = self._signal_repeated_error(now)
        if s1_active:
            signals.append(("repeated_error", self.W_REPEATED_ERROR))

        # -- Signal 2: Long Pause on Error (screen unchanged >30s after error) --
        s2_active, s2_detail = self._signal_long_pause_on_error(now)
        if s2_active:
            signals.append(("long_pause_error", self.W_LONG_PAUSE))

        # -- Signal 3: Rapid Small Screen Changes (5+ distinct in 60s) --
        s3_active, s3_detail = self._signal_rapid_changes(now)
        if s3_active:
            signals.append(("rapid_changes", self.W_RAPID_CHANGES))

        # -- Signal 4: Stale Screen with Activity (unchanged >45s, user active) --
        s4_active, s4_detail = self._signal_stale_with_activity(now)
        if s4_active:
            signals.append(("stale_active", self.W_STALE_ACTIVE))

        # -- Compute confidence --
        confidence = sum(w for _, w in signals)
        active_names = [name for name, _ in signals]
        num_active = len(signals)

        # -- Should trigger? --
        should_trigger = (
            confidence >= self._threshold
            and num_active >= self._min_signals
            and self._cooldown_expired(now)
        )

        # -- Build reason string --
        reason = ""
        if should_trigger:
            parts = []
            if s1_active:
                parts.append(s1_detail)
            if s2_active:
                parts.append(s2_detail)
            if s3_active:
                parts.append(s3_detail)
            if s4_active:
                parts.append(s4_detail)
            reason = "; ".join(parts)

        result = StruggleResult(
            confidence=confidence,
            active_signals=active_names,
            should_trigger=should_trigger,
            reason=reason,
        )

        if should_trigger:
            log.info(
                "struggle.trigger",
                confidence=round(confidence, 3),
                signals=active_names,
                reason=reason,
            )
        elif num_active > 0:
            log.debug(
                "struggle.signals_active",
                confidence=round(confidence, 3),
                signals=active_names,
                trigger=False,
            )

        return result

    # ------------------------------------------------------------------
    # Cooldown management
    # ------------------------------------------------------------------

    def record_trigger(self) -> None:
        """Record that a proactive trigger was sent.

        Starts the standard cooldown timer.
        """
        self._last_trigger_time = time.monotonic()
        self._total_triggers += 1
        log.info("struggle.trigger_recorded", total=self._total_triggers)

    def record_decline(self) -> None:
        """Record that the user declined proactive help.

        Starts the longer decline cooldown timer.
        """
        self._last_decline_time = time.monotonic()
        log.info("struggle.decline_recorded")

    def _cooldown_expired(self, now: float) -> bool:
        """Check if both cooldown timers have expired."""
        if self._last_trigger_time > 0:
            if now - self._last_trigger_time < self._cooldown:
                return False
        if self._last_decline_time > 0:
            if now - self._last_decline_time < self._decline_cooldown:
                return False
        return True

    # ------------------------------------------------------------------
    # Force trigger (demo mode)
    # ------------------------------------------------------------------

    def force_trigger(self) -> StruggleResult:
        """Force a struggle trigger regardless of signals.

        Used in demo mode with the F4 hotkey for reliable demos.
        Returns a StruggleResult that always has should_trigger=True.
        """
        log.info("struggle.force_trigger", demo_mode=self._demo_mode)
        return StruggleResult(
            confidence=1.0,
            active_signals=["manual_trigger"],
            should_trigger=True,
            reason="Manual trigger via F4 hotkey (demo mode)",
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all signal state. Useful on session reconnect."""
        self._frame_history.clear()
        self._current_hash = None
        self._last_change_time = time.monotonic()
        self._error_detected = False
        self._error_detected_time = 0.0
        self._last_activity_time = time.monotonic()
        # Do NOT reset cooldown timers — those should persist
        log.info("struggle.reset")

    # ------------------------------------------------------------------
    # Private — Individual signal evaluators
    # ------------------------------------------------------------------

    def _signal_repeated_error(self, now: float) -> tuple[bool, str]:
        """Signal 1: Same screen content appears 3+ times in 2-minute window
        while an error has been detected in Gemini responses.

        Requires error_detected=True so the signal fires only when the
        developer keeps seeing the same error screen (compile error, test
        failure, crash).  Without error context, a static screen is just 
        idle — not necessarily struggle.

        When OCR text hashes are available, uses those instead of raw
        JPEG hashes.  Text hashes are immune to pixel-level noise
        (cursor blink, clock tick, notification badge) so the signal
        is much more reliable.
        """
        if not self._frame_history:
            return False, ""

        # Only fire when Gemini has identified an error context
        if not self._error_detected:
            return False, ""

        # Decide which hash to use: prefer OCR text hash, fall back to JPEG
        # Check if any recent frame has a text_hash — if so, use text hashes
        cutoff = now - self.REPEATED_ERROR_WINDOW
        has_text_hashes = any(
            r.text_hash is not None
            for r in self._frame_history
            if r.timestamp >= cutoff
        )

        # Count occurrences of each hash in the window
        hash_counts: dict[str, int] = {}
        for record in self._frame_history:
            if record.timestamp >= cutoff:
                # Use text_hash when available (more robust), else JPEG hash
                h = (record.text_hash if has_text_hashes and record.text_hash else record.hash)
                hash_counts[h] = hash_counts.get(h, 0) + 1

        # Find the most repeated hash
        max_count = 0
        for h, c in hash_counts.items():
            if c > max_count:
                max_count = c

        active = max_count >= self.REPEATED_ERROR_COUNT
        hash_type = "text" if has_text_hashes else "pixel"
        detail = (
            f"same error screen ({hash_type}) seen {max_count}x in {int(self.REPEATED_ERROR_WINDOW)}s"
            if active else ""
        )
        return active, detail

    def _signal_long_pause_on_error(self, now: float) -> tuple[bool, str]:
        """Signal 2: Screen unchanged >30s after Gemini mentioned an error.

        Indicates the developer is staring at an error and not making
        progress — they might be stuck reading / understanding it.

        Error state auto-expires after 2 minutes if no new error keywords
        appear in Gemini responses (time-based decay instead of clearing
        on every non-error response).
        """
        if not self._error_detected:
            return False, ""

        # Auto-expire error state after 2 minutes of no new error keywords
        ERROR_STATE_TTL = 120.0  # seconds
        if now - self._error_detected_time > ERROR_STATE_TTL:
            log.debug("struggle.error_expired", reason="ttl_exceeded")
            self._error_detected = False
            return False, ""

        time_since_change = now - self._last_change_time
        active = time_since_change >= self.LONG_PAUSE_SECONDS

        detail = (
            f"screen unchanged for {int(time_since_change)}s after error detected"
            if active else ""
        )
        return active, detail

    def _signal_rapid_changes(self, now: float) -> tuple[bool, str]:
        """Signal 3: 5+ distinct screen hashes within 60 seconds.

        Indicates rapid trial-and-error editing — the developer is
        making frequent small changes trying to fix something.
        """
        if not self._frame_history:
            return False, ""

        cutoff = now - self.RAPID_CHANGE_WINDOW
        recent_hashes: set[str] = set()
        for record in self._frame_history:
            if record.timestamp >= cutoff:
                recent_hashes.add(record.hash)

        distinct_count = len(recent_hashes)
        active = distinct_count >= self.RAPID_CHANGE_COUNT

        detail = (
            f"{distinct_count} distinct screens in {int(self.RAPID_CHANGE_WINDOW)}s"
            if active else ""
        )
        return active, detail

    def _signal_stale_with_activity(self, now: float) -> tuple[bool, str]:
        """Signal 4: Screen unchanged >45s but user was recently active.

        Indicates the developer is thinking/searching elsewhere without
        making visible progress on screen — possibly stuck.
        """
        time_since_change = now - self._last_change_time
        time_since_activity = now - self._last_activity_time

        # Screen must be stale AND user must have been active recently (within 30s)
        screen_stale = time_since_change >= self.STALE_ACTIVE_SECONDS
        user_active = time_since_activity < 30.0

        active = screen_stale and user_active

        detail = (
            f"screen unchanged {int(time_since_change)}s but user active "
            f"{int(time_since_activity)}s ago"
            if active else ""
        )
        return active, detail
