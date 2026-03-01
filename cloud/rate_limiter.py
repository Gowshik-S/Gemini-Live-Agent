"""
Token-bucket rate limiter with priority levels and degradation ladder.

Budget: 30 RPM for the Gemini Live API.  Priorities allow graceful
degradation under load -- lower-priority calls are shed first.
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import Dict, List

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Priority levels (lower number = higher priority)
# ---------------------------------------------------------------------------


class Priority(IntEnum):
    SESSION = 0       # Live session keep-alive / core relay
    USER_ASK = 1      # Explicit user question
    PROACTIVE = 2     # Rio-initiated proactive suggestions
    BACKGROUND = 3    # Background analysis, memory writes, etc.


# ---------------------------------------------------------------------------
# Degradation thresholds (RPM)
# ---------------------------------------------------------------------------

class DegradationLevel(IntEnum):
    NORMAL = 0        # 0-20 RPM  -- full functionality
    CAUTION = 1       # 20-25 RPM -- reduce non-essential
    EMERGENCY = 2     # 25-29 RPM -- voice-only
    CRITICAL = 3      # 30 RPM    -- queue non-session calls


_THRESHOLDS: List[tuple[DegradationLevel, int]] = [
    (DegradationLevel.CRITICAL, 30),
    (DegradationLevel.EMERGENCY, 25),
    (DegradationLevel.CAUTION, 20),
    (DegradationLevel.NORMAL, 0),
]

# Maximum priority allowed at each degradation level.
# If a call's priority is *higher* (numerically larger) than the cutoff,
# it is rejected.
_PRIORITY_CUTOFFS: Dict[DegradationLevel, int] = {
    DegradationLevel.NORMAL: Priority.BACKGROUND,     # everything allowed
    DegradationLevel.CAUTION: Priority.PROACTIVE,      # drop background
    DegradationLevel.EMERGENCY: Priority.USER_ASK,     # drop proactive+bg
    DegradationLevel.CRITICAL: Priority.SESSION,        # session only
}


class RateLimiter:
    """Token-bucket rate limiter with priority-aware degradation.

    The bucket refills once per minute.  Each ``record_call`` consumes one
    token.  ``can_call`` checks whether the current degradation level permits
    the given priority.
    """

    def __init__(self, budget_rpm: int = 30) -> None:
        self._budget = budget_rpm
        self._calls: List[float] = []  # timestamps of calls within window
        self._window_s = 60.0
        self._log = logger.bind(component="rate_limiter")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def can_call(self, priority: int) -> bool:
        """Return True if a call at *priority* is allowed right now."""
        self._prune()
        level = self._degradation_level()
        cutoff = _PRIORITY_CUTOFFS[level]
        allowed = priority <= cutoff

        if not allowed:
            self._log.warning(
                "rate_limiter.rejected",
                priority=priority,
                level=level.name,
                rpm=len(self._calls),
            )
        return allowed

    def record_call(self, priority: int) -> None:
        """Record that a call was made."""
        now = time.monotonic()
        self._calls.append(now)
        self._log.debug(
            "rate_limiter.record",
            priority=priority,
            rpm=len(self._calls),
            level=self._degradation_level().name,
        )

    def try_acquire(self, priority: int) -> bool:
        """Atomically check permission and record the call if allowed.

        Combines ``can_call()`` + ``record_call()`` into a single
        operation to prevent race conditions between the check and
        the record.

        Returns True if the call was permitted (and recorded).
        """
        self._prune()
        level = self._degradation_level()
        cutoff = _PRIORITY_CUTOFFS[level]
        allowed = priority <= cutoff

        if allowed:
            now = time.monotonic()
            self._calls.append(now)
            self._log.debug(
                "rate_limiter.acquired",
                priority=priority,
                rpm=len(self._calls),
                level=level.name,
            )
        else:
            self._log.warning(
                "rate_limiter.rejected",
                priority=priority,
                level=level.name,
                rpm=len(self._calls),
            )
        return allowed

    def get_usage(self) -> Dict[str, object]:
        """Return current usage stats for the dashboard."""
        self._prune()
        rpm = len(self._calls)
        level = self._degradation_level()
        return {
            "rpm": rpm,
            "budget": self._budget,
            "utilization_pct": round(rpm / self._budget * 100, 1) if self._budget else 0,
            "degradation_level": level.name,
            "allowed_max_priority": _PRIORITY_CUTOFFS[level],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Remove call timestamps older than the 60-second window."""
        cutoff = time.monotonic() - self._window_s
        # Fast path: if oldest entry is still valid, nothing to prune
        if self._calls and self._calls[0] >= cutoff:
            return
        self._calls = [t for t in self._calls if t >= cutoff]

    def _degradation_level(self) -> DegradationLevel:
        rpm = len(self._calls)
        for level, threshold in _THRESHOLDS:
            if rpm >= threshold:
                return level
        return DegradationLevel.NORMAL
