"""
Model router for Rio cloud service.

Routes requests between Gemini Flash (Live API) and Gemini Pro.
Pro escalation is stubbed for L0-L3; wired up in L4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RoutingRecord:
    """Tracks which model handled a request, for dashboard telemetry."""
    timestamp: float
    model: str
    route_reason: str
    latency_ms: Optional[float] = None


class ModelRouter:
    """Routes requests to the appropriate Gemini model.

    Flash (via Live API) handles all real-time interactions.
    Pro escalation is available for deep analysis tasks (L4+).
    """

    def __init__(
        self,
        api_key: str = "",
        rate_limiter=None,
        pro_rpm_budget: int = 5,
    ) -> None:
        self._api_key = api_key
        self._rate_limiter = rate_limiter
        self._pro_rpm_budget = pro_rpm_budget
        self._pro_rpm_used = 0
        self._pro_rpm_window_start = time.time()
        self._history: List[RoutingRecord] = []
        self._inject_callback: Optional[Callable[[str], Coroutine]] = None
        self._log = logger.bind(component="model_router")

    # ------------------------------------------------------------------
    # Inject callback (set by ws_rio_live per-connection)
    # ------------------------------------------------------------------

    def set_inject_callback(
        self, callback: Callable[[str], Coroutine]
    ) -> None:
        """Register the async callback used to inject Pro results into the
        active Gemini session.  Called once per WebSocket connection."""
        self._inject_callback = callback

    # ------------------------------------------------------------------
    # Live / Flash (active in L0+)
    # ------------------------------------------------------------------

    def record_flash_call(self, reason: str = "live_relay") -> None:
        """Record that a request was handled by Flash via the Live session."""
        record = RoutingRecord(
            timestamp=time.time(),
            model="gemini-2.5-flash",
            route_reason=reason,
        )
        self._history.append(record)
        self._log.debug("router.flash", reason=reason)

    # ------------------------------------------------------------------
    # Pro escalation (L4)
    # ------------------------------------------------------------------

    def should_use_pro(self, text: str) -> bool:
        """Decide whether this request warrants Pro escalation.

        Heuristics (L4):
          - Text contains deep-analysis keywords
          - Pro RPM budget not exhausted

        Returns False in L0-L3 (stub).
        """
        # Reset RPM window every 60 seconds
        now = time.time()
        if now - self._pro_rpm_window_start >= 60:
            self._pro_rpm_used = 0
            self._pro_rpm_window_start = now

        if self._pro_rpm_used >= self._pro_rpm_budget:
            return False

        # Keyword heuristic — expand in L4
        pro_keywords = {
            "architecture", "design review", "refactor", "explain in depth",
            "why does", "root cause", "analyze", "analyse", "deep dive",
            # Vision-heavy deep analysis keywords
            "review my code", "find bugs", "security audit",
            "optimize", "detailed analysis", "explain everything",
        }
        text_lower = text.lower()
        return any(kw in text_lower for kw in pro_keywords)

    async def call_pro(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Call Gemini Pro for deep analysis.

        L0-L3: Returns None (stub).
        L4: Will call the Pro model and return the analysis string.
        """
        self._log.debug(
            "router.call_pro.stub",
            prompt_len=len(prompt),
            note="Pro escalation not implemented until L4",
        )
        # Track RPM usage even for stub so budget logic is exercised
        self._pro_rpm_used += 1
        record = RoutingRecord(
            timestamp=time.time(),
            model="gemini-2.5-pro",
            route_reason="pro_escalation_stub",
        )
        self._history.append(record)
        return None

    async def inject_pro_result(self, analysis: str) -> None:
        """Inject a Pro analysis result into the active Live session.

        Calls the per-connection inject callback registered via
        ``set_inject_callback()``.  No-op if no callback is set.
        """
        if self._inject_callback is None:
            self._log.debug("router.inject_pro_result.no_callback")
            return
        self._log.debug(
            "router.inject_pro_result",
            analysis_len=len(analysis),
        )
        try:
            await self._inject_callback(analysis)
        except Exception:
            self._log.exception("router.inject_pro_result.error")

    # ------------------------------------------------------------------
    # Dashboard telemetry
    # ------------------------------------------------------------------

    def get_routing_stats(self) -> Dict[str, object]:
        """Return routing statistics for the dashboard."""
        # Only keep last 100 records in memory
        if len(self._history) > 200:
            self._history = self._history[-100:]

        flash_count = sum(1 for r in self._history if "flash" in r.model)
        pro_count = sum(1 for r in self._history if "pro" in r.model)
        total = len(self._history)

        return {
            "total_requests": total,
            "flash_count": flash_count,
            "pro_count": pro_count,
            "flash_pct": round(flash_count / total * 100, 1) if total else 0,
            "pro_pct": round(pro_count / total * 100, 1) if total else 0,
            "last_model": self._history[-1].model if self._history else None,
            "pro_rpm_current": self._pro_rpm_used,
            "pro_rpm_budget": self._pro_rpm_budget,
        }
