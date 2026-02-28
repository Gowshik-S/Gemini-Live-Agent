"""
Model router for Rio cloud service.

L0 placeholder -- all requests go through the Gemini Live API (Flash).
Pro escalation will be added in L4 (Day 13).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

    For L0 everything is handled by Flash via the Live API session.
    The Pro escalation path (``call_pro``, ``inject_pro_result``) is
    stubbed out and will be wired up in L4.
    """

    def __init__(self) -> None:
        self._history: List[RoutingRecord] = []
        self._log = logger.bind(component="model_router")

    # ------------------------------------------------------------------
    # Live / Flash (active in L0)
    # ------------------------------------------------------------------

    def record_flash_call(self, reason: str = "live_relay") -> None:
        """Record that a request was handled by Flash via the Live session."""
        record = RoutingRecord(
            timestamp=time.time(),
            model="gemini-2.0-flash",
            route_reason=reason,
        )
        self._history.append(record)
        self._log.debug("router.flash", reason=reason)

    # ------------------------------------------------------------------
    # Pro escalation stubs (L4)
    # ------------------------------------------------------------------

    async def call_pro(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> None:
        """Placeholder -- will call Gemini Pro for deep analysis in L4.

        Returns None until implemented.
        """
        self._log.debug(
            "router.call_pro.stub",
            prompt_len=len(prompt),
            note="Pro escalation not implemented until L4",
        )
        return None

    async def inject_pro_result(self, analysis: str) -> None:
        """Placeholder -- inject a Pro analysis result into the Live session.

        Will be used in L4 to enrich Flash context with Pro insights.
        """
        self._log.debug(
            "router.inject_pro_result.stub",
            analysis_len=len(analysis),
            note="Pro injection not implemented until L4",
        )

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
        }
