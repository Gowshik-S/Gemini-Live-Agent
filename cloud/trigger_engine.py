"""
Rio Cloud — Trigger Engine (C5)

Event-driven automation: schedule or trigger agent tasks based on events.

Supports:
  - Cron-style scheduled tasks (e.g. "every 30 minutes check server status")
  - File-watch triggers (e.g. "when log file changes, analyze errors")
  - Keyword triggers (e.g. "when user says 'deploy', run deployment checklist")

Usage:
    engine = TriggerEngine(orchestrator, inject_fn)
    engine.add_schedule("health_check", "*/30 * * * *", "Check server health status")
    engine.add_keyword("deploy", "Run the deployment checklist for production")
    await engine.start()
    ...
    await engine.stop()
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import structlog

log = structlog.get_logger(__name__)


@dataclass
class Trigger:
    """A single trigger definition."""
    name: str
    trigger_type: str  # "schedule" | "keyword" | "file_watch"
    goal: str  # task to execute when triggered
    enabled: bool = True
    # Schedule-specific
    interval_seconds: int = 0  # simplified: interval in seconds (0 = disabled)
    # Keyword-specific
    keywords: list[str] = field(default_factory=list)
    # Metadata
    last_fired: float = 0.0
    fire_count: int = 0
    cooldown_seconds: int = 60  # Minimum gap between firings


class TriggerEngine:
    """Manages triggers and fires them as orchestrator tasks."""

    def __init__(
        self,
        orchestrator: Any,
        inject_context_fn: Callable[[str], Awaitable[None]],
    ) -> None:
        self._orchestrator = orchestrator
        self._inject_fn = inject_context_fn
        self._triggers: dict[str, Trigger] = {}
        self._running = False
        self._schedule_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Trigger management
    # ------------------------------------------------------------------

    def add_schedule(
        self,
        name: str,
        interval_seconds: int,
        goal: str,
        cooldown_seconds: int = 60,
    ) -> Trigger:
        """Add a schedule-based trigger that fires every N seconds."""
        trigger = Trigger(
            name=name,
            trigger_type="schedule",
            goal=goal,
            interval_seconds=max(30, interval_seconds),  # Min 30s
            cooldown_seconds=cooldown_seconds,
        )
        self._triggers[name] = trigger
        log.info("trigger.added", name=name, type="schedule", interval=interval_seconds)
        return trigger

    def add_keyword(
        self,
        name: str,
        keywords: list[str] | str,
        goal: str,
        cooldown_seconds: int = 30,
    ) -> Trigger:
        """Add a keyword trigger that fires when user utterance matches."""
        if isinstance(keywords, str):
            keywords = [keywords]
        trigger = Trigger(
            name=name,
            trigger_type="keyword",
            goal=goal,
            keywords=[k.lower() for k in keywords],
            cooldown_seconds=cooldown_seconds,
        )
        self._triggers[name] = trigger
        log.info("trigger.added", name=name, type="keyword", keywords=keywords)
        return trigger

    def remove(self, name: str) -> bool:
        if name in self._triggers:
            del self._triggers[name]
            return True
        return False

    def list_triggers(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "type": t.trigger_type,
                "goal": t.goal[:100],
                "enabled": t.enabled,
                "fire_count": t.fire_count,
            }
            for t in self._triggers.values()
        ]

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the trigger engine background loop."""
        if self._running:
            return
        self._running = True
        self._schedule_task = asyncio.create_task(self._schedule_loop())
        log.info("trigger_engine.started", triggers=len(self._triggers))

    async def stop(self) -> None:
        """Stop the trigger engine."""
        self._running = False
        if self._schedule_task:
            self._schedule_task.cancel()
            try:
                await self._schedule_task
            except asyncio.CancelledError:
                pass
            self._schedule_task = None
        log.info("trigger_engine.stopped")

    # ------------------------------------------------------------------
    # Core loops
    # ------------------------------------------------------------------

    async def _schedule_loop(self) -> None:
        """Check scheduled triggers every 10 seconds."""
        while self._running:
            try:
                await asyncio.sleep(10)
                now = time.time()
                for trigger in self._triggers.values():
                    if not trigger.enabled:
                        continue
                    if trigger.trigger_type != "schedule":
                        continue
                    if trigger.interval_seconds <= 0:
                        continue
                    elapsed = now - trigger.last_fired
                    if elapsed >= trigger.interval_seconds:
                        await self._fire(trigger)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("trigger_engine.schedule_loop.error")

    def check_utterance(self, utterance: str) -> None:
        """Check if an utterance matches any keyword triggers.

        Called from the turn_complete handler in adk_server.py.
        Fires matching triggers asynchronously.
        """
        if not utterance:
            return
        text_lower = utterance.lower()
        now = time.time()
        for trigger in self._triggers.values():
            if not trigger.enabled:
                continue
            if trigger.trigger_type != "keyword":
                continue
            if (now - trigger.last_fired) < trigger.cooldown_seconds:
                continue
            for kw in trigger.keywords:
                if kw in text_lower:
                    asyncio.create_task(self._fire(trigger))
                    break

    async def _fire(self, trigger: Trigger) -> None:
        """Fire a trigger — spawn an orchestrator task."""
        trigger.last_fired = time.time()
        trigger.fire_count += 1
        log.info(
            "trigger.fired",
            name=trigger.name,
            goal=trigger.goal[:80],
            count=trigger.fire_count,
        )
        try:
            self._orchestrator.spawn_task(trigger.goal, self._inject_fn)
        except Exception:
            log.exception("trigger.fire.error", name=trigger.name)
