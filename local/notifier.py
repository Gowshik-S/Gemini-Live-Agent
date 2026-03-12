"""
Rio Local — Notifier (F5)

Simple notification system for alerting the user when they're away.
Currently supports Telegram Bot API.

Configure in config.yaml:
    rio:
      notifications:
        telegram:
          bot_token: "YOUR_BOT_TOKEN"
          chat_id: "YOUR_CHAT_ID"

Or via environment variables:
    RIO_TELEGRAM_BOT_TOKEN=...
    RIO_TELEGRAM_CHAT_ID=...
"""

from __future__ import annotations

import os
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_TELEGRAM_API = "https://api.telegram.org"


class Notifier:
    """Send notifications to the user via Telegram."""

    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat_id: str = "",
    ) -> None:
        self._telegram_token = (
            telegram_token or os.environ.get("RIO_TELEGRAM_BOT_TOKEN", "")
        )
        self._telegram_chat_id = (
            telegram_chat_id or os.environ.get("RIO_TELEGRAM_CHAT_ID", "")
        )
        self._enabled = bool(self._telegram_token and self._telegram_chat_id)
        if self._enabled:
            log.info("notifier.telegram.enabled")
        else:
            log.debug("notifier.telegram.disabled", note="Set RIO_TELEGRAM_BOT_TOKEN and RIO_TELEGRAM_CHAT_ID")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def send(self, message: str) -> dict[str, Any]:
        """Send a notification message."""
        if not self._enabled:
            return {"success": False, "error": "Notifications not configured"}
        return await self._send_telegram(message)

    async def _send_telegram(self, message: str) -> dict[str, Any]:
        """Send a message via Telegram Bot API."""
        try:
            import httpx

            url = f"{_TELEGRAM_API}/bot{self._telegram_token}/sendMessage"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json={
                    "chat_id": self._telegram_chat_id,
                    "text": message[:4096],  # Telegram limit
                    "parse_mode": "Markdown",
                })
                if resp.status_code == 200:
                    return {"success": True, "message": "Notification sent"}
                else:
                    return {"success": False, "error": f"Telegram API error: {resp.status_code}"}
        except ImportError:
            # Fallback to urllib if httpx not available
            import urllib.request
            import json

            url = f"{_TELEGRAM_API}/bot{self._telegram_token}/sendMessage"
            data = json.dumps({
                "chat_id": self._telegram_chat_id,
                "text": message[:4096],
            }).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=10)
                return {"success": True, "message": "Notification sent"}
            except Exception as exc:
                return {"success": False, "error": str(exc)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}
