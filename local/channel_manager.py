"""
Rio Local — Channel Manager (Priority 4.2)

Minimal channel plugin layer focused on Telegram + WhatsApp.
Provides a single API for sending notifications and draft updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DraftHandle:
    channel: str
    message_id: str


class BaseChannel:
    name: str = "base"

    @property
    def enabled(self) -> bool:
        return False

    async def send(self, text: str) -> dict[str, Any]:
        return {"success": False, "error": "channel disabled"}

    async def send_typing(self) -> None:
        return

    async def start_draft(self, text: str) -> DraftHandle | None:
        result = await self.send(text)
        if not result.get("success"):
            return None
        msg_id = str(result.get("message_id", ""))
        if not msg_id:
            return None
        return DraftHandle(channel=self.name, message_id=msg_id)

    async def update_draft(self, handle: DraftHandle, text: str) -> dict[str, Any]:
        return await self.send(text)

    async def finish_draft(self, handle: DraftHandle, text: str) -> dict[str, Any]:
        return await self.update_draft(handle, text)


class TelegramChannel(BaseChannel):
    name = "telegram"

    def __init__(self, bot: Any | None) -> None:
        self._bot = bot

    @property
    def enabled(self) -> bool:
        return bool(self._bot and getattr(self._bot, "enabled", False))

    async def send(self, text: str) -> dict[str, Any]:
        if not self.enabled:
            return {"success": False, "error": "telegram disabled"}
        result = await self._bot.send(text)
        if result.get("success"):
            data = result.get("data", {}) or {}
            if isinstance(data, dict) and data.get("message_id") is not None:
                result["message_id"] = data.get("message_id")
        return result

    async def send_typing(self) -> None:
        if self.enabled:
            await self._bot.send_typing()

    async def update_draft(self, handle: DraftHandle, text: str) -> dict[str, Any]:
        if not self.enabled:
            return {"success": False, "error": "telegram disabled"}
        try:
            message_id = int(handle.message_id)
        except Exception:
            return await self.send(text)
        return await self._bot.edit_message(message_id, text)


class ChannelManager:
    """Telegram + WhatsApp focused manager (no extra channels by design)."""

    def __init__(self, channels: list[BaseChannel] | None = None) -> None:
        self._channels: dict[str, BaseChannel] = {}
        for channel in channels or []:
            self.register(channel)

    def register(self, channel: BaseChannel) -> None:
        self._channels[channel.name] = channel

    def get(self, name: str) -> BaseChannel | None:
        return self._channels.get(name)

    def enabled_channels(self) -> list[str]:
        return [name for name, channel in self._channels.items() if channel.enabled]

    async def send(self, text: str, channel: str | None = None) -> dict[str, Any]:
        if channel:
            ch = self._channels.get(channel)
            if ch and ch.enabled:
                return await ch.send(text)
            return {"success": False, "error": f"channel '{channel}' unavailable"}
        for ch in self._channels.values():
            if ch.enabled:
                return await ch.send(text)
        return {"success": False, "error": "no enabled channels"}

    async def send_all(self, text: str) -> dict[str, Any]:
        sent = 0
        errors: list[str] = []
        for name, ch in self._channels.items():
            if not ch.enabled:
                continue
            result = await ch.send(text)
            if result.get("success"):
                sent += 1
            else:
                errors.append(f"{name}: {result.get('error', 'unknown error')}")
        return {"success": sent > 0, "sent": sent, "errors": errors}

    async def start_draft(self, text: str, channel: str) -> DraftHandle | None:
        ch = self._channels.get(channel)
        if ch is None or not ch.enabled:
            return None
        return await ch.start_draft(text)

    async def update_draft(self, handle: DraftHandle, text: str) -> dict[str, Any]:
        ch = self._channels.get(handle.channel)
        if ch is None or not ch.enabled:
            return {"success": False, "error": "draft channel unavailable"}
        return await ch.update_draft(handle, text)

    async def finish_draft(self, handle: DraftHandle, text: str) -> dict[str, Any]:
        ch = self._channels.get(handle.channel)
        if ch is None or not ch.enabled:
            return {"success": False, "error": "draft channel unavailable"}
        return await ch.finish_draft(handle, text)
