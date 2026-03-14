"""
Rio Local — WhatsApp Channel (Priority 4.2)

Uses WhatsApp Cloud API for outbound notifications.
Draft updates are emulated by sending incremental update messages because
WhatsApp messages cannot be edited after send.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from .channel_manager import BaseChannel, DraftHandle
except Exception:
    from channel_manager import BaseChannel, DraftHandle


class WhatsAppChannel(BaseChannel):
    name = "whatsapp"

    def __init__(
        self,
        access_token: str = "",
        phone_number_id: str = "",
        to_number: str = "",
    ) -> None:
        self._access_token = access_token or os.environ.get("RIO_WHATSAPP_ACCESS_TOKEN", "")
        self._phone_number_id = phone_number_id or os.environ.get("RIO_WHATSAPP_PHONE_NUMBER_ID", "")
        self._to_number = to_number or os.environ.get("RIO_WHATSAPP_TO_NUMBER", "")

    @property
    def enabled(self) -> bool:
        return bool(self._access_token and self._phone_number_id and self._to_number)

    async def send(self, text: str) -> dict[str, Any]:
        if not self.enabled:
            return {"success": False, "error": "whatsapp disabled"}

        url = f"https://graph.facebook.com/v20.0/{self._phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": self._to_number,
            "type": "text",
            "text": {"body": text[:4096]},
        }
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        try:
            import httpx

            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(url, headers=headers, json=payload)
                data = resp.json() if resp.text else {}
                if 200 <= resp.status_code < 300:
                    message_id = ""
                    messages = data.get("messages", []) if isinstance(data, dict) else []
                    if messages and isinstance(messages[0], dict):
                        message_id = str(messages[0].get("id", ""))
                    return {"success": True, "data": data, "message_id": message_id}
                return {"success": False, "error": f"whatsapp api {resp.status_code}: {str(data)[:400]}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def send_typing(self) -> None:
        return

    async def start_draft(self, text: str) -> DraftHandle | None:
        result = await self.send(text)
        if not result.get("success"):
            return None
        return DraftHandle(channel=self.name, message_id=str(result.get("message_id", "")))

    async def update_draft(self, handle: DraftHandle, text: str) -> dict[str, Any]:
        return await self.send(f"Update: {text[:3900]}")

    async def finish_draft(self, handle: DraftHandle, text: str) -> dict[str, Any]:
        return await self.send(f"Done: {text[:3900]}")
