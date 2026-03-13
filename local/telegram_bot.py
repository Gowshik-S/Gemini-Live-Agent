"""
Rio Local — Bidirectional Telegram Bot

Long-polling Telegram bot that receives messages from the user and routes
them to the orchestrator as tasks executed on the PC workspace. Also sends
outbound notifications, progress updates, and task results.

Reference: OpenClaw's src/telegram/ for the pattern.

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

import asyncio
import os
from typing import Any, Awaitable, Callable

import structlog

log = structlog.get_logger(__name__)

_TELEGRAM_API = "https://api.telegram.org"


class TelegramBot:
    """Bidirectional Telegram bot using long polling.

    Inbound: receives messages, routes to a callback (typically orchestrator.spawn_task).
    Outbound: sends results/notifications and progress updates back to the same chat.
    
    Features:
    - Task routing: messages are executed as PC workspace tasks
    - Progress tracking: heartbeat messages forwarded while tasks run
    - Result delivery: task results sent back to Telegram when done
    - Approval: yes/no replies resolve pending tool approval requests
    - /tasks command: list currently running tasks
    - /cancel command: cancel all running tasks
    """

    def __init__(
        self,
        token: str = "",
        chat_id: str = "",
        on_message: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._token = token or os.environ.get("RIO_TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("RIO_TELEGRAM_CHAT_ID", "")
        self._on_message = on_message
        self._running = False
        self._poll_task: asyncio.Task | None = None
        self._offset = 0

        # Approval gate integration
        self._pending_approval = False
        self._on_approval: Callable[[bool], Awaitable[None]] | None = None

        # Status callback for /tasks command
        self._status_fn: Callable[[], str] | None = None

        self.enabled = bool(self._token and self._chat_id)
        if self.enabled:
            log.info("telegram_bot.enabled", chat_id=self._chat_id[:6] + "...")
        else:
            log.debug("telegram_bot.disabled",
                       note="Set RIO_TELEGRAM_BOT_TOKEN and RIO_TELEGRAM_CHAT_ID")

    def set_message_handler(
        self, handler: Callable[[str], Awaitable[None]],
    ) -> None:
        """Set the async callback for incoming messages (routes to orchestrator)."""
        self._on_message = handler

    def set_status_fn(self, fn: Callable[[], str]) -> None:
        """Set a sync callable that returns current task status string."""
        self._status_fn = fn

    def set_approval_handler(
        self, handler: Callable[[bool], Awaitable[None]],
    ) -> None:
        """Set the async callback for approval responses (yes/no)."""
        self._on_approval = handler

    def notify_approval_pending(self, tool_name: str) -> None:
        """Called by the approval gate to alert user via Telegram."""
        self._pending_approval = True
        asyncio.create_task(
            self.send(
                f"⚠️ *Approval Required*\n\n"
                f"Rio wants to run: `{tool_name}`\n\n"
                f"Reply *yes* to allow or *no* to deny."
            )
        )

    async def send(self, text: str, parse_mode: str = "Markdown") -> dict[str, Any]:
        """Send a message to the configured chat."""
        if not self.enabled:
            return {"success": False, "error": "Telegram not configured"}
        return await self._api_call("sendMessage", {
            "chat_id": self._chat_id,
            "text": text[:4096],
            "parse_mode": parse_mode,
        })

    async def send_typing(self) -> None:
        """Send 'typing' chat action."""
        if not self.enabled:
            return
        await self._api_call("sendChatAction", {
            "chat_id": self._chat_id,
            "action": "typing",
        })

    async def edit_message(self, message_id: int, text: str) -> dict[str, Any]:
        """Edit a previously sent message (for live progress updates)."""
        if not self.enabled:
            return {"success": False}
        return await self._api_call("editMessageText", {
            "chat_id": self._chat_id,
            "message_id": message_id,
            "text": text[:4096],
            "parse_mode": "Markdown",
        })

    def start(self) -> None:
        """Start the long-polling loop as a background task."""
        if not self.enabled or self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="telegram-poll",
        )
        log.info("telegram_bot.polling_started")

    def stop(self) -> None:
        """Stop the long-polling loop."""
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            self._poll_task = None
        log.info("telegram_bot.polling_stopped")

    # ------------------------------------------------------------------
    # Long-polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Continuously poll Telegram for new messages."""
        while self._running:
            try:
                updates = await self._get_updates()
                for update in updates:
                    await self._handle_update(update)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("telegram_bot.poll_error", error=str(exc))
                await asyncio.sleep(5)  # Back off on errors

    async def _get_updates(self) -> list[dict]:
        """Fetch new updates via getUpdates (long polling, 30s timeout)."""
        result = await self._api_call("getUpdates", {
            "offset": self._offset,
            "timeout": 30,
            "allowed_updates": ["message"],
        })
        if result.get("success") and isinstance(result.get("data"), list):
            return result["data"]
        return []

    async def _handle_update(self, update: dict) -> None:
        """Process a single Telegram update."""
        update_id = update.get("update_id", 0)
        self._offset = max(self._offset, update_id + 1)

        message = update.get("message", {})
        text = message.get("text", "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))

        # Only accept messages from the configured chat
        if chat_id != self._chat_id:
            log.debug("telegram_bot.ignored_chat", chat_id=chat_id)
            return

        if not text:
            return

        # --- Approval gate: resolve pending yes/no ---
        if self._pending_approval and self._on_approval:
            lower = text.lower()
            if lower in ("yes", "no", "approve", "deny", "ok", "cancel"):
                approved = lower in ("yes", "approve", "ok")
                self._pending_approval = False
                try:
                    await self._on_approval(approved)
                except Exception as exc:
                    log.warning("telegram_bot.approval_handler_error", error=str(exc))
                reply = "✅ Approved, proceeding..." if approved else "❌ Denied, cancelling."
                await self.send(reply)
                return

        # Handle /commands
        if text.startswith("/"):
            await self._handle_command(text)
            return

        log.info("telegram_bot.message_received", text=text[:80])

        # Route to message handler (orchestrator) — executes on the PC workspace
        if self._on_message:
            try:
                await self.send_typing()
                await self._on_message(text)
            except Exception as exc:
                log.warning("telegram_bot.handler_error", error=str(exc))
                await self.send(f"❌ Error processing: {exc}")

    async def _handle_command(self, text: str) -> None:
        """Handle Telegram bot commands."""
        cmd = text.split()[0].lower().split("@")[0]  # Strip @botname suffix

        if cmd == "/start":
            await self.send(
                "👋 Hi! I'm *Rio*, your AI assistant running on your PC.\n\n"
                "Any message you send me will be *executed as a task on your computer*.\n\n"
                "*Commands:*\n"
                "/tasks — Show running tasks\n"
                "/cancel — Cancel all tasks\n"
                "/status — Check connection\n"
                "/help — Show this message"
            )
        elif cmd == "/help":
            await self.send(
                "💡 *How to use Rio via Telegram:*\n\n"
                "Just send any message and I'll run it on your PC!\n\n"
                "*Examples:*\n"
                "• Open Chrome and go to youtube.com\n"
                "• Take a screenshot\n"
                "• Create a folder called Projects on Desktop\n"
                "• Search for the latest AI news\n"
                "• What files are in my Downloads folder?\n\n"
                "*During tasks:*\n"
                "• I'll send you progress updates\n"
                "• Reply *yes/no* when I ask for approval\n"
                "• Use /cancel to stop running tasks"
            )
        elif cmd == "/tasks":
            if self._status_fn:
                status = self._status_fn()
                await self.send(f"📋 *Active Tasks:*\n\n{status}")
            else:
                await self.send("📋 No task tracker connected.")
        elif cmd == "/status":
            if self._status_fn:
                status = self._status_fn()
                has_tasks = "running" in status.lower() or "active" in status.lower()
                icon = "🟢" if has_tasks else "✅"
                await self.send(f"{icon} *Rio is online*\n\n{status}")
            else:
                await self.send("✅ Rio is online and connected to your PC.")
        elif cmd == "/cancel":
            if self._on_message:
                await self._on_message("cancel all tasks")
            await self.send("🛑 Cancellation requested for all running tasks.")
        else:
            await self.send(f"❓ Unknown command: `{cmd}`. Try /help")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _api_call(
        self, method: str, params: dict,
    ) -> dict[str, Any]:
        """Call the Telegram Bot API."""
        url = f"{_TELEGRAM_API}/bot{self._token}/{method}"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=35) as client:
                resp = await client.post(url, json=params)
                data = resp.json()
                if data.get("ok"):
                    return {"success": True, "data": data.get("result")}
                return {"success": False, "error": data.get("description", "API error")}
        except ImportError:
            # Fallback to urllib
            import json
            import urllib.request
            req_data = json.dumps(params).encode()
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=35) as resp:
                    data = json.loads(resp.read())
                    if data.get("ok"):
                        return {"success": True, "data": data.get("result")}
                    return {"success": False, "error": data.get("description", "API error")}
            except Exception as exc:
                return {"success": False, "error": str(exc)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

