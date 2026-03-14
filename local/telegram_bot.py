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
        """Set the async callback for incoming task messages (routes to orchestrator)."""
        self._on_message = handler

    def set_chat_handler(
        self, handler: Callable[[str], Awaitable[None]],
    ) -> None:
        """Set the async callback for conversational (non-task) messages.

        These are injected into the live session as context so the model
        responds conversationally instead of spawning a task.
        """
        self._on_chat_message = handler

    def set_command_handler(
        self, handler: Callable[[str, list[str]], Awaitable[None]],
    ) -> None:
        """Set the async callback for incoming /slash commands."""
        self._on_command = handler

    def set_status_fn(self, fn: Callable[[], str]) -> None:
        """Set a sync callable that returns current task status string."""
        self._status_fn = fn

    def set_approval_handler(
        self, handler: Callable[[bool], Awaitable[None]],
    ) -> None:
        """Set the async callback for approval responses (yes/no)."""
        self._on_approval = handler

    def notify_approval_pending(self, tool_name: str, tool_args: dict | None = None) -> None:
        """Called by the approval gate to alert user via Telegram."""
        self._pending_approval = True
        args_str = ""
        if tool_args:
            # Truncate long args for readability
            summary = ", ".join(f"{k}={str(v)[:40]}" for k, v in tool_args.items())
            args_str = f"\nArgs: `{summary}`"

        asyncio.create_task(
            self.send(
                f"⚠️ *Approval Required*\n\n"
                f"Rio wants to run: `{tool_name}`{args_str}\n\n"
                f"Reply *yes* to allow or *no* to deny."
            )
        )

    async def send(self, text: str, parse_mode: str = "Markdown") -> dict[str, Any]:
        """Send a message to the configured chat."""
        if not self.enabled:
            return {"success": False, "error": "Telegram not configured"}
        # Escape markdown characters if needed (best effort)
        # Note: MarkdownV2 is more strict, standard Markdown is usually fine
        return await self._api_call("sendMessage", {
            "chat_id": self._chat_id,
            "text": text[:4096],
            "parse_mode": parse_mode,
        })

    async def send_typing(self, action: str = "typing") -> None:
        """Send 'typing', 'upload_photo', etc. chat action."""
        if not self.enabled:
            return
        await self._api_call("sendChatAction", {
            "chat_id": self._chat_id,
            "action": action,
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
        # Register bot commands so `/` shows autocomplete in Telegram
        asyncio.create_task(self._register_commands())
        # Send a brief "online" message to verify chat_id
        asyncio.create_task(self.send("🚀 *Rio is now online* (PC connection active)"))

    async def _register_commands(self) -> None:
        """Register bot commands with Telegram's BotFather API.

        This enables the `/` autocomplete menu in Telegram clients.
        Called automatically on start().
        """
        commands = [
            {"command": "help", "description": "Show available commands"},
            {"command": "tasks", "description": "List currently running tasks"},
            {"command": "cancel", "description": "Cancel all running tasks"},
            {"command": "status", "description": "Show Rio system status"},
            {"command": "screenshot", "description": "Capture PC screen"},
            {"command": "reset", "description": "Reset session (clear history & notes)"},
            {"command": "model", "description": "View or change the orchestrator model"},
            {"command": "models", "description": "List available models"},
            {"command": "memory", "description": "Search long-term memory"},
            {"command": "agents", "description": "List specialist agents"},
            {"command": "voice", "description": "Voice control help"},
        ]
        try:
            result = await self._api_call("setMyCommands", {
                "commands": commands,
            })
            if result.get("success"):
                log.info("telegram_bot.commands_registered", count=len(commands))
            else:
                log.warning("telegram_bot.commands_register_failed", result=result)
        except Exception as exc:
            log.warning("telegram_bot.commands_register_error", error=str(exc))

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
            "allowed_updates": ["message", "callback_query"],
        })
        if result.get("success") and isinstance(result.get("data"), list):
            return result["data"]
        return []

    async def _handle_update(self, update: dict) -> None:
        """Process a single Telegram update."""
        update_id = update.get("update_id", 0)
        self._offset = max(self._offset, update_id + 1)

        message = update.get("message", {})
        if not message:
            # Check for other update types (callback_query, etc)
            return

        text = message.get("text", "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))
        sender_id = str(message.get("from", {}).get("id", ""))
        username = message.get("from", {}).get("username", "unknown")

        # Only accept messages from the configured chat
        if chat_id != self._chat_id:
            log.debug("telegram_bot.ignored_chat", chat_id=chat_id, sender=username)
            # If it's a direct message to the bot, tell them their ID
            if message.get("chat", {}).get("type") == "private":
                await self._api_call("sendMessage", {
                    "chat_id": chat_id,
                    "text": (
                        f"🔒 *Access Denied*\n\n"
                        f"This Rio instance is configured for a different chat.\n\n"
                        f"Your Chat ID: `{chat_id}`\n"
                        f"Your User ID: `{sender_id}`\n\n"
                        f"Update your `.env` or `config.yaml` to use these IDs."
                    ),
                    "parse_mode": "Markdown",
                })
            return

        if not text:
            return

        # --- Approval gate: resolve pending yes/no ---
        if self._pending_approval and self._on_approval:
            lower = text.lower()
            if lower in ("yes", "no", "approve", "deny", "ok", "cancel", "y", "n"):
                approved = lower in ("yes", "approve", "ok", "y")
                self._pending_approval = False
                try:
                    await self._on_approval(approved)
                except Exception as exc:
                    log.warning("telegram_bot.approval_handler_error", error=str(exc))
                reply = "✅ *Approved*, proceeding..." if approved else "❌ *Denied*, cancelling."
                await self.send(reply)
                return

        # Handle /commands
        if text.startswith("/"):
            await self._handle_command(text)
            return

        log.info("telegram_bot.message_received", text=text[:80])

        # --- Intent classification: task vs conversation ---
        # Only route to orchestrator (spawn_task) if the message looks like a task.
        # Conversational messages go to the live session instead.
        is_task = self._classify_as_task(text)

        if is_task and self._on_message:
            try:
                await self.send_typing()
                await self._on_message(text)
            except Exception as exc:
                log.warning("telegram_bot.handler_error", error=str(exc))
                await self.send(f"❌ *Error processing:* {exc}")
        elif not is_task and hasattr(self, '_on_chat_message') and self._on_chat_message:
            try:
                await self.send_typing()
                await self._on_chat_message(text)
                # Don't auto-reply — the live model's response will come
                # through the session and be forwarded by adk_server
            except Exception as exc:
                log.warning("telegram_bot.chat_handler_error", error=str(exc))
        elif self._on_message:
            # Fallback: if no chat handler, route everything to task handler
            try:
                await self.send_typing()
                await self._on_message(text)
            except Exception as exc:
                log.warning("telegram_bot.handler_error", error=str(exc))
                await self.send(f"❌ *Error processing:* {exc}")

    def _classify_as_task(self, text: str) -> bool:
        """Classify whether a Telegram message is a task or conversation.

        Uses lightweight keyword heuristics to avoid LLM latency on every message.
        Conversational messages like "how are you", "thanks", "hello" return False.
        Task-like messages like "open chrome", "write a function" return True.
        """
        t = text.strip().lower().rstrip(".!?")

        if len(t) < 4:
            return False

        # Explicit task markers
        if t.startswith(("task:", "do:", "execute:", "please ")):
            return True

        # Resume commands
        if t in ("resume", "continue", "go on", "keep going", "go ahead"):
            return True

        # Pure greetings / conversational — NOT tasks
        _CHAT_PATTERNS = (
            "hi", "hello", "hey", "good morning", "good evening", "good night",
            "how are you", "what's up", "sup", "yo", "thanks", "thank you",
            "cool", "nice", "great", "awesome", "ok", "okay", "sure", "yeah",
            "yes", "no", "bye", "goodbye", "see you", "later", "lol", "haha",
            "what do you think", "tell me about yourself", "who are you",
            "how's it going", "what are you doing", "i'm bored",
            "tell me a joke", "sing a song", "what's your name",
        )
        if t in _CHAT_PATTERNS or any(t.startswith(p) for p in _CHAT_PATTERNS):
            return False

        # Questions without action verbs are typically conversations
        _QUESTION_STARTS = (
            "what is", "what are", "what does", "what do",
            "why ", "how does", "how is", "how are",
            "when ", "where ", "who ", "which ",
            "is it", "are there", "can you explain", "tell me about",
            "explain ", "describe ", "what happened",
        )
        if any(t.startswith(q) for q in _QUESTION_STARTS):
            # Exception: "can you open..." is a task
            _ACTION_AFTER_QUESTION = (
                "can you open", "can you run", "can you create", "can you write",
                "can you install", "can you build", "can you fix", "can you delete",
                "can you search", "can you download", "can you send", "can you click",
            )
            if any(t.startswith(a) for a in _ACTION_AFTER_QUESTION):
                return True
            return False

        # Very short messages (< 3 words) are usually chat
        if len(t.split()) < 3:
            return False

        # Action verbs at start → task
        _ACTION_VERBS = (
            "open", "go to", "goto", "navigate", "search", "create", "download",
            "install", "click", "close", "launch", "start", "browse", "find",
            "visit", "play", "stop", "delete", "move", "copy", "paste", "save",
            "upload", "send", "book", "order", "fill", "submit", "scroll", "type",
            "write", "run", "execute", "build", "deploy", "setup", "configure",
            "update", "uninstall", "rename", "drag", "edit", "fix", "read", "show",
            "switch", "set up", "look up", "check out",
        )
        for verb in _ACTION_VERBS:
            if t.startswith(verb + " ") or t.startswith(verb + ":"):
                return True

        # Task-indicating phrases anywhere
        _TASK_PHRASES = (
            "for me", "please do", "can you do", "i need you to",
            "i want you to", "go ahead and", "take over", "handle this",
            "do this", "complete this", "finish this", "do it", "get started",
        )
        for phrase in _TASK_PHRASES:
            if phrase in t:
                return True

        # URL → navigation task
        if "http://" in t or "https://" in t or "www." in t:
            return True

        return False

    async def _handle_command(self, text: str) -> None:
        """Handle Telegram bot commands."""
        parts = text.split()
        full_cmd = parts[0].lower()
        cmd = full_cmd.split("@")[0]  # Strip @botname suffix
        args = parts[1:]

        await self.send_typing()

        if cmd == "/start":
            await self.send(
                "👋 Hi! I'm *Rio*, your AI assistant running on your PC.\n\n"
                "Any message you send me will be *executed as a task on your computer*.\n\n"
                "Type /help to see what I can do."
            )
        elif cmd == "/help":
            await self.send(
                "💡 *Rio Command Guide*\n\n"
                "*Core Commands:*\n"
                "• /tasks — List running tasks\n"
                "• /cancel — Stop all tasks\n"
                "• /status — System health check\n"
                "• /reset — Start a new session (clear history)\n\n"
                "*System Controls:*\n"
                "• /model — Switch model (Flash/Pro)\n"
                "• /screenshot — Manual capture\n"
                "• /voice — Toggle voice output\n"
                "• /memory — Search long-term memory\n"
                "• /agents — List specialist agents\n"
                "• /whoami — Show ID info\n\n"
                "Just send a normal message like 'Open Chrome' to start a task!"
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
            await self.send("🛑 *Cancellation requested* for all running tasks.")
        elif cmd == "/whoami":
            await self.send(
                f"👤 *Identity Info*\n\n"
                f"Chat ID: `{self._chat_id}`\n"
                f"Status: `Authorized`"
            )
        # Route advanced commands to the registered handler (typically in adk_server.py)
        elif hasattr(self, "_on_command") and self._on_command:
            try:
                await self._on_command(cmd[1:], args)
            except Exception as exc:
                log.warning("telegram_bot.command_handler_error", error=str(exc))
                await self.send(f"❌ *Command failed:* {exc}")
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

