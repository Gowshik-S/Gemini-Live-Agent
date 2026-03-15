import asyncio
from typing import Callable, Awaitable

class BlockCoalescer:
    """
    Batches rapid UI/voice updates together to prevent notification spam.
    (OpenClaw block-reply-pipeline pattern)
    """
    def __init__(self, inject_fn: Callable[[str], Awaitable[None]], flush_interval: float = 2.0):
        self._inject_fn = inject_fn
        self._flush_interval = flush_interval
        self._buffer: list[str] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def push(self, message: str) -> None:
        """Push a message to the coalescer buffer."""
        if not self._inject_fn:
            return

        async with self._lock:
            self._buffer.append(message)
            if not self._flush_task or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        """Wait for the flush interval, then send all batched messages."""
        await asyncio.sleep(self._flush_interval)
        async with self._lock:
            if not self._buffer:
                return
            
            # Combine messages.
            # Example: "[REASONING] 🤔 I need..." + "[SYSTEM: Still working on open_app]"
            combined = "\n".join(self._buffer)
            self._buffer.clear()
            
        try:
            await self._inject_fn(combined)
        except Exception:
            pass

    async def force_flush(self) -> None:
        """Immediately flush the buffer without waiting."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        
        async with self._lock:
            if not self._buffer:
                return
            combined = "\n".join(self._buffer)
            self._buffer.clear()
            
        try:
            await self._inject_fn(combined)
        except Exception:
            pass
