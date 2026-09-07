from __future__ import annotations

import asyncio
from dataclasses import dataclass

@dataclass
class CancellationToken:
    """Cooperative cancellation signal shared down a task tree.

    Children observe the parent lazily instead of spawning a never-ending relay
    task.  This keeps short-lived runs from leaking asyncio tasks when no
    cancellation ever occurs.
    """
    _event: asyncio.Event | None = None
    _parent: "CancellationToken | None" = None

    def __post_init__(self):
        if self._event is None: self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        return bool((self._event and self._event.is_set()) or (self._parent and self._parent.cancelled))

    def cancel(self) -> None: self._event.set()

    async def wait(self) -> None:
        if self.cancelled: return
        own = asyncio.create_task(self._event.wait())
        parent = asyncio.create_task(self._parent.wait()) if self._parent else None
        try:
            await asyncio.wait({own, parent} if parent else {own}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (own, parent):
                if task and not task.done(): task.cancel()
            await asyncio.gather(*[task for task in (own, parent) if task], return_exceptions=True)

    def raise_if_cancelled(self) -> None:
        if self.cancelled: raise asyncio.CancelledError

    def child(self) -> "CancellationToken":
        return CancellationToken(_parent=self)

async def bounded(awaitable, timeout: float, token: CancellationToken | None = None):
    """Cancel the underlying awaitable on timeout or propagated cancellation."""
    task = asyncio.ensure_future(awaitable)
    watcher = asyncio.create_task(token.wait()) if token else None
    try:
        done, _ = await asyncio.wait({task, watcher} if watcher else {task}, timeout=max(0.0, timeout), return_when=asyncio.FIRST_COMPLETED)
        if not done:
            task.cancel(); await asyncio.gather(task, return_exceptions=True); raise asyncio.TimeoutError
        if watcher and watcher in done:
            task.cancel(); await asyncio.gather(task, return_exceptions=True); raise asyncio.CancelledError
        return await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if watcher:
            watcher.cancel(); await asyncio.gather(watcher, return_exceptions=True)
