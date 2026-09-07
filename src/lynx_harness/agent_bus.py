from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from .models import AgentTask

@dataclass
class BusTask:
    task: AgentTask
    status: str = "submitted"
    result: Any = None
    error: str | None = None
    _cancel: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _handler_task: asyncio.Task | None = field(default=None, repr=False)

class AgentBus:
    """Local async logical-agent bus; payloads are task contracts, not full contexts."""
    def __init__(self, handler: Callable[[AgentTask], Awaitable[Any]], max_queue: int = 100):
        if not callable(handler) or not 1<=max_queue<=10_000: raise ValueError("invalid agent bus configuration")
        self.handler, self.queue, self.tasks = handler, asyncio.Queue(maxsize=max_queue), {}
        self._worker: asyncio.Task | None = None
    async def start(self):
        if not self._worker or self._worker.done(): self._worker=asyncio.create_task(self._run())
    async def _run(self):
        while True:
            item=await self.queue.get()
            if item._cancel.is_set(): item.status="cancelled"; self.queue.task_done(); continue
            item.status="working"
            item._handler_task=asyncio.create_task(self.handler(item.task))
            try: item.result=await item._handler_task; item.status="completed" if not item._cancel.is_set() else "cancelled"
            except asyncio.CancelledError:
                item.status="cancelled"
                if item._handler_task and not item._handler_task.done(): item._handler_task.cancel()
                if item._handler_task: await asyncio.gather(item._handler_task, return_exceptions=True)
                # A canceled handler must not take down the dispatcher; a
                # cancellation of the dispatcher itself is the close signal.
                if asyncio.current_task() and asyncio.current_task().cancelling(): raise
                continue
            except Exception as exc: item.status="failed"; item.error=str(exc)[:2_000]
            finally:
                item._handler_task=None
                self.queue.task_done()
    async def submit(self, task: AgentTask) -> str:
        if not isinstance(task,AgentTask): raise TypeError("bus accepts AgentTask")
        if task.id in self.tasks: raise ValueError("task id already submitted")
        await self.start()
        item = BusTask(task)
        self.tasks[task.id] = item
        await self.queue.put(item)
        return task.id
    def status(self, task_id: str) -> BusTask:
        if task_id not in self.tasks: raise KeyError(task_id)
        return self.tasks[task_id]
    def cancel(self, task_id: str) -> None:
        item=self.status(task_id); item._cancel.set()
        if item._handler_task and not item._handler_task.done(): item._handler_task.cancel()
    async def close(self):
        if self._worker: self._worker.cancel(); await asyncio.gather(self._worker, return_exceptions=True); self._worker=None
