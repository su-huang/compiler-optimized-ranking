"""Adaptive request batching for the prediction endpoint (Phase 5).

Instead of running one forward pass per request (serialized behind a lock, per
Phase 2's fix for MPS thread-safety), incoming requests are queued and run through
the model together as a single batch. The wait window before flushing a batch is
adaptive: if requests are already piling up, flush sooner (favor latency); if the
queue is quiet, wait longer to try to fill a bigger batch (favor throughput).
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class _PendingRequest:
    item: Any
    future: "asyncio.Future"


@dataclass
class AdaptiveBatcher:
    """Batches items pulled from an asyncio queue and runs them through predict_fn.

    predict_fn(items: list) -> list[result], same length/order as items.

    The flush window scales down as more requests are already waiting: a busy
    queue flushes almost immediately (min_window_ms), a quiet queue waits longer
    (max_window_ms) hoping to accumulate a fuller batch.
    """

    predict_fn: Callable[[list], list]
    max_batch_size: int = 16
    min_window_ms: float = 2.0
    max_window_ms: float = 20.0
    busy_threshold: int = 4

    _queue: "asyncio.Queue[_PendingRequest]" = field(default_factory=asyncio.Queue, init=False)
    _worker_task: "asyncio.Task | None" = field(default=None, init=False)

    def start(self) -> None:
        self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()

    @property
    def queue_depth(self) -> int:
        """Requests currently waiting to be batched -- exported as a Prometheus gauge
        for the Phase 6 queue-depth-driven Horizontal Pod Autoscaler."""
        return self._queue.qsize()

    async def submit(self, item: Any) -> Any:
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put(_PendingRequest(item=item, future=future))
        return await future

    def _current_window_ms(self) -> float:
        depth = self._queue.qsize()
        if depth >= self.busy_threshold:
            return self.min_window_ms
        # linearly interpolate between max (empty queue) and min (busy_threshold)
        fraction = depth / self.busy_threshold
        return self.max_window_ms - fraction * (self.max_window_ms - self.min_window_ms)

    async def _collect_batch(self) -> list[_PendingRequest]:
        first = await self._queue.get()
        batch = [first]

        deadline = time.perf_counter() + self._current_window_ms() / 1000
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        while True:
            batch = await self._collect_batch()
            items = [pending.item for pending in batch]
            try:
                # predict_fn does blocking model compute; run it off the event loop so
                # other coroutines (health checks, other batchers) aren't stalled.
                results = await asyncio.to_thread(self.predict_fn, items)
            except Exception as exc:  # noqa: BLE001 - propagate to every waiter
                for pending in batch:
                    if not pending.future.done():
                        pending.future.set_exception(exc)
                continue

            for pending, result in zip(batch, results):
                if not pending.future.done():
                    pending.future.set_result(result)
