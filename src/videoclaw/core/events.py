"""Lightweight async event bus for internal VideoClaw communication."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predefined event types
# ---------------------------------------------------------------------------

TASK_STARTED: str = "task.started"
TASK_COMPLETED: str = "task.completed"
TASK_FAILED: str = "task.failed"
SHOT_GENERATED: str = "shot.generated"
COST_UPDATED: str = "cost.updated"
PROJECT_COMPLETED: str = "project.completed"

# A handler receives the event type string and an arbitrary data payload.
EventHandler = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


class EventBus:
    """Simple publish/subscribe event bus backed by asyncio.

    Subscribers are async callables invoked concurrently when an event fires.
    Exceptions in individual handlers are logged but never propagate to the
    emitter -- fire-and-forget semantics.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: EventHandler) -> None:
        """Register *callback* to be called whenever *event_type* is emitted."""
        self._handlers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: EventHandler) -> None:
        """Remove a previously registered callback."""
        try:
            self._handlers[event_type].remove(callback)
        except ValueError:
            pass

    async def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Fire *event_type* with optional *data*, notifying all subscribers."""
        payload = data or {}
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return

        tasks = [self._safe_call(h, event_type, payload) for h in handlers]
        await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    @staticmethod
    async def _safe_call(
        handler: EventHandler,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        try:
            await handler(event_type, data)
        except Exception:
            logger.exception(
                "Unhandled error in event handler for %r",
                event_type,
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

event_bus = EventBus()
"""Global event bus instance shared across the application."""
