"""Tests for the event bus."""

import asyncio

import pytest

from videoclaw.core.events import EventBus


@pytest.mark.asyncio
async def test_subscribe_and_emit():
    bus = EventBus()
    received = []

    async def handler(event_type, data):
        received.append((event_type, data))

    bus.subscribe("test.event", handler)
    await bus.emit("test.event", {"key": "value"})
    assert len(received) == 1
    assert received[0] == ("test.event", {"key": "value"})


@pytest.mark.asyncio
async def test_unsubscribe():
    bus = EventBus()
    received = []

    async def handler(event_type, data):
        received.append(1)

    bus.subscribe("x", handler)
    bus.unsubscribe("x", handler)
    await bus.emit("x")
    assert len(received) == 0


@pytest.mark.asyncio
async def test_handler_error_does_not_propagate():
    bus = EventBus()

    async def bad_handler(event_type, data):
        raise RuntimeError("boom")

    bus.subscribe("fail", bad_handler)
    # Should not raise
    await bus.emit("fail")
