"""WebSocket endpoint for real-time project progress updates."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from videoclaw.core.events import event_bus

router = APIRouter()
logger = logging.getLogger(__name__)

# Active connections: project_id -> set of websockets
_connections: dict[str, set[WebSocket]] = {}


@router.websocket("/{project_id}")
async def project_ws(websocket: WebSocket, project_id: str) -> None:
    await websocket.accept()
    _connections.setdefault(project_id, set()).add(websocket)
    logger.info("WS connected: project=%s", project_id)

    # Register event listener for this project
    async def _on_event(event_type: str, data: dict[str, Any]) -> None:
        if data.get("project_id") != project_id:
            return
        msg = json.dumps({"event": event_type, **data})
        dead: list[WebSocket] = []
        for ws in _connections.get(project_id, set()):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _connections.get(project_id, set()).discard(ws)

    # Subscribe to all relevant events
    events = [
        "task.started", "task.completed", "task.failed",
        "shot.generated", "cost.updated",
        "project.planned", "project.completed", "project.failed",
    ]
    for evt in events:
        event_bus.subscribe(evt, _on_event)

    try:
        while True:
            # Keep connection alive, accept pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        _connections.get(project_id, set()).discard(websocket)
        for evt in events:
            event_bus.unsubscribe(evt, _on_event)
        logger.info("WS disconnected: project=%s", project_id)
