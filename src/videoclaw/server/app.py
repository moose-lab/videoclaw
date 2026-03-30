"""FastAPI application -- VideoClaw API server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from videoclaw.config import get_config
from videoclaw.server.routes import generation, projects, ws

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    cfg = get_config()
    cfg.ensure_dirs()
    logger.info("VideoClaw server starting  (projects_dir=%s)", cfg.projects_dir)
    yield
    logger.info("VideoClaw server shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="VideoClaw",
        description="The Agent OS for AI Video Generation",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
    app.include_router(generation.router, prefix="/api/generate", tags=["generation"])
    app.include_router(ws.router, prefix="/ws", tags=["websocket"])

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
