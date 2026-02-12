"""Video generation endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from videoclaw.config import get_config
from videoclaw.core.director import Director
from videoclaw.core.events import event_bus
from videoclaw.core.executor import DAGExecutor
from videoclaw.core.planner import build_dag
from videoclaw.core.state import ProjectState, StateManager
from videoclaw.cost.tracker import CostTracker
from videoclaw.models.llm.litellm_wrapper import LLMClient
from videoclaw.models.registry import ModelRegistry

router = APIRouter()
logger = logging.getLogger(__name__)
_state_mgr = StateManager()


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None
    strategy: str = "auto"
    budget_usd: float | None = None


class GenerateResponse(BaseModel):
    project_id: str
    status: str
    message: str


@router.post("/", response_model=GenerateResponse)
async def start_generation(body: GenerateRequest) -> GenerateResponse:
    """Kick off a full video generation pipeline (async, non-blocking)."""
    cfg = get_config()
    ps = _state_mgr.create_project(prompt=body.prompt)

    # Launch the pipeline in background
    asyncio.create_task(_run_pipeline(ps, body, cfg))

    return GenerateResponse(
        project_id=ps.project_id,
        status="started",
        message="Pipeline launched. Connect to /ws/{project_id} for progress.",
    )


@router.get("/{project_id}/status")
async def get_status(project_id: str) -> dict:
    try:
        ps = _state_mgr.load(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "project_id": ps.project_id,
        "status": ps.status.value,
        "cost_total": ps.cost_total,
        "shots": [s.to_dict() for s in ps.storyboard],
    }


async def _run_pipeline(
    ps: ProjectState,
    body: GenerateRequest,
    cfg: Any,
) -> None:
    """Execute the full generation pipeline in background."""
    try:
        llm = LLMClient(default_model=cfg.default_llm)
        director = Director(llm=llm)

        # Step 1: Director plans the project
        ps = await director.plan(ps)
        _state_mgr.save(ps)
        await event_bus.emit("project.planned", {"project_id": ps.project_id})

        # Step 2: Build DAG and execute
        dag = build_dag(ps)
        executor = DAGExecutor(
            dag=dag,
            project_state=ps,
            event_bus=event_bus,
        )
        ps = await executor.run()
        _state_mgr.save(ps)
        await event_bus.emit("project.completed", {"project_id": ps.project_id})

    except Exception:
        logger.exception("Pipeline failed for project %s", ps.project_id)
        ps.status = "failed"
        _state_mgr.save(ps)
        await event_bus.emit("project.failed", {"project_id": ps.project_id})
