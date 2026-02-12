"""Project CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from videoclaw.core.state import ProjectState, StateManager

router = APIRouter()
_state_mgr = StateManager()


class CreateProjectRequest(BaseModel):
    prompt: str


class ProjectSummary(BaseModel):
    project_id: str
    status: str
    prompt: str
    cost_total: float
    created_at: str
    updated_at: str
    shots_count: int


def _summarise(ps: ProjectState) -> ProjectSummary:
    return ProjectSummary(
        project_id=ps.project_id,
        status=ps.status.value,
        prompt=ps.prompt,
        cost_total=ps.cost_total,
        created_at=ps.created_at,
        updated_at=ps.updated_at,
        shots_count=len(ps.storyboard),
    )


@router.post("/", response_model=ProjectSummary)
async def create_project(body: CreateProjectRequest) -> ProjectSummary:
    ps = _state_mgr.create_project(prompt=body.prompt)
    return _summarise(ps)


@router.get("/", response_model=list[ProjectSummary])
async def list_projects() -> list[ProjectSummary]:
    out: list[ProjectSummary] = []
    for pid in _state_mgr.list_projects():
        try:
            out.append(_summarise(_state_mgr.load(pid)))
        except Exception:
            continue
    return out


@router.get("/{project_id}")
async def get_project(project_id: str) -> dict:
    try:
        ps = _state_mgr.load(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
    return ps.to_dict()


@router.delete("/{project_id}")
async def delete_project(project_id: str) -> dict:
    import shutil

    project_dir = _state_mgr.projects_dir / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    shutil.rmtree(project_dir)
    return {"deleted": project_id}
