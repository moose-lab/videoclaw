"""Project state management -- tracks the lifecycle of a video project."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, fields, asdict
from enum import StrEnum
from pathlib import Path
from typing import Any

from videoclaw.config import get_config
from videoclaw.utils import _now_iso

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProjectStatus(StrEnum):
    PLANNING = "planning"
    GENERATING = "generating"
    COMPOSING = "composing"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


class ShotStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Shot:
    shot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    prompt: str = ""
    duration_seconds: float = 5.0
    model_id: str = "mock"
    status: ShotStatus = ShotStatus.PENDING
    asset_path: str | None = None
    cost: float = 0.0
    retries: int = 0
    reference_images: dict[str, str] = field(default_factory=dict)
    # Mapping: character_name → file_path (primary / front view)
    # e.g. {"林薇": "/path/to/dramas/xxx/characters/林薇_front.png"}
    multi_reference_images: dict[str, list[str]] = field(default_factory=dict)
    # Mapping: character_name → [front, three_quarter, full_body]
    # For Seedance 2.0 Universal Reference (全能参考) multi-angle consistency
    reference_image_urls: dict[str, str] = field(default_factory=dict)
    # Mapping: character_name → HTTPS URL (turnaround sheet)
    # For Seedance API via vectorspace.cn proxy which rejects base64 data URIs

    # -- Serialisation helpers -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Shot:
        data = dict(data)  # shallow copy so we don't mutate the caller's dict
        data["status"] = ShotStatus(data.get("status", "pending"))
        # Filter to known fields for backward compatibility
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)


@dataclass
class ProjectState:
    project_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    status: ProjectStatus = ProjectStatus.PLANNING
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    prompt: str = ""
    script: str | None = None
    storyboard: list[Shot] = field(default_factory=list)
    assets: dict[str, str] = field(default_factory=dict)
    cost_total: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Bump ``updated_at`` to the current UTC time."""
        self.updated_at = _now_iso()

    # -- Serialisation helpers -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "prompt": self.prompt,
            "script": self.script,
            "storyboard": [s.to_dict() for s in self.storyboard],
            "assets": self.assets,
            "cost_total": self.cost_total,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectState:
        data = dict(data)
        data["status"] = ProjectStatus(data.get("status", "planning"))
        data["storyboard"] = [Shot.from_dict(s) for s in data.get("storyboard", [])]
        return cls(**data)


# ---------------------------------------------------------------------------
# State manager
# ---------------------------------------------------------------------------

class StateManager:
    """Persists :class:`ProjectState` as JSON files on disk.

    Layout::

        {projects_dir}/{project_id}/state.json
    """

    def __init__(self, projects_dir: Path | None = None) -> None:
        self.projects_dir = projects_dir or get_config().projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, project_id: str) -> Path:
        return self.projects_dir / project_id / "state.json"

    # -- Public API ------------------------------------------------------------

    def create_project(self, prompt: str) -> ProjectState:
        """Create a new project, persist initial state, and return it."""
        state = ProjectState(prompt=prompt)
        self.save(state)
        logger.info("Created project %s", state.project_id)
        return state

    def save(self, state: ProjectState) -> Path:
        """Write *state* to disk as JSON and return the file path."""
        state.touch()
        path = self._state_path(state.project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        return path

    def load(self, project_id: str) -> ProjectState:
        """Load a project's state from disk.

        Raises:
            FileNotFoundError: if no state file exists for *project_id*.
        """
        path = self._state_path(project_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProjectState.from_dict(data)

    def update_shot(self, project_id: str, shot_id: str, **kwargs: Any) -> ProjectState:
        """Update fields on a specific shot within a project and save.

        Raises:
            KeyError: if the shot is not found.
        """
        state = self.load(project_id)
        for shot in state.storyboard:
            if shot.shot_id == shot_id:
                for key, value in kwargs.items():
                    if not hasattr(shot, key):
                        raise AttributeError(f"Shot has no attribute {key!r}")
                    setattr(shot, key, value)
                self.save(state)
                return state
        raise KeyError(f"Shot {shot_id!r} not found in project {project_id!r}")

    def list_projects(self) -> list[str]:
        """Return project IDs for all persisted projects."""
        return [
            p.name
            for p in self.projects_dir.iterdir()
            if p.is_dir() and (p / "state.json").exists()
        ]
