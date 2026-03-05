"""Data models for AI short drama series."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from videoclaw.config import get_config


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DramaStatus(StrEnum):
    DRAFT = "draft"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class EpisodeStatus(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Character:
    """A recurring character in the drama series."""

    name: str
    description: str = ""
    visual_prompt: str = ""
    voice_style: str = ""
    reference_image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Character:
        return cls(**data)


@dataclass
class Episode:
    """A single episode in a drama series."""

    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    number: int = 1
    title: str = ""
    synopsis: str = ""
    opening_hook: str = ""
    status: EpisodeStatus = EpisodeStatus.PENDING
    project_id: str | None = None
    duration_seconds: float = 60.0
    script: str | None = None
    scene_prompts: list[dict[str, Any]] = field(default_factory=list)
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        data = dict(data)
        data["status"] = EpisodeStatus(data.get("status", "pending"))
        return cls(**data)


@dataclass
class DramaSeries:
    """A complete short drama series with episodes and characters."""

    series_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    title: str = ""
    genre: str = ""
    synopsis: str = ""
    style: str = "cinematic"
    language: str = "zh"
    aspect_ratio: str = "9:16"
    target_episode_duration: float = 60.0
    total_episodes: int = 5
    status: DramaStatus = DramaStatus.DRAFT
    characters: list[Character] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    model_id: str = "mock"
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = _now_iso()

    @property
    def cost_total(self) -> float:
        return sum(ep.cost for ep in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "title": self.title,
            "genre": self.genre,
            "synopsis": self.synopsis,
            "style": self.style,
            "language": self.language,
            "aspect_ratio": self.aspect_ratio,
            "target_episode_duration": self.target_episode_duration,
            "total_episodes": self.total_episodes,
            "status": self.status.value,
            "characters": [c.to_dict() for c in self.characters],
            "episodes": [e.to_dict() for e in self.episodes],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model_id": self.model_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DramaSeries:
        data = dict(data)
        data["status"] = DramaStatus(data.get("status", "draft"))
        data["characters"] = [Character.from_dict(c) for c in data.get("characters", [])]
        data["episodes"] = [Episode.from_dict(e) for e in data.get("episodes", [])]
        return cls(**data)


# ---------------------------------------------------------------------------
# Drama state manager
# ---------------------------------------------------------------------------

class DramaManager:
    """Persists DramaSeries as JSON files on disk.

    Layout::

        {projects_dir}/dramas/{series_id}/series.json
        {projects_dir}/dramas/{series_id}/episodes/{episode_id}/
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = (base_dir or get_config().projects_dir) / "dramas"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _series_path(self, series_id: str) -> Path:
        return self.base_dir / series_id / "series.json"

    def create(self, **kwargs: Any) -> DramaSeries:
        series = DramaSeries(**kwargs)
        self.save(series)
        return series

    def save(self, series: DramaSeries) -> Path:
        series.touch()
        path = self._series_path(series.series_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(series.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, series_id: str) -> DramaSeries:
        path = self._series_path(series_id)
        if not path.exists():
            raise FileNotFoundError(f"Drama series {series_id!r} not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        return DramaSeries.from_dict(data)

    def list_series(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return [
            p.name
            for p in self.base_dir.iterdir()
            if p.is_dir() and (p / "series.json").exists()
        ]

    def delete(self, series_id: str) -> None:
        import shutil

        series_dir = self.base_dir / series_id
        if series_dir.exists():
            shutil.rmtree(series_dir)
