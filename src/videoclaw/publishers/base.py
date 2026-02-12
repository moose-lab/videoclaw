"""Publisher base protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable


class PublishStatus(StrEnum):
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass
class PublishRequest:
    video_path: Path
    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    thumbnail_path: Path | None = None
    schedule_at: str | None = None  # ISO 8601
    extra: dict = field(default_factory=dict)


@dataclass
class PublishResult:
    platform: str
    status: PublishStatus
    url: str | None = None
    platform_id: str | None = None
    error: str | None = None


@runtime_checkable
class Publisher(Protocol):
    """Protocol that all platform publishers must implement."""

    @property
    def platform_name(self) -> str: ...

    async def publish(self, request: PublishRequest) -> PublishResult: ...

    async def get_status(self, platform_id: str) -> PublishResult: ...

    async def health_check(self) -> bool: ...
