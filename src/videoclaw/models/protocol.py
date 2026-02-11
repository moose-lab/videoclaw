"""Core model adapter protocol — the extensibility foundation for VideoClaw.

Every video-generation backend (local diffusion model, cloud API, hybrid
pipeline) is exposed through the :class:`VideoModelAdapter` protocol so that
the rest of the system never couples to a concrete provider.
"""

from __future__ import annotations

import enum
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ModelCapability(enum.Enum):
    """Declares what a model adapter can do."""

    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    VIDEO_TO_VIDEO = "video_to_video"
    TEXT_TO_IMAGE = "text_to_image"
    UPSCALE = "upscale"


class ExecutionMode(enum.Enum):
    """Where computation happens."""

    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Immutable specification for a single generation job."""

    prompt: str
    negative_prompt: str = ""
    width: int = 1280
    height: int = 720
    duration_seconds: float = 5.0
    fps: int = 24
    seed: int | None = None
    reference_image: bytes | None = None
    reference_video: bytes | None = None
    style_preset: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    """The output produced by a successful generation call."""

    video_data: bytes
    format: str = "mp4"
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    model_id: str = ""


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Streamed during generation to report incremental progress."""

    progress: float  # 0.0 .. 1.0
    stage: str
    preview_frame: bytes | None = None


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VideoModelAdapter(Protocol):
    """Structural sub-typing contract for every video-generation backend.

    Implement this protocol to plug a new model into VideoClaw.  The protocol
    is :func:`runtime_checkable` so callers can verify compliance with
    ``isinstance(obj, VideoModelAdapter)``.
    """

    @property
    def model_id(self) -> str:
        """Globally unique identifier for this model (e.g. ``"sora"``)."""
        ...

    @property
    def capabilities(self) -> list[ModelCapability]:
        """List of capabilities this adapter supports."""
        ...

    @property
    def execution_mode(self) -> ExecutionMode:
        """Where the model's computation is executed."""
        ...

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run a full generation and return the finished result."""
        ...

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        """Yield :class:`ProgressEvent` instances followed by a final
        :class:`GenerationResult`.

        Adapters that do not support streaming may simply yield a single
        ``GenerationResult``.
        """
        ...

    async def estimate_cost(self, request: GenerationRequest) -> float:
        """Return the estimated cost in USD **before** running the job."""
        ...

    async def health_check(self) -> bool:
        """Return ``True`` if the backend is reachable and ready."""
        ...
