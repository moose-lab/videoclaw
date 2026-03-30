"""Base class for cloud video adapters.

Provides shared infrastructure for cloud-based video generation backends:
API key validation, Bearer auth headers, cost estimation, health checks,
and a default poll-based ``generate_stream`` implementation.

Concrete subclasses must implement ``generate`` and the protocol properties.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator


from videoclaw.models.protocol import (
    ExecutionMode,
    GenerationRequest,
    GenerationResult,
    ModelCapability,
    ProgressEvent,
)

logger = logging.getLogger(__name__)


class BaseCloudVideoAdapter:
    """Optional base for cloud video adapters.

    Provides default implementations for:
    - ``estimate_cost`` / ``_estimate_cost_for``
    - ``health_check``
    - ``generate_stream`` (wraps ``generate`` with progress events)
    - ``_ensure_api_key`` / ``_bearer_headers``

    Subclasses must set:
    - ``_api_key`` (in ``__init__``)
    - ``_COST_PER_SECOND_USD`` (class-level constant)
    - ``_ADAPTER_NAME`` (for log messages)
    """

    _COST_PER_SECOND_USD: float = 0.02
    _ADAPTER_NAME: str = "cloud"

    _api_key: str | None = None

    # ------------------------------------------------------------------
    # Protocol properties (subclass must override)
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        raise NotImplementedError

    @property
    def capabilities(self) -> list[ModelCapability]:
        raise NotImplementedError

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.CLOUD

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> None:
        """Raise if no API key is configured."""
        if not self._api_key:
            raise RuntimeError(
                f"{self._ADAPTER_NAME} API key is required. "
                f"Check environment variables or pass api_key= to the adapter."
            )

    def _bearer_headers(self) -> dict[str, str]:
        """Return standard Bearer-token auth headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _estimate_cost_for(self, request: GenerationRequest) -> float:
        """Estimate cost based on duration."""
        return round(request.duration_seconds * self._COST_PER_SECOND_USD, 4)

    # ------------------------------------------------------------------
    # Default implementations
    # ------------------------------------------------------------------

    async def estimate_cost(self, request: GenerationRequest) -> float:
        """Return the estimated cost in USD."""
        return self._estimate_cost_for(request)

    async def health_check(self) -> bool:
        """Return True if the API key is configured."""
        if not self._api_key:
            logger.warning("[%s] No API key configured", self._ADAPTER_NAME)
            return False
        return True

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run a full generation.  Must be overridden by subclasses."""
        raise NotImplementedError

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        """Default poll-based streaming that wraps ``generate``.

        Subclasses with native streaming can override this method.
        """
        yield ProgressEvent(progress=0.0, stage="submitting")
        yield ProgressEvent(progress=0.1, stage="generating")

        result = await self.generate(request)

        yield ProgressEvent(progress=1.0, stage="complete")
        yield result

    def _build_result(
        self,
        video_data: bytes,
        request: GenerationRequest,
        *,
        job_id: str = "",
        extra_metadata: dict | None = None,
    ) -> GenerationResult:
        """Build a standard GenerationResult."""
        metadata = {
            "model": self.model_id,
            "prompt": request.prompt,
            "generated_at": time.time(),
        }
        if job_id:
            metadata["job_id"] = job_id
        if extra_metadata:
            metadata.update(extra_metadata)

        return GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata=metadata,
            cost_usd=self._estimate_cost_for(request),
            model_id=self.model_id,
        )
