"""Mock video adapter — generates placeholder output for testing.

This adapter never calls an external service.  It produces a minimal byte
payload that is clearly identifiable as mock output, making it safe to use
in CI pipelines and local development.
"""

from __future__ import annotations

import asyncio
import logging
import struct
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


def _build_placeholder_mp4(
    width: int = 1280,
    height: int = 720,
    duration_seconds: float = 5.0,
    fps: int = 24,
) -> bytes:
    """Return a minimal byte sequence that pretends to be an MP4 container.

    This is **not** a valid MP4 file — it is a compact header that carries
    enough metadata for downstream code to inspect without requiring a real
    video codec.  Use ``ffprobe`` or similar to verify real files; this is
    purely a dev/test placeholder.
    """
    # We embed a recognizable magic header and the generation parameters so
    # tests can assert on the metadata without decoding a real container.
    magic = b"VCLAW_MOCK_MP4"
    payload = (
        magic
        + struct.pack(">HH", width, height)
        + struct.pack(">f", duration_seconds)
        + struct.pack(">H", fps)
    )
    # Pad to a round 1 KiB so callers that check ``len(video_data)`` get a
    # plausible (albeit tiny) file.
    return payload.ljust(1024, b"\x00")


class MockVideoAdapter:
    """Adapter that returns synthetic output instantly.

    Useful for:

    * Running the full pipeline without GPU / API keys.
    * Unit and integration tests.
    * Benchmarking orchestration overhead in isolation.
    """

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> list[ModelCapability]:
        return [ModelCapability.TEXT_TO_VIDEO, ModelCapability.IMAGE_TO_VIDEO]

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.LOCAL

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a placeholder MP4 after a tiny simulated delay."""
        logger.info(
            "[mock] generate — prompt=%r, %dx%d, %.1fs @ %dfps",
            request.prompt[:80],
            request.width,
            request.height,
            request.duration_seconds,
            request.fps,
        )

        # Simulate a small amount of work so timing-sensitive tests remain
        # realistic.
        await asyncio.sleep(0.05)

        video_data = _build_placeholder_mp4(
            width=request.width,
            height=request.height,
            duration_seconds=request.duration_seconds,
            fps=request.fps,
        )

        return GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": self.model_id,
                "prompt": request.prompt,
                "seed": request.seed,
                "generated_at": time.time(),
            },
            cost_usd=0.0,
            model_id=self.model_id,
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        """Yield synthetic progress events followed by the final result."""
        stages = [
            (0.0, "initialising"),
            (0.25, "encoding_prompt"),
            (0.50, "generating_frames"),
            (0.75, "compositing"),
            (0.90, "encoding_video"),
        ]

        for progress, stage in stages:
            yield ProgressEvent(progress=progress, stage=stage)
            await asyncio.sleep(0.02)

        result = await self.generate(request)
        yield ProgressEvent(progress=1.0, stage="complete")
        yield result

    # ------------------------------------------------------------------
    # Cost & health
    # ------------------------------------------------------------------

    async def estimate_cost(self, request: GenerationRequest) -> float:
        """Mock generation is always free."""
        return 0.0

    async def health_check(self) -> bool:
        """Mock adapter is always healthy."""
        return True
