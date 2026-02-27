"""ZhipuAI (智谱清影) video generation adapter.

This adapter wraps the ZhipuAI CogVideoX API using the official zhipuai SDK.

Environment variables:
- ZHIPU_API_KEY / VIDEOCLAW_ZHIPU_API_KEY

API Documentation: https://bigmodel.cn/dev/api/normal-model/video-model
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import AsyncIterator

import httpx
from zhipuai import ZhipuAI

from videoclaw.config import get_config
from videoclaw.models.protocol import (
    ExecutionMode,
    GenerationRequest,
    GenerationResult,
    ModelCapability,
    ProgressEvent,
)

logger = logging.getLogger(__name__)

# ZhipuAI video models
ZHIPU_MODELS = {
    "cogvideox-flash": "cogvideox-flash",  # Fast model (recommended)
    "cogvideox": "cogvideox",  # High quality model
}

# Cost estimate (USD per second) - ZhipuAI has free tier
_COST_PER_SECOND_USD = 0.015

# Timeouts
_GENERATE_TIMEOUT_S = 600.0  # 10 minutes
_POLL_INTERVAL_S = 5.0  # Poll every 5 seconds


class ZhipuVideoAdapter:
    """Adapter for ZhipuAI (智谱清影) CogVideoX video generation API.

    Uses the official zhipuai SDK for authentication and API calls.

    Parameters
    ----------
    api_key:
        ZhipuAI API Key. Falls back to ZHIPU_API_KEY or VIDEOCLAW_ZHIPU_API_KEY.
    model:
        Model identifier, "cogvideox-flash" (fast) or "cogvideox" (quality).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "cogvideox-flash",
    ) -> None:
        config = get_config()
        self._api_key = api_key or config.zhipu_api_key
        self._model = model
        self._client: ZhipuAI | None = None

    def _get_client(self) -> ZhipuAI:
        """Get or create the ZhipuAI client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "ZhipuAI API key is required. Set ZHIPU_API_KEY or "
                    "VIDEOCLAW_ZHIPU_API_KEY, or pass api_key= to ZhipuVideoAdapter."
                )
            self._client = ZhipuAI(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def capabilities(self) -> list[ModelCapability]:
        return [ModelCapability.TEXT_TO_VIDEO, ModelCapability.IMAGE_TO_VIDEO]

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.CLOUD

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Submit a generation job and poll until completion."""
        self._ensure_api_key()

        # Submit task and get task ID
        task_id = await self._create_task(request)

        # Poll for completion
        video_url = await self._poll_for_completion(task_id)

        # Download the video
        video_data = await self._download_video(video_url)

        cost = self._estimate_cost_for(request)

        return GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": self.model_id,
                "task_id": task_id,
                "prompt": request.prompt,
                "generated_at": time.time(),
            },
            cost_usd=cost,
            model_id=self.model_id,
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        """Stream progress events while polling the ZhipuAI job."""
        self._ensure_api_key()

        yield ProgressEvent(progress=0.0, stage="submitting_task")

        # Create task
        task_id = await self._create_task(request)
        yield ProgressEvent(progress=0.1, stage="task_created")

        # Poll for completion
        poll_count = 0
        video_url = None
        while True:
            video_url = await self._poll_for_completion(task_id, poll_only=True)
            if video_url:
                break

            poll_count += 1
            progress = min(0.1 + poll_count * 0.05, 0.9)
            yield ProgressEvent(progress=progress, stage="processing")

            await asyncio.sleep(_POLL_INTERVAL_S)

        yield ProgressEvent(progress=0.95, stage="downloading")

        # Download video
        video_data = await self._download_video(video_url)

        cost = self._estimate_cost_for(request)

        yield ProgressEvent(progress=1.0, stage="complete")
        yield GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": self.model_id,
                "task_id": task_id,
                "prompt": request.prompt,
                "generated_at": time.time(),
            },
            cost_usd=cost,
            model_id=self.model_id,
        )

    # ------------------------------------------------------------------
    # Cost & health
    # ------------------------------------------------------------------

    async def estimate_cost(self, request: GenerationRequest) -> float:
        return self._estimate_cost_for(request)

    async def health_check(self) -> bool:
        """Check readiness by verifying API key is set."""
        if not self._api_key:
            logger.warning("[zhipu] No API key configured")
            return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "ZhipuAI API key is required. Set ZHIPU_API_KEY or "
                "VIDEOCLAW_ZHIPU_API_KEY, or pass api_key= to ZhipuVideoAdapter."
            )

    def _build_size(self, request: GenerationRequest) -> str:
        """Build size string for ZhipuAI API."""
        # ZhipuAI supports: 1024x576, 576x1024, 1280x720, 720x1280
        w, h = request.width, request.height
        if w >= 1280:
            return "1280x720"
        elif w >= 1024:
            return "1024x576"
        elif h >= 1024:
            return "576x1024"
        else:
            return "1024x576"

    async def _create_task(self, request: GenerationRequest) -> str:
        """Create a video generation task and return the task ID."""
        client = self._get_client()
        size = self._build_size(request)

        logger.info("[zhipu] Creating task with model: %s, size: %s", self._model, size)

        # Build kwargs for API call
        kwargs: dict = {
            "model": self._model,
            "prompt": request.prompt,
            "size": size,
            "quality": "speed" if "flash" in self._model else "quality",
            "with_audio": False,
        }

        # Handle image-to-video
        if request.reference_image:
            # Convert bytes to base64 if needed
            if isinstance(request.reference_image, bytes):
                b64_image = base64.b64encode(request.reference_image).decode("utf-8")
                kwargs["image_url"] = f"data:image/png;base64,{b64_image}"
            elif "image_url" in request.extra:
                kwargs["image_url"] = request.extra["image_url"]

        # Run synchronous SDK call in thread pool
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.videos.generations(**kwargs)
            )
        except Exception as e:
            logger.error("[zhipu] Failed to create task: %s", e)
            raise

        task_id = response.id
        logger.info("[zhipu] Created generation task: %s", task_id)
        return task_id

    async def _poll_for_completion(
        self, task_id: str, poll_only: bool = False
    ) -> str | None:
        """Poll for task completion and return video URL if ready."""
        client = self._get_client()

        # Run synchronous SDK call in thread pool
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: client.videos.retrieve_videos_result(id=task_id)
            )
        except Exception as e:
            logger.error("[zhipu] Failed to poll task: %s", e)
            raise

        status = result.task_status
        logger.debug("[zhipu] Task %s status: %s", task_id, status)

        if status == "SUCCESS":
            if result.video_result and len(result.video_result) > 0:
                video_url = result.video_result[0].url
                logger.info("[zhipu] Task completed, video URL: %s", video_url[:50] + "...")
                return video_url
            else:
                raise RuntimeError(f"Task succeeded but no video_result: {result}")
        elif status == "FAILED":
            error_msg = getattr(result, "error_message", "Unknown error")
            raise RuntimeError(f"ZhipuAI generation failed (task={task_id}): {error_msg}")
        elif poll_only:
            return None
        else:
            logger.info("[zhipu] Task %s status: %s, waiting...", task_id, status)
            return None

    async def _download_video(self, video_url: str) -> bytes:
        """Download video from URL."""
        logger.info("[zhipu] Downloading video from: %s", video_url[:50] + "...")

        async with httpx.AsyncClient(timeout=_GENERATE_TIMEOUT_S) as client:
            resp = await client.get(video_url)
            resp.raise_for_status()
            return resp.content

    @staticmethod
    def _estimate_cost_for(request: GenerationRequest) -> float:
        return round(request.duration_seconds * _COST_PER_SECOND_USD, 4)


# Factory function for entry point
def create_zhipu_adapter() -> ZhipuVideoAdapter:
    """Factory function to create a ZhipuVideoAdapter instance."""
    return ZhipuVideoAdapter()
