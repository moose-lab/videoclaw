"""ZhipuAI (智谱清影) video generation adapter.

Uses the official zhipuai SDK for CogVideoX API.

Environment variables:
- ZHIPU_API_KEY / VIDEOCLAW_ZHIPU_API_KEY
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator

import httpx
from zhipuai import ZhipuAI

from videoclaw.config import get_config
from videoclaw.models.adapters.base import BaseCloudVideoAdapter
from videoclaw.models.protocol import (
    GenerationRequest,
    GenerationResult,
    ModelCapability,
    ProgressEvent,
)

logger = logging.getLogger(__name__)

ZHIPU_MODELS = {
    "cogvideox-flash": "cogvideox-flash",
    "cogvideox": "cogvideox",
}

_GENERATE_TIMEOUT_S = 600.0
_POLL_INTERVAL_S = 5.0


class ZhipuVideoAdapter(BaseCloudVideoAdapter):
    """Adapter for ZhipuAI (智谱清影) CogVideoX video generation API."""

    _COST_PER_SECOND_USD = 0.015
    _ADAPTER_NAME = "zhipu"

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
        if self._client is None:
            self._ensure_api_key()
            self._client = ZhipuAI(api_key=self._api_key)
        return self._client

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def capabilities(self) -> list[ModelCapability]:
        return [ModelCapability.TEXT_TO_VIDEO, ModelCapability.IMAGE_TO_VIDEO]

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_api_key()

        task_id = await self._create_task(request)
        video_url = await self._poll_for_completion(task_id)
        video_data = await self._download_video(video_url)

        return self._build_result(
            video_data, request,
            extra_metadata={"task_id": task_id},
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        self._ensure_api_key()

        yield ProgressEvent(progress=0.0, stage="submitting_task")

        task_id = await self._create_task(request)
        yield ProgressEvent(progress=0.1, stage="task_created")

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

        video_data = await self._download_video(video_url)

        yield ProgressEvent(progress=1.0, stage="complete")
        yield self._build_result(
            video_data, request,
            extra_metadata={"task_id": task_id},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_size(self, request: GenerationRequest) -> str:
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
        client = self._get_client()
        size = self._build_size(request)

        logger.info("[zhipu] Creating task with model: %s, size: %s", self._model, size)

        kwargs: dict = {
            "model": self._model,
            "prompt": request.prompt,
            "size": size,
            "quality": "speed" if "flash" in self._model else "quality",
            "with_audio": False,
        }

        if request.reference_image:
            if isinstance(request.reference_image, bytes):
                b64_image = base64.b64encode(request.reference_image).decode("utf-8")
                kwargs["image_url"] = f"data:image/png;base64,{b64_image}"
            elif "image_url" in request.extra:
                kwargs["image_url"] = request.extra["image_url"]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.videos.generations(**kwargs),
        )

        task_id = response.id
        logger.info("[zhipu] Created generation task: %s", task_id)
        return task_id

    async def _poll_for_completion(
        self, task_id: str, poll_only: bool = False
    ) -> str | None:
        client = self._get_client()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: client.videos.retrieve_videos_result(id=task_id),
        )

        status = result.task_status
        logger.debug("[zhipu] Task %s status: %s", task_id, status)

        if status == "SUCCESS":
            if result.video_result and len(result.video_result) > 0:
                video_url = result.video_result[0].url
                logger.info("[zhipu] Task completed, video URL: %s", video_url[:50] + "...")
                return video_url
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
        logger.info("[zhipu] Downloading video from: %s", video_url[:50] + "...")
        async with httpx.AsyncClient(timeout=_GENERATE_TIMEOUT_S) as client:
            resp = await client.get(video_url)
            resp.raise_for_status()
            return resp.content
