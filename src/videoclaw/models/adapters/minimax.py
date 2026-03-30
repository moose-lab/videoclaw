"""MiniMax (海螺AI) video generation adapter.

This adapter wraps the MiniMax video generation API (Hailuo video).

Environment variables:
- MINIMAX_API_KEY / VIDEOCLAW_MINIMAX_API_KEY
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from collections.abc import AsyncIterator

import httpx

from videoclaw.config import get_config
from videoclaw.models.adapters.base import BaseCloudVideoAdapter
from videoclaw.models.protocol import (
    GenerationRequest,
    GenerationResult,
    ModelCapability,
    ProgressEvent,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.minimax.io"
_VERSION = "v1"

MINIMAX_MODELS = {
    "minimax-hailuo-2.3": "MiniMax-Hailuo-2.3",
    "minimax-hailuo-2.3-fast": "MiniMax-Hailuo-2.3Fast",
    "minimax-hailuo-02": "MiniMax-Hailuo-02",
    "minimax-s2v-01": "S2V-01",
}

_GENERATE_TIMEOUT_S = 600.0
_LIGHT_TIMEOUT_S = 30.0
_POLL_INTERVAL_S = 10.0


class MiniMaxVideoAdapter(BaseCloudVideoAdapter):
    """Adapter for MiniMax (海螺AI) video generation API."""

    _COST_PER_SECOND_USD = 0.02
    _ADAPTER_NAME = "minimax"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "minimax-hailuo-2.3",
    ) -> None:
        config = get_config()
        self._api_key = (
            api_key
            or os.environ.get("MINIMAX_API_KEY")
            or config.minimax_api_key
        )
        self._model = model
        self._base_url = _BASE_URL

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def capabilities(self) -> list[ModelCapability]:
        if self._model == "minimax-s2v-01":
            return [ModelCapability.IMAGE_TO_VIDEO]
        return [ModelCapability.TEXT_TO_VIDEO, ModelCapability.IMAGE_TO_VIDEO]

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Submit a generation job and poll until completion."""
        self._ensure_api_key()

        task_id = await self._create_task(request)
        file_id = await self._poll_for_completion(task_id)
        video_data = await self._download_video(file_id)

        return self._build_result(
            video_data, request,
            extra_metadata={"task_id": task_id, "file_id": file_id},
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        """Stream progress events while polling the MiniMax job."""
        self._ensure_api_key()

        yield ProgressEvent(progress=0.0, stage="submitting_task")

        task_id = await self._create_task(request)
        yield ProgressEvent(progress=0.1, stage="task_created")

        poll_count = 0
        file_id = None
        while True:
            file_id = await self._poll_for_completion(task_id, poll_only=True)
            if file_id:
                break

            poll_count += 1
            progress = min(0.1 + poll_count * 0.05, 0.9)
            yield ProgressEvent(progress=progress, stage="processing")

            await asyncio.sleep(_POLL_INTERVAL_S)

        yield ProgressEvent(progress=0.95, stage="downloading")

        video_data = await self._download_video(file_id)

        yield ProgressEvent(progress=1.0, stage="complete")
        yield self._build_result(
            video_data, request,
            extra_metadata={"task_id": task_id, "file_id": file_id},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(self, request: GenerationRequest) -> dict:
        api_model = MINIMAX_MODELS.get(self._model, "MiniMax-Hailuo-2.3")

        payload: dict = {
            "model": api_model,
            "prompt": request.prompt,
        }

        duration = int(request.duration_seconds)
        payload["duration"] = 6 if duration <= 6 else 10

        if request.height >= 1080:
            payload["resolution"] = "1080P"
        else:
            payload["resolution"] = "768P"

        if request.reference_image:
            if "first_frame_image_url" in request.extra:
                payload["first_frame_image"] = request.extra["first_frame_image_url"]
            elif "first_frame_image" in request.extra:
                payload["first_frame_image"] = request.extra["first_frame_image"]
            elif "first_frame_image" not in payload:
                b64 = base64.b64encode(request.reference_image).decode("utf-8")
                payload["first_frame_image"] = f"data:image/png;base64,{b64}"

        if self._model == "minimax-s2v-01" and request.reference_image:
            if "subject_reference" in request.extra:
                payload["subject_reference"] = request.extra["subject_reference"]
            elif "subject_reference" not in payload:
                b64 = base64.b64encode(request.reference_image).decode("utf-8")
                payload["subject_reference"] = [
                    {"image": f"data:image/png;base64,{b64}"}
                ]

        logger.debug("[minimax] Built payload: %s", payload)
        return payload

    async def _create_task(self, request: GenerationRequest) -> str:
        path = f"/{_VERSION}/video_generation"
        headers = self._bearer_headers()
        payload = self._build_payload(request)

        logger.info("[minimax] Creating task with model: %s", payload.get("model"))

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S, connect=30.0),
        ) as client:
            resp = await client.post(path, headers=headers, json=payload)
            if resp.status_code != 200:
                logger.error("[minimax] API error response: %s", resp.text)
            resp.raise_for_status()

            data = resp.json()
            task_id = data.get("task_id")
            if not task_id:
                raise RuntimeError(f"Failed to get task_id from response: {data}")

            logger.info("[minimax] Created generation task %s", task_id)
            return task_id

    async def _poll_for_completion(
        self, task_id: str, poll_only: bool = False
    ) -> str | None:
        path = f"/{_VERSION}/query/video_generation"
        headers = self._bearer_headers()
        params = {"task_id": task_id}

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_LIGHT_TIMEOUT_S),
        ) as client:
            resp = await client.get(path, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status")

            if status == "Success":
                file_id = data.get("file_id")
                if not file_id:
                    raise RuntimeError(f"Task succeeded but no file_id: {data}")
                return file_id
            elif status == "Fail":
                error_msg = data.get("error_message", "Unknown error")
                raise RuntimeError(
                    f"MiniMax generation failed (task={task_id}): {error_msg}"
                )
            elif poll_only:
                logger.debug("[minimax] Task %s status: %s", task_id, status)
                return None
            else:
                logger.info("[minimax] Task %s status: %s, waiting...", task_id, status)
                return None

    async def _download_video(self, file_id: str) -> bytes:
        path = f"/{_VERSION}/files/retrieve"
        headers = self._bearer_headers()
        params = {"file_id": file_id}

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S),
        ) as client:
            resp = await client.get(path, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            download_url = data.get("file", {}).get("download_url")
            if not download_url:
                raise RuntimeError(f"No download_url in response: {data}")

            logger.info("[minimax] Downloading video from: %s", download_url[:50] + "...")

            video_resp = await client.get(download_url)
            video_resp.raise_for_status()
            return video_resp.content
