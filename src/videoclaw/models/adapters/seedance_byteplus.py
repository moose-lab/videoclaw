"""Seedance 1.5 Pro video generation adapter — BytePlus ModelArk API.

Uses the official BytePlus ModelArk endpoint (international) for Seedance 1.5
Pro video generation.

Environment variables:
- BYTEPLUS_API_KEY / VIDEOCLAW_BYTEPLUS_API_KEY
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

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

_ARK_MODEL_ID = "seedance-1-5-pro-251215"

_RESOLUTION_TO_RATIO: dict[tuple[int, int], str] = {
    (720, 1280): "9:16",
    (1280, 720): "16:9",
    (1024, 1024): "1:1",
    (768, 1024): "3:4",
    (1024, 768): "4:3",
}

_POLL_INTERVAL_S = 10.0
_POLL_TIMEOUT_S = 600.0
_HTTP_TIMEOUT_S = 30.0


class SeedanceBytePlusAdapter(BaseCloudVideoAdapter):
    """Adapter for Seedance 1.5 Pro via BytePlus ModelArk."""

    _COST_PER_SECOND_USD = 0.04
    _ADAPTER_NAME = "seedance-bp"

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        config = get_config()
        self._api_key = (
            api_key
            or os.environ.get("BYTEPLUS_API_KEY")
            or config.byteplus_api_key
        )
        self._api_base = (api_base or config.byteplus_api_base).rstrip("/")

    @property
    def model_id(self) -> str:
        return "seedance-1.5-pro"

    @property
    def capabilities(self) -> list[ModelCapability]:
        return [ModelCapability.TEXT_TO_VIDEO, ModelCapability.IMAGE_TO_VIDEO]

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_api_key()
        task_id = await self._create_task(request)
        video_url = await self._poll_until_done(task_id)
        video_data = await self._download_video(video_url)

        return self._build_result(
            video_data, request,
            extra_metadata={"task_id": task_id, "ark_model": _ARK_MODEL_ID},
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        self._ensure_api_key()

        yield ProgressEvent(progress=0.0, stage="submitting")

        task_id = await self._create_task(request)
        yield ProgressEvent(progress=0.1, stage="task_created")

        elapsed = 0.0
        video_url: str | None = None
        while elapsed < _POLL_TIMEOUT_S:
            await asyncio.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S
            status, video_url = await self._check_task(task_id)

            if status == "done" and video_url:
                break
            if status == "failed":
                raise RuntimeError(f"Seedance 1.5 Pro failed (task={task_id})")

            yield ProgressEvent(
                progress=min(0.1 + (elapsed / _POLL_TIMEOUT_S) * 0.8, 0.9),
                stage="processing",
            )
        else:
            raise TimeoutError(f"Seedance 1.5 Pro timed out after {_POLL_TIMEOUT_S}s")

        yield ProgressEvent(progress=0.95, stage="downloading")
        video_data = await self._download_video(video_url)

        yield ProgressEvent(progress=1.0, stage="complete")
        yield self._build_result(
            video_data, request,
            extra_metadata={"task_id": task_id, "ark_model": _ARK_MODEL_ID},
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_content(self, request: GenerationRequest) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]

        image_urls: list[dict[str, str]] | None = request.extra.get("image_urls")
        if image_urls:
            for img_info in image_urls:
                url = img_info.get("url", "")
                if url.startswith("http"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": url},
                    })
            return content

        if request.reference_image:
            b64 = base64.b64encode(request.reference_image).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        return content

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        res_key = (request.width, request.height)
        ratio = _RESOLUTION_TO_RATIO.get(res_key, "9:16")
        duration = max(5, min(15, int(request.duration_seconds)))

        return {
            "model": _ARK_MODEL_ID,
            "content": self._build_content(request),
            "generate_audio": False,
            "ratio": ratio,
            "duration": duration,
            "watermark": False,
        }

    async def _create_task(self, request: GenerationRequest) -> str:
        payload = self._build_payload(request)

        logger.info(
            "[seedance-bp] Creating task: model=%s ratio=%s duration=%s",
            payload["model"], payload.get("ratio"), payload.get("duration"),
        )

        max_retries = 3
        for attempt in range(max_retries):
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                resp = await client.post(
                    f"{self._api_base}/contents/generations/tasks",
                    headers=self._bearer_headers(),
                    json=payload,
                )

                if resp.status_code == 429:
                    wait = min(15.0 * (attempt + 1), 60.0)
                    logger.warning("[seedance-bp] Rate limited, waiting %.0fs", wait)
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.error("[seedance-bp] Create failed %d: %s",
                                 resp.status_code, resp.text[:500])
                resp.raise_for_status()

                data = resp.json()
                task_id = data.get("id") or data.get("task_id")
                if not task_id:
                    raise RuntimeError(f"No task_id in response: {data}")

                logger.info("[seedance-bp] Created task %s", task_id)
                return task_id

        raise RuntimeError("Seedance 1.5 Pro task creation failed after retries")

    async def _check_task(self, task_id: str) -> tuple[str, str | None]:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            resp = await client.get(
                f"{self._api_base}/contents/generations/tasks/{task_id}",
                headers=self._bearer_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        raw_status = (data.get("status") or "").lower()
        logger.debug("[seedance-bp] Task %s: %s", task_id, raw_status)

        if raw_status in ("succeeded", "success", "completed"):
            video_url = self._extract_video_url(data)
            return "done", video_url
        if raw_status in ("failed", "error", "cancelled"):
            return "failed", None
        return "processing", None

    async def _poll_until_done(self, task_id: str) -> str:
        elapsed = 0.0
        while elapsed < _POLL_TIMEOUT_S:
            await asyncio.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S

            status, video_url = await self._check_task(task_id)
            if status == "done":
                if not video_url:
                    raise RuntimeError(f"Task {task_id} done but no video URL")
                return video_url
            if status == "failed":
                raise RuntimeError(f"Seedance 1.5 Pro failed (task={task_id})")

            logger.info("[seedance-bp] Task %s: %s (%.0fs)", task_id, status, elapsed)

        raise TimeoutError(f"Timed out after {_POLL_TIMEOUT_S}s (task={task_id})")

    @staticmethod
    def _extract_video_url(data: dict[str, Any]) -> str | None:
        content = data.get("content")
        if isinstance(content, dict):
            if url := content.get("video_url"):
                return url
        if url := data.get("video_url"):
            return url
        return None

    async def _download_video(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
