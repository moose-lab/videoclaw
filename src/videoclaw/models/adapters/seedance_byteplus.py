"""Seedance 1.5 Pro video generation adapter — BytePlus ModelArk API.

Uses the official BytePlus ModelArk endpoint (international) for Seedance 1.5
Pro video generation.  Complement to the Seedance 2.0 adapter (vectorspace.cn).

API endpoints (BytePlus):
- Create task:  POST {base}/contents/generations/tasks
- Poll task:    GET  {base}/contents/generations/tasks/{task_id}

Environment variables:
- BYTEPLUS_API_KEY / VIDEOCLAW_BYTEPLUS_API_KEY
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from videoclaw.config import get_config
from videoclaw.models.protocol import (
    ExecutionMode,
    GenerationRequest,
    GenerationResult,
    ModelCapability,
    ProgressEvent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARK_MODEL_ID = "seedance-1-5-pro-251215"

_RESOLUTION_TO_RATIO: dict[tuple[int, int], str] = {
    (720, 1280): "9:16",
    (1280, 720): "16:9",
    (1024, 1024): "1:1",
    (768, 1024): "3:4",
    (1024, 768): "4:3",
}

_COST_PER_SECOND_USD = 0.04  # Seedance 1.5 Pro is slightly cheaper
_POLL_INTERVAL_S = 10.0
_POLL_TIMEOUT_S = 600.0
_HTTP_TIMEOUT_S = 30.0


class SeedanceBytePlusAdapter:
    """Adapter for Seedance 1.5 Pro via BytePlus ModelArk.

    Parameters
    ----------
    api_key:
        BytePlus ModelArk API key.  Falls back to BYTEPLUS_API_KEY or
        VIDEOCLAW_BYTEPLUS_API_KEY.
    api_base:
        Override the ModelArk base URL.
    """

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

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return "seedance-1.5-pro"

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
        self._ensure_api_key()
        task_id = await self._create_task(request)
        video_url = await self._poll_until_done(task_id)
        video_data = await self._download_video(video_url)

        return GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": _ARK_MODEL_ID,
                "task_id": task_id,
                "prompt": request.prompt[:200],
                "generated_at": time.time(),
            },
            cost_usd=self._estimate_cost_for(request),
            model_id=self.model_id,
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
        yield GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={"model": _ARK_MODEL_ID, "task_id": task_id},
            cost_usd=self._estimate_cost_for(request),
            model_id=self.model_id,
        )

    # ------------------------------------------------------------------
    # Cost & health
    # ------------------------------------------------------------------

    async def estimate_cost(self, request: GenerationRequest) -> float:
        return self._estimate_cost_for(request)

    async def health_check(self) -> bool:
        if not self._api_key:
            logger.warning("[seedance-bp] No BYTEPLUS_API_KEY configured")
            return False
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "BytePlus API key required for Seedance 1.5 Pro. "
                "Set BYTEPLUS_API_KEY or VIDEOCLAW_BYTEPLUS_API_KEY."
            )

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _build_content(self, request: GenerationRequest) -> list[dict[str, Any]]:
        """Build content array.

        BytePlus Seedance 1.5 Pro accepts HTTPS image URLs for
        image-to-video.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]

        # Pre-hosted HTTPS image URLs
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

        # Base64 images — BytePlus may support these (unlike vectorspace proxy)
        if request.reference_image:
            import base64
            b64 = base64.b64encode(request.reference_image).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        return content

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Translate GenerationRequest to BytePlus API body.

        BytePlus Seedance 1.5 Pro format::

            {
                "model": "seedance-1-5-pro-251215",
                "content": [...],
                "generate_audio": false,
                "ratio": "9:16",
                "duration": 5,
                "watermark": false
            }
        """
        res_key = (request.width, request.height)
        ratio = _RESOLUTION_TO_RATIO.get(res_key, "9:16")
        duration = max(5, min(15, int(request.duration_seconds)))

        return {
            "model": _ARK_MODEL_ID,
            "content": self._build_content(request),
            "generate_audio": False,  # 1.5 Pro audio gen is optional
            "ratio": ratio,
            "duration": duration,
            "watermark": False,
        }

    async def _create_task(self, request: GenerationRequest) -> str:
        """POST {base}/contents/generations/tasks"""
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
                    headers=self._auth_headers(),
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
        """GET {base}/contents/generations/tasks/{task_id}"""
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            resp = await client.get(
                f"{self._api_base}/contents/generations/tasks/{task_id}",
                headers=self._auth_headers(),
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
        """Extract video URL from BytePlus task result.

        BytePlus format: ``{"content": {"video_url": "https://tos-..."}}``
        """
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

    @staticmethod
    def _estimate_cost_for(request: GenerationRequest) -> float:
        return round(request.duration_seconds * _COST_PER_SECOND_USD, 4)
