"""Seedance (豆包) video generation adapter.

Wraps the Seedance 2.0 video generation API (vectorspace.cn proxy for
ByteDance Volcengine Ark).

Seedance 2.0 key capabilities:
- Text-to-video and image-to-video (first_frame / last_frame)
- Universal Reference architecture for character consistency
- Native 9:16 vertical video support (4-15 seconds per clip)
- Native audio co-generation (Dual-Branch Diffusion Transformer)

API endpoints (vectorspace.cn):
- Create task:  POST {base}/api/v1/doubao/create
- Query result: POST {base}/api/v1/doubao/get_result

Environment variables:
- ARK_API_KEY / VIDEOCLAW_ARK_API_KEY
"""

from __future__ import annotations

import asyncio
import base64
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

_DEFAULT_BASE_URL = "https://sd2.vectorspace.cn"

# Ark model identifier used by the API
_ARK_MODEL_ID = "doubao-seedance-2-0-260128"

# Width×Height → ratio string accepted by the API
_RESOLUTION_TO_RATIO: dict[tuple[int, int], str] = {
    (720, 1280): "9:16",   # TikTok vertical
    (1280, 720): "16:9",   # horizontal
    (1024, 1024): "1:1",   # square
    (768, 1024): "3:4",
    (1024, 768): "4:3",
}

# Cost estimate (USD per second)
_COST_PER_SECOND_USD = 0.05

# Polling configuration
_POLL_INTERVAL_S = 5.0
_POLL_TIMEOUT_S = 300.0  # 5 minutes max
_HTTP_TIMEOUT_S = 30.0


class SeedanceVideoAdapter:
    """Adapter for ByteDance Seedance 2.0 video generation.

    Parameters
    ----------
    api_key:
        API key for the Seedance endpoint. Falls back to ARK_API_KEY or
        VIDEOCLAW_ARK_API_KEY environment variable.
    base_url:
        Override the API base URL (defaults to vectorspace.cn proxy).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        config = get_config()
        self._api_key = (
            api_key
            or os.environ.get("ARK_API_KEY")
            or config.ark_api_key
        )
        self._base_url = (base_url or config.seedance_base_url or _DEFAULT_BASE_URL).rstrip("/")

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return "seedance-2.0"

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
        """Submit a video generation task and poll until completion."""
        self._ensure_api_key()

        task_id = await self._create_task(request)
        video_url = await self._poll_until_done(task_id)
        video_data = await self._download_video(video_url)

        cost = self._estimate_cost_for(request)

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
            cost_usd=cost,
            model_id=self.model_id,
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        """Yield progress events while polling the task."""
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
                raise RuntimeError(f"Seedance generation failed (task={task_id})")

            progress = min(0.1 + (elapsed / _POLL_TIMEOUT_S) * 0.8, 0.9)
            yield ProgressEvent(progress=progress, stage="processing")
        else:
            raise TimeoutError(
                f"Seedance generation timed out after {_POLL_TIMEOUT_S}s"
            )

        yield ProgressEvent(progress=0.95, stage="downloading")
        video_data = await self._download_video(video_url)

        cost = self._estimate_cost_for(request)

        yield ProgressEvent(progress=1.0, stage="complete")
        yield GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": _ARK_MODEL_ID,
                "task_id": task_id,
                "prompt": request.prompt[:200],
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
        """Return True if an API key is configured."""
        if not self._api_key:
            logger.warning("[seedance] No ARK_API_KEY configured")
            return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "Seedance API key required. "
                "Set ARK_API_KEY or VIDEOCLAW_ARK_API_KEY."
            )

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _build_content(self, request: GenerationRequest) -> list[dict[str, Any]]:
        """Build the multimodal ``content`` array.

        Format::

            [
                {"type": "text", "text": "prompt"},
                {"type": "image_url", "image_url": {"url": "..."}, "role": "first_frame"},
                {"type": "image_url", "image_url": {"url": "..."}, "role": "reference_image"},
            ]
        """
        content: list[dict[str, Any]] = []

        # Text prompt
        content.append({"type": "text", "text": request.prompt})

        # Primary reference image → first_frame (image-to-video)
        if request.reference_image:
            b64 = base64.b64encode(request.reference_image).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
                "role": "first_frame",
            })

        # Additional character reference images
        extra_refs: dict[str, bytes] | None = request.extra.get(
            "additional_references"
        )
        if extra_refs:
            for _name, img_bytes in extra_refs.items():
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                    "role": "reference_image",
                })

        return content

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Translate a GenerationRequest into the API request body.

        API body format::

            {
                "model": "doubao-seedance-2-0-260128",
                "content": [...],
                "generate_audio": true,
                "ratio": "9:16",
                "resolution": "720p",
                "duration": 5
            }
        """
        # Resolve aspect ratio string
        res_key = (request.width, request.height)
        ratio = _RESOLUTION_TO_RATIO.get(res_key, "9:16")

        # Duration clamped to Seedance's 4-15s range
        duration = max(4, min(15, int(request.duration_seconds)))

        payload: dict[str, Any] = {
            "model": _ARK_MODEL_ID,
            "content": self._build_content(request),
            "generate_audio": True,
            "ratio": ratio,
            "resolution": "720p",
            "duration": duration,
        }

        if request.seed is not None:
            payload["seed"] = request.seed

        return payload

    async def _create_task(self, request: GenerationRequest) -> str:
        """Submit a video generation task.

        ``POST {base}/api/v1/doubao/create``
        """
        payload = self._build_payload(request)

        logger.info(
            "[seedance] Creating task: model=%s ratio=%s duration=%s",
            payload["model"],
            payload.get("ratio"),
            payload.get("duration"),
        )

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            resp = await client.post(
                f"{self._base_url}/api/v1/doubao/create",
                headers=self._auth_headers(),
                json=payload,
            )

            if resp.status_code != 200:
                logger.error(
                    "[seedance] Task creation failed %d: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                resp.raise_for_status()

            data = resp.json()
            task_id = data.get("id") or data.get("task_id")

            if not task_id:
                raise RuntimeError(
                    f"No task_id in Seedance response: {data}"
                )

            logger.info("[seedance] Created task %s", task_id)
            return task_id

    async def _check_task(self, task_id: str) -> tuple[str, str | None]:
        """Query task status.

        ``POST {base}/api/v1/doubao/get_result`` with ``{"id": task_id}``

        Returns ``(status, video_url | None)`` where status is one of
        ``"processing"``, ``"done"``, or ``"failed"``.
        """
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            resp = await client.post(
                f"{self._base_url}/api/v1/doubao/get_result",
                headers=self._auth_headers(),
                json={"id": task_id},
            )
            resp.raise_for_status()
            data = resp.json()

        raw_status = (data.get("status") or "").lower()

        logger.debug("[seedance] Task %s status: %s", task_id, raw_status)

        if raw_status in ("succeeded", "success", "completed", "done"):
            video_url = self._extract_video_url(data)
            return "done", video_url

        if raw_status in ("failed", "error", "cancelled"):
            error = data.get("error") or "unknown"
            logger.error("[seedance] Task %s failed: %s", task_id, error)
            return "failed", None

        return "processing", None

    async def _poll_until_done(self, task_id: str) -> str:
        """Poll task status until completion, returning the video URL."""
        elapsed = 0.0
        while elapsed < _POLL_TIMEOUT_S:
            await asyncio.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S

            status, video_url = await self._check_task(task_id)

            if status == "done":
                if not video_url:
                    raise RuntimeError(
                        f"Seedance task {task_id} completed but no video URL"
                    )
                return video_url

            if status == "failed":
                raise RuntimeError(
                    f"Seedance generation failed (task={task_id})"
                )

            logger.info(
                "[seedance] Task %s: %s (%.0fs elapsed)",
                task_id, status, elapsed,
            )

        raise TimeoutError(
            f"Seedance generation timed out after {_POLL_TIMEOUT_S}s "
            f"(task={task_id})"
        )

    @staticmethod
    def _extract_video_url(data: dict[str, Any]) -> str | None:
        """Extract video URL from the task result.

        Known response format::

            {
                "id": "task_...",
                "status": "succeeded",
                "content": {"video_url": "https://ark-acg-cn-beijing..."},
                ...
            }
        """
        # Primary format: content.video_url
        content = data.get("content")
        if isinstance(content, dict):
            if url := content.get("video_url"):
                return url

        # Fallback: data.video_url
        if url := data.get("video_url"):
            return url

        # Fallback: output.video_url
        if url := data.get("output", {}).get("video_url"):
            return url

        return None

    async def _download_video(self, url: str) -> bytes:
        """Download video bytes from URL."""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    @staticmethod
    def _estimate_cost_for(request: GenerationRequest) -> float:
        return round(request.duration_seconds * _COST_PER_SECOND_USD, 4)
