"""Kling (可灵) video generation adapter.

Uses Access Key + Secret Key authentication.

Environment variables:
- KLING_ACCESS_KEY / VIDEOCLAW_KLING_ACCESS_KEY
- KLING_SECRET_KEY / VIDEOCLAW_KLING_SECRET_KEY
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from math import gcd

import httpx

from videoclaw.models.adapters.base import BaseCloudVideoAdapter
from videoclaw.utils import resolve_credential
from videoclaw.models.protocol import (
    GenerationRequest,
    GenerationResult,
    ModelCapability,
    ProgressEvent,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.klingai.com"
_VERSION = "v1"

KLING_MODELS = {
    "kling-1.0": "kling-v1",
    "kling-1.5": "kling-v1-5",
    "kling-1.6": "kling-v1",
}

_GENERATE_TIMEOUT_S = 600.0
_LIGHT_TIMEOUT_S = 15.0


class KlingVideoAdapter(BaseCloudVideoAdapter):
    """Adapter for Kling (可灵) video generation API."""

    _COST_PER_SECOND_USD = 0.03
    _ADAPTER_NAME = "kling"

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._access_key = resolve_credential(
            explicit=access_key, env_vars="KLING_ACCESS_KEY",
            config_attr="kling_access_key",
        )
        self._secret_key = resolve_credential(
            explicit=secret_key, env_vars="KLING_SECRET_KEY",
            config_attr="kling_secret_key",
        )
        # BaseCloudVideoAdapter uses _api_key for health_check
        self._api_key = self._access_key
        self._base_url = _BASE_URL

    @property
    def model_id(self) -> str:
        return "kling-1.6"

    @property
    def capabilities(self) -> list[ModelCapability]:
        return [ModelCapability.TEXT_TO_VIDEO, ModelCapability.IMAGE_TO_VIDEO]

    def _ensure_api_key(self) -> None:
        if not self._access_key or not self._secret_key:
            raise RuntimeError(
                "Kling API keys are required. Set KLING_ACCESS_KEY and "
                "KLING_SECRET_KEY (or VIDEOCLAW_ prefixed variants)."
            )

    def _auth_headers(self, method: str = "POST", path: str = None) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._access_key}",
        }

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_api_key()

        job_id = await self._create_job(request)
        video_url = await self._poll_for_completion(job_id)
        video_data = await self._download_video(video_url)

        return self._build_result(
            video_data, request, job_id=job_id,
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        self._ensure_api_key()

        yield ProgressEvent(progress=0.0, stage="submitting_job")

        job_id = await self._create_job(request)
        yield ProgressEvent(progress=0.1, stage="job_created")

        poll_count = 0
        video_url = None
        while True:
            video_url = await self._poll_for_completion(job_id, poll_only=True)
            if video_url:
                break

            poll_count += 1
            progress = min(0.1 + poll_count * 0.05, 0.9)
            yield ProgressEvent(progress=progress, stage="processing")

            await asyncio.sleep(3.0)

        yield ProgressEvent(progress=0.95, stage="downloading")
        video_data = await self._download_video(video_url)

        yield ProgressEvent(progress=1.0, stage="complete")
        yield self._build_result(
            video_data, request, job_id=job_id,
        )

    async def health_check(self) -> bool:
        if not self._access_key or not self._secret_key:
            logger.warning("[kling] No API keys configured")
            return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(self, request: GenerationRequest) -> dict:
        api_model = KLING_MODELS.get(self.model_id, "kling-v1")

        payload: dict = {
            "model": api_model,
            "prompt": request.prompt,
        }

        if request.width and request.height:
            g = gcd(request.width, request.height)
            payload["aspect_ratio"] = f"{request.width // g}:{request.height // g}"

        if request.duration_seconds:
            duration = int(request.duration_seconds)
            payload["duration"] = "5" if duration <= 5 else "10"

        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            payload["seed"] = request.seed

        if request.reference_image:
            b64 = base64.b64encode(request.reference_image).decode("utf-8")
            payload["image"] = f"data:image/png;base64,{b64}"

        logger.debug("[kling] Built payload: %s", payload)
        return payload

    async def _create_job(self, request: GenerationRequest) -> str:
        path = (
            f"/{_VERSION}/videos/image2video"
            if request.reference_image
            else f"/{_VERSION}/videos/text2video"
        )
        headers = self._auth_headers(method="POST", path=path)
        payload = self._build_payload(request)

        logger.info("[kling] Creating job with payload: %s", payload)

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S, connect=30.0),
        ) as client:
            resp = await client.post(path, headers=headers, json=payload)
            if resp.status_code != 200:
                logger.error("[kling] API error response: %s", resp.text)
            resp.raise_for_status()

            data = resp.json()

            job_id = (
                data.get("data", {}).get("task_id")
                or data.get("data", {}).get("taskId")
                or data.get("taskId")
                or data.get("task_id")
            )

            if not job_id:
                raise RuntimeError(f"Failed to get job ID from response: {data}")

            logger.info("[kling] Created generation job %s", job_id)
            return job_id

    async def _poll_for_completion(
        self, job_id: str, poll_only: bool = False
    ) -> str | None:
        path = f"/{_VERSION}/videos/text2video/{job_id}"
        headers = self._auth_headers(method="GET", path=path)

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_LIGHT_TIMEOUT_S),
        ) as client:
            resp = await client.get(path, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            status = (
                data.get("data", {}).get("task_status")
                or data.get("data", {}).get("taskStatus")
                or data.get("task_status")
                or data.get("taskStatus")
            )

            if status in ("succeed", "SUCCESS", "complete"):
                result = data.get("data", {}).get("task_result", {})
                return (
                    result.get("url")
                    or result.get("video_url")
                    or data.get("data", {}).get("url")
                )
            elif status in ("failed", "FAILED", "error"):
                error_msg = (
                    data.get("data", {}).get("task_status_msg")
                    or data.get("message")
                    or "unknown error"
                )
                raise RuntimeError(
                    f"Kling generation failed (job={job_id}): {error_msg}"
                )
            elif poll_only:
                return None
            else:
                logger.info("[kling] Job %s status: %s", job_id, status)
                return None

    async def _download_video(self, url: str) -> bytes:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S)
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
