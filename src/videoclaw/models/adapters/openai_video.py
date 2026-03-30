"""OpenAI Sora adapter — cloud video generation via the OpenAI API.

Environment variable ``OPENAI_API_KEY`` is used when no explicit key is
provided at construction time.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

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

_BASE_URL = "https://api.openai.com/v1"
_GENERATE_TIMEOUT_S = 300.0
_LIGHT_TIMEOUT_S = 15.0


class OpenAIVideoAdapter(BaseCloudVideoAdapter):
    """Adapter for OpenAI's Sora video-generation API."""

    _COST_PER_SECOND_USD = 0.05
    _ADAPTER_NAME = "sora"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _BASE_URL,
    ) -> None:
        self._api_key = resolve_credential(
            explicit=api_key, env_vars="OPENAI_API_KEY", config_attr="openai_api_key",
        )
        self._base_url = base_url.rstrip("/")

    @property
    def model_id(self) -> str:
        return "sora"

    @property
    def capabilities(self) -> list[ModelCapability]:
        return [ModelCapability.TEXT_TO_VIDEO, ModelCapability.IMAGE_TO_VIDEO]

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Submit a generation job and poll until completion."""
        self._ensure_api_key()

        headers = self._bearer_headers()
        payload = self._build_payload(request)

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S, connect=30.0),
        ) as client:
            create_resp = await client.post(
                "/v1/videos/generations", headers=headers, json=payload,
            )
            create_resp.raise_for_status()
            job = create_resp.json()
            job_id: str = job["id"]
            logger.info("[sora] Created generation job %s", job_id)

            video_url: str | None = None
            while True:
                poll_resp = await client.get(
                    f"/v1/videos/generations/{job_id}", headers=headers,
                )
                poll_resp.raise_for_status()
                status_data = poll_resp.json()
                status = status_data.get("status", "unknown")

                if status == "completed":
                    video_url = status_data["output"]["url"]
                    break
                elif status == "failed":
                    error_msg = status_data.get("error", {}).get(
                        "message", "unknown error"
                    )
                    raise RuntimeError(
                        f"Sora generation failed (job={job_id}): {error_msg}"
                    )
                await asyncio.sleep(2.0)

            dl_resp = await client.get(video_url)
            dl_resp.raise_for_status()

        return self._build_result(
            dl_resp.content, request, job_id=job_id,
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        """Stream progress events while polling the Sora job."""
        self._ensure_api_key()

        headers = self._bearer_headers()
        payload = self._build_payload(request)

        yield ProgressEvent(progress=0.0, stage="submitting_job")

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S, connect=30.0),
        ) as client:
            create_resp = await client.post(
                "/v1/videos/generations", headers=headers, json=payload,
            )
            create_resp.raise_for_status()
            job = create_resp.json()
            job_id: str = job["id"]

            yield ProgressEvent(progress=0.1, stage="job_created")

            poll_count = 0
            video_url: str | None = None
            while True:
                poll_resp = await client.get(
                    f"/v1/videos/generations/{job_id}", headers=headers,
                )
                poll_resp.raise_for_status()
                status_data = poll_resp.json()
                status = status_data.get("status", "unknown")
                poll_count += 1

                progress = min(0.1 + poll_count * 0.05, 0.9)
                yield ProgressEvent(progress=progress, stage=f"polling ({status})")

                if status == "completed":
                    video_url = status_data["output"]["url"]
                    break
                elif status == "failed":
                    error_msg = status_data.get("error", {}).get(
                        "message", "unknown error"
                    )
                    raise RuntimeError(
                        f"Sora generation failed (job={job_id}): {error_msg}"
                    )
                await asyncio.sleep(2.0)

            yield ProgressEvent(progress=0.95, stage="downloading")
            dl_resp = await client.get(video_url)
            dl_resp.raise_for_status()

        yield ProgressEvent(progress=1.0, stage="complete")
        yield self._build_result(
            dl_resp.content, request, job_id=job_id,
        )

    async def health_check(self) -> bool:
        if not self._api_key:
            logger.warning("[sora] No API key configured")
            return False
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(_LIGHT_TIMEOUT_S),
            ) as client:
                resp = await client.get(
                    "/v1/models", headers=self._bearer_headers(),
                )
                return resp.status_code == 200
        except httpx.HTTPError:
            logger.debug("[sora] Health-check HTTP error", exc_info=True)
            return False

    @staticmethod
    def _build_payload(request: GenerationRequest) -> dict:
        payload: dict = {
            "model": "sora",
            "prompt": request.prompt,
            "size": f"{request.width}x{request.height}",
            "duration": request.duration_seconds,
            "fps": request.fps,
        }
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.style_preset:
            payload["style"] = request.style_preset
        if request.extra:
            payload.update(request.extra)
        return payload
