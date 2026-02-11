"""OpenAI Sora adapter — cloud video generation via the OpenAI API.

This adapter wraps the OpenAI video-generation endpoint (Sora).  The actual
endpoint path and payload schema will be finalised once the Sora API reaches
GA; the structure below is based on the public documentation available at the
time of writing and can be adjusted with minimal changes.

Environment variable ``OPENAI_API_KEY`` is used when no explicit key is
provided at construction time.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator

import httpx

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

_BASE_URL = "https://api.openai.com/v1"

# Rough pricing estimate: USD per second of generated video.
_COST_PER_SECOND_USD = 0.05

# Default timeout for the long-running generation call.
_GENERATE_TIMEOUT_S = 300.0

# Timeout for lightweight calls (health check, etc.).
_LIGHT_TIMEOUT_S = 15.0


class OpenAIVideoAdapter:
    """Adapter for OpenAI's Sora video-generation API.

    Parameters
    ----------
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when *None*.
    base_url:
        Override the OpenAI base URL (useful for proxies / testing).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _BASE_URL,
    ) -> None:
        self._api_key: str | None = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return "sora"

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
        """Submit a generation job and poll until completion.

        The implementation follows the expected Sora API contract:

        1. ``POST /v1/videos/generations`` — create job.
        2. ``GET  /v1/videos/generations/{id}`` — poll for status.
        3. Download the resulting video from the returned URL.
        """
        self._ensure_api_key()

        headers = self._auth_headers()
        payload = self._build_payload(request)

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S, connect=30.0),
        ) as client:
            # --- Step 1: create generation job ---
            # TODO: Update endpoint path once Sora API reaches GA.
            create_resp = await client.post(
                "/v1/videos/generations",
                headers=headers,
                json=payload,
            )
            create_resp.raise_for_status()
            job = create_resp.json()
            job_id: str = job["id"]
            logger.info("[sora] Created generation job %s", job_id)

            # --- Step 2: poll until done ---
            video_url: str | None = None
            while True:
                poll_resp = await client.get(
                    f"/v1/videos/generations/{job_id}",
                    headers=headers,
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
                else:
                    # Still processing — back off briefly.
                    import asyncio

                    await asyncio.sleep(2.0)

            # --- Step 3: download video ---
            dl_resp = await client.get(video_url)
            dl_resp.raise_for_status()
            video_data = dl_resp.content

        cost = self._estimate_cost_for(request)

        return GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": self.model_id,
                "job_id": job_id,
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
        """Stream progress events while polling the Sora job.

        Because the Sora API is poll-based rather than truly streaming, we
        emit synthetic :class:`ProgressEvent` instances between polls to keep
        the caller informed.
        """
        self._ensure_api_key()

        import asyncio

        headers = self._auth_headers()
        payload = self._build_payload(request)

        yield ProgressEvent(progress=0.0, stage="submitting_job")

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S, connect=30.0),
        ) as client:
            create_resp = await client.post(
                "/v1/videos/generations",
                headers=headers,
                json=payload,
            )
            create_resp.raise_for_status()
            job = create_resp.json()
            job_id: str = job["id"]

            yield ProgressEvent(progress=0.1, stage="job_created")

            poll_count = 0
            video_url: str | None = None
            while True:
                poll_resp = await client.get(
                    f"/v1/videos/generations/{job_id}",
                    headers=headers,
                )
                poll_resp.raise_for_status()
                status_data = poll_resp.json()
                status = status_data.get("status", "unknown")
                poll_count += 1

                # Emit progress that creeps toward 0.9 during polling.
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
                else:
                    await asyncio.sleep(2.0)

            yield ProgressEvent(progress=0.95, stage="downloading")
            dl_resp = await client.get(video_url)
            dl_resp.raise_for_status()
            video_data = dl_resp.content

        cost = self._estimate_cost_for(request)

        yield ProgressEvent(progress=1.0, stage="complete")
        yield GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": self.model_id,
                "job_id": job_id,
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
        """Check readiness by verifying the API key is set and making a
        lightweight request to the OpenAI API.
        """
        if not self._api_key:
            logger.warning("[sora] No API key configured")
            return False

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(_LIGHT_TIMEOUT_S),
            ) as client:
                resp = await client.get(
                    "/v1/models",
                    headers=self._auth_headers(),
                )
                return resp.status_code == 200
        except httpx.HTTPError:
            logger.debug("[sora] Health-check HTTP error", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "OpenAI API key is required.  Set OPENAI_API_KEY or pass "
                "api_key= to OpenAIVideoAdapter."
            )

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _build_payload(request: GenerationRequest) -> dict:
        """Translate a :class:`GenerationRequest` to the Sora API body.

        TODO: Update field names once the Sora REST API stabilises.
        """
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

        # TODO: handle reference_image / reference_video as multipart uploads
        # once the API supports image-to-video.

        return payload

    @staticmethod
    def _estimate_cost_for(request: GenerationRequest) -> float:
        return round(request.duration_seconds * _COST_PER_SECOND_USD, 4)
