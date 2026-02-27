"""Kling (可灵) video generation adapter.

This adapter wraps the Kling AI video generation API.
Authentication uses Access Key + Secret Key with AWS-style signature.

Environment variables:
- KLING_ACCESS_KEY / VIDEOCLAW_KLING_ACCESS_KEY
- KLING_SECRET_KEY / VIDEOCLAW_KLING_SECRET_KEY
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from collections.abc import AsyncIterator

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

# API Constants - Updated based on Kling API documentation
_BASE_URL = "https://api.klingai.com"
_VERSION = "v1"

# Kling API models
KLING_MODELS = {
    "kling-1.0": "kling-v1",
    "kling-1.5": "kling-v1-5", 
    "kling-1.6": "kling-v1",
}

# Cost estimate (USD per second) - Kling pricing varies, using rough estimate
_COST_PER_SECOND_USD = 0.03

# Timeouts
_GENERATE_TIMEOUT_S = 600.0  # 10 minutes for video generation
_LIGHT_TIMEOUT_S = 15.0


class KlingVideoAdapter:
    """Adapter for Kling (可灵) video generation API.

    Parameters
    ----------
    access_key:
        Kling Access Key. Falls back to KLING_ACCESS_KEY or VIDEOCLAW_KLING_ACCESS_KEY.
    secret_key:
        Kling Secret Key. Falls back to KLING_SECRET_KEY or VIDEOCLAW_KLING_SECRET_KEY.
    """

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        config = get_config()
        self._access_key = (
            access_key
            or os.environ.get("KLING_ACCESS_KEY")
            or config.kling_access_key
        )
        self._secret_key = (
            secret_key
            or os.environ.get("KLING_SECRET_KEY")
            or config.kling_secret_key
        )
        self._base_url = _BASE_URL

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return "kling-1.6"

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

        # Build request and get job ID
        job_id = await self._create_job(request)

        # Poll for completion
        video_url = await self._poll_for_completion(job_id)

        # Download the video
        video_data = await self._download_video(video_url)

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
        """Stream progress events while polling the Kling job."""
        self._ensure_api_key()

        yield ProgressEvent(progress=0.0, stage="submitting_job")

        # Create job
        job_id = await self._create_job(request)
        yield ProgressEvent(progress=0.1, stage="job_created")

        # Poll for completion
        poll_count = 0
        video_url = None
        while True:
            video_url = await self._poll_for_completion(job_id, poll_only=True)
            if video_url:
                break

            poll_count += 1
            progress = min(0.1 + poll_count * 0.05, 0.9)
            yield ProgressEvent(progress=progress, stage="processing")

            import asyncio

            await asyncio.sleep(3.0)

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
        """Check readiness by verifying API keys are set."""
        if not self._access_key or not self._secret_key:
            logger.warning("[kling] No API keys configured")
            return False
        # Keys are configured - assume ready
        # Note: Actual API validation would require knowing the correct endpoint
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> None:
        if not self._access_key or not self._secret_key:
            raise RuntimeError(
                "Kling API keys are required. Set KLING_ACCESS_KEY and "
                "KLING_SECRET_KEY (or VIDEOCLAW_KLING_ACCESS_KEY and "
                "VIDEOCLAW_KLING_SECRET_KEY), or pass access_key= and "
                "secret_key= to KlingVideoAdapter."
            )

    def _auth_headers(self, method: str = "POST", path: str = None) -> dict[str, str]:
        """Generate authentication headers with Bearer token.

        Uses Access Key directly as Bearer token.
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._access_key}",
        }

    def _build_payload(self, request: GenerationRequest) -> dict:
        """Translate a GenerationRequest to the Kling API body."""
        # Use correct model name for API
        api_model = KLING_MODELS.get(self.model_id, "kling-v1")
        
        payload: dict = {
            "model": api_model,
            "prompt": request.prompt,
        }
        
        # Add optional parameters only if they have valid values
        if request.width and request.height:
            # Kling expects aspect_ratio as string like "16:9"
            from math import gcd
            g = gcd(request.width, request.height)
            payload["aspect_ratio"] = f"{request.width // g}:{request.height // g}"
        
        if request.duration_seconds:
            # Kling expects duration in specific values (5, 10)
            duration = int(request.duration_seconds)
            if duration <= 5:
                payload["duration"] = "5"
            else:
                payload["duration"] = "10"

        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            payload["seed"] = request.seed

        # Handle image to video
        if request.reference_image:
            # Would need to upload the image first
            # For now, pass as base64 if supported
            pass

        logger.debug("[kling] Built payload: %s", payload)
        return payload

    async def _create_job(self, request: GenerationRequest) -> str:
        """Create a generation job and return the job ID."""
        path = f"/{_VERSION}/videos/text2video"
        headers = self._auth_headers(method="POST", path=path)
        payload = self._build_payload(request)

        logger.info("[kling] Creating job with payload: %s", payload)
        logger.debug("[kling] Auth headers: %s", {k: v[:20] + '...' if k == 'Authorization' else v for k, v in headers.items()})
        
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S, connect=30.0),
        ) as client:
            try:
                resp = await client.post(
                    path,
                    headers=headers,
                    json=payload,
                )
                
                logger.info("[kling] API response status: %d", resp.status_code)
                
                if resp.status_code != 200:
                    logger.error("[kling] API error response: %s", resp.text)
                    resp.raise_for_status()
                    
                data = resp.json()
                logger.debug("[kling] API response data: %s", data)

                # Try different response formats
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
                
            except httpx.HTTPStatusError as e:
                logger.error("[kling] HTTP error: %s - %s", e.response.status_code, e.response.text)
                raise
            except Exception as e:
                logger.error("[kling] Failed to create job: %s", e)
                raise

    async def _poll_for_completion(
        self, job_id: str, poll_only: bool = False
    ) -> str | None:
        """Poll for job completion and return video URL if ready."""
        path = f"/{_VERSION}/videos/text2video/{job_id}"
        headers = self._auth_headers(method="GET", path=path)

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_LIGHT_TIMEOUT_S),
        ) as client:
            resp = await client.get(
                path,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            
            logger.debug("[kling] Poll response: %s", data)

            # Try different response formats
            status = (
                data.get("data", {}).get("task_status")
                or data.get("data", {}).get("taskStatus")
                or data.get("task_status")
                or data.get("taskStatus")
            )

            if status in ("succeed", "SUCCESS", "complete"):
                result = data.get("data", {}).get("task_result", {})
                video_url = (
                    result.get("url")
                    or result.get("video_url")
                    or data.get("data", {}).get("url")
                )
                return video_url
            elif status in ("failed", "FAILED", "error"):
                error_msg = (
                    data.get("data", {}).get("task_status_msg")
                    or data.get("message")
                    or "unknown error"
                )
                raise RuntimeError(f"Kling generation failed (job={job_id}): {error_msg}")
            elif poll_only:
                return None
            else:
                logger.info("[kling] Job %s status: %s", job_id, status)
                return None

    async def _download_video(self, url: str) -> bytes:
        """Download video from URL."""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S)
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    @staticmethod
    def _estimate_cost_for(request: GenerationRequest) -> float:
        return round(request.duration_seconds * _COST_PER_SECOND_USD, 4)
