"""MiniMax (海螺AI) video generation adapter.

This adapter wraps the MiniMax video generation API (Hailuo video).
Authentication uses API Key with Bearer token.

Environment variables:
- MINIMAX_API_KEY / VIDEOCLAW_MINIMAX_API_KEY

API Documentation: https://platform.minimaxi.com/docs/guides/video-generation
"""

from __future__ import annotations

import asyncio
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

# API Constants
_BASE_URL = "https://api.minimax.io"
_VERSION = "v1"

# MiniMax API models
MINIMAX_MODELS = {
    "minimax-hailuo-2.3": "MiniMax-Hailuo-2.3",
    "minimax-hailuo-2.3-fast": "MiniMax-Hailuo-2.3Fast",
    "minimax-hailuo-02": "MiniMax-Hailuo-02",
    "minimax-s2v-01": "S2V-01",  # Subject reference video
}

# Cost estimate (USD per second) - MiniMax has free tier, this is for reference
_COST_PER_SECOND_USD = 0.02

# Timeouts
_GENERATE_TIMEOUT_S = 600.0  # 10 minutes for video generation
_LIGHT_TIMEOUT_S = 30.0
_POLL_INTERVAL_S = 10.0  # Recommended polling interval


class MiniMaxVideoAdapter:
    """Adapter for MiniMax (海螺AI) video generation API.

    Parameters
    ----------
    api_key:
        MiniMax API Key. Falls back to MINIMAX_API_KEY or VIDEOCLAW_MINIMAX_API_KEY.
    model:
        Model identifier, one of "minimax-hailuo-2.3", "minimax-hailuo-2.3-fast",
        "minimax-hailuo-02", "minimax-s2v-01".
    """

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

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def capabilities(self) -> list[ModelCapability]:
        if self._model == "minimax-s2v-01":
            # Subject reference video - requires a face photo
            return [ModelCapability.IMAGE_TO_VIDEO]
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

        # Build request and get task ID
        task_id = await self._create_task(request)

        # Poll for completion
        file_id = await self._poll_for_completion(task_id)

        # Download the video
        video_data = await self._download_video(file_id)

        cost = self._estimate_cost_for(request)

        return GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": self.model_id,
                "task_id": task_id,
                "file_id": file_id,
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
        """Stream progress events while polling the MiniMax job."""
        self._ensure_api_key()

        yield ProgressEvent(progress=0.0, stage="submitting_task")

        # Create task
        task_id = await self._create_task(request)
        yield ProgressEvent(progress=0.1, stage="task_created")

        # Poll for completion
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

        # Download video
        video_data = await self._download_video(file_id)

        cost = self._estimate_cost_for(request)

        yield ProgressEvent(progress=1.0, stage="complete")
        yield GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": self.model_id,
                "task_id": task_id,
                "file_id": file_id,
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
            logger.warning("[minimax] No API key configured")
            return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "MiniMax API key is required. Set MINIMAX_API_KEY or "
                "VIDEOCLAW_MINIMAX_API_KEY, or pass api_key= to "
                "MiniMaxVideoAdapter."
            )

    def _auth_headers(self) -> dict[str, str]:
        """Generate authentication headers with Bearer token."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _build_payload(self, request: GenerationRequest) -> dict:
        """Translate a GenerationRequest to the MiniMax API body."""
        api_model = MINIMAX_MODELS.get(self._model, "MiniMax-Hailuo-2.3")

        payload: dict = {
            "model": api_model,
            "prompt": request.prompt,
        }

        # Duration: 6 or 10 seconds
        duration = int(request.duration_seconds)
        payload["duration"] = 6 if duration <= 6 else 10

        # Resolution: 1080P or 768P
        if request.height >= 1080:
            payload["resolution"] = "1080P"
        else:
            payload["resolution"] = "768P"

        # Image to video: first_frame_image
        if request.reference_image:
            if "first_frame_image_url" in request.extra:
                payload["first_frame_image"] = request.extra["first_frame_image_url"]
            elif "first_frame_image" in request.extra:
                payload["first_frame_image"] = request.extra["first_frame_image"]
            elif "first_frame_image" not in payload:
                # bytes → base64 data URI
                import base64
                b64 = base64.b64encode(request.reference_image).decode("utf-8")
                payload["first_frame_image"] = f"data:image/png;base64,{b64}"

        # Subject reference (for S2V-01 model)
        if self._model == "minimax-s2v-01" and request.reference_image:
            if "subject_reference" in request.extra:
                payload["subject_reference"] = request.extra["subject_reference"]
            elif "subject_reference" not in payload:
                import base64
                b64 = base64.b64encode(request.reference_image).decode("utf-8")
                payload["subject_reference"] = [
                    {"image": f"data:image/png;base64,{b64}"}
                ]

        logger.debug("[minimax] Built payload: %s", payload)
        return payload

    async def _create_task(self, request: GenerationRequest) -> str:
        """Create a video generation task and return the task ID."""
        path = f"/{_VERSION}/video_generation"
        headers = self._auth_headers()
        payload = self._build_payload(request)

        logger.info("[minimax] Creating task with model: %s", payload.get("model"))
        logger.debug("[minimax] Payload: %s", payload)

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

                logger.info("[minimax] API response status: %d", resp.status_code)

                if resp.status_code != 200:
                    logger.error("[minimax] API error response: %s", resp.text)
                    resp.raise_for_status()

                data = resp.json()
                logger.debug("[minimax] API response data: %s", data)

                task_id = data.get("task_id")
                if not task_id:
                    raise RuntimeError(f"Failed to get task_id from response: {data}")

                logger.info("[minimax] Created generation task %s", task_id)
                return task_id

            except httpx.HTTPStatusError as e:
                logger.error("[minimax] HTTP error: %s - %s", e.response.status_code, e.response.text)
                raise
            except Exception as e:
                logger.error("[minimax] Failed to create task: %s", e)
                raise

    async def _poll_for_completion(
        self, task_id: str, poll_only: bool = False
    ) -> str | None:
        """Poll for task completion and return file_id if ready."""
        path = f"/{_VERSION}/query/video_generation"
        headers = self._auth_headers()
        params = {"task_id": task_id}

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_LIGHT_TIMEOUT_S),
        ) as client:
            resp = await client.get(
                path,
                headers=headers,
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

            logger.debug("[minimax] Poll response: %s", data)

            status = data.get("status")

            if status == "Success":
                file_id = data.get("file_id")
                if not file_id:
                    raise RuntimeError(f"Task succeeded but no file_id in response: {data}")
                return file_id
            elif status == "Fail":
                error_msg = data.get("error_message", "Unknown error")
                raise RuntimeError(f"MiniMax generation failed (task={task_id}): {error_msg}")
            elif poll_only:
                logger.debug("[minimax] Task %s status: %s", task_id, status)
                return None
            else:
                # For non-poll mode, wait and retry
                logger.info("[minimax] Task %s status: %s, waiting...", task_id, status)
                return None

    async def _download_video(self, file_id: str) -> bytes:
        """Download video using file_id."""
        path = f"/{_VERSION}/files/retrieve"
        headers = self._auth_headers()
        params = {"file_id": file_id}

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_GENERATE_TIMEOUT_S),
        ) as client:
            # Get download URL
            resp = await client.get(
                path,
                headers=headers,
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

            download_url = data.get("file", {}).get("download_url")
            if not download_url:
                raise RuntimeError(f"No download_url in response: {data}")

            logger.info("[minimax] Downloading video from: %s", download_url[:50] + "...")

            # Download video
            video_resp = await client.get(download_url)
            video_resp.raise_for_status()
            return video_resp.content

    @staticmethod
    def _estimate_cost_for(request: GenerationRequest) -> float:
        return round(request.duration_seconds * _COST_PER_SECOND_USD, 4)


# Factory function for entry point
def create_minimax_adapter() -> MiniMaxVideoAdapter:
    """Factory function to create a MiniMaxVideoAdapter instance."""
    return MiniMaxVideoAdapter()
