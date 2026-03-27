"""Image generation via Evolink Seedream 5.0 API.

Provides an async client that submits image generation tasks, polls for
completion, downloads results, and returns local file paths.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from videoclaw.config import get_config

logger = logging.getLogger(__name__)

# Polling configuration
_POLL_INTERVAL = 5.0  # seconds
_POLL_TIMEOUT = 180.0  # seconds


class EvolinkImageGenerator:
    """Generates images via the Evolink Seedream 5.0 API."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        cfg = get_config()
        self._api_key = api_key or cfg.evolink_api_key
        self._api_base = (api_base or cfg.evolink_api_base).rstrip("/")
        if not self._api_key:
            raise ValueError(
                "Evolink API key required. Set VIDEOCLAW_EVOLINK_API_KEY or pass api_key."
            )
        # Stores the HTTPS URL of the last generated image (before download).
        # Used by CharacterDesigner to pass URLs to Seedance API which rejects
        # base64 data URIs via the vectorspace.cn proxy.
        self.last_image_url: str | None = None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        *,
        output_dir: Path,
        filename: str = "image.png",
        model: str = "doubao-seedream-5.0-lite",
        size: str = "3:4",
        quality: str = "2K",
        reference_urls: list[str] | None = None,
    ) -> Path:
        """Generate an image and save to *output_dir/filename*.

        Returns the local Path to the downloaded image.

        Parameters
        ----------
        size:
            Ratio format (``"1:1"``, ``"3:4"``, ``"16:9"``, ``"auto"``) or
            pixel format (``"2048x2048"``).  Seedream 5.0 does NOT accept
            raw pixel dimensions separated by ``:``.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": 1,
        }
        if reference_urls:
            body["image_urls"] = reference_urls

        logger.info(
            "Submitting image generation: model=%s size=%s prompt=%.60s...",
            model, size, prompt,
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Submit generation task
            resp = await client.post(
                f"{self._api_base}/images/generations",
                headers=self._headers(),
                json=body,
            )
            if resp.status_code != 200:
                logger.error(
                    "Image API error %d: %s", resp.status_code, resp.text,
                )
            resp.raise_for_status()
            task_data = resp.json()
            logger.debug("Image task response: %s", task_data)

            # Some APIs return the image URL directly; others return a task ID
            image_url = self._extract_image_url(task_data)
            if not image_url:
                task_id = task_data.get("id") or task_data.get("task_id")
                if not task_id:
                    raise RuntimeError(f"No task_id in response: {task_data}")
                image_url = await self._poll_task(client, task_id)

            # Store the HTTPS URL before download (for Seedance API passthrough)
            self.last_image_url = image_url

            # Download the image
            img_resp = await client.get(image_url, timeout=60.0)
            img_resp.raise_for_status()
            output_path.write_bytes(img_resp.content)

        logger.info("Image saved: %s", output_path)
        return output_path

    async def _poll_task(self, client: httpx.AsyncClient, task_id: str) -> str:
        """Poll until the task completes and return the image URL."""
        elapsed = 0.0
        while elapsed < _POLL_TIMEOUT:
            await asyncio.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL

            resp = await client.get(
                f"{self._api_base}/tasks/{task_id}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")

            if status in ("completed", "succeeded", "success"):
                url = self._extract_image_url(data)
                if url:
                    return url
                raise RuntimeError(f"Task completed but no image URL: {data}")

            if status in ("failed", "error"):
                raise RuntimeError(f"Image generation failed: {data}")

            logger.debug("Polling task %s: status=%s (%.0fs)", task_id, status, elapsed)

        raise TimeoutError(f"Image generation timed out after {_POLL_TIMEOUT}s")

    @staticmethod
    def _extract_image_url(data: dict[str, Any]) -> str | None:
        """Try to find an image URL in various API response formats."""
        # Format: { "data": [{"url": "..."}] }  (OpenAI-compatible)
        for item in data.get("data", []):
            if url := item.get("url"):
                return url
        # Format: { "results": ["url"] }
        for item in data.get("results", []):
            if isinstance(item, str):
                return item
            if isinstance(item, dict) and (url := item.get("url")):
                return url
        # Format: { "output": { "image_url": "..." } }
        if output := data.get("output"):
            if url := output.get("image_url"):
                return url
        return None
