"""Image generation via Evolink Seedream 5.0 API.

Provides an async client that submits image generation tasks, polls for
completion, downloads results, and returns local file paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from videoclaw.config import get_config
from videoclaw.generation.base_image import BaseImageGenerator

logger = logging.getLogger(__name__)


class EvolinkImageGenerator(BaseImageGenerator):
    """Generates images via the Evolink Seedream 5.0 API."""

    _poll_interval = 5.0
    _poll_timeout = 180.0

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
        output_path = self._ensure_dir(output_dir, filename)

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

            image_url = self._extract_image_url(task_data)
            if not image_url:
                task_id = task_data.get("id") or task_data.get("task_id")
                if not task_id:
                    raise RuntimeError(f"No task_id in response: {task_data}")
                image_url = await self._poll_until_ready(
                    client,
                    f"{self._api_base}/tasks/{task_id}",
                    self._headers(),
                )

            await self._download_and_save(client, image_url, output_path)

        logger.info("Image saved: %s", output_path)
        return output_path
