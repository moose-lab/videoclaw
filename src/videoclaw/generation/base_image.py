"""Base class for image generators.

Provides shared infrastructure: config resolution, HTTP headers, output
directory management, polling loops, and ``last_image_url`` tracking.
Concrete subclasses only need to implement :meth:`_build_request` and
optionally override :meth:`_parse_response`.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default polling settings (overridable per subclass)
_POLL_INTERVAL = 5.0
_POLL_TIMEOUT = 180.0


class BaseImageGenerator(ABC):
    """Abstract base for all image generators.

    Subclasses must implement:
    - :meth:`_build_request` — return ``(url, headers, body)``

    Subclasses may override:
    - :meth:`_parse_response` — extract image URL or bytes from API response
    - :meth:`_poll_task` — custom polling logic for async APIs
    """

    #: HTTPS URL of the last generated image (before local download).
    #: Used by CharacterDesigner to pass URLs to Seedance API.
    last_image_url: str | None = None

    # Subclass configuration
    _timeout: float = 120.0
    _poll_interval: float = _POLL_INTERVAL
    _poll_timeout: float = _POLL_TIMEOUT

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        output_dir: Path,
        filename: str = "image.png",
        **kwargs: Any,
    ) -> Path:
        """Generate an image and save to *output_dir/filename*."""
        ...

    async def _download_and_save(
        self,
        client: httpx.AsyncClient,
        image_url: str,
        output_path: Path,
    ) -> Path:
        """Download an image from *image_url* and write to *output_path*."""
        self.last_image_url = image_url
        img_resp = await client.get(image_url, timeout=60.0)
        img_resp.raise_for_status()
        output_path.write_bytes(img_resp.content)
        return output_path

    async def _poll_until_ready(
        self,
        client: httpx.AsyncClient,
        poll_url: str,
        headers: dict[str, str],
    ) -> str:
        """Poll *poll_url* until the task completes.  Returns the image URL."""
        elapsed = 0.0
        while elapsed < self._poll_timeout:
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

            resp = await client.get(poll_url, headers=headers)
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

            logger.debug("Polling: status=%s (%.0fs)", status, elapsed)

        raise TimeoutError(
            f"Image generation timed out after {self._poll_timeout}s"
        )

    @staticmethod
    def _extract_image_url(data: dict[str, Any]) -> str | None:
        """Try to find an image URL in common API response formats.

        Handles:
        - ``{"data": [{"url": "..."}]}`` (OpenAI-compatible)
        - ``{"results": ["url"]}``
        - ``{"output": {"image_url": "..."}}``
        """
        for item in data.get("data", []):
            if url := item.get("url"):
                return url
        for item in data.get("results", []):
            if isinstance(item, str):
                return item
            if isinstance(item, dict) and (url := item.get("url")):
                return url
        if output := data.get("output"):
            if url := output.get("image_url"):
                return url
        return None

    @staticmethod
    def _ensure_dir(output_dir: Path, filename: str) -> Path:
        """Create output directory and return the full output path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
