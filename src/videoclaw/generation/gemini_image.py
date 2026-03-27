"""Image generation via Google Gemini (Nano Banana 2) API.

Provides an async client that generates images using the Gemini
``generateContent`` endpoint with ``responseModalities: ["TEXT", "IMAGE"]``.

The primary model is ``gemini-3.1-flash-image-preview`` (Nano Banana 2),
Google's fast, high-quality image generation model.

Environment variables
---------------------
- ``GOOGLE_API_KEY`` / ``VIDEOCLAW_GOOGLE_API_KEY``
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from videoclaw.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Supported aspect ratios
ASPECT_RATIOS = {
    "1:1", "3:4", "4:3", "9:16", "16:9",
    "4:1", "1:4", "8:1", "1:8", "21:9",
}


class GeminiImageGenerator:
    """Generate images via Google Gemini (Nano Banana 2).

    Endpoint: ``POST {base}/models/{model}:generateContent``

    Parameters
    ----------
    api_key : str | None
        Google API key.  Falls back to ``GOOGLE_API_KEY`` or config.
    model : str
        Model name (default: ``gemini-3.1-flash-image-preview``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        config = get_config()
        self._api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or getattr(config, "google_api_key", None)
        )
        self._model = model

    async def generate(
        self,
        prompt: str,
        *,
        output_dir: Path,
        filename: str = "image.png",
        size: str = "3:4",
        **kwargs: Any,
    ) -> Path:
        """Generate an image and save to *output_dir/filename*.

        Returns the local :class:`Path` to the downloaded image.

        Parameters
        ----------
        size : str
            Aspect ratio (e.g. ``"3:4"``, ``"16:9"``).
        """
        if not self._api_key:
            raise RuntimeError(
                "Google API key required. Set GOOGLE_API_KEY or "
                "VIDEOCLAW_GOOGLE_API_KEY."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Map size aliases
        aspect_ratio = size if size in ASPECT_RATIOS else "3:4"

        body: dict[str, Any] = {
            "contents": [{
                "parts": [{"text": prompt}],
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                },
            },
        }

        url = f"{_API_BASE}/models/{self._model}:generateContent"

        logger.info(
            "[gemini-image] Generating: model=%s aspect=%s prompt=%.60s...",
            self._model, aspect_ratio, prompt,
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
                json=body,
            )
            if resp.status_code != 200:
                logger.error(
                    "[gemini-image] API error %d: %s",
                    resp.status_code, resp.text[:500],
                )
            resp.raise_for_status()

            data = resp.json()
            image_bytes = self._extract_image(data)
            if not image_bytes:
                raise RuntimeError(f"No image data in response: {data}")

            output_path.write_bytes(image_bytes)

        logger.info("[gemini-image] Saved: %s", output_path)
        return output_path

    @staticmethod
    def _extract_image(data: dict[str, Any]) -> bytes | None:
        """Extract base64-decoded image bytes from Gemini response."""
        candidates = data.get("candidates", [])
        if not candidates:
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])

        return None
