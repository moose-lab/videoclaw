"""BytePlus ModelArk image generation (Seedream) and editing (Seededit).

Provides two generators backed by the BytePlus ModelArk API
(``ark.ap-southeast.bytepluses.com``):

:class:`BytePlusImageGenerator`
    Text-to-image via Seedream 5.0 Lite / 4.5 / 4.0.  Synchronous endpoint —
    the response contains the image URL directly.

:class:`BytePlusImageEditor`
    Image-to-image editing via Seededit 3.0.  Accepts a source image (URL or
    base64) and a text instruction to produce a modified image.

Both share the same API key (``VIDEOCLAW_BYTEPLUS_API_KEY``) and base URL.

Environment variables
---------------------
- ``BYTEPLUS_API_KEY`` / ``VIDEOCLAW_BYTEPLUS_API_KEY``
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
# Model IDs
# ---------------------------------------------------------------------------

SEEDREAM_MODELS: dict[str, str] = {
    "seedream-5.0": "seedream-5-0-260128",
    "seedream-5.0-lite": "seedream-5-0-lite-260128",
    "seedream-4.5": "seedream-4-5-251128",
}

SEEDEDIT_MODEL_ID = "seededit-3-0-i2i-250628"


# ===================================================================
# BytePlusImageGenerator  (Text-to-Image — Seedream)
# ===================================================================

class BytePlusImageGenerator:
    """Generate images via BytePlus Seedream models.

    Endpoint: ``POST {base}/images/generations`` (synchronous).

    Parameters
    ----------
    api_key : str | None
        BytePlus ModelArk API key.  Falls back to env var ``BYTEPLUS_API_KEY``
        or config ``byteplus_api_key``.
    model : str
        One of ``"seedream-5.0"`` (default), ``"seedream-4.5"``,
        ``"seedream-4.0"``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "seedream-5.0",
        api_base: str | None = None,
    ) -> None:
        config = get_config()
        self._api_key = (
            api_key
            or os.environ.get("BYTEPLUS_API_KEY")
            or config.byteplus_api_key
        )
        self._api_base = (api_base or config.byteplus_api_base).rstrip("/")
        self._model_id = SEEDREAM_MODELS.get(model, SEEDREAM_MODELS["seedream-5.0"])
        # Stores the HTTPS URL of the last generated image (before download).
        # Used by CharacterDesigner to pass URLs to Seedance API.
        self.last_image_url: str | None = None

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def generate(
        self,
        prompt: str,
        *,
        output_dir: Path,
        filename: str = "image.png",
        size: str = "2K",
        output_format: str = "jpeg",
        watermark: bool = False,
    ) -> Path:
        """Generate an image and save to *output_dir/filename*.

        Returns the local :class:`Path` to the downloaded image.
        """
        if not self._api_key:
            raise RuntimeError(
                "BytePlus API key required. Set BYTEPLUS_API_KEY or "
                "VIDEOCLAW_BYTEPLUS_API_KEY."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        body: dict[str, Any] = {
            "model": self._model_id,
            "prompt": prompt,
            "size": size,
            "watermark": watermark,
        }
        # output_format only supported by Seedream 5.0 Lite
        if self._model_id == SEEDREAM_MODELS["seedream-5.0"]:
            body["output_format"] = output_format

        logger.info(
            "[byteplus-image] Generating: model=%s size=%s prompt=%.60s...",
            self._model_id, size, prompt,
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._api_base}/images/generations",
                headers=self._headers(),
                json=body,
            )
            if resp.status_code != 200:
                logger.error(
                    "[byteplus-image] API error %d: %s",
                    resp.status_code, resp.text[:500],
                )
            resp.raise_for_status()

            data = resp.json()
            image_url = self._extract_image_url(data)
            if not image_url:
                raise RuntimeError(f"No image URL in response: {data}")

            # Store HTTPS URL before download (for Seedance API passthrough)
            self.last_image_url = image_url

            # Download the image
            img_resp = await client.get(image_url, timeout=60.0)
            img_resp.raise_for_status()
            output_path.write_bytes(img_resp.content)

        logger.info("[byteplus-image] Saved: %s", output_path)
        return output_path

    @staticmethod
    def _extract_image_url(data: dict[str, Any]) -> str | None:
        """Extract image URL from the BytePlus response.

        Response format::

            {"data": [{"url": "https://tos-..."}], "usage": {...}}
        """
        for item in data.get("data", []):
            if url := item.get("url"):
                return url
        return None


# ===================================================================
# BytePlusImageEditor  (Image-to-Image — Seededit 3.0)
# ===================================================================

class BytePlusImageEditor:
    """Edit images via BytePlus Seededit 3.0.

    Endpoint: ``POST {base}/images/generations`` with ``image`` parameter.

    Accepts a source image (HTTPS URL or base64) and a text instruction
    to produce a modified version.  Useful for:
    - Character costume/appearance changes
    - Background/lighting alterations
    - Style transfers within drama scenes
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        config = get_config()
        self._api_key = (
            api_key
            or os.environ.get("BYTEPLUS_API_KEY")
            or config.byteplus_api_key
        )
        self._api_base = (api_base or config.byteplus_api_base).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def edit(
        self,
        instruction: str,
        *,
        image_url: str | None = None,
        image_bytes: bytes | None = None,
        output_dir: Path,
        filename: str = "edited.png",
        guidance_scale: float = 5.5,
        size: str = "adaptive",
    ) -> Path:
        """Edit an image and save to *output_dir/filename*.

        Provide either *image_url* (HTTPS URL) or *image_bytes* (raw bytes
        that will be base64-encoded).

        Returns the local :class:`Path` to the edited image.
        """
        if not self._api_key:
            raise RuntimeError(
                "BytePlus API key required for Seededit. Set BYTEPLUS_API_KEY "
                "or VIDEOCLAW_BYTEPLUS_API_KEY."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Resolve image source
        if image_url:
            image_src = image_url
        elif image_bytes:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            image_src = f"data:image/png;base64,{b64}"
        else:
            raise ValueError("Either image_url or image_bytes must be provided")

        body: dict[str, Any] = {
            "model": SEEDEDIT_MODEL_ID,
            "prompt": instruction,
            "image": image_src,
            "response_format": "url",
            "size": size,
            "guidance_scale": guidance_scale,
            "watermark": False,
        }

        logger.info(
            "[byteplus-edit] Editing: instruction=%.60s...",
            instruction,
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._api_base}/images/generations",
                headers=self._headers(),
                json=body,
            )
            if resp.status_code != 200:
                logger.error(
                    "[byteplus-edit] API error %d: %s",
                    resp.status_code, resp.text[:500],
                )
            resp.raise_for_status()

            data = resp.json()
            result_url = self._extract_image_url(data)
            if not result_url:
                raise RuntimeError(f"No image URL in edit response: {data}")

            img_resp = await client.get(result_url, timeout=60.0)
            img_resp.raise_for_status()
            output_path.write_bytes(img_resp.content)

        logger.info("[byteplus-edit] Saved: %s", output_path)
        return output_path

    @staticmethod
    def _extract_image_url(data: dict[str, Any]) -> str | None:
        for item in data.get("data", []):
            if url := item.get("url"):
                return url
        return None
