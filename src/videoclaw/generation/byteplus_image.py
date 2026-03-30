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
from pathlib import Path
from typing import Any

import httpx

from videoclaw.generation.base_image import BaseImageGenerator
from videoclaw.utils import resolve_credential

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


def _resolve_byteplus_credentials(
    api_key: str | None = None,
    api_base: str | None = None,
) -> tuple[str | None, str]:
    """Resolve BytePlus API key and base URL from args / env / config."""
    from videoclaw.config import get_config
    key = resolve_credential(
        explicit=api_key, env_vars="BYTEPLUS_API_KEY",
        config_attr="byteplus_api_key",
    )
    base = (api_base or get_config().byteplus_api_base).rstrip("/")
    return key, base


# ===================================================================
# BytePlusImageGenerator  (Text-to-Image — Seedream)
# ===================================================================

class BytePlusImageGenerator(BaseImageGenerator):
    """Generate images via BytePlus Seedream models.

    Endpoint: ``POST {base}/images/generations`` (synchronous).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "seedream-5.0",
        api_base: str | None = None,
    ) -> None:
        self._api_key, self._api_base = _resolve_byteplus_credentials(api_key, api_base)
        self._model_id = SEEDREAM_MODELS.get(model, SEEDREAM_MODELS["seedream-5.0"])
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
        **kwargs: Any,
    ) -> Path:
        """Generate an image and save to *output_dir/filename*."""
        if not self._api_key:
            raise RuntimeError(
                "BytePlus API key required. Set BYTEPLUS_API_KEY or "
                "VIDEOCLAW_BYTEPLUS_API_KEY."
            )

        output_path = self._ensure_dir(output_dir, filename)

        body: dict[str, Any] = {
            "model": self._model_id,
            "prompt": prompt,
            "size": size,
            "watermark": watermark,
        }
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

            await self._download_and_save(client, image_url, output_path)

        logger.info("[byteplus-image] Saved: %s", output_path)
        return output_path


# ===================================================================
# BytePlusImageEditor  (Image-to-Image — Seededit 3.0)
# ===================================================================

class BytePlusImageEditor(BaseImageGenerator):
    """Edit images via BytePlus Seededit 3.0.

    Accepts a source image (HTTPS URL or base64) and a text instruction
    to produce a modified version.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._api_key, self._api_base = _resolve_byteplus_credentials(api_key, api_base)
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
        **kwargs: Any,
    ) -> Path:
        """Edit interface conforming to BaseImageGenerator.

        Pass ``image_url`` or ``image_bytes`` via kwargs.
        """
        return await self.edit(
            prompt,
            image_url=kwargs.get("image_url"),
            image_bytes=kwargs.get("image_bytes"),
            output_dir=output_dir,
            filename=filename,
            guidance_scale=kwargs.get("guidance_scale", 5.5),
            size=kwargs.get("size", "adaptive"),
        )

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
        """
        if not self._api_key:
            raise RuntimeError(
                "BytePlus API key required for Seededit. Set BYTEPLUS_API_KEY "
                "or VIDEOCLAW_BYTEPLUS_API_KEY."
            )

        output_path = self._ensure_dir(output_dir, filename)

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

        logger.info("[byteplus-edit] Editing: instruction=%.60s...", instruction)

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

            await self._download_and_save(client, result_url, output_path)

        logger.info("[byteplus-edit] Saved: %s", output_path)
        return output_path
