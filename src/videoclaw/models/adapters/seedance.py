"""Seedance (豆包) video generation adapter.

Wraps the Seedance 2.0 video generation API (vectorspace.cn proxy for
ByteDance Volcengine Ark).

Seedance 2.0 API specification (from official docs):

Content (required) — multimodal input combinations:
  - Text only
  - Text (optional) + Image
  - Text (optional) + Video
  - Text (optional) + Image + Audio
  - Text (optional) + Image + Video
  - Text (optional) + Video + Audio
  - Text (optional) + Image + Video + Audio

Image constraints:
  - Formats: jpeg, png, webp, bmp, tiff, gif
  - Aspect ratio (W/H): (0.4, 2.5)
  - Width/Height: 300–6000px
  - Size: single < 30MB, request body < 64MB
  - Limits: first_frame: 1 | first+last_frame: 2 | reference_image: 1–9

Image roles (MUTUALLY EXCLUSIVE scenarios):
  - first_frame   — image-to-video first frame (1 image)
  - last_frame    — image-to-video last frame (must pair with first_frame)
  - reference_image — Universal Reference / 全能参考 (1–9 images)

Video constraints:
  - Formats: mp4, mov | Resolution: 480p, 720p
  - Duration: [2, 15]s, max 3 clips, total ≤ 15s
  - Size: single ≤ 50MB | FPS: [24, 60]
  - Role: reference_video only

Audio constraints:
  - Formats: wav, mp3 | Duration: [2, 15]s, max 3 segments, total ≤ 15s
  - Size: single ≤ 15MB, request body ≤ 64MB
  - CANNOT be used alone — must accompany at least 1 video or image
  - Role: reference_audio only

Other parameters:
  - generate_audio: bool (default true) — co-generate voice/SFX/BGM
  - resolution: "480p" | "720p" (default "720p")
  - ratio: "16:9"|"4:3"|"1:1"|"3:4"|"9:16"|"21:9"|"adaptive" (default "adaptive")
  - duration: int [5, 15] (default 5)
  - watermark: bool — add watermark
  - tools: object[] — web_search for improved timeliness

API endpoints (vectorspace.cn):
  - Create task:  POST {base}/api/v1/doubao/create
  - Query result: POST {base}/api/v1/doubao/get_result

Environment variables:
  - ARK_API_KEY / VIDEOCLAW_ARK_API_KEY
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://sd2.vectorspace.cn"

# Ark model identifier used by the API
# Supported models on vectorspace.cn proxy:
#   doubao-seedance-2.0-fast, doubao-seedance-2.0-fast-260128,
#   doubao-seedance-2-0, doubao-seedance-2-0-260128
_ARK_MODEL_ID = "doubao-seedance-2.0-fast-260128"

# Width×Height → ratio string accepted by the API
# 720p resolution mapping (from official docs)
_RESOLUTION_TO_RATIO: dict[tuple[int, int], str] = {
    (720, 1280): "9:16",    # TikTok vertical
    (1280, 720): "16:9",    # horizontal
    (960, 960): "1:1",      # square (720p)
    (834, 1112): "3:4",
    (1112, 834): "4:3",
    (1470, 630): "21:9",    # ultrawide
    # Legacy mappings (auto-converted)
    (1024, 1024): "1:1",
    (768, 1024): "3:4",
    (1024, 768): "4:3",
}

# Duration limits (from official docs: [5, 15])
_MIN_DURATION_S = 5
_MAX_DURATION_S = 15

# Image constraints
_MAX_REFERENCE_IMAGES = 9
_MAX_IMAGE_SIZE_BYTES = 30 * 1024 * 1024  # 30MB per image
_MAX_REQUEST_SIZE_BYTES = 64 * 1024 * 1024  # 64MB total
_IMAGE_MIN_PX = 300
_IMAGE_MAX_PX = 6000

# Text constraints
_MAX_TEXT_EN_WORDS = 1000
_MAX_TEXT_ZH_CHARS = 500

# Cost estimate (USD per second)
_COST_PER_SECOND_USD = 0.05

# Polling configuration
_POLL_INTERVAL_S = 10.0
_POLL_TIMEOUT_S = 6000.0  # 100 minutes max
_HTTP_TIMEOUT_S = 30.0


def _detect_mime_from_bytes(data: bytes) -> str:
    """Detect image MIME type from magic bytes (file signature).

    Falls back to ``image/png`` if unrecognised.
    """
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] in (b"BM",):
        return "image/bmp"
    if data[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return "image/tiff"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def _image_to_data_uri(path_or_bytes: str | bytes) -> str | None:
    """Convert a local file path or raw bytes to a base64 data URI.

    MIME type is detected from the actual file content (magic bytes),
    not from the file extension — this avoids mismatches when e.g. a
    file is named ``.png`` but contains JPEG data.

    Returns None if the file doesn't exist or exceeds size limits.
    """
    if isinstance(path_or_bytes, str):
        p = Path(path_or_bytes)
        if not p.exists():
            logger.warning("[seedance] Image file not found: %s", path_or_bytes)
            return None
        if p.stat().st_size > _MAX_IMAGE_SIZE_BYTES:
            logger.warning(
                "[seedance] Image too large (%d bytes > %d): %s",
                p.stat().st_size, _MAX_IMAGE_SIZE_BYTES, path_or_bytes,
            )
            return None
        raw = p.read_bytes()
    else:
        raw = path_or_bytes

    mime = _detect_mime_from_bytes(raw)
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


class SeedanceVideoAdapter:
    """Adapter for ByteDance Seedance 2.0 video generation.

    Supports the full multimodal content pipeline:
    - Text-to-video (text prompt only)
    - Image-to-video (first_frame / last_frame)
    - Universal Reference (reference_image, 1–9 images for character consistency)
    - Reference video (reference_video)
    - Reference audio (reference_audio, must accompany image/video)
    - Audio co-generation (generate_audio=True)

    Parameters
    ----------
    api_key:
        API key for the Seedance endpoint. Falls back to ARK_API_KEY or
        VIDEOCLAW_ARK_API_KEY environment variable.
    base_url:
        Override the API base URL (defaults to vectorspace.cn proxy).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        config = get_config()
        self._api_key = (
            api_key
            or os.environ.get("ARK_API_KEY")
            or config.ark_api_key
        )
        self._base_url = (base_url or config.seedance_base_url or _DEFAULT_BASE_URL).rstrip("/")

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return "seedance-2.0"

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
        """Submit a video generation task and poll until completion."""
        self._ensure_api_key()

        task_id = await self._create_task(request)
        video_url = await self._poll_until_done(task_id)
        video_data = await self._download_video(video_url)

        cost = self._estimate_cost_for(request)

        return GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": _ARK_MODEL_ID,
                "task_id": task_id,
                "prompt": request.prompt[:200],
                "generated_at": time.time(),
            },
            cost_usd=cost,
            model_id=self.model_id,
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[ProgressEvent | GenerationResult]:
        """Yield progress events while polling the task."""
        self._ensure_api_key()

        yield ProgressEvent(progress=0.0, stage="submitting")

        task_id = await self._create_task(request)
        yield ProgressEvent(progress=0.1, stage="task_created")

        elapsed = 0.0
        video_url: str | None = None
        while elapsed < _POLL_TIMEOUT_S:
            await asyncio.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S

            status, video_url = await self._check_task(task_id)

            if status == "done" and video_url:
                break
            if status == "failed":
                raise RuntimeError(f"Seedance generation failed (task={task_id})")

            progress = min(0.1 + (elapsed / _POLL_TIMEOUT_S) * 0.8, 0.9)
            yield ProgressEvent(progress=progress, stage="processing")
        else:
            raise TimeoutError(
                f"Seedance generation timed out after {_POLL_TIMEOUT_S}s"
            )

        yield ProgressEvent(progress=0.95, stage="downloading")
        video_data = await self._download_video(video_url)

        cost = self._estimate_cost_for(request)

        yield ProgressEvent(progress=1.0, stage="complete")
        yield GenerationResult(
            video_data=video_data,
            format="mp4",
            duration_seconds=request.duration_seconds,
            metadata={
                "model": _ARK_MODEL_ID,
                "task_id": task_id,
                "prompt": request.prompt[:200],
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
        """Return True if an API key is configured."""
        if not self._api_key:
            logger.warning("[seedance] No ARK_API_KEY configured")
            return False
        return True

    # ------------------------------------------------------------------
    # Content builder (multimodal)
    # ------------------------------------------------------------------

    def _build_content(self, request: GenerationRequest) -> list[dict[str, Any]]:
        """Build the multimodal ``content`` array for the API request.

        Handles all supported input combinations:
        - Text (always included if non-empty)
        - Images: reference_image (Universal Reference, 1–9), first_frame, last_frame
        - Video: reference_video
        - Audio: reference_audio (must accompany image/video)

        Image role scenarios are MUTUALLY EXCLUSIVE:
        - first_frame / last_frame — image-to-video mode
        - reference_image — Universal Reference mode (全能参考)

        Data flow:
        - ``request.extra["image_paths"]``: list[dict] with local file paths
          → converted to base64 data URIs
        - ``request.extra["image_urls"]``: list[dict] with HTTPS URLs
          → passed through directly
        - ``request.reference_image``: bytes → converted to base64 data URI
        - ``request.extra["additional_references"]``: dict[name, bytes]
          → converted to base64 data URIs
        - ``request.extra["reference_videos"]``: list[dict] with HTTPS URLs
        - ``request.extra["reference_audios"]``: list[dict] with HTTPS URLs
        """
        content: list[dict[str, Any]] = []
        ref_image_count = 0
        has_first_frame = False

        # --- 1. Text prompt ---
        if request.prompt:
            # Truncate to API limits
            words = request.prompt.split()
            if len(words) > _MAX_TEXT_EN_WORDS:
                truncated = " ".join(words[:_MAX_TEXT_EN_WORDS])
                logger.warning(
                    "[seedance] Prompt truncated from %d to %d words",
                    len(words), _MAX_TEXT_EN_WORDS,
                )
            else:
                truncated = request.prompt
            content.append({"type": "text", "text": truncated})

        # --- 2a. Image paths (local files → base64 data URIs) ---
        # This is the PRIMARY path for character reference images from
        # the drama pipeline (CharacterDesigner → Shot.reference_images)
        image_paths: list[dict[str, str]] | None = request.extra.get("image_paths")
        if image_paths:
            for img_info in image_paths:
                path = img_info.get("path", "")
                role = img_info.get("role", "reference_image")

                if role == "first_frame":
                    has_first_frame = True
                if role == "reference_image":
                    if ref_image_count >= _MAX_REFERENCE_IMAGES:
                        logger.warning(
                            "[seedance] Max reference images (%d) reached, skipping: %s",
                            _MAX_REFERENCE_IMAGES, path,
                        )
                        continue
                    ref_image_count += 1

                data_uri = _image_to_data_uri(path)
                if data_uri:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                        "role": role,
                    })
                    logger.debug(
                        "[seedance] Added image %s (role=%s) from %s",
                        ref_image_count, role, Path(path).name,
                    )

        # --- 2b. Image URLs (HTTPS — passed through directly) ---
        image_urls: list[dict[str, str]] | None = request.extra.get("image_urls")
        if image_urls:
            for img_info in image_urls:
                url = img_info.get("url", "")
                role = img_info.get("role", "reference_image")

                if not url.startswith("http"):
                    # Might be a local path — try converting to data URI
                    data_uri = _image_to_data_uri(url)
                    if data_uri:
                        url = data_uri
                    else:
                        continue

                if role == "first_frame":
                    has_first_frame = True
                if role == "reference_image":
                    if ref_image_count >= _MAX_REFERENCE_IMAGES:
                        continue
                    ref_image_count += 1

                content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                    "role": role,
                })

        # --- 2c. Raw bytes: primary reference image ---
        if request.reference_image and ref_image_count < _MAX_REFERENCE_IMAGES:
            data_uri = _image_to_data_uri(request.reference_image)
            if data_uri:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                    "role": "reference_image",
                })
                ref_image_count += 1
                logger.info(
                    "[seedance] Added primary reference image as base64 (%d bytes)",
                    len(request.reference_image),
                )

        # --- 2d. Raw bytes: additional reference images ---
        extra_refs: dict[str, bytes] | None = request.extra.get("additional_references")
        if extra_refs:
            for char_name, img_bytes in extra_refs.items():
                if ref_image_count >= _MAX_REFERENCE_IMAGES:
                    logger.warning(
                        "[seedance] Max reference images reached, skipping %s",
                        char_name,
                    )
                    break
                data_uri = _image_to_data_uri(img_bytes)
                if data_uri:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                        "role": "reference_image",
                    })
                    ref_image_count += 1
                    logger.info(
                        "[seedance] Added reference for %s as base64 (%d bytes)",
                        char_name, len(img_bytes),
                    )

        # --- 3. Reference videos ---
        ref_videos: list[dict[str, str]] | None = request.extra.get("reference_videos")
        if ref_videos:
            for vid_info in ref_videos:
                url = vid_info.get("url", "")
                if url:
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": url},
                        "role": "reference_video",
                    })

        # --- 4. Reference audios (must accompany image/video) ---
        ref_audios: list[dict[str, str]] | None = request.extra.get("reference_audios")
        if ref_audios:
            has_media = ref_image_count > 0 or ref_videos
            if not has_media:
                logger.warning(
                    "[seedance] reference_audio requires at least 1 image/video — skipping"
                )
            else:
                for aud_info in ref_audios:
                    url = aud_info.get("url", "")
                    if url:
                        content.append({
                            "type": "audio_url",
                            "audio_url": {"url": url},
                            "role": "reference_audio",
                        })

        # --- Validation: mutual exclusivity ---
        if has_first_frame and ref_image_count > 0:
            # first_frame and reference_image are mutually exclusive
            # Prefer reference_image (Universal Reference) for drama pipeline
            logger.warning(
                "[seedance] first_frame and reference_image are mutually exclusive! "
                "Keeping reference_image (Universal Reference mode)."
            )
            content = [
                c for c in content
                if not (c.get("role") in ("first_frame", "last_frame"))
            ]

        if ref_image_count > 0:
            logger.info(
                "[seedance] Universal Reference mode: %d reference images loaded",
                ref_image_count,
            )
        elif not any(c.get("role") for c in content if isinstance(c, dict)):
            logger.info("[seedance] Text-to-video mode (no reference images)")

        return content

    # ------------------------------------------------------------------
    # Payload builder
    # ------------------------------------------------------------------

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Translate a GenerationRequest into the API request body.

        API body format::

            {
                "model": "doubao-seedance-2-0-260128",
                "content": [...],
                "generate_audio": true,
                "ratio": "9:16",
                "resolution": "720p",
                "duration": 5,
                "watermark": false
            }
        """
        # Resolve aspect ratio string
        res_key = (request.width, request.height)
        ratio = _RESOLUTION_TO_RATIO.get(res_key, "9:16")

        # Duration clamped to Seedance 2.0's [5, 15] range (from official docs)
        duration = max(_MIN_DURATION_S, min(_MAX_DURATION_S, int(request.duration_seconds)))

        # Resolution: default 720p for quality, allow override via extra
        resolution = request.extra.get("resolution", "720p")

        payload: dict[str, Any] = {
            "model": _ARK_MODEL_ID,
            "content": self._build_content(request),
            "generate_audio": request.extra.get("generate_audio", True),
            "ratio": ratio,
            "resolution": resolution,
            "duration": duration,
            "watermark": False,
        }

        if request.seed is not None:
            payload["seed"] = request.seed

        return payload

    # ------------------------------------------------------------------
    # API communication
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "Seedance API key required. "
                "Set ARK_API_KEY or VIDEOCLAW_ARK_API_KEY."
            )

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def _create_task(self, request: GenerationRequest) -> str:
        """Submit a video generation task with 429 retry.

        ``POST {base}/api/v1/doubao/create``
        """
        payload = self._build_payload(request)

        # Log content summary (not full base64)
        content_summary = []
        for c in payload.get("content", []):
            ctype = c.get("type", "?")
            role = c.get("role", "")
            if ctype == "text":
                content_summary.append(f"text({len(c.get('text', ''))}ch)")
            elif ctype == "image_url":
                url = c.get("image_url", {}).get("url", "")
                is_b64 = url.startswith("data:")
                content_summary.append(f"img({role},{'b64' if is_b64 else 'url'})")
            elif ctype == "video_url":
                content_summary.append(f"video({role})")
            elif ctype == "audio_url":
                content_summary.append(f"audio({role})")

        logger.info(
            "[seedance] Creating task: model=%s ratio=%s duration=%s content=[%s]",
            payload["model"],
            payload.get("ratio"),
            payload.get("duration"),
            ", ".join(content_summary),
        )

        max_retries = 5
        for attempt in range(max_retries):
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                resp = await client.post(
                    f"{self._base_url}/api/v1/doubao/create",
                    headers=self._auth_headers(),
                    json=payload,
                )

                if resp.status_code == 429:
                    wait = min(15.0 * (attempt + 1), 60.0)
                    logger.warning(
                        "[seedance] Rate limited (429), waiting %.0fs before retry %d/%d",
                        wait, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.error(
                        "[seedance] Task creation failed %d: %s",
                        resp.status_code,
                        resp.text[:500],
                    )
                    resp.raise_for_status()

                data = resp.json()
                task_id = data.get("id") or data.get("task_id")

                if not task_id:
                    raise RuntimeError(
                        f"No task_id in Seedance response: {data}"
                    )

                logger.info("[seedance] Created task %s", task_id)
                return task_id

        raise RuntimeError("Seedance task creation failed: rate limited after all retries")

    async def _check_task(self, task_id: str) -> tuple[str, str | None]:
        """Query task status.

        ``POST {base}/api/v1/doubao/get_result`` with ``{"id": task_id}``

        Returns ``(status, video_url | None)`` where status is one of
        ``"processing"``, ``"done"``, or ``"failed"``.
        """
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            resp = await client.post(
                f"{self._base_url}/api/v1/doubao/get_result",
                headers=self._auth_headers(),
                json={"id": task_id},
            )
            resp.raise_for_status()
            data = resp.json()

        raw_status = (data.get("status") or "").lower()

        logger.debug("[seedance] Task %s status: %s", task_id, raw_status)

        if raw_status in ("succeeded", "success", "completed", "done"):
            video_url = self._extract_video_url(data)
            return "done", video_url

        if raw_status in ("failed", "error", "cancelled"):
            error = data.get("error") or "unknown"
            logger.error("[seedance] Task %s failed: %s", task_id, error)
            return "failed", None

        return "processing", None

    async def _poll_until_done(self, task_id: str) -> str:
        """Poll task status until completion, returning the video URL."""
        elapsed = 0.0
        while elapsed < _POLL_TIMEOUT_S:
            await asyncio.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S

            status, video_url = await self._check_task(task_id)

            if status == "done":
                if not video_url:
                    raise RuntimeError(
                        f"Seedance task {task_id} completed but no video URL"
                    )
                return video_url

            if status == "failed":
                raise RuntimeError(
                    f"Seedance generation failed (task={task_id})"
                )

            logger.info(
                "[seedance] Task %s: %s (%.0fs elapsed)",
                task_id, status, elapsed,
            )

        raise TimeoutError(
            f"Seedance generation timed out after {_POLL_TIMEOUT_S}s "
            f"(task={task_id})"
        )

    @staticmethod
    def _extract_video_url(data: dict[str, Any]) -> str | None:
        """Extract video URL from the task result.

        Known response format::

            {
                "id": "task_...",
                "status": "succeeded",
                "content": {"video_url": "https://ark-acg-cn-beijing..."},
                ...
            }
        """
        # Primary format: content.video_url
        content = data.get("content")
        if isinstance(content, dict):
            if url := content.get("video_url"):
                return url

        # Fallback: data.video_url
        if url := data.get("video_url"):
            return url

        # Fallback: output.video_url
        if url := data.get("output", {}).get("video_url"):
            return url

        return None

    async def _download_video(self, url: str) -> bytes:
        """Download video bytes from URL."""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    @staticmethod
    def _estimate_cost_for(request: GenerationRequest) -> float:
        return round(request.duration_seconds * _COST_PER_SECOND_USD, 4)
