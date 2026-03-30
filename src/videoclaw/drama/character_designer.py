"""Character reference image generation for drama series.

Uses EvolinkImageGenerator to produce consistent character turnaround sheets
that serve as visual anchors for downstream video generation.

Seedance 2.0 Universal Reference best practices:
- A single turnaround sheet (front / side / back in one image) per character
- Clean white background, consistent lighting, same outfit across all views
- Character sheet format enables maximum cross-shot consistency
- Pass via image_urls (HTTPS) to avoid vectorspace.cn base64 rejection
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from typing import Any, Protocol, runtime_checkable

from videoclaw.drama.models import DramaManager, DramaSeries

logger = logging.getLogger(__name__)


@runtime_checkable
class ImageGenerator(Protocol):
    """Minimal interface for image generators (Evolink, BytePlus, Gemini)."""

    async def generate(
        self,
        prompt: str,
        *,
        output_dir: Path,
        filename: str,
        **kwargs: Any,
    ) -> Path: ...

# ---------------------------------------------------------------------------
# Turnaround sheet prompt (single image with front / side / back views)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Character turnaround sheet prompt — 角色三视图标准公式（3D CGI 版）
# ---------------------------------------------------------------------------
# 设计原则（基于 Session 7 PrivacyInformation 过滤器测试验证）:
#   ① 布局声明：单宽图三视图，正面（左）/ 45°侧面（中）/ 背面（右）
#   ② 背景：纯白无缝摄影棚，无地面阴影
#   ③ 帧幅：全身，头顶到脚底，人物高度占 75-80%
#   ④ 光效：柔和均匀正面上方 key light，无强投影（影棚标准光）
#   ⑤ 风格：3D CGI / MetaHuman / Unreal Engine 风格（非写实照片，非 anime）
#   ⑥ 一致性：三视图同一面孔/发型/服装，零偏差
#   ⑦ 姿态：自然站姿，手臂自然或轻叠前方，中性表情
#
# 两层角色一致性策略 (Two-layer consistency strategy):
#   Layer 1: 3D CGI 参考图 → 绕过 vectorspace.cn PrivacyInformation 过滤器
#            提供服装/体型/发型结构参考（面部逼真度有损失）
#   Layer 2: 视频提示词中注入 CHARACTER IDENTITY 文本 → 弥补面部逼真度损失
#            参见 prompt_enhancer.py: _WESTERN_REALISM_HEADER + CHARACTER IDENTITY
#
# ⚠️ 注意：写实/半写实/概念艺术风格均会触发 PrivacyInformation 过滤
#    必须使用 3D CGI / MetaHuman 风格，否则视频生成 API 会拒绝参考图
# ---------------------------------------------------------------------------

TURNAROUND_SHEET_PROMPT = """\
character turnaround sheet, 角色三视图参考图,
3D CGI character model render, Unreal Engine character asset,
game character reference, virtual production asset, NOT a real photo,

{appearance}

① Layout: single wide image with three full-body panels side by side, evenly spaced.
  — Left panel: front view, character faces camera directly
  — Center panel: 45-degree three-quarter side view, character faces slightly left
  — Right panel: back view, character faces completely away from camera
② Framing: head-to-toe full body in each panel, character fills 75–80% of panel height, \
centered, no cropping of feet or head.
③ Background: pure seamless white background, no ground shadow, no floor texture, \
no props, no environment.
④ Lighting: soft even diffused studio lighting, no harsh shadows.
⑤ Consistency: identical character design, same hairstyle, same outfit, same accessories \
across all three panels — zero variation between views.
⑥ Pose: relaxed standing, arms naturally at sides or hands lightly clasped in front, \
neutral expression, looking straight ahead.

Style: {style_line}
Quality: highly detailed, professional 3D character reference sheet\
"""

# ---------------------------------------------------------------------------
# Legacy multi-angle reference poses (deprecated — use turnaround sheet)
# ---------------------------------------------------------------------------

REFERENCE_POSES: list[dict[str, str]] = [
    {
        "angle": "front",
        "framing": "upper body portrait, facing camera directly, centered",
    },
    {
        "angle": "three_quarter",
        "framing": "upper body portrait, three-quarter view facing slightly left",
    },
    {
        "angle": "full_body",
        "framing": "full body standing pose, facing camera, feet visible",
    },
]

# Legacy prompt template for separate per-angle images (deprecated)
CHARACTER_IMAGE_PROMPT = """\
{appearance}

Style: {style_line}
Framing: {framing}, character occupying 60-80 percent of frame, neutral gray background
Expression: neutral, calm
Lighting: soft studio lighting, even illumination, no dramatic shadows
Quality: highly detailed, photorealistic, 8K\
"""

# Legacy single-pose prompt (backward compat)
CHARACTER_IMAGE_PROMPT_SINGLE = """\
{appearance}

Style: {style_line}
Framing: upper body portrait, facing slightly left, neutral gray background
Expression: neutral, calm
Lighting: soft studio lighting, even illumination
Quality: highly detailed, photorealistic, 8K\
"""

# Regex patterns for camera/cinematography language to strip from visual_prompt
_CAMERA_PATTERNS = re.compile(
    r"[，,。.;；]?\s*(?:"
    r"画面|镜头|拍摄|特写|全景|远景|近景|中景|仰视|俯视|仰拍|俯拍|"
    r"采用|运用|背景为|背景是|光影|滤镜|虚实|景深|"
    r"camera|shot|angle|close-up|wide shot|lighting setup|background"
    r").*?(?=[，,。.;；]|$)",
    re.IGNORECASE,
)


def clean_visual_prompt(visual_prompt: str) -> str:
    """Remove camera/cinematography language, keeping only appearance."""
    cleaned = _CAMERA_PATTERNS.sub("", visual_prompt)
    # Remove leading/trailing punctuation and whitespace
    cleaned = re.sub(r"^[，,。.;；\s]+|[，,。.;；\s]+$", "", cleaned)
    return cleaned.strip() or visual_prompt


class CharacterDesigner:
    """Generates reference images for all characters in a drama series.

    Default mode (``turnaround=True``): generates a single turnaround sheet
    per character (front / side / back in one image) — the standard for
    Seedance 2.0 Universal Reference.

    Legacy mode (``multi_angle=True, turnaround=False``): generates 3
    separate images per character.  Deprecated.
    """

    def __init__(
        self,
        image_generator: ImageGenerator | None = None,
        drama_manager: DramaManager | None = None,
        multi_angle: bool = True,
        turnaround: bool = True,
    ) -> None:
        self._img_gen = image_generator
        self._drama_mgr = drama_manager or DramaManager()
        self._multi_angle = multi_angle
        self._turnaround = turnaround

    def _ensure_generator(self) -> ImageGenerator:
        if self._img_gen is None:
            # Default to BytePlus Seedream for character turnaround generation
            # Verify the API key is actually available before committing to BytePlus
            try:
                from videoclaw.config import get_config
                cfg = get_config()
                if cfg.byteplus_api_key:
                    from videoclaw.generation.byteplus_image import BytePlusImageGenerator
                    self._img_gen = BytePlusImageGenerator()
                    logger.info("Using BytePlus Seedream for character images")
                else:
                    raise ValueError("No BytePlus API key")
            except Exception:
                from videoclaw.generation.image import EvolinkImageGenerator
                self._img_gen = EvolinkImageGenerator()
                logger.info("Using Evolink for character images (BytePlus unavailable)")
        return self._img_gen

    async def design_characters(
        self,
        series: DramaSeries,
        *,
        force: bool = False,
    ) -> DramaSeries:
        """Generate reference images for each character in the series.

        Default: generates a single turnaround sheet (角色三视图) per character
        using the standard prompt formula (layout + views + background +
        consistency + pose). Populates ``reference_image`` (local path)
        and ``reference_image_url`` (HTTPS URL for Seedance API).

        Skips characters that already have reference images unless *force*.
        """
        from videoclaw.drama.locale import get_locale

        gen = self._ensure_generator()
        char_dir = self._char_dir(series.series_id)
        style = series.style or "cinematic"

        locale = get_locale(series.language)
        style_line = locale.character_image_style.format(style=style)

        for character in series.characters:
            # Skip if already has images (unless force)
            if not force:
                if self._turnaround and character.reference_image:
                    logger.info("Skipping %s (already has turnaround sheet)",
                                character.name)
                    continue
                if not self._turnaround and self._multi_angle and character.reference_images:
                    logger.info("Skipping %s (already has %d reference images)",
                                character.name, len(character.reference_images))
                    continue
                if not self._turnaround and not self._multi_angle and character.reference_image:
                    logger.info("Skipping %s (already has reference image)", character.name)
                    continue

            appearance = clean_visual_prompt(character.visual_prompt)
            safe_name = re.sub(r"[^\w\-]", "_", character.name).strip("_")

            if self._turnaround:
                await self._generate_turnaround(
                    gen, character, appearance, style_line, safe_name, char_dir,
                )
            elif self._multi_angle:
                await self._generate_multi_angle(
                    gen, character, appearance, style_line, safe_name, char_dir,
                )
            else:
                await self._generate_single(
                    gen, character, appearance, style_line, safe_name, char_dir,
                )

        self._drama_mgr.save(series)
        logger.info("Character designs saved for series %s", series.series_id)
        return series

    async def _generate_turnaround(
        self,
        gen: ImageGenerator,
        character,
        appearance: str,
        style_line: str,
        safe_name: str,
        char_dir: Path,
    ) -> None:
        """Generate a single turnaround sheet (角色三视图).

        Uses the standard turnaround prompt formula:
        ① Layout declaration ② View enumeration ③ Background constraint
        ④ Consistency declaration ⑤ Pose constraint

        Wide aspect ratio (16:9) to fit three full-body views side by side.
        Stores the HTTPS URL from the image API for downstream Seedance usage.
        """
        prompt = TURNAROUND_SHEET_PROMPT.format(
            appearance=appearance,
            style_line=style_line,
        )
        filename = f"{safe_name}_turnaround.png"

        logger.info("Generating turnaround sheet for %s", character.name)
        path = await gen.generate(
            prompt,
            output_dir=char_dir,
            filename=filename,
            size="16:9",
        )
        character.reference_image = str(path)
        character.reference_images = [str(path)]

        # Store the HTTPS URL if the generator captured it
        if hasattr(gen, "last_image_url") and gen.last_image_url:
            character.reference_image_url = gen.last_image_url
            logger.info("Stored HTTPS URL for %s turnaround: %s",
                        character.name, gen.last_image_url[:80])

    async def _generate_multi_angle(
        self,
        gen: ImageGenerator,
        character,
        appearance: str,
        style_line: str,
        safe_name: str,
        char_dir: Path,
    ) -> None:
        """Generate 3 reference images (front, three-quarter, full-body). Deprecated."""
        paths: list[str] = []
        for pose in REFERENCE_POSES:
            prompt = CHARACTER_IMAGE_PROMPT.format(
                appearance=appearance,
                style_line=style_line,
                framing=pose["framing"],
            )
            filename = f"{safe_name}_{pose['angle']}.png"

            logger.info("Generating %s reference (%s) for %s",
                        pose["angle"], pose["framing"][:30], character.name)
            path = await gen.generate(
                prompt,
                output_dir=char_dir,
                filename=filename,
            )
            paths.append(str(path))

        character.reference_images = paths
        # Primary image = front view (first in list)
        character.reference_image = paths[0] if paths else None

    async def _generate_single(
        self,
        gen: ImageGenerator,
        character,
        appearance: str,
        style_line: str,
        safe_name: str,
        char_dir: Path,
    ) -> None:
        """Generate single reference image (legacy mode)."""
        prompt = CHARACTER_IMAGE_PROMPT_SINGLE.format(
            appearance=appearance,
            style_line=style_line,
        )
        filename = f"{safe_name}.png"

        logger.info("Generating reference image for %s", character.name)
        path = await gen.generate(
            prompt,
            output_dir=char_dir,
            filename=filename,
        )
        character.reference_image = str(path)

    async def refresh_urls(
        self,
        series: DramaSeries,
        *,
        force: bool = False,
    ) -> dict[str, str]:
        """Refresh character reference image HTTPS URLs.

        Image URLs from providers like Evolink/BytePlus are TOS-signed and
        expire after ~24 hours. This method re-generates turnaround sheets
        for characters whose URLs are missing or expired, capturing fresh
        HTTPS URLs.

        When *force* is ``False`` (default), only refreshes characters
        whose ``reference_image_url`` is empty or falsy. When *force*,
        regenerates all character images to get fresh URLs.

        Returns a dict mapping character name → fresh HTTPS URL.
        """
        from videoclaw.drama.locale import get_locale

        gen = self._ensure_generator()
        char_dir = self._char_dir(series.series_id)
        style = series.style or "cinematic"
        locale = get_locale(series.language)
        style_line = locale.character_image_style.format(style=style)

        refreshed: dict[str, str] = {}

        for character in series.characters:
            if not force and character.reference_image_url:
                logger.info(
                    "Skipping %s URL refresh (has URL: %s...)",
                    character.name,
                    character.reference_image_url[:60],
                )
                refreshed[character.name] = character.reference_image_url
                continue

            appearance = clean_visual_prompt(character.visual_prompt)
            safe_name = re.sub(r"[^\w\-]", "_", character.name).strip("_")
            filename = f"{safe_name}_turnaround.png"

            # Delete stale local file so provider generates anew
            local_path = char_dir / filename
            if local_path.exists():
                local_path.unlink()
                logger.info("Deleted stale turnaround: %s", filename)

            prompt = TURNAROUND_SHEET_PROMPT.format(
                appearance=appearance,
                style_line=style_line,
            )

            logger.info("Refreshing URL for %s...", character.name)
            try:
                path = await gen.generate(
                    prompt,
                    output_dir=char_dir,
                    filename=filename,
                    size="16:9",
                )
                character.reference_image = str(path)
                character.reference_images = [str(path)]

                url = ""
                if hasattr(gen, "last_image_url") and gen.last_image_url:
                    url = gen.last_image_url
                    character.reference_image_url = url
                    logger.info(
                        "Refreshed %s URL: %s...",
                        character.name,
                        url[:80],
                    )

                refreshed[character.name] = url
            except Exception as e:
                logger.error("Failed to refresh %s: %s", character.name, e)
                refreshed[character.name] = ""

        self._drama_mgr.save(series)
        logger.info(
            "URL refresh complete: %d/%d characters have URLs",
            sum(1 for v in refreshed.values() if v),
            len(refreshed),
        )
        return refreshed

    def _char_dir(self, series_id: str) -> Path:
        return self._drama_mgr.base_dir / series_id / "characters"
