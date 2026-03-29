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
# Character turnaround sheet prompt — 角色三视图标准公式
# ---------------------------------------------------------------------------
# 设计原则（参考 AI短剧行业标准三视图规范）:
#   ① 布局声明：单宽图三视图，正面（左）/ 45°侧面（中）/ 背面（右）
#   ② 背景：纯白无缝摄影棚，无地面阴影
#   ③ 帧幅：全身，头顶到脚底，人物高度占 75-80%
#   ④ 光效：柔和均匀正面上方 key light，无强投影（影棚标准光）
#   ⑤ 风格：写实电影感（photorealistic cinema），非插画/anime
#   ⑥ 一致性：三视图同一面孔/发型/服装，零偏差
#   ⑦ 姿态：自然站姿，手臂自然或轻叠前方，中性表情
#
# ⚠️ 注意：此模版生成写实风格图，用于 Seedance 角色参考图输入。
#    如遇 PrivacyInformation 过滤，说明人脸过于逼真，调整 style_line 降低真实度。
# ---------------------------------------------------------------------------

TURNAROUND_SHEET_PROMPT = """\
character reference sheet for film drama production, 角色三视图参考图,
fictional character design asset, AI-generated character,

{appearance}

① Layout: single wide image containing three full-body views side by side, \
evenly spaced, no overlap.
  — Left panel: front view, character faces camera directly
  — Center panel: 45-degree three-quarter side view, character faces slightly left
  — Right panel: back view, character faces completely away from camera
② Framing: head-to-toe full body in each panel, character fills 75–80% of panel height, \
centered, no cropping of feet or head.
③ Background: pure seamless white studio background, no ground shadow, \
no floor texture, no props, no environment.
④ Lighting: soft diffused studio lighting, even front-overhead key light, \
gentle falloff, no harsh shadows, professional studio photography quality.
⑤ Consistency: strictly identical face, same hairstyle, same outfit, same accessories \
across all three panels — zero variation in appearance between views.
⑥ Pose: relaxed natural standing pose, arms loosely at sides or hands lightly clasped \
in front, feet shoulder-width apart, neutral calm expression.

Style: {style_line}
Quality: 8K, highly detailed, sharp focus, professional film production character asset\
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
            try:
                from videoclaw.generation.byteplus_image import BytePlusImageGenerator
                self._img_gen = BytePlusImageGenerator()
            except Exception:
                from videoclaw.generation.image import EvolinkImageGenerator
                self._img_gen = EvolinkImageGenerator()
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

    def _char_dir(self, series_id: str) -> Path:
        return self._drama_mgr.base_dir / series_id / "characters"
