"""Character reference image generation for drama series.

Uses EvolinkImageGenerator to produce consistent character portraits
that serve as visual anchors for downstream video generation.

Seedance 2.0 Universal Reference best practices:
- 1-4 reference images per character for maximum consistency
- Single subject, neutral pose, plain background, clear lighting
- Three views recommended: front, three-quarter, full-body
- Character should occupy 60-80% of the frame
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from videoclaw.drama.models import DramaManager, DramaSeries
from videoclaw.generation.image import EvolinkImageGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi-angle reference poses (Seedance 2.0 Universal Reference)
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

# Prompt template for character reference images
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

    When ``multi_angle=True`` (default), generates 3 reference images per
    character (front, three-quarter, full-body) following Seedance 2.0's
    Universal Reference best practices.
    """

    def __init__(
        self,
        image_generator: EvolinkImageGenerator | None = None,
        drama_manager: DramaManager | None = None,
        multi_angle: bool = True,
    ) -> None:
        self._img_gen = image_generator
        self._drama_mgr = drama_manager or DramaManager()
        self._multi_angle = multi_angle

    def _ensure_generator(self) -> EvolinkImageGenerator:
        if self._img_gen is None:
            self._img_gen = EvolinkImageGenerator()
        return self._img_gen

    async def design_characters(
        self,
        series: DramaSeries,
        *,
        force: bool = False,
    ) -> DramaSeries:
        """Generate reference images for each character in the series.

        When ``multi_angle`` is enabled, produces 3 images per character
        (front / three-quarter / full-body) and populates both
        ``reference_images`` (list) and ``reference_image`` (primary front).

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
                if self._multi_angle and character.reference_images:
                    logger.info("Skipping %s (already has %d reference images)",
                                character.name, len(character.reference_images))
                    continue
                if not self._multi_angle and character.reference_image:
                    logger.info("Skipping %s (already has reference image)", character.name)
                    continue

            appearance = clean_visual_prompt(character.visual_prompt)
            safe_name = re.sub(r"[^\w\-]", "_", character.name).strip("_")

            if self._multi_angle:
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

    async def _generate_multi_angle(
        self,
        gen: EvolinkImageGenerator,
        character,
        appearance: str,
        style_line: str,
        safe_name: str,
        char_dir: Path,
    ) -> None:
        """Generate 3 reference images (front, three-quarter, full-body)."""
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
        gen: EvolinkImageGenerator,
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
