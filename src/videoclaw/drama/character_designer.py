"""Character reference image generation for drama series.

Uses EvolinkImageGenerator to produce consistent character portraits
that serve as visual anchors for downstream video generation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from videoclaw.drama.models import DramaManager, DramaSeries
from videoclaw.generation.image import EvolinkImageGenerator

logger = logging.getLogger(__name__)

# Prompt template for character reference images
CHARACTER_IMAGE_PROMPT = """\
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
    """Generates reference images for all characters in a drama series."""

    def __init__(
        self,
        image_generator: EvolinkImageGenerator | None = None,
        drama_manager: DramaManager | None = None,
    ) -> None:
        self._img_gen = image_generator
        self._drama_mgr = drama_manager or DramaManager()

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

        Skips characters that already have a reference_image unless *force* is True.
        """
        from videoclaw.drama.locale import get_locale

        gen = self._ensure_generator()
        char_dir = self._char_dir(series.series_id)
        style = series.style or "cinematic"

        locale = get_locale(series.language)
        style_line = locale.character_image_style.format(style=style)

        for character in series.characters:
            if character.reference_image and not force:
                logger.info("Skipping %s (already has reference image)", character.name)
                continue

            appearance = clean_visual_prompt(character.visual_prompt)
            prompt = CHARACTER_IMAGE_PROMPT.format(
                appearance=appearance,
                style_line=style_line,
            )

            safe_name = re.sub(r"[^\w\-]", "_", character.name).strip("_")
            filename = f"{safe_name}.png"

            logger.info("Generating reference image for %s", character.name)
            path = await gen.generate(
                prompt,
                output_dir=char_dir,
                filename=filename,
            )
            character.reference_image = str(path)

        self._drama_mgr.save(series)
        logger.info("Character designs saved for series %s", series.series_id)
        return series

    def _char_dir(self, series_id: str) -> Path:
        return self._drama_mgr.base_dir / series_id / "characters"
