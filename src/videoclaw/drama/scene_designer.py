"""Scene/environment reference image generation for drama series.

Extracts unique locations from episode scenes and generates reference images
to ensure visual consistency across shots set in the same environment.

Analogous to :class:`CharacterDesigner` but for backgrounds/settings.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from videoclaw.drama.models import DramaManager, DramaSeries, Episode
from videoclaw.generation.image import EvolinkImageGenerator

logger = logging.getLogger(__name__)

# Prompt template for scene/environment reference images
SCENE_IMAGE_PROMPT = """\
{description}

Style: {style_line}
Framing: wide establishing shot, no people, empty scene
Lighting: {lighting}
Atmosphere: cinematic, atmospheric, highly detailed
Quality: photorealistic, 8K, clean composition\
"""


class SceneLocation:
    """A unique location extracted from episode scenes."""

    __slots__ = ("name", "description", "reference_image")

    def __init__(
        self,
        name: str,
        description: str = "",
        reference_image: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.reference_image = reference_image

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "reference_image": self.reference_image,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneLocation:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            reference_image=data.get("reference_image"),
        )


def extract_locations(episodes: list[Episode]) -> list[SceneLocation]:
    """Extract unique scene locations from episode scene descriptions.

    Groups scenes by normalised location keywords and returns a deduplicated
    list of :class:`SceneLocation` objects.
    """
    seen: dict[str, SceneLocation] = {}

    for ep in episodes:
        for scene in ep.scenes:
            # Use the first sentence of the visual_prompt as location key
            loc_key = _extract_location_key(scene.visual_prompt or scene.description)
            if not loc_key:
                continue

            normalised = _normalise_key(loc_key)
            if normalised not in seen:
                seen[normalised] = SceneLocation(
                    name=normalised,
                    description=loc_key,
                )

    return list(seen.values())


def _extract_location_key(text: str) -> str:
    """Extract the setting/environment portion from a visual prompt.

    Heuristic: take the first clause before any character or action description.
    """
    if not text:
        return ""
    # Take text up to first comma or period, or first 80 chars
    for sep in (",", ".", "，", "。", ";", "；"):
        idx = text.find(sep)
        if 10 < idx < 100:
            return text[:idx].strip()
    return text[:80].strip()


def _normalise_key(text: str) -> str:
    """Normalise a location key for deduplication."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:60]


class SceneDesigner:
    """Generates reference images for unique scene locations in a drama series."""

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

    async def design_scenes(
        self,
        series: DramaSeries,
        *,
        force: bool = False,
    ) -> list[SceneLocation]:
        """Extract unique locations and generate reference images.

        Returns the list of :class:`SceneLocation` objects with paths populated.
        Also stores them in ``series.metadata["locations"]``.
        """
        from videoclaw.drama.locale import get_locale

        gen = self._ensure_generator()
        scene_dir = self._scene_dir(series.series_id)
        style = series.style or "cinematic"
        locale = get_locale(series.language)
        style_line = locale.character_image_style.format(style=style)

        # Extract locations from all episodes
        locations = extract_locations(series.episodes)
        if not locations:
            logger.info("No unique locations extracted from episodes")
            return []

        logger.info("Extracted %d unique locations", len(locations))

        for loc in locations:
            if loc.reference_image and not force:
                logger.info("Skipping location %r (already has image)", loc.name)
                continue

            prompt = SCENE_IMAGE_PROMPT.format(
                description=loc.description,
                style_line=style_line,
                lighting="natural lighting, golden hour" if "outdoor" in loc.description.lower()
                else "soft interior lighting, warm tones",
            )

            safe_name = re.sub(r"[^\w\-]", "_", loc.name).strip("_")[:50]
            filename = f"scene_{safe_name}.png"

            logger.info("Generating scene reference for %r", loc.name)
            path = await gen.generate(
                prompt,
                output_dir=scene_dir,
                filename=filename,
                size="16:9",  # wide establishing shot
            )
            loc.reference_image = str(path)

        # Persist to series metadata
        series.metadata["locations"] = [loc.to_dict() for loc in locations]
        self._drama_mgr.save(series)

        logger.info("Scene designs saved for series %s (%d locations)",
                     series.series_id, len(locations))
        return locations

    def _scene_dir(self, series_id: str) -> Path:
        return self._drama_mgr.base_dir / series_id / "scenes"
