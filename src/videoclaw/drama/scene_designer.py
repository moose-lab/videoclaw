"""Scene/environment and prop reference image generation for drama series.

Extracts unique locations and key props from episode scenes and generates
reference images to ensure visual consistency across shots set in the same
environment or featuring the same objects.

Analogous to :class:`CharacterDesigner` but for backgrounds/settings and props.

Production flow::

    剧本分镜 → 角色参考图 → **场景/物品参考图** → 视频生成
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from videoclaw.drama.models import DramaManager, DramaSeries, Episode

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

# Prompt template for prop/item reference images
PROP_IMAGE_PROMPT = """\
Product photography of {description}.
Isolated on white background, studio lighting, multiple angles if possible.
Highly detailed, photorealistic, clean composition.
Style: {style_line}\
"""


class SceneLocation:
    """A unique location extracted from episode scenes."""

    __slots__ = ("description", "name", "reference_image", "reference_image_url")

    def __init__(
        self,
        name: str,
        description: str = "",
        reference_image: str | None = None,
        reference_image_url: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.reference_image = reference_image
        self.reference_image_url = reference_image_url

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "reference_image": self.reference_image,
            "reference_image_url": self.reference_image_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneLocation:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            reference_image=data.get("reference_image"),
            reference_image_url=data.get("reference_image_url"),
        )


class PropAsset:
    """A key prop or item that needs visual consistency across shots."""

    __slots__ = ("description", "name", "reference_image", "reference_image_url", "scenes_used")

    def __init__(
        self,
        name: str,
        description: str = "",
        scenes_used: list[str] | None = None,
        reference_image: str | None = None,
        reference_image_url: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.scenes_used = scenes_used or []
        self.reference_image = reference_image
        self.reference_image_url = reference_image_url

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scenes_used": self.scenes_used,
            "reference_image": self.reference_image,
            "reference_image_url": self.reference_image_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PropAsset:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            scenes_used=data.get("scenes_used", []),
            reference_image=data.get("reference_image"),
            reference_image_url=data.get("reference_image_url"),
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


def extract_props(episodes: list[Episode]) -> list[PropAsset]:
    """Extract key props/items from episode scenes that need visual consistency.

    Identifies props mentioned multiple times across scenes or in important
    dramatic moments (e.g. a letter, a weapon, a vehicle, a specific object).
    """
    # Count prop mentions across scenes
    prop_mentions: dict[str, dict[str, Any]] = {}

    for ep in episodes:
        for scene in ep.scenes:
            text = f"{scene.visual_prompt} {scene.dialogue} {scene.description}".lower()
            scene_id = scene.scene_id

            # Extract props via keyword patterns
            for prop_name, pattern in _PROP_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    if prop_name not in prop_mentions:
                        prop_mentions[prop_name] = {
                            "description": _extract_prop_description(prop_name, text),
                            "scenes": [],
                        }
                    if scene_id not in prop_mentions[prop_name]["scenes"]:
                        prop_mentions[prop_name]["scenes"].append(scene_id)

    # Only keep props that appear in 2+ scenes (need consistency) or are dramatically important
    props: list[PropAsset] = []
    for name, info in prop_mentions.items():
        if len(info["scenes"]) >= 2:
            props.append(PropAsset(
                name=name,
                description=info["description"],
                scenes_used=info["scenes"],
            ))

    return props


# Common prop patterns for Western drama
_PROP_PATTERNS: list[tuple[str, str]] = [
    ("brochure", r"\bbrochure\b"),
    ("name_badge", r"\b(name\s*badge|lanyard|name\s*tag)\b"),
    ("letter", r"\bletter\b"),
    ("phone", r"\b(phone|smartphone|cellphone)\b"),
    ("gun", r"\b(gun|pistol|rifle|weapon)\b"),
    ("car", r"\b(car|vehicle|limousine|suv)\b"),
    ("ring", r"\b(ring|engagement\s*ring|wedding\s*ring)\b"),
    ("necklace", r"\b(necklace|pendant|jewel)\b"),
    ("document", r"\b(document|contract|deed|papers?)\b"),
    ("photograph", r"\b(photo|photograph|picture\s*frame)\b"),
    ("key", r"\b(key|key\s*card)\b"),
    ("bag", r"\b(bag|purse|briefcase|suitcase)\b"),
]


def _extract_prop_description(prop_name: str, context: str) -> str:
    """Extract a short description of the prop from its context."""
    # Try to find the prop in context and grab surrounding words
    pattern = rf"(\w[\w\s]{{0,30}}?\b{re.escape(prop_name)}\b[\w\s]{{0,30}})"
    match = re.search(pattern, context, re.IGNORECASE)
    if match:
        return match.group(1).strip()[:80]
    return prop_name


def _extract_location_key(text: str) -> str:
    """Extract the setting/environment portion from a visual prompt."""
    if not text:
        return ""
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
    """Generates reference images for scene locations and key props.

    After storyboard decomposition, call ``design_scenes()`` and
    ``design_props()`` to generate visual references that ensure
    consistency when the same location or prop appears across multiple shots.
    """

    def __init__(
        self,
        image_generator: Any | None = None,
        drama_manager: DramaManager | None = None,
    ) -> None:
        self._img_gen = image_generator
        self._drama_mgr = drama_manager or DramaManager()

    def _ensure_generator(self) -> Any:
        if self._img_gen is None:
            try:
                from videoclaw.config import get_config
                cfg = get_config()
                if cfg.byteplus_api_key:
                    from videoclaw.generation.byteplus_image import BytePlusImageGenerator
                    self._img_gen = BytePlusImageGenerator()
                    logger.info("Using BytePlus Seedream for scene/prop images")
                else:
                    raise ValueError("No BytePlus API key")
            except Exception:
                from videoclaw.generation.evolink_image import EvolinkImageGenerator
                self._img_gen = EvolinkImageGenerator()
                logger.info("Using Evolink for scene/prop images (BytePlus unavailable)")
        return self._img_gen

    async def design_scenes(
        self,
        series: DramaSeries,
        *,
        force: bool = False,
    ) -> list[SceneLocation]:
        """Extract unique locations and generate reference images.

        Returns the list of :class:`SceneLocation` objects with paths populated.
        Also stores them in ``series.metadata["locations"]`` and updates
        the ConsistencyManifest's ``scene_references``.
        """
        from videoclaw.drama.locale import get_locale

        gen = self._ensure_generator()
        scene_dir = self._scene_dir(series.series_id)
        style = series.style or "cinematic"
        locale = get_locale(series.language)
        style_line = locale.character_image_style.format(style=style)

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
                size="16:9",
            )
            loc.reference_image = str(path)
            loc.reference_image_url = getattr(gen, "last_image_url", None)

        # Persist to series metadata and consistency manifest
        series.metadata["locations"] = [loc.to_dict() for loc in locations]
        if series.consistency_manifest:
            for loc in locations:
                if loc.reference_image:
                    series.consistency_manifest.scene_references[loc.name] = loc.reference_image
        self._drama_mgr.save(series)

        logger.info("Scene designs saved for series %s (%d locations)",
                     series.series_id, len(locations))
        return locations

    async def design_props(
        self,
        series: DramaSeries,
        *,
        force: bool = False,
    ) -> list[PropAsset]:
        """Extract key props and generate reference images.

        Only generates images for props that appear in 2+ scenes (need
        consistency). Stores results in ``series.metadata["props"]`` and
        updates the ConsistencyManifest's ``prop_references``.
        """
        from videoclaw.drama.locale import get_locale

        gen = self._ensure_generator()
        prop_dir = self._prop_dir(series.series_id)
        style = series.style or "cinematic"
        locale = get_locale(series.language)
        style_line = locale.character_image_style.format(style=style)

        props = extract_props(series.episodes)
        if not props:
            logger.info("No recurring props found needing consistency images")
            return []

        logger.info("Extracted %d recurring props: %s",
                     len(props), [p.name for p in props])

        for prop in props:
            if prop.reference_image and not force:
                logger.info("Skipping prop %r (already has image)", prop.name)
                continue

            prompt = PROP_IMAGE_PROMPT.format(
                description=prop.description,
                style_line=style_line,
            )

            safe_name = re.sub(r"[^\w\-]", "_", prop.name).strip("_")[:50]
            filename = f"prop_{safe_name}.png"

            logger.info("Generating prop reference for %r (used in %d scenes)",
                         prop.name, len(prop.scenes_used))
            path = await gen.generate(
                prompt,
                output_dir=prop_dir,
                filename=filename,
                size="1:1",
            )
            prop.reference_image = str(path)
            prop.reference_image_url = getattr(gen, "last_image_url", None)

        # Persist to series metadata and consistency manifest
        series.metadata["props"] = [p.to_dict() for p in props]
        if series.consistency_manifest:
            for prop in props:
                if prop.reference_image:
                    series.consistency_manifest.prop_references[prop.name] = prop.reference_image
        self._drama_mgr.save(series)

        logger.info("Prop designs saved for series %s (%d props)",
                     series.series_id, len(props))
        return props

    def _scene_dir(self, series_id: str) -> Path:
        return self._drama_mgr.base_dir / series_id / "scenes"

    def _prop_dir(self, series_id: str) -> Path:
        return self._drama_mgr.base_dir / series_id / "props"
