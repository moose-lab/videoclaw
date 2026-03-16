"""PromptEnhancer — programmatically enriches visual prompts with character
descriptions, shot/camera labels, and style tags.

Ensures every scene's ``visual_prompt`` includes complete character appearance,
shot scale, camera movement, and series style — regardless of LLM output quality.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from videoclaw.drama.models import ShotScale

if TYPE_CHECKING:
    from videoclaw.drama.models import DramaScene, DramaSeries, Episode

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

SHOT_SCALE_LABELS: dict[ShotScale, str] = {
    ShotScale.CLOSE_UP: "close-up shot",
    ShotScale.MEDIUM_CLOSE: "medium close-up shot",
    ShotScale.MEDIUM: "medium shot",
    ShotScale.WIDE: "wide shot",
    ShotScale.EXTREME_WIDE: "extreme wide shot",
}

CAMERA_MOVEMENT_LABELS: dict[str, str] = {
    "static": "static camera",
    "pan_left": "camera panning left",
    "pan_right": "camera panning right",
    "dolly_in": "camera dollying in",
    "tracking": "tracking shot",
    "crane_up": "crane shot moving up",
    "handheld": "handheld camera",
}

# Models that only accept English prompts — whitelist approach so new
# domestic models work without touching this list.
_ENGLISH_ONLY_PREFIXES = ("sora", "runway", "pika", "veo")

# CJK + full-width character ranges
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+')


# ---------------------------------------------------------------------------
# PromptEnhancer
# ---------------------------------------------------------------------------

class PromptEnhancer:
    """Enriches scene visual prompts with character, camera, and style info.

    Parameters
    ----------
    strip_chinese:
        - ``None`` (default): auto-detect per ``model_id`` using
          :pymethod:`should_strip_chinese`.
        - ``True``: always strip CJK characters from output.
        - ``False``: never strip CJK characters.
    """

    def __init__(self, strip_chinese: bool | None = None) -> None:
        self._strip_chinese = strip_chinese

    # -- public API ---------------------------------------------------------

    def should_strip_chinese(self, model_id: str) -> bool:
        """Return ``True`` if CJK characters should be removed for *model_id*.

        When ``strip_chinese`` was set explicitly at init, that value wins.
        Otherwise only known English-only model prefixes trigger stripping.
        """
        if self._strip_chinese is not None:
            return self._strip_chinese
        return any(model_id.startswith(p) for p in _ENGLISH_ONLY_PREFIXES)

    def enhance_scene_prompt(self, scene: DramaScene, series: DramaSeries) -> str:
        """Build an enhanced visual prompt for a single *scene*.

        Steps:
        1. Start with ``scene.visual_prompt``
        2. Inject character appearance descriptions
        3. Prepend shot scale / camera movement labels
        4. Conditionally strip CJK characters
        5. Append series style tag line
        """
        parts: list[str] = []

        # 1. Shot scale + camera movement header
        shot_label = SHOT_SCALE_LABELS.get(scene.shot_scale, "") if scene.shot_scale else ""
        cam_label = CAMERA_MOVEMENT_LABELS.get(scene.camera_movement, scene.camera_movement)
        header_parts = [p for p in (shot_label, cam_label) if p]
        if header_parts:
            parts.append(", ".join(header_parts) + ".")

        # 2. Original visual prompt
        if scene.visual_prompt:
            parts.append(scene.visual_prompt.rstrip(".") + ".")

        # 3. Character descriptions
        char_map = {c.name: c for c in series.characters}
        for name in scene.characters_present:
            char = char_map.get(name)
            if char and char.visual_prompt:
                parts.append(f"Character: {name} — {char.visual_prompt.rstrip('.')}.")

        # 4. Conditionally strip CJK
        text = " ".join(parts)
        if self.should_strip_chinese(series.model_id):
            text = self._strip_cjk(text)

        # 5. Style tag line
        style_tag = f"Style: {series.style}, vertical {series.aspect_ratio}, cinematic lighting."
        text = text.rstrip() + " " + style_tag

        return text.strip()

    def enhance_all_scenes(self, episode: Episode, series: DramaSeries) -> Episode:
        """Enhance all scenes in *episode* in-place. Returns the mutated episode."""
        for scene in episode.scenes:
            scene.visual_prompt = self.enhance_scene_prompt(scene, series)
        return episode

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _strip_cjk(text: str) -> str:
        """Remove CJK / full-width characters and collapse resulting whitespace."""
        text = _CJK_RE.sub("", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
