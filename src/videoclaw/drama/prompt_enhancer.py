"""PromptEnhancer — generates Seedance 2.0 optimised visual prompts for drama scenes.

Follows the five-part director-style prompt anatomy::

    Camera (shot + movement) → Subject (character + action) → Scene → Style → Constraints

Ensures every scene's ``visual_prompt`` includes complete character appearance,
Seedance-friendly camera vocabulary, style anchors, and realism modifiers —
regardless of LLM output quality.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from videoclaw.drama.models import ShotScale

if TYPE_CHECKING:
    from videoclaw.drama.models import DramaScene, DramaSeries, Episode

# ---------------------------------------------------------------------------
# Seedance 2.0 camera vocabulary
# ---------------------------------------------------------------------------

# Shot scale → Seedance-optimised phrasing (use lens-bucket feel, not mm)
SHOT_SCALE_LABELS: dict[ShotScale, str] = {
    ShotScale.CLOSE_UP: "close-up shot, telephoto feel",
    ShotScale.MEDIUM_CLOSE: "medium close-up shot, normal lens feel",
    ShotScale.MEDIUM: "medium shot, normal lens feel",
    ShotScale.WIDE: "wide shot, wide-angle feel",
    ShotScale.EXTREME_WIDE: "extreme wide establishing shot, wide-angle feel",
}

# camera_movement → Seedance-friendly single-verb phrasing
# Rule: ONE motion verb per shot — never stack multiple movements
CAMERA_MOVEMENT_LABELS: dict[str, str] = {
    "static": "locked-off static camera",
    "pan_left": "slow pan left",
    "pan_right": "slow pan right",
    "dolly_in": "slow dolly in",
    "tracking": "gimbal-smooth tracking shot",
    "crane_up": "slow crane up",
    "handheld": "handheld camera, slight sway",
}

# Shot scale → best camera pairing (Seedance 2.0 guide)
_SHOT_CAMERA_AFFINITY: dict[ShotScale, set[str]] = {
    ShotScale.CLOSE_UP: {"dolly_in", "static"},      # tiny push-ins or locked
    ShotScale.MEDIUM_CLOSE: {"handheld", "tracking"}, # personal or polished
    ShotScale.MEDIUM: {"handheld", "tracking", "dolly_in"},
    ShotScale.WIDE: {"static", "dolly_in", "crane_up"},
    ShotScale.EXTREME_WIDE: {"static", "crane_up"},
}

# Models that require English-only prompts
_ENGLISH_ONLY_PREFIXES = ("sora", "runway", "pika", "veo", "seedance")

# CJK + full-width character ranges
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+')

# Realism modifiers appended to every Seedance prompt
_REALISM_MODIFIERS = "film grain, natural lighting, shallow depth of field"

# Default constraints (3-5 items max per Seedance best practice)
_DEFAULT_CONSTRAINTS = (
    "no text overlays, no watermarks, no extra characters, "
    "anatomically correct hands, no sudden camera jumps"
)


# ---------------------------------------------------------------------------
# PromptEnhancer
# ---------------------------------------------------------------------------

class PromptEnhancer:
    """Enriches scene visual prompts with Seedance 2.0 optimised structure.

    Output follows the five-part director-style anatomy::

        Camera → Subject → Scene → Style → Constraints

    Parameters
    ----------
    strip_chinese:
        - ``None`` (default): auto-detect per ``model_id``.
        - ``True``: always strip CJK characters.
        - ``False``: never strip.
    """

    def __init__(self, strip_chinese: bool | None = None) -> None:
        self._strip_chinese = strip_chinese

    # -- public API ---------------------------------------------------------

    def should_strip_chinese(self, model_id: str) -> bool:
        """Return ``True`` if CJK characters should be removed for *model_id*."""
        if self._strip_chinese is not None:
            return self._strip_chinese
        return any(model_id.startswith(p) for p in _ENGLISH_ONLY_PREFIXES)

    def enhance_scene_prompt(self, scene: DramaScene, series: DramaSeries) -> str:
        """Build a Seedance 2.0 optimised visual prompt for a single *scene*.

        Five-part structure:
        1. **Camera** — shot scale + single camera movement verb
        2. **Subject** — character appearance (repeated every scene for consistency)
        3. **Scene** — original visual_prompt (setting + action + lighting)
        4. **Style** — series style anchor + realism modifiers
        5. **Constraints** — 3-5 bans to prevent artifacts
        """
        parts: list[str] = []

        # --- 1. Camera ---
        shot_label = (
            SHOT_SCALE_LABELS.get(scene.shot_scale, "")
            if scene.shot_scale
            else ""
        )
        cam_label = CAMERA_MOVEMENT_LABELS.get(
            scene.camera_movement, scene.camera_movement or ""
        )
        camera_parts = [p for p in (shot_label, cam_label) if p]
        if camera_parts:
            parts.append(", ".join(camera_parts) + ".")

        # --- 2. Subject (character descriptions — repeated for consistency) ---
        char_map = {c.name: c for c in series.characters}
        for name in scene.characters_present:
            char = char_map.get(name)
            if char and char.visual_prompt:
                parts.append(
                    f"Same character {name} — {char.visual_prompt.rstrip('.')}."
                )

        # --- 3. Scene (original visual prompt — setting + action + mood) ---
        if scene.visual_prompt:
            parts.append(scene.visual_prompt.rstrip(".") + ".")

        # --- 4. Conditionally strip CJK ---
        text = " ".join(parts)
        if self.should_strip_chinese(series.model_id):
            text = self._strip_cjk(text)

        # --- 5. Style anchor + realism modifiers ---
        style_tag = (
            f"Style: {series.style}, vertical {series.aspect_ratio}, "
            f"{_REALISM_MODIFIERS}."
        )
        text = text.rstrip() + " " + style_tag

        # --- 6. Constraints ---
        text = text.rstrip() + " " + f"Constraints: {_DEFAULT_CONSTRAINTS}."

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
