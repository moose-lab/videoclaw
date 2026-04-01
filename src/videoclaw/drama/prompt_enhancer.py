"""PromptEnhancer — generates Seedance 2.0 optimised visual prompts for drama scenes.

Follows the five-part director-style prompt anatomy::

    Camera (shot + movement) → Subject (character + action) → Scene → Style → Constraints

Plus a **text directive layer** that instructs Seedance to render in-video text:

- **Dialogue subtitles** — ``[Show subtitle at bottom: "..."]``
- **Narration subtitles** — ``[Narrator speaks: "...". Show subtitle: "..."]``
- **Title cards** — ``[Show large centered title text: "..."]``
- **Inner monologue** — ``[Character thinks: "...". Show subtitle: "..."]``
- **Character name cards** — ``[Show name card at bottom: "NAME, AGE — Role"]``

Ensures every scene's ``visual_prompt`` includes complete character appearance,
Seedance-friendly camera vocabulary, style anchors, and realism modifiers —
regardless of LLM output quality.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from videoclaw.drama.models import ShotScale

if TYPE_CHECKING:
    from videoclaw.drama.models import Character, DramaScene, DramaSeries, Episode

# ---------------------------------------------------------------------------
# Reference marker constants
# ---------------------------------------------------------------------------

_MAX_ENGLISH_WORDS = 1000
_MAX_CHINESE_CHARS = 500
_CJK_CHAR_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
_REF_KEY_RE = re.compile(r'[^a-zA-Z0-9]')


def _to_ref_key(name: str) -> str:
    """Convert a human-readable name to a ref key for ``[ref:key]`` markers.

    - Lowercase
    - Non-alphanumeric → underscore
    - Strip leading/trailing underscores

    Examples::

        "Ivy Angel"       → "ivy_angel"
        "poolside_night"  → "poolside_night"
        "Colton Black"    → "colton_black"
    """
    key = _REF_KEY_RE.sub("_", name.lower()).strip("_")
    # Collapse consecutive underscores
    key = re.sub(r"_+", "_", key)
    return key

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
# Note: "no text overlays" was removed because the text directive layer
# explicitly instructs Seedance to render subtitles, name cards, and title
# cards. The constraint "no unwanted text" covers random/hallucinated text
# while allowing the intentional text directives to work.
_DEFAULT_CONSTRAINTS = (
    "no unwanted text or random letters, no watermarks, no extra characters, "
    "anatomically correct hands, no sudden camera jumps"
)

# ---------------------------------------------------------------------------
# Western drama two-layer character consistency strategy
# ---------------------------------------------------------------------------
# Layer 1: 3D CGI / MetaHuman-style turnaround image → passes PrivacyInformation
#           filter on vectorspace.cn proxy; provides costume/body structure ref.
# Layer 2: CHARACTER IDENTITY text block → photorealistic appearance anchor
#           that compensates for the stylized ref image's reduced facial fidelity.
#
# Usage: prepended to visual_prompt when series.language == "en" and characters
#        are present in the scene. The compact identity string uses the character's
#        visual_prompt (first 150 chars) as a concise appearance anchor.
# ---------------------------------------------------------------------------

_WESTERN_REALISM_HEADER = (
    "Generate as photorealistic live-action film with REAL HUMAN ACTORS. "
    "Photorealistic skin, real hair, realistic fabric and lighting. "
    "NOT cartoon, NOT anime, NOT illustration. "
    "Netflix drama cinematography. "
)

# ---------------------------------------------------------------------------
# Dramatic tension modifiers for hook / cliffhanger scenes
# ---------------------------------------------------------------------------
# 黄金3秒法则: first shot must grab viewer within 3 seconds
# 悬念收尾: last shot must leave viewer wanting the next episode
# ---------------------------------------------------------------------------

_HOOK_MODIFIERS = (
    "HIGH DRAMATIC IMPACT opening shot. "
    "Immediately convey conflict, suspense, or visual spectacle within the first 3 seconds. "
    "Dynamic composition, intense emotion, cinematic urgency."
)

_CLIFFHANGER_MODIFIERS = (
    "CLIFFHANGER ending shot. "
    "End on unresolved tension — freeze at the moment of maximum suspense. "
    "The viewer must feel compelled to watch the next episode. "
    "Dramatic lighting shift, intense close-up on pivotal action."
)

# ---------------------------------------------------------------------------
# Scene continuity hints for adjacent shots
# ---------------------------------------------------------------------------

_CONTINUITY_PREFIX = (
    "Continuing seamlessly from the previous shot"
    " — same location, same lighting, consistent character positions."
)


# ---------------------------------------------------------------------------
# PromptEnhancer
# ---------------------------------------------------------------------------

class PromptEnhancer:
    """Enriches scene visual prompts with Seedance 2.0 optimised structure.

    Output follows the five-part director-style anatomy::

        Camera → Subject → Scene → Style → Constraints → Text Directives

    The text directive layer instructs Seedance to render on-screen text
    (subtitles, name cards, title cards) directly in the generated video.
    This matches Seedance 2.0's ``generate_audio: true`` workflow where
    dialogue is baked into the video — external TTS/subtitle overlays are
    not needed.

    Parameters
    ----------
    strip_chinese:
        - ``None`` (default): auto-detect per ``model_id``.
        - ``True``: always strip CJK characters.
        - ``False``: never strip.
    """

    def __init__(self, strip_chinese: bool | None = None) -> None:
        self._strip_chinese = strip_chinese
        # Track which characters have been introduced across scenes
        # within a single enhance_all_scenes() call.
        self._chars_introduced: set[str] = set()
        # Previous scene context for continuity hints
        self._prev_scene: DramaScene | None = None
        # Total scene count and current index for hook/cliffhanger detection
        self._total_scenes: int = 0
        self._scene_index: int = 0
        # Learned constraints from audit feedback (experience feedback / 经验反哺)
        self._learned_constraints: list[str] = []

    # -- public API ---------------------------------------------------------

    def should_strip_chinese(self, model_id: str) -> bool:
        """Return ``True`` if CJK characters should be removed for *model_id*."""
        if self._strip_chinese is not None:
            return self._strip_chinese
        return any(model_id.startswith(p) for p in _ENGLISH_ONLY_PREFIXES)

    def load_audit_constraints(self, series_dir: Path, min_count: int = 3) -> list[str]:
        """Read the audit log's frequent defects and map them to prompt constraints.

        Scans ``{series_dir}/audit_logs/ep*_audit.jsonl`` for defects that
        appeared >= *min_count* times, then maps common patterns to actionable
        constraint strings.  Unknown defects are included literally as
        ``"AVOID: {defect_text}"``.

        The loaded constraints are automatically injected via
        :meth:`inject_learned_constraints`.

        Returns the list of constraint strings (empty if no log exists).
        """
        from videoclaw.drama.vision_auditor import AuditLog

        log_dir = series_dir / "audit_logs"
        if not log_dir.is_dir():
            return []

        # Aggregate frequent defects from all episode logs in the directory
        all_defects: list[str] = []
        for log_file in sorted(log_dir.glob("ep*_audit.jsonl")):
            audit_log = AuditLog(log_file)
            all_defects.extend(audit_log.get_frequent_defects(min_count))

        if not all_defects:
            return []

        # De-duplicate while preserving order
        seen: set[str] = set()
        unique_defects: list[str] = []
        for d in all_defects:
            if d not in seen:
                seen.add(d)
                unique_defects.append(d)

        constraints: list[str] = []
        for defect in unique_defects:
            dl = defect.lower()
            if "hand" in dl or "finger" in dl:
                constraints.append(
                    "Ensure anatomically correct hands with exactly 5 fingers per hand"
                )
            elif "temporal_break" in dl or "flicker" in dl:
                constraints.append(
                    "Maintain visual stability with no abrupt changes between frames"
                )
            elif "character" in dl and "missing" in dl:
                constraints.append(
                    "All specified characters must be clearly visible in frame"
                )
            elif "scene" in dl and "mismatch" in dl:
                constraints.append(
                    "Scene must exactly match the described location and time of day"
                )
            else:
                constraints.append(f"AVOID: {defect}")

        # De-duplicate mapped constraints (multiple defects may map to same constraint)
        seen_constraints: set[str] = set()
        deduped: list[str] = []
        for c in constraints:
            if c not in seen_constraints:
                seen_constraints.add(c)
                deduped.append(c)

        self.inject_learned_constraints(deduped)
        return deduped

    def inject_learned_constraints(self, constraints: list[str]) -> None:
        """Store learned constraints to be appended in :meth:`enhance_scene_prompt`.

        These constraints come from audit feedback (经验反哺) and are injected
        into the ``Constraints:`` section of every subsequent prompt.
        """
        self._learned_constraints = list(constraints)

    def enhance_scene_prompt(
        self,
        scene: DramaScene,
        series: DramaSeries,
        *,
        available_refs: dict[str, dict[str, str]] | None = None,
    ) -> str:
        """Build a Seedance 2.0 optimised visual prompt for a single *scene*.

        Six-part structure:
        0. **Realism header** — (Western only) photorealistic + CHARACTER IDENTITY
        1. **Camera** — shot scale + single camera movement verb
        2. **Subject** — character appearance (repeated every scene for consistency)
        3. **Scene** — original visual_prompt (setting + action + lighting)
        4. **Style** — series style anchor + realism modifiers
        5. **Constraints** — 3-5 bans to prevent artifacts
        6. **Text directives** — subtitle / name card / title card / inner monologue

        The text directive layer instructs Seedance to render on-screen text
        directly in the generated video. This is critical for:

        - Dialogue subtitles visible at the bottom of the frame
        - Character name cards on first close-up appearance
        - Title cards for dramatic narration
        - Inner monologue styling (italicised thought subtitles)

        Additionally injects:

        - **Dramatic tension** for hook (first) and cliffhanger (last) shots
        - **Continuity hints** linking adjacent shots for visual coherence
        - **[ref:key] markers** when *available_refs* is provided

        Parameters
        ----------
        available_refs:
            Optional dict with keys ``"characters"``, ``"scenes"``, ``"props"``.
            Each value maps ``name -> URL-or-path``. When provided, ``[ref:key]``
            markers are inserted after matching character/scene/prop descriptions.
        """
        parts: list[str] = []
        char_map = {c.name: c for c in series.characters}

        # --- 0a. Dramatic tension modifiers (hook / cliffhanger) ---
        shot_role = getattr(scene, "shot_role", "normal")
        if shot_role == "hook" or self._scene_index == 0:
            parts.append(_HOOK_MODIFIERS)
        elif shot_role == "cliffhanger" or (
            self._total_scenes > 0 and self._scene_index == self._total_scenes - 1
        ):
            parts.append(_CLIFFHANGER_MODIFIERS)

        # --- 0b. Scene continuity (link to previous shot) ---
        if self._prev_scene is not None and self._scene_index > 0:
            # Only add continuity hint when scenes share location or characters
            prev_chars = set(self._prev_scene.characters_present or [])
            curr_chars = set(scene.characters_present or [])
            shared_chars = prev_chars & curr_chars
            if shared_chars:
                parts.append(_CONTINUITY_PREFIX)

        # --- 0c. Western drama: realism header + CHARACTER IDENTITY (two-layer strategy) ---
        char_refs = (available_refs or {}).get("characters", {})
        if series.language == "en" and scene.characters_present:
            char_ids = []
            for name in scene.characters_present:
                char = char_map.get(name)
                if char and char.visual_prompt:
                    vp_compact = char.visual_prompt[:150].rstrip(",. ")
                    entry = f"{name}: {vp_compact}"
                    ref_key = _to_ref_key(name)
                    if char_refs and ref_key in {_to_ref_key(k) for k in char_refs}:
                        entry += f" [ref:{ref_key}]"
                    char_ids.append(entry)
            if char_ids:
                parts.append(
                    _WESTERN_REALISM_HEADER
                    + "CHARACTER IDENTITY: "
                    + "; ".join(char_ids)
                    + "."
                )

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
        # For Western drama, character identity already injected in part 0.
        # For Chinese drama, use the standard "Same character" format.
        if series.language != "en":
            for name in scene.characters_present:
                char = char_map.get(name)
                if char and char.visual_prompt:
                    entry = f"Same character {name} — {char.visual_prompt.rstrip('.')}."
                    ref_key = _to_ref_key(name)
                    if char_refs and ref_key in {_to_ref_key(k) for k in char_refs}:
                        entry += f" [ref:{ref_key}]"
                    parts.append(entry)

        # --- 3. Scene (original visual prompt — setting + action + mood) ---
        scene_refs = (available_refs or {}).get("scenes", {})
        prop_refs = (available_refs or {}).get("props", {})
        if scene.visual_prompt:
            scene_text = scene.visual_prompt.rstrip(".") + "."
            # --- Scene ref marker ---
            if scene_refs:
                vp_lower = scene.visual_prompt.lower()
                desc_lower = scene.description.lower()
                for scene_name in scene_refs:
                    skey = _to_ref_key(scene_name)
                    # Match if the scene key (with underscores as spaces) appears
                    # in the visual_prompt or description
                    match_text = skey.replace("_", " ")
                    if match_text in vp_lower or match_text in desc_lower:
                        scene_text += f" [ref:{skey}]"
                        break  # Only ONE scene ref per shot
            # --- Prop ref markers ---
            if prop_refs:
                vp_lower = scene.visual_prompt.lower()
                for prop_name in prop_refs:
                    pkey = _to_ref_key(prop_name)
                    match_text = pkey.replace("_", " ")
                    if match_text in vp_lower:
                        scene_text += f" [ref:{pkey}]"
            parts.append(scene_text)

        # --- 4. Text directives (subtitle / name card / title card / inner monologue) ---
        text_directives = self._build_text_directives(scene, char_map, series)
        if text_directives:
            parts.extend(text_directives)

        # --- 5. Conditionally strip CJK ---
        text = " ".join(parts)
        if self.should_strip_chinese(series.model_id):
            text = self._strip_cjk(text)

        # --- 6. Style anchor + realism modifiers ---
        style_tag = (
            f"Style: {series.style}, vertical {series.aspect_ratio}, "
            f"{_REALISM_MODIFIERS}."
        )
        text = text.rstrip() + " " + style_tag

        # --- 7. Constraints ---
        constraints = _DEFAULT_CONSTRAINTS
        if self._learned_constraints:
            learned = ", ".join(self._learned_constraints)
            constraints = f"{constraints}, {learned}"
        text = text.rstrip() + " " + f"Constraints: {constraints}."

        # --- 8. Enforce text length limits ---
        language = series.language or "en"
        text = self._enforce_text_length(text, language)

        return text.strip()

    def enhance_all_scenes(
        self,
        episode: Episode,
        series: DramaSeries,
        *,
        available_refs: dict[str, dict[str, str]] | None = None,
    ) -> Episode:
        """Enhance all scenes in *episode* in-place. Returns the mutated episode.

        Resets all per-episode tracking state:
        - Character introduction tracker (name cards)
        - Scene index counter (hook / cliffhanger detection)
        - Previous scene reference (continuity hints)
        """
        self._chars_introduced = set()
        self._prev_scene = None
        self._total_scenes = len(episode.scenes)
        self._scene_index = 0

        for scene in episode.scenes:
            enhanced = self.enhance_scene_prompt(
                scene, series, available_refs=available_refs,
            )
            scene.enhanced_visual_prompt = enhanced
            self._prev_scene = scene
            self._scene_index += 1

        return episode

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _enforce_text_length(text: str, language: str) -> str:
        """Enforce hard text length limits.

        - English (or any non-Chinese): max *_MAX_ENGLISH_WORDS* words
          (excluding ``[ref:key]`` markers).
        - Chinese: max *_MAX_CHINESE_CHARS* CJK characters.

        When a prompt exceeds the limit, the visual description section (the
        middle bulk of the prompt) is shortened while preserving camera
        directives, style, constraints, and all ``[ref:key]`` markers.
        """
        if language == "zh":
            cjk_chars = _CJK_CHAR_RE.findall(text)
            if len(cjk_chars) <= _MAX_CHINESE_CHARS:
                return text
            # Find the visual description section to truncate.
            # Strategy: locate "Style:" marker, keep everything from Style:
            # onward plus camera/header prefix. Shorten the middle.
            style_idx = text.find("Style:")
            if style_idx == -1:
                # No structure found — hard truncate CJK chars
                return text
            prefix_end = 0
            # Find end of camera/header section (first sentence after any
            # hook/continuity/character identity block)
            for marker in ("CHARACTER IDENTITY:", "Same character"):
                idx = text.find(marker)
                if idx != -1:
                    # Find end of this section (next sentence boundary)
                    end = text.find(". ", idx)
                    if end != -1:
                        prefix_end = max(prefix_end, end + 2)
            suffix = text[style_idx:]
            middle = text[prefix_end:style_idx].rstrip()
            # Extract ref markers from middle to preserve them
            ref_markers = re.findall(r'\[ref:[a-zA-Z0-9_]+\]', middle)
            # Count CJK chars in prefix + suffix
            fixed_cjk = len(_CJK_CHAR_RE.findall(text[:prefix_end])) + len(
                _CJK_CHAR_RE.findall(suffix)
            )
            budget = _MAX_CHINESE_CHARS - fixed_cjk
            if budget <= 0:
                return text[:prefix_end].rstrip() + " " + suffix
            # Truncate middle to fit budget
            truncated: list[str] = []
            count = 0
            for ch in middle:
                if _CJK_CHAR_RE.match(ch):
                    if count >= budget:
                        break
                    count += 1
                truncated.append(ch)
            middle_new = "".join(truncated).rstrip()
            # Re-append any ref markers that were lost
            for marker in ref_markers:
                if marker not in middle_new:
                    middle_new += f" {marker}"
            return text[:prefix_end] + middle_new + " " + suffix
        else:
            # English word count (exclude [ref:key] markers)
            stripped = re.sub(r'\[ref:[a-zA-Z0-9_]+\]', '', text)
            words = stripped.split()
            if len(words) <= _MAX_ENGLISH_WORDS:
                return text
            # Need to shorten — find visual description section
            style_idx = text.find("Style:")
            if style_idx == -1:
                return text
            # Preserve everything from "Style:" onward (style + constraints)
            suffix = text[style_idx:]
            # Find the prefix to preserve (camera + character identity)
            prefix_end = 0
            for marker in ("CHARACTER IDENTITY:", "Same character"):
                idx = text.find(marker)
                if idx != -1:
                    end = text.find(". ", idx)
                    if end != -1:
                        prefix_end = max(prefix_end, end + 2)
            # Also preserve camera line
            cam_end = text.find(".", 0, style_idx)
            if cam_end != -1 and cam_end < prefix_end:
                pass  # prefix_end already covers it
            elif cam_end != -1 and prefix_end == 0:
                prefix_end = cam_end + 2
            prefix = text[:prefix_end]
            middle = text[prefix_end:style_idx].rstrip()
            # Extract ref markers from middle
            ref_markers = re.findall(r'\[ref:[a-zA-Z0-9_]+\]', middle)
            # Count words in prefix + suffix (excluding markers)
            prefix_clean = re.sub(r'\[ref:[a-zA-Z0-9_]+\]', '', prefix)
            suffix_clean = re.sub(r'\[ref:[a-zA-Z0-9_]+\]', '', suffix)
            fixed_words = len(prefix_clean.split()) + len(suffix_clean.split())
            budget = _MAX_ENGLISH_WORDS - fixed_words
            if budget <= 0:
                result = prefix.rstrip() + " " + suffix
                return result
            # Truncate middle to budget words, preserving ref markers
            middle_clean = re.sub(r'\[ref:[a-zA-Z0-9_]+\]', '\x00', middle)
            tokens = middle_clean.split()
            kept: list[str] = []
            wcount = 0
            for tok in tokens:
                if tok == '\x00':
                    kept.append(tok)  # Always keep marker placeholders
                    continue
                if wcount >= budget:
                    break
                kept.append(tok)
                wcount += 1
            middle_new = " ".join(kept)
            # Restore ref markers
            for marker in ref_markers:
                middle_new = middle_new.replace('\x00', marker, 1)
            # Remove any remaining placeholders
            middle_new = middle_new.replace('\x00', '').strip()
            return prefix + middle_new + " " + suffix

    def _build_text_directives(
        self,
        scene: DramaScene,
        char_map: dict[str, Character],
        series: DramaSeries,
    ) -> list[str]:
        """Build Seedance text rendering directives for a scene.

        Returns a list of prompt fragments that instruct Seedance to render
        on-screen text elements (subtitles, name cards, title cards).

        Text directives follow Seedance positioning conventions:

        - **Subtitles** — bottom center of frame (standard subtitle position)
        - **Name cards** — lower-third beside the character, ABOVE the subtitle
          area to avoid overlap (任务要求: 角色介绍和字幕互相不应遮挡)
        - **Title cards** — large centered text on screen

        These directives are placed AFTER the visual scene description so
        they don't interfere with Seedance's scene composition.
        """
        directives: list[str] = []
        # --- Check if this scene will have a subtitle ---
        dialogue = (scene.dialogue or "").strip()
        narration = (scene.narration or "").strip()
        narration_type = getattr(scene, "narration_type", "voiceover")

        # --- Character name card (first close-up/medium-close appearance) ---
        # 名牌永远紧贴角色身旁，竖排文字，不与底部字幕冲突。
        # Position: vertically written text placed RIGHT NEXT TO the character,
        # never at the bottom center where subtitles go.
        intro_scales = {ShotScale.CLOSE_UP, ShotScale.MEDIUM_CLOSE}
        if scene.shot_scale in intro_scales:
            # Determine focal character for name card
            focal: str | None = None
            speaker = (scene.speaking_character or "").strip()
            if speaker and speaker in char_map:
                focal = speaker
            elif len(scene.characters_present) == 1 and scene.characters_present[0] in char_map:
                focal = scene.characters_present[0]

            if focal and focal not in self._chars_introduced:
                char = char_map[focal]
                name_card = self._format_name_card(char)
                directives.append(
                    f'[Show character name card VERTICALLY written, placed right next to '
                    f'the character\'s body (not at bottom). '
                    f'Text reads top-to-bottom: {name_card}. '
                    f'The name card must stay close to the character, never overlap with '
                    f'the bottom subtitle area]'
                )
                self._chars_introduced.add(focal)

        # --- Narration (voiceover or title card) ---
        if narration:
            if narration_type == "title_card":
                directives.append(
                    f'[Show large centered title text on screen: "{narration}"]'
                )
            else:
                # Voiceover narration with subtitle at bottom center
                directives.append(
                    f'[Narrator speaks: "{narration}". '
                    f'Show subtitle at bottom center of screen: "{narration}"]'
                )

        # --- Dialogue (spoken dialogue or inner monologue) ---
        if dialogue:
            raw_speaker = scene.speaking_character or "Character"
            line_type = getattr(scene, "dialogue_line_type", "dialogue")

            # Handle multi-speaker scenes: "GUEST 1, GUEST 2" or "Colton Black, Ivy Angel"
            # Parse comma / " & " separated names and pick the primary speaker
            speakers = [
                s.strip()
                for s in re.split(r"[,，]|\s*&\s*", raw_speaker)
                if s.strip()
            ]
            # De-duplicate while preserving order
            seen: set[str] = set()
            unique_speakers: list[str] = []
            for s in speakers:
                if s not in seen:
                    seen.add(s)
                    unique_speakers.append(s)
            primary_speaker = unique_speakers[0] if unique_speakers else "Character"

            if line_type == "inner_monologue":
                directives.append(
                    f'[{primary_speaker} thinks (inner monologue): "{dialogue}". '
                    f'Show subtitle at bottom center of screen: "{dialogue}"]'
                )
            else:
                directives.append(
                    f'[{primary_speaker} speaks: "{dialogue}". '
                    f'Show subtitle at bottom center of screen: "{dialogue}"]'
                )

        return directives

    @staticmethod
    def _format_name_card(char: Character) -> str:
        """Format a character name card string.

        Tries to extract age and role from the character description.
        Falls back to just the character name if no structured info is available.
        """
        name = char.name

        # Try to extract age from description (e.g. "26-year-old", "mid-twenties")
        age_match = re.search(
            r'(\d{2})\s*[-–]?\s*(?:year|岁)',
            char.description,
            re.IGNORECASE,
        )
        age_str = age_match.group(1) if age_match else ""

        # Try to extract a short role descriptor (first phrase before comma or period)
        role = ""
        desc = char.description.strip()
        if desc:
            # Take first meaningful fragment (up to 30 chars, before first comma/period)
            first_fragment = re.split(r'[,，.。;；]', desc)[0].strip()
            if len(first_fragment) <= 40:
                role = first_fragment
            else:
                role = first_fragment[:37] + "..."

        parts = [name]
        if age_str:
            parts[0] = f"{name}, {age_str}"
        if role:
            return f'"{" — ".join([parts[0], role])}"'
        return f'"{parts[0]}"'

    @staticmethod
    def _strip_cjk(text: str) -> str:
        """Remove CJK / full-width characters and collapse resulting whitespace."""
        text = _CJK_RE.sub("", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
