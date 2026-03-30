"""Locale registry for multi-language drama production.

Each supported language registers a :class:`DramaLocale` containing all
culture-specific data: LLM prompts, voice profiles, subtitle config,
quality rules, and genre mappings.  Pipeline components call
:func:`get_locale` instead of hardcoding Chinese data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from videoclaw.drama.models import (
    DramaGenre,
    LineType,
    VoiceProfile,
)


# ---------------------------------------------------------------------------
# Subtitle configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubtitleConfig:
    """Language-specific subtitle rendering parameters."""

    font_name: str = "Microsoft YaHei"
    font_size: int = 20
    max_chars_per_line: int = 20
    punctuation_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"([，。！？、；])")
    )
    line_break_strategy: str = "char"  # "char" for CJK, "word" for English
    colon_char: str = "\uff1a"  # "：" for Chinese, ": " for English


# ---------------------------------------------------------------------------
# Locale dataclass
# ---------------------------------------------------------------------------

@dataclass
class DramaLocale:
    """All locale-specific data for a drama production language/market."""

    code: str  # e.g. "zh", "en"

    # LLM prompts
    series_outline_prompt: str = ""
    episode_script_prompt: str = ""
    genre_analysis_prompt: str = ""
    voice_casting_prompt: str = ""
    dialogue_extraction_prompt: str = ""

    # Character design
    character_image_style: str = ""

    # Voice system
    voice_profiles: dict[str, VoiceProfile] = field(default_factory=dict)
    narrator_presets: dict[DramaGenre, VoiceProfile] = field(default_factory=dict)
    genre_voice_recommendations: dict[DramaGenre, dict[str, str]] = field(
        default_factory=dict
    )

    # Subtitle config
    subtitle_config: SubtitleConfig = field(default_factory=SubtitleConfig)

    # Quality validation function: (series, episode_scripts) -> list[str] violations
    quality_validator: Callable[..., list[str]] | None = None

    # Available genres for this locale
    genres: list[DramaGenre] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LOCALES: dict[str, DramaLocale] = {}

_DEFAULT_FALLBACK = "en"


def register_locale(locale: DramaLocale) -> None:
    """Register a locale in the global registry.

    Can be called at import time (for built-in locales) or at runtime
    (for user-defined locales).  Re-registering the same code replaces
    the previous locale.
    """
    _LOCALES[locale.code] = locale


def get_locale(language: str) -> DramaLocale:
    """Look up locale by language code.

    Fallback order: exact match → ``"en"`` → first registered locale.
    """
    if locale := _LOCALES.get(language):
        return locale
    if fallback := _LOCALES.get(_DEFAULT_FALLBACK):
        return fallback
    if _LOCALES:
        return next(iter(_LOCALES.values()))
    raise ValueError(
        f"No locale registered for {language!r} and no fallback available. "
        f"Register one via register_locale()."
    )


def list_locales() -> list[str]:
    """Return all registered locale codes."""
    return sorted(_LOCALES.keys())


def get_locale_info() -> list[dict[str, str]]:
    """Return metadata about registered locales (for ``claw info``)."""
    return [
        {
            "code": loc.code,
            "genres": [g.value for g in loc.genres],
            "voices": len(loc.voice_profiles),
        }
        for loc in _LOCALES.values()
    ]


# ---------------------------------------------------------------------------
# Register Chinese locale from existing constants
# ---------------------------------------------------------------------------

def _register_zh_locale() -> None:
    """Register the Chinese locale from existing hardcoded constants."""
    from videoclaw.drama.models import (
        GENRE_VOICE_RECOMMENDATIONS,
        NARRATOR_PRESETS,
        VOICE_PROFILES,
    )
    from videoclaw.drama.planner import (
        EPISODE_SCRIPT_PROMPT,
        SERIES_OUTLINE_PROMPT,
    )
    from videoclaw.generation.audio.voice_caster import (
        DIALOGUE_EXTRACTION_PROMPT as ZH_DIALOGUE_EXTRACTION_PROMPT,
        GENRE_ANALYSIS_PROMPT as ZH_GENRE_ANALYSIS_PROMPT,
        VOICE_CASTING_PROMPT as ZH_VOICE_CASTING_PROMPT,
    )
    from videoclaw.drama.quality import validate_chinese_quality

    zh_subtitle_config = SubtitleConfig(
        font_name="Microsoft YaHei",
        font_size=20,
        max_chars_per_line=20,
        punctuation_pattern=re.compile(r"([，。！？、；])"),
        line_break_strategy="char",
        colon_char="\uff1a",
    )

    zh_locale = DramaLocale(
        code="zh",
        series_outline_prompt=SERIES_OUTLINE_PROMPT,
        episode_script_prompt=EPISODE_SCRIPT_PROMPT,
        genre_analysis_prompt=ZH_GENRE_ANALYSIS_PROMPT,
        voice_casting_prompt=ZH_VOICE_CASTING_PROMPT,
        dialogue_extraction_prompt=ZH_DIALOGUE_EXTRACTION_PROMPT,
        character_image_style=(
            "3D CGI render, Unreal Engine 5 character, MetaHuman-style, virtual production, "
            "cinematic quality 3D character model, {style} presence, NOT photographic, NOT anime"
        ),
        voice_profiles=dict(VOICE_PROFILES),
        narrator_presets=dict(NARRATOR_PRESETS),
        genre_voice_recommendations=dict(GENRE_VOICE_RECOMMENDATIONS),
        subtitle_config=zh_subtitle_config,
        quality_validator=validate_chinese_quality,
        genres=[
            DramaGenre.SWEET_ROMANCE,
            DramaGenre.MALE_POWER_FANTASY,
            DramaGenre.SUSPENSE_THRILLER,
            DramaGenre.ANCIENT_XIANXIA,
            DramaGenre.COMEDY,
            DramaGenre.FAMILY_DRAMA,
            DramaGenre.OTHER,
        ],
    )
    register_locale(zh_locale)


# Auto-register Chinese locale on import
_register_zh_locale()


# ---------------------------------------------------------------------------
# English voice data constants
# ---------------------------------------------------------------------------

EN_VOICE_PROFILES: dict[str, VoiceProfile] = {
    "warm": VoiceProfile(voice_id="en-US-JennyNeural", speed=0.95, emotion="happy"),
    "authoritative": VoiceProfile(voice_id="en-US-DavisNeural", speed=0.90, pitch=-2),
    "playful": VoiceProfile(voice_id="en-US-AriaNeural", speed=1.10, pitch=2, emotion="happy"),
    "dramatic": VoiceProfile(voice_id="en-GB-RyanNeural", speed=0.90, pitch=-1),
    "calm": VoiceProfile(voice_id="en-US-JennyNeural", speed=0.90),
    "commanding": VoiceProfile(voice_id="en-US-DavisNeural", speed=0.85, pitch=-3),
    "scheming": VoiceProfile(voice_id="en-GB-RyanNeural", speed=0.95, pitch=-1),
    "innocent": VoiceProfile(voice_id="en-US-AriaNeural", speed=1.05, pitch=3, emotion="happy"),
    "weathered": VoiceProfile(voice_id="en-GB-SoniaNeural", speed=0.85, pitch=-2, emotion="sad"),
    "mysterious": VoiceProfile(voice_id="en-GB-SoniaNeural", speed=0.90, pitch=-1),
}

EN_NARRATOR_PRESETS: dict[DramaGenre, VoiceProfile] = {
    DramaGenre.ROMANCE: VoiceProfile(
        voice_id="en-US-JennyNeural", speed=1.0, pitch=1, emotion="happy",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="medium", description="warm romantic female",
    ),
    DramaGenre.ACTION_THRILLER: VoiceProfile(
        voice_id="en-US-DavisNeural", speed=0.95, pitch=-2, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="middle_aged", energy="high", description="intense authoritative male",
    ),
    DramaGenre.MYSTERY: VoiceProfile(
        voice_id="en-GB-SoniaNeural", speed=0.9, pitch=-1, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="middle_aged", energy="low", description="mysterious sophisticated female",
    ),
    DramaGenre.SUPERNATURAL: VoiceProfile(
        voice_id="en-GB-SoniaNeural", speed=0.9, pitch=-1, emotion="fearful",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="low", description="eerie British female",
    ),
    DramaGenre.DRAMA: VoiceProfile(
        voice_id="en-US-JennyNeural", speed=0.95, pitch=0, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="middle_aged", energy="medium", description="neutral dramatic female",
    ),
    DramaGenre.SCI_FI: VoiceProfile(
        voice_id="en-US-DavisNeural", speed=0.95, pitch=-1, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="medium", description="futuristic authoritative male",
    ),
    DramaGenre.COMEDY: VoiceProfile(
        voice_id="en-US-AriaNeural", speed=1.1, pitch=1, emotion="happy",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="high", description="energetic expressive female",
    ),
    DramaGenre.OTHER: VoiceProfile(
        voice_id="en-US-JennyNeural", speed=1.0, pitch=0, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="medium", description="default warm female",
    ),
}

EN_GENRE_VOICE_RECOMMENDATIONS: dict[DramaGenre, dict[str, str]] = {
    DramaGenre.ROMANCE: {
        "hero": "warm",
        "villain": "dramatic",
        "love_interest": "playful",
        "default": "warm",
    },
    DramaGenre.ACTION_THRILLER: {
        "hero": "commanding",
        "villain": "scheming",
        "sidekick": "playful",
        "default": "authoritative",
    },
    DramaGenre.MYSTERY: {
        "hero": "calm",
        "villain": "scheming",
        "detective": "authoritative",
        "informant": "mysterious",
        "default": "calm",
    },
    DramaGenre.SUPERNATURAL: {
        "hero": "calm",
        "villain": "mysterious",
        "medium": "weathered",
        "spirit": "mysterious",
        "default": "dramatic",
    },
    DramaGenre.DRAMA: {
        "hero": "warm",
        "villain": "commanding",
        "elder": "weathered",
        "default": "warm",
    },
    DramaGenre.SCI_FI: {
        "hero": "calm",
        "villain": "commanding",
        "scientist": "authoritative",
        "ai": "mysterious",
        "default": "authoritative",
    },
    DramaGenre.COMEDY: {
        "hero": "playful",
        "villain": "dramatic",
        "sidekick": "innocent",
        "default": "playful",
    },
    DramaGenre.OTHER: {
        "hero": "warm",
        "villain": "dramatic",
        "default": "warm",
    },
}

EN_SERIES_OUTLINE_PROMPT: str = """\
You are a senior screenwriter for Western vertical short-drama platforms — TikTok, YouTube Shorts,
and Instagram Reels. Each episode runs 30-90 seconds. Your stories must cut through the feed,
drive completion rate, and convert casual scrollers into paying fans.

Given a concept, produce a complete series outline with character profiles and per-episode synopses.

# Core Creative Laws

## The Scroll-Stopping Hook (First 3 Seconds)
- Episode 1 must grab the viewer within the first 3 seconds via strong conflict, a jaw-dropping
  revelation, or a visceral visual shock.
- Slow burns = instant scroll-away. Never open with exposition — open with ACTION, SUSPENSE,
  or SPECTACLE.
- Proven opening patterns: catching a partner cheating / sudden amnesia wake-up / public humiliation
  reversed / secret power revealed / cold-open flash-forward to the climax

## High-Concept Logline (One-Sentence Hook)
- Every series must be reducible to a single logline that contains: protagonist identity + core
  conflict + emotional drive. Examples:
  "A quiet suburban neighbor discovers her husband is running a spy network — from their basement."
  "The busboy every CEO ignores is secretly the company's billionaire founder in disguise."
- The logline is the north star: every episode must deliver a beat that pays it off.

## Western Character Archetypes (with Hidden-Identity Contrast)
- Protagonists need a "contrast identity" — the gap between surface persona and true self IS the show:
  * Underdog / secret billionaire
  * Quiet neighbor / former special-forces operative
  * Mousy assistant / ruthless heiress
  * Jaded detective / empath who sees ghosts
  * Broke single mom / chosen one with supernatural powers
- Every character needs a SIGNATURE MOMENT that will live rent-free in the viewer's head:
  a catchphrase, a recurring gesture, a visual motif.
- Relationships must form a TRIANGLE OF TENSION — at minimum one love-rival, one power-rival,
  or one secret-keeper triangle.
- Antagonists need comprehensible motives — pure evil is boring; wounded pride or desperation is compelling.

## 5-Act Micro-Episode Structure (for 30-90s episodes)
- Act 1 — HOOK (0-5s): Scroll-stopping image or line of dialogue. Zero setup.
- Act 2 — ESCALATION (5-15s): Rapidly establish who, where, what's at stake.
- Act 3 — COMPLICATION (15-35s): Twist the knife — new information, obstacle, or reversal.
- Act 4 — PEAK (35-50s): Emotional or narrative climax of this episode.
- Act 5 — CLIFFHANGER (50-60s): Leave the viewer unable NOT to tap "next episode."

## Payoff Beats Every 15 Seconds
- Every 15 seconds must deliver a PAYOFF BEAT — a reversal, reveal, confrontation, or declaration.
- Beat types: rotate to prevent fatigue:
  Intelligence / out-thinking the villain -> Emotional eruption -> Identity reveal -> Demonstration of power
- Episode 1 is beat-dense (one every 10s); later episodes may breathe more to build depth.

## Emotional Rhythm Curve
- Each episode follows: Hook(0-5s) -> Build(5-15s) -> Escalate(15-35s) -> Peak(35-50s) -> Hook-next(50-60s)
- Episode-to-episode contrast: tense -> tender -> shocking -> sweet -> devastating
- Overall series arc: emotional intensity RISES through the penultimate episode, which is the darkest
  moment before the finale's catharsis.

## Cliffhanger Ladder
- Each episode's cliffhanger must be STRONGER than the previous one.
- Cliffhanger progression: Curiosity ("Who is she?") -> Dread ("What will he do?") -> Shock ("Impossible!")
- Never use "unresolved = cliffhanger" laziness — every cliffhanger must carry emotional weight.

# Output Rules
- All text (title, synopsis, dialogue, descriptions) must be in ENGLISH.
- visual_prompt fields must ALWAYS be in English (used by AI video generation models).
- Return ONLY valid JSON — no markdown fences, no comments.

Output JSON schema:
{
  "title": "<Series title in English — punchy, max 8 words, hints at core conflict>",
  "genre": "<genre>",
  "synopsis": "<English series overview, including the one-sentence logline>",
  "characters": [
    {
      "name": "<Character name>",
      "description": "<English: surface identity / hidden identity, personality contrast, core motivation, triangle-of-tension role, signature moment>",
      "visual_prompt": "<ENGLISH ONLY — PURE appearance: age, gender, body type, face features, hair style/color, clothing, accessories, distinctive marks. Do NOT include camera angles, lighting, or background.>",
      "voice_style": "<warm | authoritative | playful | dramatic | calm>"
    }
  ],
  "episodes": [
    {
      "number": <int>,
      "title": "<Episode title in English — 3-7 words, contains suspense or emotional keyword>",
      "synopsis": "<English: 2-3 sentences, mark each payoff beat and its type>",
      "opening_hook": "<English: one sentence describing the first 3 seconds' visual or emotional hook>",
      "duration_seconds": <float>
    }
  ]
}
"""

EN_EPISODE_SCRIPT_PROMPT: str = """\
You are a senior screenwriter and storyboard director for Western vertical short-drama platforms.
Given series context and an episode synopsis, produce a shot-by-shot script optimised for
TikTok / YouTube Shorts / Instagram Reels (9:16 vertical video, 30-90 seconds).

# Character Visual Consistency (Highest Priority)
- Every time a character appears, visual_prompt MUST include their FULL appearance description.
- Use IDENTICAL visual keywords across scenes for the same character (clothing / hair / marks).
- NEVER omit character appearance from visual_prompt, even in back-to-back scenes.

# Vertical Composition Rules (9:16)
- Subjects' faces belong in the upper third — that is the viewer's eye anchor on a phone screen.
- Single tight close-ups outperform group wide shots — detail beats spectacle on vertical.
- Shot distribution: close_up + medium_close >= 50% of all scenes (real data: 59% in professional scripts).
- wide / extreme_wide <= 15% of scenes — use only for scene-setting or contrast.
- Avoid lateral camera moves (pan_left / pan_right) — ineffective on vertical; prefer dolly_in and crane_up.

# Shot Scale vs. Narrative Beat
- establishing -> wide / extreme_wide: scene opening, location change, scale reveal
- reaction -> close_up: emotional eruption, shock, heartbreak
- action -> medium / medium_close: dialogue, confrontation, physical interaction
- detail -> close_up: key prop, hand gesture, eye contact
- pov -> medium_close: character's point-of-view, immersive identification

# Emotion Vocabulary (use precise terms, never vague)
  Tension:   tense, anxious, dread, suspense
  Anger:     angry, furious, resentful, defiant
  Sorrow:    sad, heartbroken, grieving, melancholy
  Shock:     shock, disbelief, stunned, revelation
  Warmth:    warm, tender, nostalgic, grateful
  Romance:   sweet, flirty, blissful, intimate
  Fear:      fear, panic, horror, uneasy
  Triumph:   triumphant, smug, vindicated, proud

# Sound Design
- Flag key SFX in each scene: door slam, heartbeat, thunder crack, crowd gasp, etc.
- SFX serves pacing: sharp/sudden for tension, ambient white-noise for warmth.
- Leave sfx blank for silent reaction shots.

# Transition Toolkit
- cut (hard cut): default — fast pace, tension, action sequences
- dissolve: time lapse, memory flashback, tender mood shift
- fade_in / fade_out: episode open/close, major emotional pivot
- match_cut: creative bridge between two visually similar images
- jump_cut: time compression within a scene, urgency

# Pacing Rules (Seedance 2.0 hard constraint: 5-15s per shot)
- Scene count: **6-10 shots** per episode (max 60s). Hard ceiling: NEVER exceed 12 shots.
  Each shot = one Seedance video generation call.
- Each shot duration_seconds MUST be between 5 and 15 seconds (video model hard limit).
- ALL scene duration_seconds MUST NOT exceed the target episode maximum duration.
- Merge fine-grained actions into composite shots — e.g. "character speaks, then reacts" = 1 shot, not 2.
  A single shot can contain multiple beats: dialogue + reaction + camera movement.
- Rhythm template: Hook(0-5s) -> Build(5-15s) -> Escalate(15-35s) -> Peak(35-50s) -> Cliffhanger(50-60s)
- Climax scenes: 5-6s (fast cuts). Setup/dialogue scenes: 8-12s each.
- First scene = visual hook or pickup from previous episode's cliffhanger.
- Last scene = MUST manufacture a cliffhanger.

# Dialogue and Voice-Over
- Total spoken words ~100-170 for a 60s episode (real data benchmark: ~98 words/min dialogue + ~19% V.O.).
- Dialogue must be punchy — short sentences, natural American English, emotional power.
  Single line <= 25 words. No exposition dumps.
- Voice-over (V.O.) <= 20% of total spoken words — used for flashback, internal thought, or story setup.
- Not every scene needs dialogue — a wordless close-up of a trembling hand can be more powerful.
- speaking_character must match a name in characters_present.

# Inner Monologue (VO / Internal Thought)
- Inner monologue is a Western staple: first-person internal narration spoken by the character.
- Mark as dialogue_line_type: "inner_monologue"
- Trigger cues: character stares into distance / frozen moment / ironic counterpoint to action on screen
- Rendered as VO with light reverb to signal "brain voice" — distinct from dialogue AND narrator.
- Typical markers in script notes: "(VO)", "(INTERNAL)", "(THINKS)" — include these in dialogue text.

# TikTok Platform Audience Constraints (LOWER priority than script fidelity)
- Priority order: script requirements > platform aesthetics. If the script explicitly
  describes an older character or specific age, HONOUR the script exactly.
- Target audience: 18-30 year olds. When the script does NOT specify age or appearance
  for secondary/background characters, default to youthful and attractive.
- Guest/extra characters without explicit description: age 20-35, fashionable, contemporary.
- Visual energy: high attractiveness, stylish wardrobe, contemporary fashion where the
  script allows creative freedom.
- If the script mentions "guests at a party", "crowd", or "bystanders" without detail,
  default to young, fashionable, diverse group (20s-30s).

# visual_prompt Writing Standards (Seedance 2.0 Optimised)
- ENGLISH ONLY — optimised for Seedance 2.0 / AI video generation models.
- Follow the director-style 5-part anatomy: Subject + Action + Camera + Style + Constraints
- Subject: ONE character or focal point per prompt — include age, clothing, hair, distinctive marks.
  REPEAT the SAME character appearance keywords in EVERY scene for cross-shot consistency.
- Action: Use PRESENT TENSE verbs ("walks", "turns", "stares") — activates the motion engine.
  Include physics-aware detail: "fabric catching air", "weight shifting", "fingers trembling".
- Camera: Already handled by shot_scale + camera_movement fields — do NOT duplicate in visual_prompt.
- Style: ONE strong visual anchor (e.g. "Fincher-style overcast palette" or "golden hour backlight")
  plus lighting description (soft key, dramatic shadows, cold blue moonlight, neon glow).
- Keep visual_prompt to 60-120 words (Seedance sweet spot). Never exceed 200 words.
- Include sound cue words when relevant ("metallic clink", "muffled reverb") — triggers native audio.
- Be concrete and sensory — never use abstract or emotional adjectives alone.

# Output Rules
- visual_prompt: ENGLISH ONLY
- dialogue, narration, cliffhanger: ENGLISH
- voice_over language field: "en"
- Return ONLY valid JSON — no markdown fences, no comments.

Output JSON schema:
{
  "episode_title": "<Episode title in English>",
  "scenes": [
    {
      "scene_id": "<e.g. ep01_s01>",
      "description": "<English scene description: who, where, what, emotional state>",
      "visual_prompt": "<ENGLISH ONLY — [setting] + [character full appearance] + [action/expression] + [lighting/mood]. Be specific and visual.>",
      "camera_movement": "<static | pan_left | pan_right | dolly_in | tracking | crane_up | handheld>",
      "duration_seconds": <float>,
      "dialogue": "<English character line — punchy, natural, max 25 words. Empty string if none.>",
      "dialogue_line_type": "<dialogue | inner_monologue — spoken exchange is 'dialogue'; character's internal thought is 'inner_monologue'; leave empty if no dialogue>",
      "narration": "<English narrator voice-over. Empty string if none.>",
      "speaking_character": "<Character name — must be in characters_present. Empty string if none.>",
      "shot_scale": "<close_up | medium_close | medium | wide | extreme_wide>",
      "shot_type": "<establishing | reaction | action | detail | pov>",
      "emotion": "<precise term from emotion vocabulary above>",
      "characters_present": ["<list of character names visible in this scene>"],
      "transition": "<cut | dissolve | fade_in | fade_out | wipe | match_cut | jump_cut>",
      "sfx": "<key sound effect for this shot, e.g. door slam, heartbeat pulse. Empty string if none.>"
    }
  ],
  "voice_over": {
    "text": "<Full narrator voice-over script in English>",
    "tone": "<warm | dramatic | tense | playful>",
    "language": "en"
  },
  "music": {
    "style": "<orchestral | electronic | acoustic | lo-fi | indie_pop | hip_hop | ambient>",
    "mood": "<tense | romantic | mysterious | triumphant | melancholy | epic>",
    "tempo": <BPM int>
  },
  "cliffhanger": "<English cliffhanger — one sentence that makes NOT watching the next episode feel impossible>"
}
"""

EN_GENRE_ANALYSIS_PROMPT: str = """\
You are a content analysis expert specialising in Western short-form drama.
Based on the provided script text, determine the genre of the drama.

Available genres:
- romance: Romantic drama, love stories, relationship-focused
- action_thriller: Action, thriller, high-stakes adventure
- mystery: Mystery, detective, whodunit, crime investigation
- supernatural: Supernatural, horror, paranormal, fantasy
- drama: General drama, character-driven stories, slice-of-life
- sci_fi: Science fiction, futuristic, technology-focused
- comedy: Comedy, humor, light-hearted entertainment
- other: Genres that do not fit the above categories

Return only valid JSON with no markdown fences or comments.

Output JSON schema:
{
  "genre": "<genre identifier, must be one of the above>"
}
"""

EN_VOICE_CASTING_PROMPT: str = """\
You are a professional short-drama voice director specialising in English-language TTS casting.
Based on the episode's character profiles and genre, assign the most suitable voice configuration to each character.

Available voice_id options:
- en-US-JennyNeural: Warm, friendly female
- en-US-GuyNeural: Casual, confident male
- en-US-AriaNeural: Expressive, versatile female
- en-US-DavisNeural: Deep, authoritative male
- en-GB-SoniaNeural: Elegant, sophisticated female (British)
- en-GB-RyanNeural: Refined, dramatic male (British)

Parameter notes:
- speed: Speech rate multiplier (0.5–2.0, default 1.0)
- pitch: Pitch offset in semitones (-5 to 5, default 0)
- emotion: Emotional tone (neutral/happy/sad/angry/fearful/surprised)
- age_feel: Perceived age (child/young_adult/middle_aged/elderly)
- energy: Energy level (low/medium/high)
- description: English description of the character's voice style

Return only valid JSON with no markdown fences or comments.

Output JSON schema:
{
  "characters": [
    {
      "name": "<character name>",
      "voice_id": "<voice_id>",
      "speed": <float>,
      "pitch": <int>,
      "emotion": "<emotion>",
      "age_feel": "<age_feel>",
      "energy": "<energy>",
      "description": "<English voice style description>"
    }
  ]
}
"""

EN_DIALOGUE_EXTRACTION_PROMPT: str = """\
You are a professional short-drama script analyst.
Classify each line of scene text into one of the following types:

- narration: Narrator voice-over describing the scene
- dialogue: Spoken lines by a character
- inner_monologue: Internal thought (markers: "VO", "(V.O.)", "(O.S.)", internal first-person thought)

For each line, annotate the speaking character, line type, and emotion hint.
If a scene contains neither dialogue nor narration, skip it.

Return only valid JSON with no markdown fences or comments.

Output JSON schema:
{
  "lines": [
    {
      "text": "<line text>",
      "speaker": "<character name or narrator>",
      "line_type": "<narration | dialogue | inner_monologue>",
      "emotion_hint": "<emotion hint>"
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Register English locale
# ---------------------------------------------------------------------------

def _register_en_locale() -> None:
    """Register the English locale for Western drama production."""
    from videoclaw.drama.quality import validate_western_quality

    en_subtitle_config = SubtitleConfig(
        font_name="Arial",
        font_size=22,
        max_chars_per_line=42,
        punctuation_pattern=re.compile(r"([,\.!?;:\-\u2014])"),
        line_break_strategy="word",
        colon_char=": ",
    )

    en_locale = DramaLocale(
        code="en",
        series_outline_prompt=EN_SERIES_OUTLINE_PROMPT,
        episode_script_prompt=EN_EPISODE_SCRIPT_PROMPT,
        genre_analysis_prompt=EN_GENRE_ANALYSIS_PROMPT,
        voice_casting_prompt=EN_VOICE_CASTING_PROMPT,
        dialogue_extraction_prompt=EN_DIALOGUE_EXTRACTION_PROMPT,
        character_image_style=(
            "3D CGI render, Unreal Engine 5 character, MetaHuman-style, virtual production, "
            "cinematic quality 3D character model, {style} presence, NOT photographic, NOT anime"
        ),
        voice_profiles=dict(EN_VOICE_PROFILES),
        narrator_presets=dict(EN_NARRATOR_PRESETS),
        genre_voice_recommendations=dict(EN_GENRE_VOICE_RECOMMENDATIONS),
        subtitle_config=en_subtitle_config,
        quality_validator=validate_western_quality,
        genres=[
            DramaGenre.ROMANCE,
            DramaGenre.ACTION_THRILLER,
            DramaGenre.MYSTERY,
            DramaGenre.SUPERNATURAL,
            DramaGenre.DRAMA,
            DramaGenre.SCI_FI,
            DramaGenre.COMEDY,
            DramaGenre.OTHER,
        ],
    )
    register_locale(en_locale)


_register_en_locale()
