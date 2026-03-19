"""Locale registry for multi-language drama production.

Each supported language registers a :class:`DramaLocale` containing all
culture-specific data: LLM prompts, voice profiles, subtitle config,
quality rules, and genre mappings.  Pipeline components call
:func:`get_locale` instead of hardcoding Chinese data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from videoclaw.drama.models import (
    DramaGenre,
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


def register_locale(locale: DramaLocale) -> None:
    """Register a locale in the global registry."""
    _LOCALES[locale.code] = locale


def get_locale(language: str) -> DramaLocale:
    """Look up locale by language code, falling back to ``"zh"``."""
    locale = _LOCALES.get(language)
    if locale is not None:
        return locale
    # Fallback to Chinese
    zh = _LOCALES.get("zh")
    if zh is not None:
        return zh
    raise ValueError(
        f"No locale registered for {language!r} and no 'zh' fallback available"
    )


def list_locales() -> list[str]:
    """Return all registered locale codes."""
    return list(_LOCALES.keys())


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
            "{style} Chinese drama character portrait, modern Asian aesthetics"
        ),
        voice_profiles=dict(VOICE_PROFILES),
        narrator_presets=dict(NARRATOR_PRESETS),
        genre_voice_recommendations=dict(GENRE_VOICE_RECOMMENDATIONS),
        subtitle_config=zh_subtitle_config,
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
