"""AI Short Drama orchestration — multi-episode series generation."""

from videoclaw.drama.locale import (
    DramaLocale,
    get_locale,
    get_locale_info,
    list_locales,
    register_locale,
)
from videoclaw.drama.models import (
    GENRE_VOICE_RECOMMENDATIONS,
    NARRATOR_PRESETS,
    AudioSegment,
    AudioType,
    Character,
    DialogueLine,
    DramaGenre,
    DramaManager,
    DramaScene,
    DramaSeries,
    DramaStatus,
    Episode,
    EpisodeStatus,
    LineType,
    VoiceProfile,
    recommend_voice_style,
)
from videoclaw.drama.prompt_enhancer import PromptEnhancer
from videoclaw.drama.runner import build_episode_dag

__all__ = [
    "GENRE_VOICE_RECOMMENDATIONS",
    "NARRATOR_PRESETS",
    "AudioSegment",
    "AudioType",
    "Character",
    "DialogueLine",
    "DramaGenre",
    "DramaLocale",
    "DramaManager",
    "DramaScene",
    "DramaSeries",
    "DramaStatus",
    "Episode",
    "EpisodeStatus",
    "LineType",
    "PromptEnhancer",
    "VoiceProfile",
    "build_episode_dag",
    "get_locale",
    "get_locale_info",
    "list_locales",
    "recommend_voice_style",
    "register_locale",
]
