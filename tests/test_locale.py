"""Tests for the locale registry module."""

from videoclaw.drama.locale import (
    DramaLocale,
    SubtitleConfig,
    get_locale,
    list_locales,
    register_locale,
)
from videoclaw.drama.models import DramaGenre, VoiceProfile


def test_zh_locale_registered():
    """Chinese locale is auto-registered on import."""
    locale = get_locale("zh")
    assert locale.code == "zh"
    assert locale.series_outline_prompt != ""
    assert locale.episode_script_prompt != ""


def test_zh_locale_has_voice_profiles():
    locale = get_locale("zh")
    assert "warm" in locale.voice_profiles
    assert "authoritative" in locale.voice_profiles
    assert isinstance(locale.voice_profiles["warm"], VoiceProfile)


def test_zh_locale_has_narrator_presets():
    locale = get_locale("zh")
    assert DramaGenre.SWEET_ROMANCE in locale.narrator_presets
    assert DramaGenre.SUSPENSE_THRILLER in locale.narrator_presets


def test_zh_locale_has_genre_voice_recommendations():
    locale = get_locale("zh")
    assert DramaGenre.ANCIENT_XIANXIA in locale.genre_voice_recommendations


def test_zh_locale_subtitle_config():
    locale = get_locale("zh")
    sc = locale.subtitle_config
    assert sc.font_name == "Microsoft YaHei"
    assert sc.max_chars_per_line == 20
    assert sc.line_break_strategy == "char"
    assert sc.colon_char == "\uff1a"


def test_zh_locale_has_prompts():
    locale = get_locale("zh")
    assert "竖屏短剧" in locale.series_outline_prompt
    assert "分镜" in locale.episode_script_prompt
    assert locale.genre_analysis_prompt != ""
    assert locale.voice_casting_prompt != ""
    assert locale.dialogue_extraction_prompt != ""


def test_zh_locale_character_image_style():
    locale = get_locale("zh")
    assert "Chinese drama" in locale.character_image_style


def test_zh_locale_genres():
    locale = get_locale("zh")
    assert DramaGenre.SWEET_ROMANCE in locale.genres
    assert DramaGenre.ANCIENT_XIANXIA in locale.genres


def test_fallback_to_zh():
    """Unknown language codes fall back to Chinese locale."""
    locale = get_locale("fr")
    assert locale.code == "zh"


def test_list_locales():
    codes = list_locales()
    assert "zh" in codes


def test_register_custom_locale():
    custom = DramaLocale(
        code="test_lang",
        series_outline_prompt="test prompt",
    )
    register_locale(custom)
    assert get_locale("test_lang").series_outline_prompt == "test prompt"
    # Clean up
    from videoclaw.drama.locale import _LOCALES
    _LOCALES.pop("test_lang", None)


def test_subtitle_config_defaults():
    sc = SubtitleConfig()
    assert sc.font_name == "Microsoft YaHei"
    assert sc.font_size == 20
    assert sc.max_chars_per_line == 20
    assert sc.line_break_strategy == "char"


def test_drama_genre_western_entries():
    """Western genres are available in the enum."""
    assert DramaGenre.ROMANCE == "romance"
    assert DramaGenre.ACTION_THRILLER == "action_thriller"
    assert DramaGenre.MYSTERY == "mystery"
    assert DramaGenre.SUPERNATURAL == "supernatural"
    assert DramaGenre.DRAMA == "drama"
    assert DramaGenre.SCI_FI == "sci_fi"
