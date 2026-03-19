"""Tests for English voice system (Stream B2)."""

import pytest
from unittest.mock import AsyncMock

from videoclaw.drama.locale import (
    EN_GENRE_VOICE_RECOMMENDATIONS,
    EN_NARRATOR_PRESETS,
    EN_VOICE_PROFILES,
    get_locale,
)
from videoclaw.drama.models import (
    Character,
    DramaGenre,
    DramaSeries,
    Episode,
    DramaScene,
    LineType,
    VoiceProfile,
    assign_voice_profile,
)
from videoclaw.generation.audio.voice_caster import VoiceCaster


# ---------------------------------------------------------------------------
# B2.1 / B2.2 — English voice profiles
# ---------------------------------------------------------------------------

class TestEnglishVoiceProfiles:
    def test_warm_uses_jenny_neural(self):
        assert EN_VOICE_PROFILES["warm"].voice_id == "en-US-JennyNeural"

    def test_authoritative_uses_davis_neural(self):
        assert EN_VOICE_PROFILES["authoritative"].voice_id == "en-US-DavisNeural"

    def test_playful_uses_aria_neural(self):
        assert EN_VOICE_PROFILES["playful"].voice_id == "en-US-AriaNeural"

    def test_dramatic_uses_ryan_neural(self):
        assert EN_VOICE_PROFILES["dramatic"].voice_id == "en-GB-RyanNeural"

    def test_calm_uses_jenny_neural(self):
        assert EN_VOICE_PROFILES["calm"].voice_id == "en-US-JennyNeural"

    def test_commanding_uses_davis_neural(self):
        assert EN_VOICE_PROFILES["commanding"].voice_id == "en-US-DavisNeural"

    def test_scheming_uses_ryan_neural(self):
        assert EN_VOICE_PROFILES["scheming"].voice_id == "en-GB-RyanNeural"

    def test_innocent_uses_aria_neural(self):
        assert EN_VOICE_PROFILES["innocent"].voice_id == "en-US-AriaNeural"

    def test_weathered_uses_sonia_neural(self):
        assert EN_VOICE_PROFILES["weathered"].voice_id == "en-GB-SoniaNeural"

    def test_mysterious_uses_sonia_neural(self):
        assert EN_VOICE_PROFILES["mysterious"].voice_id == "en-GB-SoniaNeural"

    def test_speed_values(self):
        assert EN_VOICE_PROFILES["warm"].speed == 0.95
        assert EN_VOICE_PROFILES["playful"].speed == 1.10
        assert EN_VOICE_PROFILES["commanding"].speed == 0.85

    def test_all_profiles_are_voice_profile_instances(self):
        for key, vp in EN_VOICE_PROFILES.items():
            assert isinstance(vp, VoiceProfile), f"{key!r} should be a VoiceProfile"


# ---------------------------------------------------------------------------
# B2.2 — English narrator presets
# ---------------------------------------------------------------------------

class TestEnglishNarratorPresets:
    WESTERN_GENRES = [
        DramaGenre.ROMANCE,
        DramaGenre.ACTION_THRILLER,
        DramaGenre.MYSTERY,
        DramaGenre.SUPERNATURAL,
        DramaGenre.DRAMA,
        DramaGenre.SCI_FI,
        DramaGenre.COMEDY,
        DramaGenre.OTHER,
    ]

    def test_all_western_genres_present(self):
        for genre in self.WESTERN_GENRES:
            assert genre in EN_NARRATOR_PRESETS, f"{genre} missing from EN_NARRATOR_PRESETS"

    def test_all_presets_are_narrator_role(self):
        for genre, vp in EN_NARRATOR_PRESETS.items():
            assert vp.role_name == "narrator", f"{genre} preset should have role_name='narrator'"

    def test_all_presets_have_narration_line_type(self):
        for genre, vp in EN_NARRATOR_PRESETS.items():
            assert vp.line_type == LineType.NARRATION, f"{genre} preset should have line_type=NARRATION"

    def test_romance_narrator_uses_jenny(self):
        assert EN_NARRATOR_PRESETS[DramaGenre.ROMANCE].voice_id == "en-US-JennyNeural"

    def test_action_thriller_narrator_uses_davis(self):
        assert EN_NARRATOR_PRESETS[DramaGenre.ACTION_THRILLER].voice_id == "en-US-DavisNeural"

    def test_mystery_narrator_uses_sonia(self):
        assert EN_NARRATOR_PRESETS[DramaGenre.MYSTERY].voice_id == "en-GB-SoniaNeural"

    def test_comedy_narrator_uses_aria(self):
        assert EN_NARRATOR_PRESETS[DramaGenre.COMEDY].voice_id == "en-US-AriaNeural"


# ---------------------------------------------------------------------------
# B2.3 — English genre voice recommendations
# ---------------------------------------------------------------------------

class TestEnglishGenreVoiceRecommendations:
    def test_romance_has_default(self):
        assert "default" in EN_GENRE_VOICE_RECOMMENDATIONS[DramaGenre.ROMANCE]

    def test_mystery_has_detective(self):
        assert "detective" in EN_GENRE_VOICE_RECOMMENDATIONS[DramaGenre.MYSTERY]

    def test_action_has_hero_and_villain(self):
        rec = EN_GENRE_VOICE_RECOMMENDATIONS[DramaGenre.ACTION_THRILLER]
        assert "hero" in rec
        assert "villain" in rec

    def test_supernatural_has_default(self):
        assert "default" in EN_GENRE_VOICE_RECOMMENDATIONS[DramaGenre.SUPERNATURAL]

    def test_all_values_reference_valid_profiles(self):
        for genre, mapping in EN_GENRE_VOICE_RECOMMENDATIONS.items():
            for archetype, style in mapping.items():
                assert style in EN_VOICE_PROFILES, (
                    f"{genre}/{archetype} references unknown style {style!r}"
                )


# ---------------------------------------------------------------------------
# B2.4 — English locale registration
# ---------------------------------------------------------------------------

class TestEnglishLocaleRegistration:
    def test_en_locale_registered(self):
        locale = get_locale("en")
        assert locale.code == "en"

    def test_en_locale_has_voice_profiles(self):
        locale = get_locale("en")
        assert "warm" in locale.voice_profiles
        assert locale.voice_profiles["warm"].voice_id == "en-US-JennyNeural"

    def test_en_locale_has_narrator_presets(self):
        locale = get_locale("en")
        assert DramaGenre.ROMANCE in locale.narrator_presets
        assert DramaGenre.MYSTERY in locale.narrator_presets

    def test_en_locale_has_genre_recommendations(self):
        locale = get_locale("en")
        assert DramaGenre.ACTION_THRILLER in locale.genre_voice_recommendations

    def test_en_locale_has_genre_analysis_prompt(self):
        locale = get_locale("en")
        assert "romance" in locale.genre_analysis_prompt
        assert "mystery" in locale.genre_analysis_prompt
        # Should be English, not Chinese
        assert "甜宠" not in locale.genre_analysis_prompt

    def test_en_locale_has_voice_casting_prompt(self):
        locale = get_locale("en")
        assert "en-US-JennyNeural" in locale.voice_casting_prompt
        assert "en-US-DavisNeural" in locale.voice_casting_prompt
        assert "en-GB-SoniaNeural" in locale.voice_casting_prompt

    def test_en_locale_has_dialogue_extraction_prompt(self):
        locale = get_locale("en")
        prompt = locale.dialogue_extraction_prompt
        assert "narration" in prompt
        assert "dialogue" in prompt
        assert "inner_monologue" in prompt
        # English markers instead of Chinese
        assert "VO" in prompt
        assert "心想" not in prompt

    def test_en_locale_genres(self):
        locale = get_locale("en")
        assert DramaGenre.ROMANCE in locale.genres
        assert DramaGenre.SCI_FI in locale.genres
        assert DramaGenre.COMEDY in locale.genres

    def test_en_subtitle_config(self):
        locale = get_locale("en")
        sc = locale.subtitle_config
        assert sc.line_break_strategy == "word"
        assert sc.colon_char == ": "
        assert sc.max_chars_per_line > 20  # wider than Chinese


# ---------------------------------------------------------------------------
# B2.5 — VoiceCaster with English prompts
# ---------------------------------------------------------------------------

class TestVoiceCasterEnglishPrompts:
    @pytest.mark.asyncio
    async def test_analyze_genre_english_sends_english_prompt(self):
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = '{"genre": "romance"}'
        caster = VoiceCaster(llm=mock_llm)

        genre = await caster.analyze_genre("A romantic story...", language="en")
        assert genre == DramaGenre.ROMANCE

        # Verify the system prompt was English locale prompt
        call_messages = mock_llm.chat.call_args[1]["messages"]
        system_msg = call_messages[0]["content"]
        assert "romance" in system_msg
        assert "Western" in system_msg or "English" in system_msg or "drama" in system_msg

    @pytest.mark.asyncio
    async def test_analyze_genre_english_user_message_is_english(self):
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = '{"genre": "mystery"}'
        caster = VoiceCaster(llm=mock_llm)

        await caster.analyze_genre("A detective story...", language="en")

        call_messages = mock_llm.chat.call_args[1]["messages"]
        user_msg = call_messages[1]["content"]
        # English user message should say "Script text" not "剧本文本"
        assert "Script text" in user_msg
        assert "剧本文本" not in user_msg

    @pytest.mark.asyncio
    async def test_analyze_genre_zh_user_message_is_chinese(self):
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = '{"genre": "other"}'
        caster = VoiceCaster(llm=mock_llm)

        await caster.analyze_genre("剧本内容...", language="zh")

        call_messages = mock_llm.chat.call_args[1]["messages"]
        user_msg = call_messages[1]["content"]
        assert "剧本文本" in user_msg

    @pytest.mark.asyncio
    async def test_cast_voices_english_uses_locale_prompt(self):
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = '{"characters": [{"name": "Alice", "voice_id": "en-US-JennyNeural", "speed": 1.0, "pitch": 0, "emotion": "happy", "age_feel": "young_adult", "energy": "medium", "description": "warm lead"}]}'
        caster = VoiceCaster(llm=mock_llm)
        series = DramaSeries(
            characters=[Character(name="Alice", description="Warm heroine")],
        )

        voice_map = await caster.cast_voices(series, DramaGenre.ROMANCE, language="en")

        # Narrator should use English locale preset
        assert "narrator" in voice_map
        narrator = voice_map["narrator"]
        assert narrator.line_type == LineType.NARRATION
        # English romance narrator should be JennyNeural
        assert narrator.voice_id == "en-US-JennyNeural"

        # Character should be in the map
        assert "Alice" in voice_map

        # Verify English system prompt was used
        call_messages = mock_llm.chat.call_args[1]["messages"]
        system_msg = call_messages[0]["content"]
        assert "en-US-JennyNeural" in system_msg

    @pytest.mark.asyncio
    async def test_cast_voices_english_user_message_is_english(self):
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = '{"characters": []}'
        caster = VoiceCaster(llm=mock_llm)
        series = DramaSeries(
            characters=[Character(name="Bob", description="Villain")],
        )

        await caster.cast_voices(series, DramaGenre.ACTION_THRILLER, language="en")

        call_messages = mock_llm.chat.call_args[1]["messages"]
        user_msg = call_messages[1]["content"]
        assert "Genre:" in user_msg
        assert "Characters:" in user_msg
        assert "剧情类型" not in user_msg

    @pytest.mark.asyncio
    async def test_extract_dialogue_lines_english_uses_locale_prompt(self):
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = '{"lines": [{"text": "Hello there", "speaker": "narrator", "line_type": "narration", "emotion_hint": "neutral"}]}'
        caster = VoiceCaster(llm=mock_llm)

        episode = Episode(
            number=1,
            scenes=[
                DramaScene(scene_id="s01", narration="The city at night..."),
            ],
        )
        voice_map = {"narrator": VoiceProfile(role_name="narrator")}

        lines = await caster.extract_dialogue_lines(episode, voice_map, language="en")

        assert len(lines) == 1
        assert lines[0].text == "Hello there"

        # Verify English system prompt
        call_messages = mock_llm.chat.call_args[1]["messages"]
        system_msg = call_messages[0]["content"]
        assert "narration" in system_msg
        assert "VO" in system_msg

    @pytest.mark.asyncio
    async def test_extract_dialogue_lines_english_user_prefix(self):
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = '{"lines": []}'
        caster = VoiceCaster(llm=mock_llm)

        episode = Episode(
            number=1,
            scenes=[
                DramaScene(scene_id="s01", dialogue="Hi there", speaking_character="Alice"),
            ],
        )
        voice_map = {"Alice": VoiceProfile(role_name="Alice")}

        await caster.extract_dialogue_lines(episode, voice_map, language="en")

        call_messages = mock_llm.chat.call_args[1]["messages"]
        user_msg = call_messages[1]["content"]
        assert "Scene s01:" in user_msg
        assert "Dialogue:" in user_msg
        assert "Speaking character:" in user_msg
        assert "场景" not in user_msg


# ---------------------------------------------------------------------------
# B2.6 — assign_voice_profile locale-aware
# ---------------------------------------------------------------------------

class TestAssignVoiceProfileLocaleAware:
    def test_zh_warm_maps_to_friendly_person(self):
        char = Character(name="李明", description="warm hero", voice_style="warm")
        result = assign_voice_profile(char, language="zh")
        assert result.voice_profile is not None
        assert result.voice_profile.voice_id == "Friendly_Person"

    def test_en_warm_maps_to_jenny_neural(self):
        char = Character(name="Alice", description="warm heroine", voice_style="warm")
        result = assign_voice_profile(char, language="en")
        assert result.voice_profile is not None
        assert result.voice_profile.voice_id == "en-US-JennyNeural"

    def test_en_authoritative_maps_to_davis_neural(self):
        char = Character(name="Boss", description="authoritative boss", voice_style="authoritative")
        result = assign_voice_profile(char, language="en")
        assert result.voice_profile.voice_id == "en-US-DavisNeural"

    def test_en_dramatic_maps_to_ryan_neural(self):
        char = Character(name="Hero", description="dramatic hero", voice_style="dramatic")
        result = assign_voice_profile(char, language="en")
        assert result.voice_profile.voice_id == "en-GB-RyanNeural"

    def test_en_commanding_maps_to_davis_neural(self):
        char = Character(name="Commander", description="commanding", voice_style="commanding")
        result = assign_voice_profile(char, language="en")
        assert result.voice_profile.voice_id == "en-US-DavisNeural"

    def test_en_unknown_style_falls_back_to_warm(self):
        char = Character(name="Unknown", description="unknown style", voice_style="nonexistent_style")
        result = assign_voice_profile(char, language="en")
        # Should fall back to warm (en-US-JennyNeural)
        assert result.voice_profile is not None
        assert result.voice_profile.voice_id == "en-US-JennyNeural"

    def test_skips_character_with_existing_profile(self):
        existing = VoiceProfile(voice_id="custom-voice", role_name="Alice")
        char = Character(name="Alice", voice_style="warm", voice_profile=existing)
        result = assign_voice_profile(char, language="en")
        # Should not overwrite existing profile
        assert result.voice_profile.voice_id == "custom-voice"

    def test_default_language_is_zh(self):
        """Calling without language arg should use Chinese profiles."""
        char = Character(name="李明", voice_style="warm")
        result = assign_voice_profile(char)
        assert result.voice_profile.voice_id == "Friendly_Person"
