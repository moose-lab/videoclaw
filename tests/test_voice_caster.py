"""Tests for VoiceCaster — LLM-powered voice casting for multi-role TTS."""

import pytest
from unittest.mock import AsyncMock

from videoclaw.drama.models import (
    Character,
    DialogueLine,
    DramaGenre,
    DramaScene,
    DramaSeries,
    Episode,
    LineType,
    NARRATOR_PRESETS,
    VoiceProfile,
)
from videoclaw.generation.audio.voice_caster import VoiceCaster


@pytest.mark.asyncio
async def test_analyze_genre_returns_enum():
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = '{"genre": "suspense_thriller"}'
    caster = VoiceCaster(llm=mock_llm)
    genre = await caster.analyze_genre("黑暗走廊...")
    assert genre == DramaGenre.SUSPENSE_THRILLER


@pytest.mark.asyncio
async def test_analyze_genre_fallback():
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = '{"genre": "unknown"}'
    caster = VoiceCaster(llm=mock_llm)
    genre = await caster.analyze_genre("...")
    assert genre == DramaGenre.OTHER


@pytest.mark.asyncio
async def test_cast_voices_includes_narrator():
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = '{"characters": [{"name": "林薇", "voice_id": "Lively_Girl", "speed": 1.05, "pitch": 2, "emotion": "happy", "age_feel": "young_adult", "energy": "high", "description": "活泼少女"}]}'
    caster = VoiceCaster(llm=mock_llm)
    series = DramaSeries(
        characters=[Character(name="林薇", description="活泼少女")],
        episodes=[
            Episode(
                number=1,
                scenes=[DramaScene(dialogue="你好", speaking_character="林薇")],
            )
        ],
    )
    voice_map = await caster.cast_voices(series, DramaGenre.SWEET_ROMANCE)
    assert "narrator" in voice_map
    assert voice_map["narrator"].line_type == LineType.NARRATION
    assert "林薇" in voice_map


@pytest.mark.asyncio
async def test_extract_dialogue_lines_skips_empty():
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = '{"lines": [{"text": "你怎么来了", "speaker": "林薇", "line_type": "dialogue", "emotion_hint": "surprised"}]}'
    caster = VoiceCaster(llm=mock_llm)
    episode = Episode(
        number=1,
        scenes=[
            DramaScene(scene_id="s01", dialogue="你怎么来了", speaking_character="林薇"),
            DramaScene(scene_id="s02", dialogue="", narration=""),
        ],
    )
    voice_map = {
        "narrator": NARRATOR_PRESETS[DramaGenre.OTHER],
        "林薇": VoiceProfile(voice_id="Lively_Girl", role_name="林薇"),
    }
    lines = await caster.extract_dialogue_lines(episode, voice_map)
    assert len(lines) >= 1
    assert all(isinstance(l, DialogueLine) for l in lines)
    assert not any(l.scene_id == "s02" for l in lines)
