"""Tests for drama planner (planner.py)."""

import json
import pytest

from unittest.mock import AsyncMock

from videoclaw.drama.models import (
    Character,
    DramaScene,
    DramaSeries,
    Episode,
    ShotScale,
    ShotType,
)
from videoclaw.drama.planner import DramaPlanner


# ---------------------------------------------------------------------------
# Mock LLM → DramaScene parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_script_episode_parses_mock_llm_response():
    """script_episode should parse mock LLM JSON into DramaScene objects."""
    mock_llm_response = json.dumps({
        "episode_title": "命运来电",
        "scenes": [
            {
                "scene_id": "ep01_s01",
                "description": "深夜办公室，林晓接到神秘电话",
                "visual_prompt": "Modern office at night, young Chinese woman in business suit, short black hair, looking shocked at phone, dramatic lighting from desk lamp",
                "camera_movement": "dolly_in",
                "duration_seconds": 5.0,
                "dialogue": "喂？你是谁？",
                "narration": "",
                "speaking_character": "林晓",
                "shot_scale": "medium_close",
                "shot_type": "action",
                "emotion": "suspense",
                "characters_present": ["林晓"],
                "transition": "fade_in",
            },
            {
                "scene_id": "ep01_s02",
                "description": "林晓震惊地看着手机屏幕",
                "visual_prompt": "Close-up of young Chinese woman's face, short black hair, eyes wide with shock, phone screen illuminating her face, dark office background",
                "camera_movement": "static",
                "duration_seconds": 3.0,
                "dialogue": "",
                "narration": "那一刻，她的世界彻底改变了",
                "speaking_character": "",
                "shot_scale": "close_up",
                "shot_type": "reaction",
                "emotion": "shock",
                "characters_present": ["林晓"],
                "transition": "cut",
            },
        ],
        "voice_over": {"text": "那一刻，她的世界彻底改变了", "tone": "dramatic", "language": "zh"},
        "music": {"style": "orchestral", "mood": "mysterious", "tempo": 90},
        "cliffhanger": "电话那头的声音，竟然是她自己",
    }, ensure_ascii=False)

    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=mock_llm_response)

    planner = DramaPlanner(llm=mock_llm)
    series = DramaSeries(title="测试剧", characters=[Character(name="林晓")])
    episode = Episode(number=1, title="命运来电", synopsis="深夜接到神秘电话", duration_seconds=8.0)

    script_data = await planner.script_episode(series, episode)

    assert len(episode.scenes) == 2
    assert episode.scenes[0].scene_id == "ep01_s01"
    assert episode.scenes[0].speaking_character == "林晓"
    assert episode.scenes[0].shot_scale == ShotScale.MEDIUM_CLOSE
    assert episode.scenes[0].shot_type == ShotType.ACTION
    assert episode.scenes[0].emotion == "suspense"
    assert episode.scenes[0].characters_present == ["林晓"]
    assert episode.scenes[0].transition == "fade_in"

    assert episode.scenes[1].shot_scale == ShotScale.CLOSE_UP
    assert episode.scenes[1].shot_type == ShotType.REACTION
    assert episode.scenes[1].narration == "那一刻，她的世界彻底改变了"

    assert episode.script is not None
    assert "命运来电" in episode.script
    assert script_data["cliffhanger"] == "电话那头的声音，竟然是她自己"


# ---------------------------------------------------------------------------
# Duration validation / adjustment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_script_episode_adjusts_duration_proportionally():
    """When scene durations deviate >5s from target, planner should scale them."""
    mock_llm_response = json.dumps({
        "episode_title": "时长测试",
        "scenes": [
            {
                "scene_id": "ep01_s01",
                "description": "场景1",
                "visual_prompt": "scene one",
                "camera_movement": "static",
                "duration_seconds": 20.0,
                "dialogue": "",
                "narration": "",
                "speaking_character": "",
                "shot_scale": "medium",
                "shot_type": "action",
                "emotion": "tense",
                "characters_present": [],
                "transition": "cut",
            },
            {
                "scene_id": "ep01_s02",
                "description": "场景2",
                "visual_prompt": "scene two",
                "camera_movement": "static",
                "duration_seconds": 40.0,
                "dialogue": "",
                "narration": "",
                "speaking_character": "",
                "shot_scale": "wide",
                "shot_type": "establishing",
                "emotion": "warm",
                "characters_present": [],
                "transition": "cut",
            },
        ],
        "voice_over": {"text": "", "tone": "warm", "language": "zh"},
        "music": {"style": "acoustic", "mood": "romantic", "tempo": 80},
        "cliffhanger": "测试悬念",
    }, ensure_ascii=False)

    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=mock_llm_response)

    planner = DramaPlanner(llm=mock_llm)
    series = DramaSeries(title="测试剧")
    episode = Episode(number=1, title="时长测试", synopsis="测试", duration_seconds=30.0)

    await planner.script_episode(series, episode)

    # Original: 20+40=60s, target=30s → scale by 0.5
    total = sum(s.duration_seconds for s in episode.scenes)
    assert abs(total - 30.0) <= 2.0, f"Expected ~30s, got {total}s"
    # Ratio should be preserved: scene1 should be ~half of scene2
    assert episode.scenes[0].duration_seconds < episode.scenes[1].duration_seconds


@pytest.mark.asyncio
async def test_script_episode_no_adjustment_within_tolerance():
    """When scene durations are within ±5s of target, no adjustment should occur."""
    mock_llm_response = json.dumps({
        "episode_title": "精准时长",
        "scenes": [
            {
                "scene_id": "ep01_s01",
                "description": "场景",
                "visual_prompt": "scene",
                "camera_movement": "static",
                "duration_seconds": 28.0,
                "dialogue": "",
                "narration": "",
                "speaking_character": "",
                "shot_scale": "medium",
                "shot_type": "action",
                "emotion": "tense",
                "characters_present": [],
                "transition": "cut",
            },
        ],
        "voice_over": {"text": "", "tone": "warm", "language": "zh"},
        "music": {"style": "acoustic", "mood": "romantic", "tempo": 80},
        "cliffhanger": "测试",
    }, ensure_ascii=False)

    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=mock_llm_response)

    planner = DramaPlanner(llm=mock_llm)
    series = DramaSeries(title="测试剧")
    episode = Episode(number=1, title="精准", synopsis="测试", duration_seconds=30.0)

    await planner.script_episode(series, episode)

    # 28s is within 5s of 30s target — no scaling
    assert episode.scenes[0].duration_seconds == 28.0


@pytest.mark.asyncio
async def test_parse_json_strips_markdown_fences():
    """_parse_json should handle markdown code fences around JSON."""
    planner = DramaPlanner()

    fenced = '```json\n{"key": "value"}\n```'
    result = planner._parse_json(fenced)
    assert result == {"key": "value"}

    bare = '{"key": "value"}'
    result = planner._parse_json(bare)
    assert result == {"key": "value"}

    with pytest.raises(ValueError, match="invalid JSON"):
        planner._parse_json("not json at all")
