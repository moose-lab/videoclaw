"""Tests for Western (English) drama planner and locale wiring (Stream B1)."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, call

from videoclaw.drama.locale import EN_EPISODE_SCRIPT_PROMPT, EN_SERIES_OUTLINE_PROMPT, get_locale
from videoclaw.drama.models import (
    Character,
    DramaGenre,
    DramaScene,
    DramaSeries,
    Episode,
    ShotScale,
    ShotType,
)
from videoclaw.drama.planner import (
    EPISODE_SCRIPT_PROMPT,
    SERIES_OUTLINE_PROMPT,
    DramaPlanner,
)


# ---------------------------------------------------------------------------
# Sample English drama data — "The Neighbor" (suburban thriller)
# ---------------------------------------------------------------------------

EN_SERIES_OUTLINE_RESPONSE = json.dumps({
    "title": "The Neighbor",
    "genre": "action_thriller",
    "synopsis": (
        "A mild-mannered suburban nurse realizes her new next-door neighbor "
        "is a ghost from her classified past — and he knows everything."
    ),
    "characters": [
        {
            "name": "Claire",
            "description": (
                "Surface: overworked ER nurse, exhausted single mom. "
                "Hidden: former CIA field analyst who faked her death. "
                "Signature moment: snaps a zip-tie with one hand while calmly answering a phone call."
            ),
            "visual_prompt": (
                "Late 30s White woman, athletic build, auburn hair in a loose bun, "
                "wearing worn blue scrubs with a coffee stain, no makeup, tired eyes"
            ),
            "voice_style": "calm",
        },
        {
            "name": "Marcus",
            "description": (
                "Surface: friendly contractor who just moved in next door. "
                "Hidden: NSA handler sent to bring Claire back in — or eliminate her. "
                "Signature moment: his measured smile never reaches his eyes."
            ),
            "visual_prompt": (
                "Early 40s Black man, broad-shouldered, close-cropped hair, "
                "wearing a clean grey Henley and work boots, calm neutral expression"
            ),
            "voice_style": "authoritative",
        },
    ],
    "episodes": [
        {
            "number": 1,
            "title": "New Neighbor",
            "synopsis": (
                "Claire comes home to find Marcus already moved in next door. "
                "He hands her a package addressed to her classified alias — a name no civilian should know. "
                "[REVEAL beat: package label / SHOCK beat: her frozen expression]"
            ),
            "opening_hook": "A friendly man waves from the driveway holding a package with a name Claire hasn't used in seven years.",
            "duration_seconds": 60.0,
        },
        {
            "number": 2,
            "title": "He Knows",
            "synopsis": (
                "Marcus quotes her old mission report verbatim over the fence. "
                "Claire realizes she's been found — and has no safe house left. "
                "[INTELLIGENCE beat: mission quote / DREAD beat: her escape route is blocked]"
            ),
            "opening_hook": "Marcus recites Claire's classified field name as casually as a weather greeting.",
            "duration_seconds": 60.0,
        },
    ],
})

EN_EPISODE_SCRIPT_RESPONSE = json.dumps({
    "episode_title": "New Neighbor",
    "scenes": [
        {
            "scene_id": "ep01_s01",
            "description": "Suburban street at dusk. Claire pulls into her driveway, exhausted after a double shift.",
            "visual_prompt": (
                "Wide shot of a quiet American suburb at golden hour, "
                "late-30s auburn-haired woman in blue scrubs stepping out of an old Honda Civic, "
                "tired eyes scanning the street, warm amber sunlight, cinematic"
            ),
            "camera_movement": "dolly_in",
            "duration_seconds": 5.0,
            "dialogue": "",
            "dialogue_line_type": "",
            "narration": "Seven years of silence. She thought she'd earned it.",
            "speaking_character": "",
            "shot_scale": "wide",
            "shot_type": "establishing",
            "emotion": "tense",
            "characters_present": ["Claire"],
            "transition": "fade_in",
            "sfx": "car door, distant lawnmower",
        },
        {
            "scene_id": "ep01_s02",
            "description": "Marcus stands at the fence between properties, smiling, holding a padded envelope.",
            "visual_prompt": (
                "Medium shot of a suburban front yard at dusk, "
                "broad-shouldered Black man in grey Henley and work boots leaning on a white picket fence, "
                "holding a brown padded envelope, calm neutral expression, warm side lighting"
            ),
            "camera_movement": "static",
            "duration_seconds": 4.0,
            "dialogue": "Hey, neighbor. This came for you. I signed for it — hope that's okay.",
            "dialogue_line_type": "dialogue",
            "narration": "",
            "speaking_character": "Marcus",
            "shot_scale": "medium",
            "shot_type": "action",
            "emotion": "suspense",
            "characters_present": ["Claire", "Marcus"],
            "transition": "cut",
            "sfx": "",
        },
        {
            "scene_id": "ep01_s03",
            "description": "Claire's face as she reads the label. The name printed is OPERATIVE WREN — her buried alias.",
            "visual_prompt": (
                "Extreme close-up of a padded envelope label, "
                "the name OPERATIVE WREN printed in block letters, "
                "a woman's fingers tightening around the edge, dramatic shadows, cinematic"
            ),
            "camera_movement": "dolly_in",
            "duration_seconds": 3.0,
            "dialogue": "(VO) That name died with my old life.",
            "dialogue_line_type": "inner_monologue",
            "narration": "",
            "speaking_character": "Claire",
            "shot_scale": "close_up",
            "shot_type": "detail",
            "emotion": "shock",
            "characters_present": ["Claire"],
            "transition": "cut",
            "sfx": "heartbeat pulse",
        },
        {
            "scene_id": "ep01_s04",
            "description": "Claire snaps her head up, staring at Marcus. His expression hasn't changed.",
            "visual_prompt": (
                "Close-up of a late-30s White woman's face, auburn hair loose, "
                "blue scrubs, wide eyes locked on someone off-frame, "
                "jaw tight with controlled fear, golden-hour side light, atmospheric"
            ),
            "camera_movement": "static",
            "duration_seconds": 3.0,
            "dialogue": "Who gave you this?",
            "dialogue_line_type": "dialogue",
            "narration": "",
            "speaking_character": "Claire",
            "shot_scale": "close_up",
            "shot_type": "reaction",
            "emotion": "dread",
            "characters_present": ["Claire"],
            "transition": "cut",
            "sfx": "",
        },
        {
            "scene_id": "ep01_s05",
            "description": "Marcus tilts his head with that same calm smile — and says nothing.",
            "visual_prompt": (
                "Medium close-up of a broad-shouldered Black man in grey Henley, "
                "standing at a white picket fence, a slow deliberate head tilt, "
                "faint unreadable smile, deep amber backlight, cinematic moody"
            ),
            "camera_movement": "dolly_in",
            "duration_seconds": 3.5,
            "dialogue": "",
            "dialogue_line_type": "",
            "narration": "",
            "speaking_character": "",
            "shot_scale": "medium_close",
            "shot_type": "reaction",
            "emotion": "suspense",
            "characters_present": ["Marcus"],
            "transition": "cut",
            "sfx": "ambient cicadas, sudden silence",
        },
    ],
    "voice_over": {
        "text": "Seven years of silence. She thought she'd earned it.",
        "tone": "tense",
        "language": "en",
    },
    "music": {
        "style": "electronic",
        "mood": "tense",
        "tempo": 95,
    },
    "cliffhanger": "Marcus smiles and says, 'We've been watching you for three years, Wren.'",
})


# ---------------------------------------------------------------------------
# B1.5: Tests
# ---------------------------------------------------------------------------


def test_en_locale_registered():
    """English locale is auto-registered on import."""
    locale = get_locale("en")
    assert locale.code == "en"


def test_en_locale_has_series_outline_prompt():
    """English locale must include a non-empty series_outline_prompt."""
    locale = get_locale("en")
    assert locale.series_outline_prompt != ""
    assert "TikTok" in locale.series_outline_prompt
    assert "scroll" in locale.series_outline_prompt.lower()


def test_en_locale_has_episode_script_prompt():
    """English locale must include a non-empty episode_script_prompt."""
    locale = get_locale("en")
    assert locale.episode_script_prompt != ""
    assert "inner_monologue" in locale.episode_script_prompt
    assert "9:16" in locale.episode_script_prompt


def test_en_series_outline_prompt_has_western_archetypes():
    """EN_SERIES_OUTLINE_PROMPT must reference Western character archetypes."""
    assert "billionaire" in EN_SERIES_OUTLINE_PROMPT.lower()
    assert "logline" in EN_SERIES_OUTLINE_PROMPT.lower()
    assert "5-Act" in EN_SERIES_OUTLINE_PROMPT or "5-act" in EN_SERIES_OUTLINE_PROMPT.lower()


def test_en_episode_script_prompt_has_inner_monologue():
    """EN_EPISODE_SCRIPT_PROMPT must describe inner_monologue conventions."""
    assert "inner_monologue" in EN_EPISODE_SCRIPT_PROMPT
    assert "dialogue_line_type" in EN_EPISODE_SCRIPT_PROMPT
    # Should reference VO / internal markers instead of Chinese markers
    assert "(VO)" in EN_EPISODE_SCRIPT_PROMPT or "VO" in EN_EPISODE_SCRIPT_PROMPT


def test_en_episode_script_prompt_no_chinese_markers():
    """EN_EPISODE_SCRIPT_PROMPT must not contain Chinese-specific markers."""
    assert "心想" not in EN_EPISODE_SCRIPT_PROMPT
    assert "暗道" not in EN_EPISODE_SCRIPT_PROMPT


def test_en_locale_subtitle_config():
    """English subtitle config should use English-appropriate settings."""
    locale = get_locale("en")
    sc = locale.subtitle_config
    assert sc.font_name == "Arial"
    assert sc.font_size == 22
    assert sc.max_chars_per_line == 42
    assert sc.line_break_strategy == "word"
    assert sc.colon_char == ": "


def test_en_locale_character_image_style():
    """English locale character_image_style should reference Hollywood aesthetics."""
    locale = get_locale("en")
    assert "Hollywood" in locale.character_image_style
    assert "Western drama" in locale.character_image_style


def test_en_locale_genres():
    """English locale should include Western genre set."""
    locale = get_locale("en")
    assert DramaGenre.ROMANCE in locale.genres
    assert DramaGenre.ACTION_THRILLER in locale.genres
    assert DramaGenre.MYSTERY in locale.genres
    assert DramaGenre.SUPERNATURAL in locale.genres
    assert DramaGenre.DRAMA in locale.genres
    assert DramaGenre.SCI_FI in locale.genres
    assert DramaGenre.COMEDY in locale.genres
    assert DramaGenre.OTHER in locale.genres
    # Chinese-specific genres should NOT be in the English locale
    assert DramaGenre.SWEET_ROMANCE not in locale.genres
    assert DramaGenre.ANCIENT_XIANXIA not in locale.genres


def test_zh_locale_still_uses_chinese_prompt():
    """Chinese locale must still have Chinese-language prompts (no regression)."""
    locale = get_locale("zh")
    assert "竖屏短剧" in locale.series_outline_prompt
    assert "分镜" in locale.episode_script_prompt


# ---------------------------------------------------------------------------
# Planner wiring: plan_series uses locale prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_series_uses_en_prompt_for_english_series():
    """plan_series must send EN_SERIES_OUTLINE_PROMPT to LLM when language='en'."""
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=EN_SERIES_OUTLINE_RESPONSE)

    planner = DramaPlanner(llm=mock_llm)
    series = DramaSeries(
        title="The Neighbor",
        synopsis="A nurse discovers her new neighbor knows her buried CIA past.",
        language="en",
        total_episodes=2,
        target_episode_duration=60.0,
    )

    result = await planner.plan_series(series)

    # Verify the English prompt was used
    call_kwargs = mock_llm.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    system_msg = next(m["content"] for m in messages if m["role"] == "system")
    assert system_msg == EN_SERIES_OUTLINE_PROMPT
    assert system_msg != SERIES_OUTLINE_PROMPT  # must not use Chinese prompt

    # Verify English data is parsed correctly
    assert result.title == "The Neighbor"
    assert len(result.characters) == 2
    assert result.characters[0].name == "Claire"
    assert len(result.episodes) == 2
    assert result.episodes[0].title == "New Neighbor"


@pytest.mark.asyncio
async def test_plan_series_uses_zh_prompt_for_chinese_series():
    """plan_series must send SERIES_OUTLINE_PROMPT (Chinese) when language='zh'."""
    zh_response = json.dumps({
        "title": "命运",
        "genre": "sweet_romance",
        "synopsis": "一段跨越命运的爱情故事",
        "characters": [{"name": "林晓", "description": "女主", "visual_prompt": "young woman", "voice_style": "warm"}],
        "episodes": [{"number": 1, "title": "初见", "synopsis": "两人相遇", "opening_hook": "偶遇", "duration_seconds": 60.0}],
    }, ensure_ascii=False)

    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=zh_response)

    planner = DramaPlanner(llm=mock_llm)
    series = DramaSeries(
        title="命运",
        synopsis="甜蜜爱情剧",
        language="zh",
        total_episodes=1,
    )

    await planner.plan_series(series)

    call_kwargs = mock_llm.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    system_msg = next(m["content"] for m in messages if m["role"] == "system")
    assert system_msg == SERIES_OUTLINE_PROMPT
    assert "竖屏短剧" in system_msg


# ---------------------------------------------------------------------------
# Planner wiring: script_episode uses locale prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_script_episode_uses_en_prompt_for_english_series():
    """script_episode must send EN_EPISODE_SCRIPT_PROMPT when language='en'."""
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=EN_EPISODE_SCRIPT_RESPONSE)

    planner = DramaPlanner(llm=mock_llm)
    series = DramaSeries(
        title="The Neighbor",
        language="en",
        characters=[
            Character(name="Claire", visual_prompt="auburn-haired woman in scrubs"),
            Character(name="Marcus", visual_prompt="broad-shouldered man in grey Henley"),
        ],
    )
    episode = Episode(
        number=1,
        title="New Neighbor",
        synopsis="Claire discovers her new neighbor knows her buried alias.",
        duration_seconds=18.5,
    )

    script_data = await planner.script_episode(series, episode)

    # Verify EN prompt was used
    call_kwargs = mock_llm.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    system_msg = next(m["content"] for m in messages if m["role"] == "system")
    assert system_msg == EN_EPISODE_SCRIPT_PROMPT
    assert system_msg != EPISODE_SCRIPT_PROMPT

    # Verify English content is parsed into DramaScene objects
    assert len(episode.scenes) == 5
    assert episode.scenes[0].scene_id == "ep01_s01"
    assert episode.scenes[0].shot_scale == ShotScale.WIDE
    assert episode.scenes[0].shot_type == ShotType.ESTABLISHING
    assert episode.scenes[0].emotion == "tense"

    # Inner monologue in scene 3
    assert episode.scenes[2].dialogue_line_type == "inner_monologue"
    assert episode.scenes[2].speaking_character == "Claire"
    assert "(VO)" in episode.scenes[2].dialogue

    # English cliffhanger
    assert "Wren" in script_data["cliffhanger"]
    assert episode.script is not None


@pytest.mark.asyncio
async def test_script_episode_uses_zh_prompt_for_chinese_series():
    """script_episode must send EPISODE_SCRIPT_PROMPT (Chinese) when language='zh'."""
    zh_script = json.dumps({
        "episode_title": "命运来电",
        "scenes": [
            {
                "scene_id": "ep01_s01",
                "description": "深夜办公室",
                "visual_prompt": "dark office, woman at desk",
                "camera_movement": "static",
                "duration_seconds": 8.0,
                "dialogue": "喂？",
                "dialogue_line_type": "dialogue",
                "narration": "",
                "speaking_character": "林晓",
                "shot_scale": "medium_close",
                "shot_type": "action",
                "emotion": "suspense",
                "characters_present": ["林晓"],
                "transition": "cut",
                "sfx": "",
            }
        ],
        "voice_over": {"text": "深夜", "tone": "dramatic", "language": "zh"},
        "music": {"style": "orchestral", "mood": "mysterious", "tempo": 90},
        "cliffhanger": "电话那头是她自己的声音",
    }, ensure_ascii=False)

    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=zh_script)

    planner = DramaPlanner(llm=mock_llm)
    series = DramaSeries(title="测试剧", language="zh", characters=[Character(name="林晓")])
    episode = Episode(number=1, title="命运来电", synopsis="测试", duration_seconds=8.0)

    await planner.script_episode(series, episode)

    call_kwargs = mock_llm.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    system_msg = next(m["content"] for m in messages if m["role"] == "system")
    assert system_msg == EPISODE_SCRIPT_PROMPT
    assert "分镜" in system_msg


# ---------------------------------------------------------------------------
# EN prompt JSON schema completeness checks
# ---------------------------------------------------------------------------


def test_en_series_outline_prompt_has_json_schema():
    """EN_SERIES_OUTLINE_PROMPT must document the same JSON fields as the Chinese version."""
    for field in ("title", "genre", "synopsis", "characters", "episodes",
                  "visual_prompt", "voice_style", "opening_hook", "duration_seconds"):
        assert field in EN_SERIES_OUTLINE_PROMPT, f"Missing field '{field}' in EN_SERIES_OUTLINE_PROMPT"


def test_en_episode_script_prompt_has_json_schema():
    """EN_EPISODE_SCRIPT_PROMPT must document the full scene JSON schema."""
    for field in ("scene_id", "description", "visual_prompt", "camera_movement",
                  "duration_seconds", "dialogue", "dialogue_line_type", "narration",
                  "speaking_character", "shot_scale", "shot_type", "emotion",
                  "characters_present", "transition", "sfx",
                  "voice_over", "music", "cliffhanger"):
        assert field in EN_EPISODE_SCRIPT_PROMPT, f"Missing field '{field}' in EN_EPISODE_SCRIPT_PROMPT"
