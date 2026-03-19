"""Tests for drama quality validators — Western and Chinese locale rules.

Covers:
- validate_western_quality: valid drama passes, each rule fires on bad input
- validate_chinese_quality: valid Chinese drama passes
- DramaQualityValidator: dispatches to correct validator by language
"""

from __future__ import annotations

import pytest

from videoclaw.drama.models import Character, DramaSeries
from videoclaw.drama.quality import (
    DramaQualityValidator,
    validate_chinese_quality,
    validate_western_quality,
)


# ---------------------------------------------------------------------------
# Helpers — minimal valid fixtures
# ---------------------------------------------------------------------------

def _make_western_series(**overrides) -> DramaSeries:
    """Return a minimal valid Western/English DramaSeries."""
    defaults = dict(
        title="Crimson Vows",
        genre="romance",
        synopsis=(
            "A disgraced detective secretly working to clear her name falls for the very "
            "billionaire she is investigating, but hidden loyalties threaten to destroy them both."
        ),
        language="en",
        characters=[
            Character(
                name="Elena Cross",
                description=(
                    "On the surface a confident PI but secretly a suspended detective "
                    "working to clear her name. Beneath her charm lies a fierce determination."
                ),
                visual_prompt="Tall woman in her 30s, auburn hair, sharp green eyes, leather jacket",
            ),
            Character(
                name="Marcus Vale",
                description=(
                    "A polished billionaire who appears untouchable but actually harbours "
                    "hidden guilt over his father's crimes."
                ),
                visual_prompt="Tall man in his 40s, silver-streaked hair, intense dark eyes, tailored suit",
            ),
        ],
    )
    defaults.update(overrides)
    return DramaSeries(**defaults)


def _make_western_episode_scripts(
    first_scene_duration: float = 4.0,
    include_cliffhanger: bool = True,
    cjk_in_visual: bool = False,
    dialogue_words: int = 10,
) -> dict[int, dict]:
    """Build a minimal 1-episode scripts dict that passes all Western rules by default."""
    visual = "Dimly lit office, woman in leather jacket standing over a desk of files, dramatic shadows"
    if cjk_in_visual:
        visual = "美丽的女人 standing in an office"

    speaking_character = "Elena Cross"
    characters_present = ["Elena Cross"]

    dialogue_text = " ".join(["word"] * dialogue_words)

    scenes = [
        # Scene 1 — hook (≤5s)
        {
            "scene_id": "ep01_s01",
            "description": "Elena finds a key piece of evidence",
            "visual_prompt": visual,
            "camera_movement": "dolly_in",
            "duration_seconds": first_scene_duration,
            "dialogue": "",
            "narration": "She had been running for three years.",
            "speaking_character": "",
            "shot_scale": "close_up",
            "shot_type": "detail",
            "emotion": "shock",
            "characters_present": ["Elena Cross"],
            "transition": "fade_in",
        },
        # Scene 2 — emotional peak
        {
            "scene_id": "ep01_s02",
            "description": "Elena confronts Marcus",
            "visual_prompt": "Two people in tense standoff in a penthouse, city lights behind",
            "camera_movement": "static",
            "duration_seconds": 8.0,
            "dialogue": dialogue_text,
            "narration": "",
            "speaking_character": speaking_character,
            "shot_scale": "medium_close",
            "shot_type": "reaction",
            "emotion": "triumphant",
            "characters_present": characters_present,
            "transition": "cut",
        },
        # Scene 3 — another close-up to keep vertical framing ratio up
        {
            "scene_id": "ep01_s03",
            "description": "Elena stares at the photo",
            "visual_prompt": "Extreme close-up of woman's eyes reflecting a photograph, tension",
            "camera_movement": "static",
            "duration_seconds": 5.0,
            "dialogue": "",
            "narration": "",
            "speaking_character": "",
            "shot_scale": "close_up",
            "shot_type": "detail",
            "emotion": "fearful",
            "characters_present": ["Elena Cross"],
            "transition": "cut",
        },
        # Scene 4 — medium shot
        {
            "scene_id": "ep01_s04",
            "description": "Wide establishing shot of the city",
            "visual_prompt": "Wide aerial shot of city skyline at dusk, dramatic clouds",
            "camera_movement": "crane_up",
            "duration_seconds": 3.0,
            "dialogue": "",
            "narration": "",
            "speaking_character": "",
            "shot_scale": "medium_close",
            "shot_type": "establishing",
            "emotion": "tense",
            "characters_present": [],
            "transition": "cut",
        },
    ]

    script: dict = {
        "episode_title": "The First Lie",
        "scenes": scenes,
        "voice_over": {"text": "She had been running for three years.", "tone": "dramatic", "language": "en"},
        "music": {"style": "orchestral", "mood": "tense", "tempo": 110},
    }
    if include_cliffhanger:
        script["cliffhanger"] = "Elena discovers the key belongs to a safe she thought was destroyed — along with everyone who knew about it."

    return {1: script}


def _make_chinese_series(**overrides) -> DramaSeries:
    defaults = dict(
        title="这个王妃太狂野",
        genre="穿越/古装",
        synopsis="现代女特工林薇穿越成被休弃的废物王妃，众人等着看笑话，殊不知她前世是顶级杀手。这一世，她要让所有欺辱她的人跪着叫姐姐。",
        language="zh",
        characters=[
            Character(
                name="林薇",
                description=(
                    "表面身份：被休弃的废物王妃；隐藏身份：穿越而来的现代女特工。"
                    "记忆点：每次反杀前都会微微一笑。"
                ),
                visual_prompt="Young Chinese woman, early 20s, delicate oval face, long black hair",
            ),
        ],
    )
    defaults.update(overrides)
    return DramaSeries(**defaults)


def _make_chinese_episode_scripts() -> dict[int, dict]:
    return {
        1: {
            "episode_title": "废妃重生",
            "scenes": [
                {
                    "scene_id": "ep01_s01",
                    "description": "林薇醒来",
                    "visual_prompt": "Dark room, young woman lying on straw, dramatic shadows",
                    "camera_movement": "dolly_in",
                    "duration_seconds": 4.0,
                    "dialogue": "",
                    "narration": "她睁开眼的那一刻",
                    "speaking_character": "",
                    "shot_scale": "close_up",
                    "shot_type": "detail",
                    "emotion": "dread",
                    "characters_present": ["林薇"],
                    "transition": "fade_in",
                },
                {
                    "scene_id": "ep01_s02",
                    "description": "林薇反击",
                    "visual_prompt": "Young woman executing a swift wrist-lock, dynamic motion",
                    "camera_movement": "handheld",
                    "duration_seconds": 4.0,
                    "dialogue": "你确定？",
                    "narration": "",
                    "speaking_character": "林薇",
                    "shot_scale": "medium_close",
                    "shot_type": "action",
                    "emotion": "triumphant",
                    "characters_present": ["林薇"],
                    "transition": "cut",
                },
                {
                    "scene_id": "ep01_s03",
                    "description": "林薇微笑",
                    "visual_prompt": "Close-up of young woman's knowing smile",
                    "camera_movement": "static",
                    "duration_seconds": 3.0,
                    "dialogue": "",
                    "narration": "",
                    "speaking_character": "",
                    "shot_scale": "close_up",
                    "shot_type": "reaction",
                    "emotion": "smug",
                    "characters_present": ["林薇"],
                    "transition": "cut",
                },
                {
                    "scene_id": "ep01_s04",
                    "description": "慕容雪反应",
                    "visual_prompt": "Silhouette of woman in pink silk hanfu, ominous",
                    "camera_movement": "static",
                    "duration_seconds": 4.0,
                    "dialogue": "",
                    "narration": "",
                    "speaking_character": "",
                    "shot_scale": "medium_close",
                    "shot_type": "reaction",
                    "emotion": "shock",
                    "characters_present": ["慕容雪"],
                    "transition": "cut",
                },
            ],
            "voice_over": {"text": "她睁开眼的那一刻，一切都改变了。", "tone": "dramatic", "language": "zh"},
            "music": {"style": "chinese_traditional", "mood": "mysterious", "tempo": 100},
            "cliffhanger": "林薇眼中映着的现代城市倒影——她到底是谁？",
        }
    }


# ===========================================================================
# Western quality validator tests
# ===========================================================================

class TestValidateWesternQuality:

    def test_valid_western_drama_no_violations(self):
        series = _make_western_series()
        scripts = _make_western_episode_scripts()
        violations = validate_western_quality(series, scripts)
        assert violations == [], f"Expected no violations, got: {violations}"

    # Rule 1: logline word count
    def test_synopsis_too_short(self):
        series = _make_western_series(synopsis="Too short.")
        violations = validate_western_quality(series, _make_western_episode_scripts())
        assert any("Synopsis word count" in v for v in violations), violations

    def test_synopsis_too_long(self):
        long_synopsis = " ".join(["word"] * 200)
        series = _make_western_series(synopsis=long_synopsis)
        violations = validate_western_quality(series, _make_western_episode_scripts())
        assert any("Synopsis word count" in v for v in violations), violations

    def test_synopsis_in_range_passes(self):
        # 10 words is the lower boundary
        series = _make_western_series(synopsis="She falls in love with the enemy agent she must stop.")
        violations = validate_western_quality(series, _make_western_episode_scripts())
        assert not any("Synopsis word count" in v for v in violations), violations

    # Rule 2: character duality
    def test_character_missing_duality(self):
        series = _make_western_series(
            characters=[
                Character(
                    name="Flat Character",
                    description="A nice person who likes coffee.",
                    visual_prompt="Person with coffee",
                )
            ]
        )
        violations = validate_western_quality(series, _make_western_episode_scripts())
        assert any("duality" in v.lower() for v in violations), violations

    def test_character_with_secretly_passes(self):
        series = _make_western_series(
            characters=[
                Character(
                    name="Agent X",
                    description="A mild-mannered librarian who secretly runs an underground network.",
                    visual_prompt="Quiet person in a library",
                )
            ]
        )
        violations = validate_western_quality(series, _make_western_episode_scripts())
        assert not any("duality" in v.lower() for v in violations), violations

    # Rule 3: scroll-stopping hook
    def test_first_scene_too_long(self):
        scripts = _make_western_episode_scripts(first_scene_duration=8.0)
        violations = validate_western_quality(_make_western_series(), scripts)
        assert any("first scene" in v.lower() for v in violations), violations

    def test_first_scene_exactly_5s_passes(self):
        scripts = _make_western_episode_scripts(first_scene_duration=5.0)
        violations = validate_western_quality(_make_western_series(), scripts)
        assert not any("first scene" in v.lower() for v in violations), violations

    # Rule 4: payoff density — tested indirectly (valid fixture has enough peaks)

    # Rule 5: cliffhanger
    def test_missing_cliffhanger(self):
        scripts = _make_western_episode_scripts(include_cliffhanger=False)
        violations = validate_western_quality(_make_western_series(), scripts)
        assert any("cliffhanger" in v.lower() for v in violations), violations

    def test_empty_cliffhanger(self):
        scripts = _make_western_episode_scripts(include_cliffhanger=True)
        scripts[1]["cliffhanger"] = ""
        violations = validate_western_quality(_make_western_series(), scripts)
        assert any("cliffhanger" in v.lower() for v in violations), violations

    # Rule 6: vertical framing — valid fixture has 3/4 close shots = 75%, passes

    def test_vertical_framing_fails(self):
        scripts = _make_western_episode_scripts()
        # Override all shots to wide
        for scene in scripts[1]["scenes"]:
            scene["shot_scale"] = "wide"
        violations = validate_western_quality(_make_western_series(), scripts)
        assert any("vertical" in v.lower() or "close_up" in v.lower() for v in violations), violations

    # Rule 7: character consistency
    def test_speaking_character_not_in_present(self):
        """Speaker named but characters_present lists someone else entirely."""
        scripts = _make_western_episode_scripts()
        # Set speaking_character to someone not in characters_present
        scripts[1]["scenes"][1]["speaking_character"] = "Elena Cross"
        scripts[1]["scenes"][1]["characters_present"] = ["Marcus Vale"]  # Elena absent
        violations = validate_western_quality(_make_western_series(), scripts)
        assert any("speaking_character" in v for v in violations), violations

    # Rule 8: dialogue density
    def test_dialogue_too_dense(self):
        # 4 scenes of ~5s each = 20s total; limit = 40 * 20/60 ≈ 13 words; use 50 words
        scripts = _make_western_episode_scripts(dialogue_words=50)
        violations = validate_western_quality(_make_western_series(), scripts)
        assert any("dialogue density" in v.lower() for v in violations), violations

    def test_dialogue_within_limit_passes(self):
        scripts = _make_western_episode_scripts(dialogue_words=5)
        violations = validate_western_quality(_make_western_series(), scripts)
        assert not any("dialogue density" in v.lower() for v in violations), violations

    # Rule 9: emotion vocabulary
    def test_missing_emotion(self):
        scripts = _make_western_episode_scripts()
        scripts[1]["scenes"][0]["emotion"] = ""
        violations = validate_western_quality(_make_western_series(), scripts)
        assert any("emotion" in v.lower() for v in violations), violations

    # Rule 10: English visual prompts — no CJK
    def test_cjk_in_visual_prompt(self):
        scripts = _make_western_episode_scripts(cjk_in_visual=True)
        violations = validate_western_quality(_make_western_series(), scripts)
        assert any("CJK" in v or "visual_prompt" in v for v in violations), violations

    def test_english_only_visual_prompt_passes(self):
        scripts = _make_western_episode_scripts(cjk_in_visual=False)
        violations = validate_western_quality(_make_western_series(), scripts)
        assert not any("CJK" in v for v in violations), violations


# ===========================================================================
# Chinese quality validator tests
# ===========================================================================

class TestValidateChineseQuality:

    def test_valid_chinese_drama_no_violations(self):
        series = _make_chinese_series()
        scripts = _make_chinese_episode_scripts()
        violations = validate_chinese_quality(series, scripts)
        assert violations == [], f"Expected no violations, got: {violations}"

    def test_synopsis_too_short(self):
        series = _make_chinese_series(synopsis="太短")
        violations = validate_chinese_quality(series, _make_chinese_episode_scripts())
        assert any("synopsis" in v.lower() or "太短" in v for v in violations), violations

    def test_character_missing_dual_identity(self):
        series = _make_chinese_series(
            characters=[
                Character(
                    name="普通角色",
                    description="一个普通的人。",
                    visual_prompt="A person",
                )
            ]
        )
        violations = validate_chinese_quality(series, _make_chinese_episode_scripts())
        assert any("反差" in v or "表面" in v or "隐藏" in v for v in violations), violations

    def test_missing_cliffhanger_zh(self):
        scripts = _make_chinese_episode_scripts()
        scripts[1]["cliffhanger"] = ""
        violations = validate_chinese_quality(_make_chinese_series(), scripts)
        assert any("cliffhanger" in v for v in violations), violations

    def test_cjk_in_visual_prompt_zh(self):
        scripts = _make_chinese_episode_scripts()
        scripts[1]["scenes"][0]["visual_prompt"] = "美丽的女人 in a dark room"
        violations = validate_chinese_quality(_make_chinese_series(), scripts)
        assert any("visual_prompt" in v or "中文" in v for v in violations), violations


# ===========================================================================
# DramaQualityValidator dispatcher tests
# ===========================================================================

class TestDramaQualityValidator:

    def test_dispatches_to_western_validator_for_en(self):
        """DramaQualityValidator calls validate_western_quality for language='en'."""
        validator = DramaQualityValidator()
        series = _make_western_series()
        scripts = _make_western_episode_scripts()
        violations = validator.validate(series, scripts)
        # Valid Western drama should pass
        assert violations == [], f"Unexpected violations: {violations}"

    def test_dispatches_to_chinese_validator_for_zh(self):
        """DramaQualityValidator calls validate_chinese_quality for language='zh'."""
        validator = DramaQualityValidator()
        series = _make_chinese_series()
        scripts = _make_chinese_episode_scripts()
        violations = validator.validate(series, scripts)
        assert violations == [], f"Unexpected violations: {violations}"

    def test_western_bad_input_produces_violations_via_dispatcher(self):
        validator = DramaQualityValidator()
        series = _make_western_series()
        scripts = _make_western_episode_scripts(include_cliffhanger=False)
        violations = validator.validate(series, scripts)
        assert any("cliffhanger" in v.lower() for v in violations), violations

    def test_chinese_bad_input_produces_violations_via_dispatcher(self):
        validator = DramaQualityValidator()
        series = _make_chinese_series(synopsis="短")
        scripts = _make_chinese_episode_scripts()
        violations = validator.validate(series, scripts)
        assert len(violations) > 0

    def test_unknown_language_falls_back_to_zh(self):
        """Unknown language code falls back to 'zh' locale (and its validator)."""
        validator = DramaQualityValidator()
        series = _make_chinese_series()
        series.language = "ja"  # unknown, should fall back to zh
        scripts = _make_chinese_episode_scripts()
        # Should not raise; uses zh validator
        violations = validator.validate(series, scripts)
        assert isinstance(violations, list)
