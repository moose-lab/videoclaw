"""Tests for PromptEnhancer — visual prompt enrichment."""

from videoclaw.drama.models import (
    Character,
    DramaScene,
    DramaSeries,
    Episode,
    ShotScale,
)
from videoclaw.drama.prompt_enhancer import PromptEnhancer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(model_id: str = "minimax-hailuo-2.3", style: str = "cinematic", **kw) -> DramaSeries:
    defaults = dict(
        title="Test Drama",
        genre="sweet_romance",
        style=style,
        aspect_ratio="9:16",
        model_id=model_id,
        characters=[
            Character(name="Lin Yue", visual_prompt="young woman, long black hair, white hanfu dress"),
            Character(name="Zhao Ming", visual_prompt="tall man, sharp jawline, black martial robe"),
        ],
    )
    defaults.update(kw)
    return DramaSeries(**defaults)


def _make_scene(**kw) -> DramaScene:
    defaults = dict(
        scene_id="ep01_s01",
        description="They meet at the bridge.",
        visual_prompt="两人在古桥上对视，微风吹过",
        camera_movement="dolly_in",
        shot_scale=ShotScale.MEDIUM,
        characters_present=["Lin Yue", "Zhao Ming"],
    )
    defaults.update(kw)
    return DramaScene(**defaults)


# ---------------------------------------------------------------------------
# Tests: character injection
# ---------------------------------------------------------------------------

class TestCharacterInjection:
    def test_enhanced_prompt_contains_character_visual_prompts(self):
        enhancer = PromptEnhancer(strip_chinese=False)
        result = enhancer.enhance_scene_prompt(_make_scene(), _make_series())
        assert "Lin Yue" in result
        assert "long black hair" in result
        assert "Zhao Ming" in result
        assert "black martial robe" in result

    def test_missing_character_is_skipped(self):
        scene = _make_scene(characters_present=["Lin Yue", "Unknown Person"])
        enhancer = PromptEnhancer(strip_chinese=False)
        result = enhancer.enhance_scene_prompt(scene, _make_series())
        assert "Lin Yue" in result
        assert "Unknown Person" not in result

    def test_no_characters_present(self):
        scene = _make_scene(characters_present=[])
        enhancer = PromptEnhancer(strip_chinese=False)
        result = enhancer.enhance_scene_prompt(scene, _make_series())
        # Should still have visual prompt, shot info, and style
        assert "dolly" in result.lower()
        assert "cinematic lighting" in result


# ---------------------------------------------------------------------------
# Tests: Chinese stripping
# ---------------------------------------------------------------------------

class TestChineseStripping:
    def test_strip_chinese_true_removes_cjk(self):
        enhancer = PromptEnhancer(strip_chinese=True)
        result = enhancer.enhance_scene_prompt(_make_scene(), _make_series())
        # Original visual_prompt had Chinese; it should be gone
        assert "古桥" not in result
        assert "对视" not in result

    def test_strip_chinese_false_keeps_cjk(self):
        enhancer = PromptEnhancer(strip_chinese=False)
        result = enhancer.enhance_scene_prompt(_make_scene(), _make_series())
        assert "古桥" in result

    def test_auto_detect_sora_strips_chinese(self):
        enhancer = PromptEnhancer(strip_chinese=None)
        series = _make_series(model_id="sora-turbo")
        result = enhancer.enhance_scene_prompt(_make_scene(), series)
        assert "古桥" not in result

    def test_auto_detect_minimax_keeps_chinese(self):
        enhancer = PromptEnhancer(strip_chinese=None)
        series = _make_series(model_id="minimax-hailuo-2.3")
        result = enhancer.enhance_scene_prompt(_make_scene(), series)
        assert "古桥" in result

    def test_auto_detect_runway_strips(self):
        enhancer = PromptEnhancer()
        assert enhancer.should_strip_chinese("runway-gen4") is True

    def test_auto_detect_pika_strips(self):
        enhancer = PromptEnhancer()
        assert enhancer.should_strip_chinese("pika-2.0") is True

    def test_auto_detect_veo_strips(self):
        enhancer = PromptEnhancer()
        assert enhancer.should_strip_chinese("veo-3") is True

    def test_auto_detect_unknown_model_keeps(self):
        enhancer = PromptEnhancer()
        assert enhancer.should_strip_chinese("some-new-model") is False


# ---------------------------------------------------------------------------
# Tests: shot scale and camera movement labels
# ---------------------------------------------------------------------------

class TestShotAndCameraLabels:
    def test_shot_scale_mapped(self):
        enhancer = PromptEnhancer(strip_chinese=False)
        scene = _make_scene(shot_scale=ShotScale.CLOSE_UP)
        result = enhancer.enhance_scene_prompt(scene, _make_series())
        assert "close-up shot" in result

    def test_camera_movement_mapped(self):
        enhancer = PromptEnhancer(strip_chinese=False)
        scene = _make_scene(camera_movement="tracking")
        result = enhancer.enhance_scene_prompt(scene, _make_series())
        assert "tracking shot" in result

    def test_no_shot_scale(self):
        enhancer = PromptEnhancer(strip_chinese=False)
        scene = _make_scene(shot_scale=None, camera_movement="static")
        result = enhancer.enhance_scene_prompt(scene, _make_series())
        assert "static camera" in result
        # Should not crash or include empty labels
        assert ",," not in result

    def test_all_shot_scales(self):
        enhancer = PromptEnhancer(strip_chinese=False)
        for scale in ShotScale:
            scene = _make_scene(shot_scale=scale)
            result = enhancer.enhance_scene_prompt(scene, _make_series())
            assert "shot" in result  # all labels contain "shot"


# ---------------------------------------------------------------------------
# Tests: style tag
# ---------------------------------------------------------------------------

class TestStyleTag:
    def test_style_tag_appended(self):
        enhancer = PromptEnhancer(strip_chinese=False)
        result = enhancer.enhance_scene_prompt(_make_scene(), _make_series())
        assert "Style: cinematic, vertical 9:16, cinematic lighting." in result

    def test_custom_style(self):
        enhancer = PromptEnhancer(strip_chinese=False)
        series = _make_series(style="anime")
        result = enhancer.enhance_scene_prompt(_make_scene(), series)
        assert "Style: anime" in result


# ---------------------------------------------------------------------------
# Tests: enhance_all_scenes
# ---------------------------------------------------------------------------

class TestEnhanceAllScenes:
    def test_mutates_all_scenes(self):
        series = _make_series()
        episode = Episode(
            number=1,
            title="First Meeting",
            scenes=[
                _make_scene(scene_id="ep01_s01"),
                _make_scene(scene_id="ep01_s02", visual_prompt="forest path scene"),
            ],
        )
        enhancer = PromptEnhancer(strip_chinese=False)
        result = enhancer.enhance_all_scenes(episode, series)

        assert result is episode  # mutates in place
        for scene in episode.scenes:
            assert "Character: Lin Yue" in scene.visual_prompt
            assert "cinematic lighting" in scene.visual_prompt

    def test_returns_episode(self):
        enhancer = PromptEnhancer()
        episode = Episode(number=1, scenes=[_make_scene()])
        result = enhancer.enhance_all_scenes(episode, _make_series())
        assert isinstance(result, Episode)
