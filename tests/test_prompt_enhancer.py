"""Tests for PromptEnhancer — visual prompt enrichment."""

import re

from videoclaw.drama.models import (
    Character,
    DramaScene,
    DramaSeries,
    Episode,
    ShotScale,
)
from videoclaw.drama.prompt_enhancer import (
    PromptEnhancer,
    _MAX_ENGLISH_WORDS,
    _MAX_CHINESE_CHARS,
    _CJK_CHAR_RE,
    _to_ref_key,
)


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
        assert "Style: cinematic" in result


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
        assert "Style: cinematic, vertical 9:16" in result

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
            # enhanced_visual_prompt should contain enriched content
            assert "Lin Yue" in scene.enhanced_visual_prompt
            assert "Style: cinematic" in scene.enhanced_visual_prompt
        # original visual_prompt must be preserved (not overwritten)
        assert episode.scenes[0].visual_prompt == "两人在古桥上对视，微风吹过"
        assert episode.scenes[1].visual_prompt == "forest path scene"

    def test_effective_prompt_returns_enhanced_when_available(self):
        series = _make_series()
        episode = Episode(number=1, scenes=[_make_scene()])
        enhancer = PromptEnhancer(strip_chinese=False)
        enhancer.enhance_all_scenes(episode, series)

        scene = episode.scenes[0]
        assert scene.effective_prompt == scene.enhanced_visual_prompt
        assert scene.effective_prompt != scene.visual_prompt

    def test_effective_prompt_falls_back_to_original(self):
        scene = _make_scene()
        assert scene.enhanced_visual_prompt == ""
        assert scene.effective_prompt == scene.visual_prompt

    def test_returns_episode(self):
        enhancer = PromptEnhancer()
        episode = Episode(number=1, scenes=[_make_scene()])
        result = enhancer.enhance_all_scenes(episode, _make_series())
        assert isinstance(result, Episode)


# ---------------------------------------------------------------------------
# Helpers for Western drama tests
# ---------------------------------------------------------------------------

def _make_western_series(**kw) -> DramaSeries:
    defaults = dict(
        title="Satan in a Suit",
        genre="thriller",
        style="cinematic",
        aspect_ratio="9:16",
        model_id="seedance-2.0",
        language="en",
        characters=[
            Character(
                name="Ivy Angel",
                description="26-year-old ambitious young woman",
                visual_prompt="young Caucasian woman, 26, long auburn hair, green eyes, fitted black dress",
            ),
            Character(
                name="Colton Black",
                description="35-year-old mysterious businessman",
                visual_prompt="tall Caucasian man, 35, dark slicked-back hair, sharp jaw, tailored charcoal suit",
            ),
        ],
    )
    defaults.update(kw)
    return DramaSeries(**defaults)


def _make_western_scene(**kw) -> DramaScene:
    defaults = dict(
        scene_id="ep01_s05",
        description="Ivy confronts Colton by the poolside at night.",
        visual_prompt="Ivy stands by the illuminated poolside at night, confronting Colton under dim amber lights",
        camera_movement="dolly_in",
        shot_scale=ShotScale.MEDIUM,
        characters_present=["Ivy Angel", "Colton Black"],
    )
    defaults.update(kw)
    return DramaScene(**defaults)


# ---------------------------------------------------------------------------
# Tests: _to_ref_key helper
# ---------------------------------------------------------------------------

class TestToRefKey:
    def test_simple_name(self):
        assert _to_ref_key("Ivy Angel") == "ivy_angel"

    def test_already_key_format(self):
        assert _to_ref_key("poolside_night") == "poolside_night"

    def test_colton_black(self):
        assert _to_ref_key("Colton Black") == "colton_black"

    def test_special_characters(self):
        assert _to_ref_key("  Hello--World!!  ") == "hello_world"

    def test_consecutive_underscores_collapsed(self):
        assert _to_ref_key("a   b") == "a_b"


# ---------------------------------------------------------------------------
# Tests: [ref:key] markers
# ---------------------------------------------------------------------------

class TestRefMarkers:
    def test_character_ref_marker_western(self):
        """Western drama with available character ref inserts [ref:key] in output."""
        enhancer = PromptEnhancer()
        scene = _make_western_scene()
        series = _make_western_series()
        refs = {
            "characters": {
                "Ivy Angel": "https://cdn.example.com/ivy.jpg",
                "Colton Black": "https://cdn.example.com/colton.jpg",
            },
        }
        result = enhancer.enhance_scene_prompt(scene, series, available_refs=refs)
        assert "[ref:ivy_angel]" in result
        assert "[ref:colton_black]" in result

    def test_scene_ref_marker(self):
        """Scene ref available and matching visual_prompt text inserts marker."""
        enhancer = PromptEnhancer()
        scene = _make_western_scene()
        series = _make_western_series()
        refs = {
            "scenes": {
                "poolside": "https://cdn.example.com/pool.jpg",
            },
        }
        result = enhancer.enhance_scene_prompt(scene, series, available_refs=refs)
        assert "[ref:poolside]" in result

    def test_no_marker_when_no_ref_available(self):
        """No available_refs means no [ref:...] markers in the output."""
        enhancer = PromptEnhancer()
        scene = _make_western_scene()
        series = _make_western_series()
        result = enhancer.enhance_scene_prompt(scene, series)
        assert "[ref:" not in result

    def test_prop_ref_marker(self):
        """Prop key matches visual_prompt text and inserts marker."""
        enhancer = PromptEnhancer()
        scene = _make_western_scene(
            visual_prompt="Ivy holds a champagne glass by the poolside, amber lights reflect on the water",
        )
        series = _make_western_series()
        refs = {
            "props": {
                "champagne glass": "https://cdn.example.com/glass.jpg",
            },
        }
        result = enhancer.enhance_scene_prompt(scene, series, available_refs=refs)
        assert "[ref:champagne_glass]" in result

    def test_backward_compat_no_available_refs(self):
        """Calling without available_refs still works (no crash, no markers)."""
        enhancer = PromptEnhancer()
        scene = _make_western_scene()
        series = _make_western_series()
        # Call without keyword argument at all
        result = enhancer.enhance_scene_prompt(scene, series)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "[ref:" not in result

    def test_enhance_all_scenes_passes_refs(self):
        """enhance_all_scenes forwards available_refs to each scene."""
        enhancer = PromptEnhancer()
        series = _make_western_series()
        episode = Episode(
            number=1,
            title="Poolside",
            scenes=[_make_western_scene()],
        )
        refs = {
            "characters": {
                "Ivy Angel": "https://cdn.example.com/ivy.jpg",
            },
        }
        enhancer.enhance_all_scenes(episode, series, available_refs=refs)
        assert "[ref:ivy_angel]" in episode.scenes[0].enhanced_visual_prompt

    def test_scene_ref_only_one_per_shot(self):
        """Even with multiple matching scene refs, only one is inserted."""
        enhancer = PromptEnhancer()
        scene = _make_western_scene(
            visual_prompt="Poolside at the rooftop lounge, amber lights flicker",
            description="Rooftop poolside scene",
        )
        series = _make_western_series()
        refs = {
            "scenes": {
                "poolside": "https://cdn.example.com/pool.jpg",
                "rooftop": "https://cdn.example.com/roof.jpg",
            },
        }
        result = enhancer.enhance_scene_prompt(scene, series, available_refs=refs)
        scene_markers = re.findall(r'\[ref:(?:poolside|rooftop)\]', result)
        # Only ONE scene ref per shot
        assert len(scene_markers) == 1

    def test_chinese_character_ref_marker(self):
        """Chinese drama with available character ref inserts [ref:key]."""
        enhancer = PromptEnhancer(strip_chinese=False)
        scene = _make_scene()  # Chinese drama scene
        series = _make_series(model_id="minimax-hailuo-2.3")
        refs = {
            "characters": {
                "Lin Yue": "https://cdn.example.com/lin_yue.jpg",
            },
        }
        result = enhancer.enhance_scene_prompt(scene, series, available_refs=refs)
        assert "[ref:lin_yue]" in result


# ---------------------------------------------------------------------------
# Tests: text length enforcement
# ---------------------------------------------------------------------------

class TestTextLengthEnforcement:
    def test_english_prompt_within_limit(self):
        """Normal English prompt stays within 1000 words."""
        enhancer = PromptEnhancer()
        scene = _make_western_scene()
        series = _make_western_series()
        result = enhancer.enhance_scene_prompt(scene, series)
        # Count words (excluding ref markers)
        cleaned = re.sub(r'\[ref:[a-zA-Z0-9_]+\]', '', result)
        word_count = len(cleaned.split())
        assert word_count <= _MAX_ENGLISH_WORDS

    def test_chinese_prompt_within_limit(self):
        """Long Chinese prompt is truncated to at most 500 CJK characters."""
        enhancer = PromptEnhancer(strip_chinese=False)
        # Build a visual_prompt with 600+ CJK characters
        long_cjk = "这是一个很长的场景描述。" * 80  # ~800 CJK chars
        scene = _make_scene(visual_prompt=long_cjk)
        series = _make_series(model_id="minimax-hailuo-2.3")
        result = enhancer.enhance_scene_prompt(scene, series)
        cjk_count = len(_CJK_CHAR_RE.findall(result))
        assert cjk_count <= _MAX_CHINESE_CHARS

    def test_long_english_prompt_truncated(self):
        """An 800-word visual prompt + enhancer overhead stays within 1000 words."""
        enhancer = PromptEnhancer()
        # Generate an ~800 word visual prompt
        words = ["dramatic", "tension", "fills", "the", "dimly", "lit", "room",
                 "as", "shadows", "dance"]
        long_vp = " ".join(words * 80)  # ~800 words
        scene = _make_western_scene(visual_prompt=long_vp)
        series = _make_western_series()
        result = enhancer.enhance_scene_prompt(scene, series)
        cleaned = re.sub(r'\[ref:[a-zA-Z0-9_]+\]', '', result)
        word_count = len(cleaned.split())
        assert word_count <= _MAX_ENGLISH_WORDS

    def test_enforce_preserves_ref_markers(self):
        """Even after truncation, ref markers are preserved."""
        enhancer = PromptEnhancer()
        words = ["dramatic", "tension", "fills", "the", "dimly", "lit", "room",
                 "as", "shadows", "dance"]
        long_vp = " ".join(words * 80)
        scene = _make_western_scene(visual_prompt=long_vp)
        series = _make_western_series()
        refs = {
            "characters": {
                "Ivy Angel": "https://cdn.example.com/ivy.jpg",
            },
        }
        result = enhancer.enhance_scene_prompt(scene, series, available_refs=refs)
        # Ref marker should survive even with truncation
        assert "[ref:ivy_angel]" in result
        # And total word count still within limit
        cleaned = re.sub(r'\[ref:[a-zA-Z0-9_]+\]', '', result)
        assert len(cleaned.split()) <= _MAX_ENGLISH_WORDS

    def test_enforce_preserves_style_and_constraints(self):
        """Truncation preserves Style and Constraints sections."""
        enhancer = PromptEnhancer()
        words = ["dramatic", "scene", "with", "intense", "visual", "detail",
                 "and", "cinematic", "mood", "lighting"]
        long_vp = " ".join(words * 80)
        scene = _make_western_scene(visual_prompt=long_vp)
        series = _make_western_series()
        result = enhancer.enhance_scene_prompt(scene, series)
        assert "Style: cinematic" in result
        assert "Constraints:" in result
