"""Tests for videoclaw.drama.prompt_segments."""

from __future__ import annotations

import pytest

from videoclaw.drama.models import ShotScale
from videoclaw.drama.prompt_segments import (
    MAX_REFERENCE_IMAGES,
    ContentBuilder,
    PromptSegment,
    PromptSegmenter,
    ReferenceMedia,
    allocate_reference_slots,
)


# ---------------------------------------------------------------------------
# ReferenceMedia construction
# ---------------------------------------------------------------------------


class TestReferenceMedia:
    def test_character_with_url(self):
        ref = ReferenceMedia(ref_type="character", key="Alice", url="https://cdn.example.com/alice.jpg")
        assert ref.ref_type == "character"
        assert ref.key == "Alice"
        assert ref.url == "https://cdn.example.com/alice.jpg"
        assert ref.path is None

    def test_scene_with_path(self):
        ref = ReferenceMedia(ref_type="scene", key="poolside", path="/assets/pool.png")
        assert ref.ref_type == "scene"
        assert ref.key == "poolside"
        assert ref.url is None
        assert ref.path == "/assets/pool.png"

    def test_prop_with_both(self):
        ref = ReferenceMedia(ref_type="prop", key="knife", url="https://example.com/knife.png", path="/tmp/knife.png")
        assert ref.ref_type == "prop"
        assert ref.url == "https://example.com/knife.png"
        assert ref.path == "/tmp/knife.png"

    def test_defaults_are_none(self):
        ref = ReferenceMedia(ref_type="character", key="Bob")
        assert ref.url is None
        assert ref.path is None

    def test_slots_no_dict(self):
        ref = ReferenceMedia(ref_type="character", key="X")
        assert not hasattr(ref, "__dict__")


# ---------------------------------------------------------------------------
# PromptSegment construction
# ---------------------------------------------------------------------------


class TestPromptSegment:
    def test_text_only(self):
        seg = PromptSegment(text="A beautiful sunset.")
        assert seg.text == "A beautiful sunset."
        assert seg.reference is None

    def test_with_reference(self):
        ref = ReferenceMedia(ref_type="character", key="Hero", url="https://example.com/hero.jpg")
        seg = PromptSegment(text="Hero enters the room.", reference=ref)
        assert seg.reference is ref

    def test_slots_no_dict(self):
        seg = PromptSegment(text="hi")
        assert not hasattr(seg, "__dict__")


# ---------------------------------------------------------------------------
# allocate_reference_slots
# ---------------------------------------------------------------------------


class TestAllocateReferenceSlots:
    _chars = {
        "Alice": "https://cdn.example.com/alice.jpg",
        "Bob": "https://cdn.example.com/bob.jpg",
        "Carol": "https://cdn.example.com/carol.jpg",
        "Dave": "https://cdn.example.com/dave.jpg",
        "Eve": "https://cdn.example.com/eve.jpg",
        "Frank": "https://cdn.example.com/frank.jpg",
        "Grace": "https://cdn.example.com/grace.jpg",
    }
    _scenes = {
        "poolside": "https://cdn.example.com/pool.jpg",
        "lobby": "https://cdn.example.com/lobby.jpg",
        "office": "https://cdn.example.com/office.jpg",
        "rooftop": "https://cdn.example.com/roof.jpg",
        "garden": "https://cdn.example.com/garden.jpg",
    }
    _props = {
        "knife": "https://cdn.example.com/knife.jpg",
        "phone": "https://cdn.example.com/phone.jpg",
        "briefcase": "https://cdn.example.com/briefcase.jpg",
    }

    def _avail(self, chars=None, scenes=None, props=None):
        return {
            "characters": chars if chars is not None else self._chars,
            "scenes": scenes if scenes is not None else self._scenes,
            "props": props if props is not None else self._props,
        }

    def test_never_exceeds_9(self):
        result = allocate_reference_slots(None, self._avail())
        assert len(result) <= MAX_REFERENCE_IMAGES

    def test_close_up_char_heavy(self):
        result = allocate_reference_slots(ShotScale.CLOSE_UP, self._avail())
        chars = [r for r in result if r.ref_type == "character"]
        scenes = [r for r in result if r.ref_type == "scene"]
        props = [r for r in result if r.ref_type == "prop"]
        assert len(chars) == 6
        assert len(scenes) == 2
        assert len(props) == 1
        assert len(result) == 9

    def test_wide_scene_heavy(self):
        result = allocate_reference_slots(ShotScale.WIDE, self._avail())
        chars = [r for r in result if r.ref_type == "character"]
        scenes = [r for r in result if r.ref_type == "scene"]
        props = [r for r in result if r.ref_type == "prop"]
        assert len(chars) == 3
        assert len(scenes) == 4
        assert len(props) == 2
        assert len(result) == 9

    def test_extreme_wide(self):
        result = allocate_reference_slots(ShotScale.EXTREME_WIDE, self._avail())
        chars = [r for r in result if r.ref_type == "character"]
        scenes = [r for r in result if r.ref_type == "scene"]
        assert len(chars) == 2
        assert len(scenes) == 5

    def test_medium_close(self):
        result = allocate_reference_slots(ShotScale.MEDIUM_CLOSE, self._avail())
        chars = [r for r in result if r.ref_type == "character"]
        scenes = [r for r in result if r.ref_type == "scene"]
        props = [r for r in result if r.ref_type == "prop"]
        assert len(chars) == 5
        assert len(scenes) == 3
        assert len(props) == 1

    def test_medium(self):
        result = allocate_reference_slots(ShotScale.MEDIUM, self._avail())
        chars = [r for r in result if r.ref_type == "character"]
        scenes = [r for r in result if r.ref_type == "scene"]
        props = [r for r in result if r.ref_type == "prop"]
        assert len(chars) == 4
        assert len(scenes) == 3
        assert len(props) == 2

    def test_none_scale_uses_default(self):
        result = allocate_reference_slots(None, self._avail())
        chars = [r for r in result if r.ref_type == "character"]
        scenes = [r for r in result if r.ref_type == "scene"]
        props = [r for r in result if r.ref_type == "prop"]
        assert len(chars) == 4
        assert len(scenes) == 3
        assert len(props) == 2

    def test_speaking_character_first(self):
        result = allocate_reference_slots(
            ShotScale.MEDIUM, self._avail(), speaking_character="Grace"
        )
        chars = [r for r in result if r.ref_type == "character"]
        assert chars[0].key == "Grace"

    def test_speaking_character_not_in_available_ignored(self):
        result = allocate_reference_slots(
            ShotScale.MEDIUM, self._avail(), speaking_character="Unknown"
        )
        # Should still return a valid result without crashing
        assert len(result) <= MAX_REFERENCE_IMAGES

    def test_empty_available(self):
        result = allocate_reference_slots(ShotScale.MEDIUM, {})
        assert result == []

    def test_partial_available_respects_budget(self):
        # Only 2 characters available but budget is 4
        available = {
            "characters": {"A": "https://a.com/a.jpg", "B": "https://b.com/b.jpg"},
            "scenes": {},
            "props": {},
        }
        result = allocate_reference_slots(ShotScale.MEDIUM, available)
        chars = [r for r in result if r.ref_type == "character"]
        assert len(chars) == 2  # capped by actual availability

    def test_url_detection(self):
        available = {
            "characters": {"hero": "https://cdn.example.com/hero.jpg"},
            "scenes": {"cave": "/local/cave.png"},
            "props": {},
        }
        result = allocate_reference_slots(ShotScale.MEDIUM, available)
        char_ref = next(r for r in result if r.key == "hero")
        scene_ref = next(r for r in result if r.key == "cave")
        assert char_ref.url == "https://cdn.example.com/hero.jpg"
        assert char_ref.path is None
        assert scene_ref.path == "/local/cave.png"
        assert scene_ref.url is None

    def test_speaking_character_not_duplicated(self):
        result = allocate_reference_slots(
            ShotScale.MEDIUM, self._avail(), speaking_character="Alice"
        )
        chars = [r for r in result if r.ref_type == "character"]
        alice_entries = [r for r in chars if r.key == "Alice"]
        assert len(alice_entries) == 1


# ---------------------------------------------------------------------------
# PromptSegmenter.parse
# ---------------------------------------------------------------------------


class TestPromptSegmenterParse:
    def _ref(self, key: str) -> ReferenceMedia:
        return ReferenceMedia(ref_type="character", key=key, url=f"https://example.com/{key}.jpg")

    def test_no_markers_single_segment(self):
        result = PromptSegmenter.parse("A hero walks into the bar.", {})
        assert len(result) == 1
        assert result[0].text == "A hero walks into the bar."
        assert result[0].reference is None

    def test_single_marker(self):
        ref_map = {"alice": self._ref("alice")}
        result = PromptSegmenter.parse("Alice enters the room [ref:alice] looking confident.", ref_map)
        assert len(result) == 2
        assert result[0].reference is ref_map["alice"]
        assert result[1].text == "looking confident."
        assert result[1].reference is None

    def test_multiple_markers(self):
        ref_map = {"alice": self._ref("alice"), "bob": self._ref("bob")}
        text = "Scene starts [ref:alice] Alice speaks [ref:bob] Bob replies."
        result = PromptSegmenter.parse(text, ref_map)
        assert len(result) == 3
        assert result[0].reference is ref_map["alice"]
        assert result[1].reference is ref_map["bob"]
        assert result[2].reference is None
        assert result[2].text == "Bob replies."

    def test_unknown_marker_stripped(self):
        ref_map = {}  # no known keys
        text = "Alice enters [ref:unknown_key] looking confident."
        result = PromptSegmenter.parse(text, ref_map)
        # All text merged into segments without reference
        combined = " ".join(s.text for s in result if s.text)
        assert "unknown_key" not in combined
        assert "[ref:" not in combined
        assert "Alice enters" in combined
        assert "looking confident" in combined

    def test_unknown_marker_not_in_output_keys(self):
        ref_map = {}
        text = "[ref:ghost] Some text."
        result = PromptSegmenter.parse(text, ref_map)
        for seg in result:
            assert seg.reference is None

    def test_trailing_text_becomes_plain_segment(self):
        ref_map = {"hero": self._ref("hero")}
        text = "The hero [ref:hero] stands tall in the evening light."
        result = PromptSegmenter.parse(text, ref_map)
        last = result[-1]
        assert last.reference is None
        assert "stands tall" in last.text

    def test_marker_only_no_surrounding_text(self):
        ref_map = {"hero": self._ref("hero")}
        result = PromptSegmenter.parse("[ref:hero]", ref_map)
        refs = [s for s in result if s.reference is not None]
        assert len(refs) == 1

    def test_empty_text(self):
        result = PromptSegmenter.parse("", {})
        assert len(result) == 1
        assert result[0].text == ""

    def test_mixed_known_unknown_markers(self):
        ref_map = {"alice": self._ref("alice")}
        text = "Text [ref:alice] known [ref:nobody] unknown end"
        result = PromptSegmenter.parse(text, ref_map)
        # alice known → segment, nobody unknown → stripped
        known_refs = [s for s in result if s.reference is not None]
        assert len(known_refs) == 1
        assert known_refs[0].reference.key == "alice"


# ---------------------------------------------------------------------------
# PromptSegmenter.strip_markers
# ---------------------------------------------------------------------------


class TestPromptSegmenterStripMarkers:
    def test_removes_single_marker(self):
        assert PromptSegmenter.strip_markers("Hello [ref:alice] world") == "Hello world"

    def test_removes_multiple_markers(self):
        result = PromptSegmenter.strip_markers("[ref:alice] Hi [ref:bob] bye [ref:carol]")
        assert "[ref:" not in result
        assert "Hi" in result
        assert "bye" in result

    def test_no_markers_unchanged(self):
        text = "No markers here."
        assert PromptSegmenter.strip_markers(text) == text

    def test_collapses_spaces(self):
        result = PromptSegmenter.strip_markers("A  [ref:x]  B")
        assert "  " not in result
        assert result == "A B"

    def test_strips_leading_trailing_spaces(self):
        result = PromptSegmenter.strip_markers("  [ref:x] text  ")
        assert result == result.strip()


# ---------------------------------------------------------------------------
# ContentBuilder.build
# ---------------------------------------------------------------------------


class TestContentBuilderBuild:
    def test_text_only_segments(self):
        segs = [PromptSegment(text="Hello world.")]
        result = ContentBuilder.build(segs)
        assert result == [{"type": "text", "text": "Hello world."}]

    def test_interleaved_text_and_image(self):
        ref = ReferenceMedia(ref_type="character", key="alice", url="https://cdn.example.com/alice.jpg")
        segs = [
            PromptSegment(text="Alice enters.", reference=ref),
            PromptSegment(text="She smiles."),
        ]
        result = ContentBuilder.build(segs)
        assert result[0] == {"type": "text", "text": "Alice enters."}
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "https://cdn.example.com/alice.jpg"
        assert result[1]["role"] == "reference_image"
        assert result[2] == {"type": "text", "text": "She smiles."}

    def test_path_only_ref_not_included(self):
        ref = ReferenceMedia(ref_type="scene", key="cave", path="/local/cave.png")
        segs = [PromptSegment(text="In the cave.", reference=ref)]
        result = ContentBuilder.build(segs)
        types = [e["type"] for e in result]
        assert "image_url" not in types

    def test_max_9_images_enforced(self):
        refs = [
            ReferenceMedia(ref_type="character", key=f"c{i}", url=f"https://example.com/{i}.jpg")
            for i in range(12)
        ]
        segs = [PromptSegment(text=f"Seg {i}", reference=refs[i]) for i in range(12)]
        result = ContentBuilder.build(segs)
        image_entries = [e for e in result if e["type"] == "image_url"]
        assert len(image_entries) == MAX_REFERENCE_IMAGES

    def test_empty_text_skipped(self):
        ref = ReferenceMedia(ref_type="character", key="hero", url="https://example.com/hero.jpg")
        segs = [PromptSegment(text="", reference=ref)]
        result = ContentBuilder.build(segs)
        # Empty text shouldn't generate a text entry
        text_entries = [e for e in result if e["type"] == "text"]
        assert len(text_entries) == 0
        image_entries = [e for e in result if e["type"] == "image_url"]
        assert len(image_entries) == 1

    def test_empty_segments_list(self):
        result = ContentBuilder.build([])
        assert result == []

    def test_multiple_text_only_segments(self):
        segs = [PromptSegment(text="First."), PromptSegment(text="Second.")]
        result = ContentBuilder.build(segs)
        assert len(result) == 2
        assert all(e["type"] == "text" for e in result)

    def test_no_ref_segments_produce_no_images(self):
        segs = [PromptSegment(text=f"Seg {i}.") for i in range(5)]
        result = ContentBuilder.build(segs)
        image_entries = [e for e in result if e["type"] == "image_url"]
        assert image_entries == []


# ---------------------------------------------------------------------------
# ContentBuilder.collect_path_refs
# ---------------------------------------------------------------------------


class TestContentBuilderCollectPathRefs:
    def test_path_only_returned(self):
        ref = ReferenceMedia(ref_type="scene", key="cave", path="/local/cave.png")
        segs = [PromptSegment(text="In the cave.", reference=ref)]
        path_refs = ContentBuilder.collect_path_refs(segs)
        assert len(path_refs) == 1
        assert path_refs[0] is ref

    def test_url_ref_not_returned(self):
        ref = ReferenceMedia(ref_type="character", key="alice", url="https://example.com/alice.jpg")
        segs = [PromptSegment(text="Alice.", reference=ref)]
        path_refs = ContentBuilder.collect_path_refs(segs)
        assert path_refs == []

    def test_no_ref_segment_not_returned(self):
        segs = [PromptSegment(text="Plain text.")]
        assert ContentBuilder.collect_path_refs(segs) == []

    def test_mixed_refs_only_path_returned(self):
        url_ref = ReferenceMedia(ref_type="character", key="alice", url="https://example.com/alice.jpg")
        path_ref = ReferenceMedia(ref_type="scene", key="cave", path="/local/cave.png")
        both_ref = ReferenceMedia(ref_type="prop", key="knife", url="https://example.com/k.jpg", path="/local/k.png")
        segs = [
            PromptSegment(text="A", reference=url_ref),
            PromptSegment(text="B", reference=path_ref),
            PromptSegment(text="C", reference=both_ref),
        ]
        path_refs = ContentBuilder.collect_path_refs(segs)
        # Only the path-only ref (no URL) should be returned
        assert len(path_refs) == 1
        assert path_refs[0] is path_ref

    def test_empty_list(self):
        assert ContentBuilder.collect_path_refs([]) == []


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """End-to-end: PromptEnhancer → Segmenter → ContentBuilder."""

    def test_enhance_parse_build(self):
        """Full flow: enhance with markers → parse → build content array."""
        from videoclaw.drama.models import Character, DramaScene, DramaSeries, ShotScale
        from videoclaw.drama.prompt_enhancer import PromptEnhancer, _to_ref_key

        series = DramaSeries(
            title="Integration Test",
            model_id="seedance-2.0",
            language="en",
            style="cinematic",
            aspect_ratio="9:16",
            characters=[
                Character(name="Ivy Angel", visual_prompt="blonde hair, server uniform, green eyes"),
            ],
        )
        scene = DramaScene(
            scene_id="ep01_s01",
            visual_prompt="Poolside at night, blue tiles, string lights overhead.",
            shot_scale=ShotScale.CLOSE_UP,
            camera_movement="dolly_in",
            characters_present=["Ivy Angel"],
            speaking_character="Ivy Angel",
            duration_seconds=5.0,
        )

        available_refs = {
            "characters": {"Ivy Angel": "https://x.com/ivy.png"},
            "scenes": {"poolside_night": "https://x.com/pool.png"},
            "props": {},
        }

        # Step 1: Enhance with ref markers
        enhancer = PromptEnhancer(strip_chinese=True)
        enhanced = enhancer.enhance_scene_prompt(scene, series, available_refs=available_refs)
        assert "[ref:" in enhanced

        # Step 2: Allocate slots + build ref map
        allocated = allocate_reference_slots(
            ShotScale.CLOSE_UP,
            available_refs,
            speaking_character="Ivy Angel",
        )
        ref_map = {_to_ref_key(r.key): r for r in allocated}

        # Step 3: Parse into segments
        segments = PromptSegmenter.parse(enhanced, ref_map)
        assert len(segments) >= 2  # at least character desc + scene/style

        # Step 4: Build Seedance content array
        content = ContentBuilder.build(segments)
        text_entries = [c for c in content if c["type"] == "text"]
        image_entries = [c for c in content if c["type"] == "image_url"]
        assert len(text_entries) >= 1
        assert len(image_entries) >= 1
        # Verify images are interleaved (not all at the end)
        first_image_idx = next(i for i, c in enumerate(content) if c["type"] == "image_url")
        last_text_idx = max(i for i, c in enumerate(content) if c["type"] == "text")
        assert first_image_idx < last_text_idx

    def test_text_length_respected_in_content(self):
        """Content text entries should not exceed word limits after enhancement."""
        import re as _re
        from videoclaw.drama.models import Character, DramaScene, DramaSeries
        from videoclaw.drama.prompt_enhancer import PromptEnhancer

        series = DramaSeries(
            title="Length Test",
            model_id="seedance-2.0",
            language="en",
            style="cinematic",
            aspect_ratio="9:16",
            characters=[
                Character(name="Test", visual_prompt="a person"),
            ],
        )
        long_desc = "The scene shows a beautiful landscape with many details. " * 100
        scene = DramaScene(
            scene_id="ep01_s01",
            visual_prompt=long_desc,
            characters_present=["Test"],
            duration_seconds=5.0,
        )

        enhancer = PromptEnhancer(strip_chinese=True)
        enhanced = enhancer.enhance_scene_prompt(scene, series)
        segments = PromptSegmenter.parse(enhanced, {})
        content = ContentBuilder.build(segments)

        total_words = sum(
            len(c["text"].split())
            for c in content
            if c["type"] == "text"
        )
        assert total_words <= 1000

    def test_no_refs_produces_text_only_content(self):
        """When no refs available, output is text-only content array."""
        from videoclaw.drama.models import Character, DramaScene, DramaSeries
        from videoclaw.drama.prompt_enhancer import PromptEnhancer

        series = DramaSeries(
            title="No Ref Test",
            model_id="seedance-2.0",
            language="en",
            style="cinematic",
            aspect_ratio="9:16",
            characters=[
                Character(name="Test", visual_prompt="a person"),
            ],
        )
        scene = DramaScene(
            scene_id="ep01_s01",
            visual_prompt="A simple scene.",
            characters_present=["Test"],
            duration_seconds=5.0,
        )

        enhancer = PromptEnhancer(strip_chinese=True)
        enhanced = enhancer.enhance_scene_prompt(scene, series)
        # No ref markers
        assert "[ref:" not in enhanced

        segments = PromptSegmenter.parse(enhanced, {})
        content = ContentBuilder.build(segments)
        # Should be text-only
        image_entries = [c for c in content if c["type"] == "image_url"]
        assert len(image_entries) == 0
        text_entries = [c for c in content if c["type"] == "text"]
        assert len(text_entries) >= 1
