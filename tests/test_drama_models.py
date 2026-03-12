"""Tests for drama data models (models.py)."""

import pytest

from videoclaw.drama.models import (
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
    EpisodeAudioManifest,
    EpisodeStatus,
    LineType,
    NARRATOR_PRESETS,
    ShotScale,
    ShotType,
    VoiceProfile,
    VOICE_PROFILES,
    assign_voice_profile,
)


# ---------------------------------------------------------------------------
# DramaSeries
# ---------------------------------------------------------------------------


def test_drama_series_roundtrip():
    """Serialise a DramaSeries and deserialise it back."""
    series = DramaSeries(
        title="Test Drama",
        genre="thriller",
        synopsis="A test synopsis",
        total_episodes=3,
        characters=[
            Character(name="Alice", description="The hero", visual_prompt="young woman, black hair"),
            Character(name="Bob", description="The villain", visual_prompt="tall man, scar"),
        ],
        episodes=[
            Episode(number=1, title="The Beginning", synopsis="It all starts here"),
            Episode(number=2, title="The Middle", synopsis="Things escalate"),
            Episode(number=3, title="The End", synopsis="Resolution"),
        ],
    )

    data = series.to_dict()
    restored = DramaSeries.from_dict(data)

    assert restored.title == "Test Drama"
    assert restored.genre == "thriller"
    assert len(restored.characters) == 2
    assert len(restored.episodes) == 3
    assert restored.characters[0].name == "Alice"
    assert restored.episodes[1].title == "The Middle"
    assert restored.status == DramaStatus.DRAFT


def test_drama_cost_total():
    """cost_total should sum episode costs."""
    series = DramaSeries(
        episodes=[
            Episode(number=1, cost=0.25),
            Episode(number=2, cost=0.50),
            Episode(number=3, cost=0.10),
        ]
    )
    assert series.cost_total == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# DramaManager
# ---------------------------------------------------------------------------


def test_drama_manager_crud(tmp_path):
    """Create, save, load, list, and delete a drama series."""
    mgr = DramaManager(base_dir=tmp_path)

    series = mgr.create(
        title="CRUD Drama",
        synopsis="Test CRUD",
        genre="comedy",
        total_episodes=2,
    )
    assert series.series_id

    ids = mgr.list_series()
    assert series.series_id in ids

    loaded = mgr.load(series.series_id)
    assert loaded.title == "CRUD Drama"
    assert loaded.genre == "comedy"
    assert loaded.total_episodes == 2

    loaded.episodes.append(Episode(number=1, title="Pilot"))
    mgr.save(loaded)
    reloaded = mgr.load(series.series_id)
    assert len(reloaded.episodes) == 1
    assert reloaded.episodes[0].title == "Pilot"

    mgr.delete(series.series_id)
    assert series.series_id not in mgr.list_series()


def test_drama_manager_load_nonexistent(tmp_path):
    """Loading a non-existent series raises FileNotFoundError."""
    mgr = DramaManager(base_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        mgr.load("does-not-exist")


# ---------------------------------------------------------------------------
# Episode & DramaScene
# ---------------------------------------------------------------------------


def test_episode_roundtrip():
    """Serialise and deserialise an Episode with DramaScene objects."""
    ep = Episode(
        number=1,
        title="Pilot",
        synopsis="The story begins",
        status=EpisodeStatus.GENERATING,
        scenes=[
            DramaScene(
                scene_id="ep01_s01",
                visual_prompt="A dark alley",
                duration_seconds=5.0,
                emotion="tense",
                characters_present=["Alice"],
                transition="cut",
            ),
            DramaScene(
                scene_id="ep01_s02",
                visual_prompt="A door opens",
                duration_seconds=3.0,
                shot_scale=ShotScale.CLOSE_UP,
                shot_type=ShotType.REACTION,
                speaking_character="Alice",
                emotion="shock",
                characters_present=["Alice", "Bob"],
                transition="dissolve",
            ),
        ],
        cost=0.42,
    )

    data = ep.to_dict()
    restored = Episode.from_dict(data)

    assert restored.number == 1
    assert restored.title == "Pilot"
    assert restored.status == EpisodeStatus.GENERATING
    assert len(restored.scenes) == 2
    assert restored.scenes[0].visual_prompt == "A dark alley"
    assert restored.scenes[0].emotion == "tense"
    assert restored.scenes[0].characters_present == ["Alice"]
    assert restored.scenes[0].transition == "cut"
    assert restored.scenes[1].shot_scale == ShotScale.CLOSE_UP
    assert restored.scenes[1].shot_type == ShotType.REACTION
    assert restored.scenes[1].speaking_character == "Alice"
    assert restored.scenes[1].characters_present == ["Alice", "Bob"]
    assert restored.cost == pytest.approx(0.42)


def test_drama_scene_from_dict_ignores_unknown_keys():
    """from_dict should silently drop keys not in DramaScene fields."""
    data = {
        "scene_id": "ep01_s01",
        "visual_prompt": "test",
        "unknown_field": "should be ignored",
        "another_unknown": 42,
    }
    scene = DramaScene.from_dict(data)
    assert scene.scene_id == "ep01_s01"
    assert scene.visual_prompt == "test"


def test_drama_scene_from_dict_invalid_enum_falls_back_to_none():
    """from_dict should set invalid enum values to None instead of crashing."""
    data = {
        "scene_id": "ep04_s07",
        "visual_prompt": "test",
        "shot_scale": "extreme_close_up",  # invalid — not in ShotScale enum
        "shot_type": "dolly_zoom",  # invalid — not in ShotType enum
    }
    scene = DramaScene.from_dict(data)
    assert scene.scene_id == "ep04_s07"
    assert scene.shot_scale is None
    assert scene.shot_type is None


def test_shot_enums_values():
    """ShotScale and ShotType should have exactly the spec-defined values."""
    assert set(ShotScale) == {
        ShotScale.CLOSE_UP, ShotScale.MEDIUM_CLOSE,
        ShotScale.MEDIUM, ShotScale.WIDE, ShotScale.EXTREME_WIDE,
    }
    assert set(ShotType) == {
        ShotType.ESTABLISHING, ShotType.REACTION,
        ShotType.ACTION, ShotType.DETAIL, ShotType.POV,
    }


def test_drama_scene_asset_tracking_roundtrip():
    """DramaScene asset tracking fields should serialize correctly."""
    scene = DramaScene(
        scene_id="ep01_s01",
        visual_prompt="A dark alley",
        video_asset_path="/projects/dramas/abc/ep01/ep01_s01.mp4",
        dialogue_audio_path="/projects/dramas/abc/ep01/ep01_s01_dialogue.mp3",
        narration_audio_path="/projects/dramas/abc/ep01/ep01_s01_narration.mp3",
        scene_status="completed",
    )
    data = scene.to_dict()
    assert data["video_asset_path"] == "/projects/dramas/abc/ep01/ep01_s01.mp4"
    assert data["scene_status"] == "completed"

    restored = DramaScene.from_dict(data)
    assert restored.video_asset_path == "/projects/dramas/abc/ep01/ep01_s01.mp4"
    assert restored.dialogue_audio_path == "/projects/dramas/abc/ep01/ep01_s01_dialogue.mp3"
    assert restored.narration_audio_path == "/projects/dramas/abc/ep01/ep01_s01_narration.mp3"
    assert restored.scene_status == "completed"

    empty = DramaScene()
    assert empty.video_asset_path is None
    assert empty.dialogue_audio_path is None
    assert empty.narration_audio_path is None
    assert empty.scene_status == "pending"


# ---------------------------------------------------------------------------
# AudioSegment & EpisodeAudioManifest
# ---------------------------------------------------------------------------


def test_audio_segment_roundtrip():
    """AudioSegment should serialize and deserialize correctly."""
    seg = AudioSegment(
        segment_id="seg01",
        scene_id="ep01_s01",
        audio_type=AudioType.DIALOGUE,
        text="你好世界",
        character_name="Alice",
        audio_path="/tmp/seg01.mp3",
        start_time=5.0,
        duration_seconds=3.0,
        volume=0.9,
    )
    data = seg.to_dict()
    assert data["audio_type"] == "dialogue"
    restored = AudioSegment.from_dict(data)
    assert restored.audio_type == AudioType.DIALOGUE
    assert restored.character_name == "Alice"
    assert restored.start_time == 5.0


def test_episode_audio_manifest_roundtrip():
    """EpisodeAudioManifest with segments should survive serialization."""
    manifest = EpisodeAudioManifest(
        episode_id="ep01",
        segments=[
            AudioSegment(segment_id="s1", scene_id="ep01_s01", audio_type=AudioType.DIALOGUE, text="台词", duration_seconds=3.0),
            AudioSegment(segment_id="s2", scene_id="ep01_s01", audio_type=AudioType.NARRATION, text="旁白", start_time=3.0, duration_seconds=5.0),
        ],
        total_duration=8.0,
        mixed_audio_path="/tmp/ep01_mixed.mp3",
    )
    data = manifest.to_dict()
    assert len(data["segments"]) == 2
    assert data["total_duration"] == 8.0

    restored = EpisodeAudioManifest.from_dict(data)
    assert restored.episode_id == "ep01"
    assert len(restored.segments) == 2
    assert restored.segments[0].audio_type == AudioType.DIALOGUE
    assert restored.segments[1].audio_type == AudioType.NARRATION
    assert restored.mixed_audio_path == "/tmp/ep01_mixed.mp3"


# ---------------------------------------------------------------------------
# VoiceProfile & Character voice mapping
# ---------------------------------------------------------------------------


def test_voice_profile_roundtrip():
    """VoiceProfile should serialize and deserialize correctly."""
    vp = VoiceProfile(voice_id="Determined_Man", speed=0.9, pitch=-1, emotion="neutral", volume=1.0)
    data = vp.to_dict()
    restored = VoiceProfile.from_dict(data)
    assert restored.voice_id == "Determined_Man"
    assert restored.speed == 0.9
    assert restored.pitch == -1
    assert restored.emotion == "neutral"


def test_character_with_voice_profile_roundtrip():
    """Character with VoiceProfile should survive serialization."""
    c = Character(
        name="Alice",
        description="The hero",
        voice_style="dramatic",
        voice_profile=VoiceProfile(voice_id="Determined_Man", speed=0.9, pitch=-1),
    )
    data = c.to_dict()
    restored = Character.from_dict(data)
    assert restored.name == "Alice"
    assert restored.voice_profile is not None
    assert restored.voice_profile.voice_id == "Determined_Man"
    assert restored.voice_profile.pitch == -1


def test_character_without_voice_profile_roundtrip():
    """Character without VoiceProfile should not include it in dict."""
    c = Character(name="Bob", description="The villain")
    data = c.to_dict()
    assert "voice_profile" not in data
    restored = Character.from_dict(data)
    assert restored.voice_profile is None


def test_assign_voice_profile():
    """assign_voice_profile should map voice_style to correct profile."""
    c = Character(name="Alice", voice_style="dramatic")
    assign_voice_profile(c)
    assert c.voice_profile is not None
    assert c.voice_profile.voice_id == "Determined_Man"
    assert c.voice_profile.pitch == -1


def test_assign_voice_profile_preserves_existing():
    """assign_voice_profile should not overwrite an existing profile."""
    existing = VoiceProfile(voice_id="Custom_Voice", speed=1.5)
    c = Character(name="Alice", voice_style="dramatic", voice_profile=existing)
    assign_voice_profile(c)
    assert c.voice_profile.voice_id == "Custom_Voice"
    assert c.voice_profile.speed == 1.5


def test_assign_voice_profile_unknown_style_falls_back():
    """Unknown voice_style should fall back to 'warm' profile."""
    c = Character(name="Alice", voice_style="unknown_style")
    assign_voice_profile(c)
    assert c.voice_profile is not None
    assert c.voice_profile.voice_id == "Friendly_Person"


def test_voice_profiles_all_styles_covered():
    """Every VOICE_PROFILES entry should produce a valid, distinct VoiceProfile."""
    expected_styles = {"warm", "authoritative", "playful", "dramatic", "calm"}
    assert set(VOICE_PROFILES.keys()) == expected_styles

    for style, profile in VOICE_PROFILES.items():
        c = Character(name=f"test_{style}", voice_style=style)
        assign_voice_profile(c)
        assert c.voice_profile is not None
        assert c.voice_profile.voice_id == profile.voice_id
        assert c.voice_profile.speed == profile.speed
        assert c.voice_profile.pitch == profile.pitch
        assert c.voice_profile.emotion == profile.emotion

    # Verify all voice_ids are distinct
    voice_ids = [p.voice_id for p in VOICE_PROFILES.values()]
    assert len(voice_ids) == len(set(voice_ids)), "Each style must map to a unique voice_id"


# ---------------------------------------------------------------------------
# LineType, DramaGenre, DialogueLine, NARRATOR_PRESETS, extended VoiceProfile
# ---------------------------------------------------------------------------


def test_line_type_enum():
    assert LineType.NARRATION == "narration"
    assert LineType.DIALOGUE == "dialogue"
    assert LineType.INNER_MONOLOGUE == "inner_monologue"


def test_drama_genre_enum():
    assert DramaGenre.SWEET_ROMANCE == "sweet_romance"
    assert DramaGenre.MALE_POWER_FANTASY == "male_power_fantasy"
    assert DramaGenre.SUSPENSE_THRILLER == "suspense_thriller"


def test_voice_profile_extended_fields():
    vp = VoiceProfile(voice_id="Test", role_name="林小姐", line_type=LineType.DIALOGUE, age_feel="young_adult", energy="high", description="活泼女声")
    d = vp.to_dict()
    assert d["role_name"] == "林小姐"
    assert d["line_type"] == "dialogue"
    restored = VoiceProfile.from_dict(d)
    assert restored.role_name == "林小姐"
    assert restored.line_type == LineType.DIALOGUE


def test_dialogue_line_roundtrip():
    line = DialogueLine(text="你怎么在这里？", speaker="林小姐", line_type=LineType.DIALOGUE, scene_id="ep01_s03", emotion_hint="shocked")
    d = line.to_dict()
    restored = DialogueLine.from_dict(d)
    assert restored.speaker == "林小姐"
    assert restored.line_type == LineType.DIALOGUE


def test_narrator_presets_by_genre():
    for genre in DramaGenre:
        assert genre in NARRATOR_PRESETS
        preset = NARRATOR_PRESETS[genre]
        assert preset.line_type == LineType.NARRATION
        assert preset.role_name == "narrator"


def test_audio_type_has_inner_monologue():
    assert AudioType.INNER_MONOLOGUE == "inner_monologue"


def test_audio_segment_with_line_type():
    seg = AudioSegment(text="他不可能知道", character_name="萧衍", audio_type=AudioType.INNER_MONOLOGUE, line_type=LineType.INNER_MONOLOGUE)
    d = seg.to_dict()
    assert d["line_type"] == "inner_monologue"
    restored = AudioSegment.from_dict(d)
    assert restored.line_type == LineType.INNER_MONOLOGUE
