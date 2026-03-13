"""Tests for drama runner (runner.py)."""

from videoclaw.drama.models import (
    Character,
    DramaScene,
    DramaSeries,
    Episode,
    ShotScale,
    ShotType,
    VoiceProfile,
)
from videoclaw.drama.runner import build_episode_dag


# ---------------------------------------------------------------------------
# DAG structure
# ---------------------------------------------------------------------------


def test_build_episode_dag():
    """build_episode_dag should produce a valid DAG from episode DramaScene objects."""
    series = DramaSeries(
        title="Test",
        model_id="mock",
    )
    ep = Episode(
        number=1,
        title="Pilot",
        scenes=[
            DramaScene(scene_id="ep01_s01", visual_prompt="shot 1", duration_seconds=5.0),
            DramaScene(scene_id="ep01_s02", visual_prompt="shot 2", duration_seconds=5.0),
        ],
    )

    dag, state = build_episode_dag(ep, series)

    # script_gen, storyboard, 2x video, 2x per_scene_tts, subtitle_gen, music, compose, render = 10
    assert len(dag.nodes) == 10
    assert len(state.storyboard) == 2
    assert state.storyboard[0].shot_id == "ep01_s01"
    assert state.storyboard[1].model_id == "mock"
    assert ep.project_id == state.project_id


# ---------------------------------------------------------------------------
# Shot ID ↔ scene_id correspondence
# ---------------------------------------------------------------------------


def test_shot_ids_match_scene_ids():
    """Each Shot.shot_id in the storyboard should correspond to its DramaScene.scene_id."""
    series = DramaSeries(title="ID Test", model_id="mock")
    scenes = [
        DramaScene(scene_id="ep02_s01", visual_prompt="A", duration_seconds=3.0),
        DramaScene(scene_id="ep02_s02", visual_prompt="B", duration_seconds=4.0),
        DramaScene(scene_id="ep02_s03", visual_prompt="C", duration_seconds=5.0),
    ]
    ep = Episode(number=2, title="Shot ID Test", scenes=scenes)

    _, state = build_episode_dag(ep, series)

    assert len(state.storyboard) == 3
    for shot, scene in zip(state.storyboard, scenes):
        assert shot.shot_id == scene.scene_id
        assert shot.prompt == scene.visual_prompt
        assert shot.duration_seconds == scene.duration_seconds


def test_build_episode_dag_auto_generates_scene_ids():
    """When scenes have empty scene_id, build_episode_dag should auto-generate IDs."""
    series = DramaSeries(title="Auto ID", model_id="mock")
    ep = Episode(
        number=3,
        title="Auto",
        scenes=[
            DramaScene(scene_id="", visual_prompt="shot 1", duration_seconds=5.0),
            DramaScene(scene_id="", visual_prompt="shot 2", duration_seconds=5.0),
        ],
    )

    _, state = build_episode_dag(ep, series)

    assert state.storyboard[0].shot_id == "ep03_s01"
    assert state.storyboard[1].shot_id == "ep03_s02"


def test_build_episode_dag_metadata():
    """DAG state metadata should include series/episode identifiers."""
    series = DramaSeries(title="Meta Test", model_id="mock", series_id="test_series_123")
    ep = Episode(
        number=1,
        episode_id="test_ep_456",
        title="Metadata",
        scenes=[DramaScene(scene_id="ep01_s01", visual_prompt="test", duration_seconds=5.0)],
    )

    _, state = build_episode_dag(ep, series)

    assert state.metadata["series_id"] == "test_series_123"
    assert state.metadata["episode_id"] == "test_ep_456"
    assert state.metadata["episode_number"] == 1
    assert state.metadata["style"] == "cinematic"
    assert state.metadata["aspect_ratio"] == "9:16"


# ---------------------------------------------------------------------------
# Voice map wiring
# ---------------------------------------------------------------------------


def test_build_episode_dag_includes_voice_map():
    """build_episode_dag should store voice_map in state.metadata."""
    series = DramaSeries(
        title="Voice Test",
        model_id="mock",
        characters=[
            Character(
                name="林薇",
                description="活泼少女",
                voice_profile=VoiceProfile(
                    voice_id="Lively_Girl", role_name="林薇", speed=1.05, pitch=2,
                ),
            ),
            Character(name="萧衍", description="霸总"),  # no voice_profile
        ],
    )
    ep = Episode(
        number=1,
        scenes=[
            DramaScene(scene_id="s01", dialogue="你来了", speaking_character="林薇"),
        ],
    )

    _, state = build_episode_dag(ep, series)

    assert "voice_map" in state.metadata
    voice_map = state.metadata["voice_map"]
    assert "林薇" in voice_map
    assert voice_map["林薇"]["voice_id"] == "Lively_Girl"
    assert voice_map["林薇"]["speed"] == 1.05
    # Characters without voice_profile should not appear
    assert "萧衍" not in voice_map


def test_build_episode_dag_scenes_have_dialogue_line_type():
    """TTS node params should include dialogue_line_type per scene."""
    series = DramaSeries(title="LineType Test", model_id="mock")
    ep = Episode(
        number=1,
        scenes=[
            DramaScene(
                scene_id="s01",
                dialogue="你来了",
                speaking_character="林薇",
                dialogue_line_type="dialogue",
            ),
            DramaScene(
                scene_id="s02",
                dialogue="不可能",
                speaking_character="萧衍",
                dialogue_line_type="inner_monologue",
            ),
            DramaScene(
                scene_id="s03",
                narration="夜幕降临",
            ),
        ],
    )

    dag, _ = build_episode_dag(ep, series)

    # Per-scene TTS nodes carry individual scene data
    assert dag.nodes["tts_s01"].params["scene"]["dialogue_line_type"] == "dialogue"
    assert dag.nodes["tts_s02"].params["scene"]["dialogue_line_type"] == "inner_monologue"
    assert dag.nodes["tts_s03"].params["scene"]["dialogue_line_type"] == "dialogue"  # default
