"""Tests for drama runner (runner.py)."""

from videoclaw.drama.models import (
    DramaScene,
    DramaSeries,
    Episode,
    ShotScale,
    ShotType,
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

    # Should have: script_gen, storyboard, 2x video, tts, music, compose, render = 8 nodes
    assert len(dag.nodes) == 8
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
