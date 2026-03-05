"""Tests for the drama orchestration module."""

import json
import pytest

from videoclaw.drama.models import (
    Character,
    DramaManager,
    DramaSeries,
    DramaStatus,
    Episode,
    EpisodeStatus,
)


# ---------------------------------------------------------------------------
# DramaSeries model tests
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
# DramaManager persistence tests
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

    # List
    ids = mgr.list_series()
    assert series.series_id in ids

    # Load
    loaded = mgr.load(series.series_id)
    assert loaded.title == "CRUD Drama"
    assert loaded.genre == "comedy"
    assert loaded.total_episodes == 2

    # Update and save
    loaded.episodes.append(Episode(number=1, title="Pilot"))
    mgr.save(loaded)
    reloaded = mgr.load(series.series_id)
    assert len(reloaded.episodes) == 1
    assert reloaded.episodes[0].title == "Pilot"

    # Delete
    mgr.delete(series.series_id)
    assert series.series_id not in mgr.list_series()


def test_drama_manager_load_nonexistent(tmp_path):
    """Loading a non-existent series raises FileNotFoundError."""
    mgr = DramaManager(base_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        mgr.load("does-not-exist")


# ---------------------------------------------------------------------------
# Episode model tests
# ---------------------------------------------------------------------------


def test_episode_roundtrip():
    """Serialise and deserialise an Episode with scene prompts."""
    ep = Episode(
        number=1,
        title="Pilot",
        synopsis="The story begins",
        status=EpisodeStatus.GENERATING,
        scene_prompts=[
            {"scene_id": "ep01_s01", "visual_prompt": "A dark alley", "duration_seconds": 5.0},
            {"scene_id": "ep01_s02", "visual_prompt": "A door opens", "duration_seconds": 3.0},
        ],
        cost=0.42,
    )

    data = ep.to_dict()
    restored = Episode.from_dict(data)

    assert restored.number == 1
    assert restored.title == "Pilot"
    assert restored.status == EpisodeStatus.GENERATING
    assert len(restored.scene_prompts) == 2
    assert restored.cost == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Drama runner (build_episode_dag) tests
# ---------------------------------------------------------------------------


def test_build_episode_dag():
    """build_episode_dag should produce a valid DAG from episode scene prompts."""
    from videoclaw.drama.runner import build_episode_dag

    series = DramaSeries(
        title="Test",
        model_id="mock",
    )
    ep = Episode(
        number=1,
        title="Pilot",
        scene_prompts=[
            {"scene_id": "ep01_s01", "visual_prompt": "shot 1", "duration_seconds": 5.0},
            {"scene_id": "ep01_s02", "visual_prompt": "shot 2", "duration_seconds": 5.0},
        ],
    )

    dag, state = build_episode_dag(ep, series)

    # Should have: script_gen, storyboard, 2x video, tts, music, compose, render = 8 nodes
    assert len(dag.nodes) == 8
    assert len(state.storyboard) == 2
    assert state.storyboard[0].shot_id == "ep01_s01"
    assert state.storyboard[1].model_id == "mock"
    assert ep.project_id == state.project_id
