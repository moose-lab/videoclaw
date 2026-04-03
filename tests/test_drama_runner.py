"""Tests for drama runner (runner.py)."""

import pytest

from videoclaw.drama.models import (
    Character,
    DramaScene,
    DramaSeries,
    Episode,
    ShotScale,
    ShotType,
    VoiceProfile,
)
from videoclaw.drama.runner import build_episode_dag, build_scene_regen_dag


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

    # script_gen, storyboard, scene_validate, 2x video, 2x per_scene_tts, subtitle_gen, music, compose, render = 11
    assert len(dag.nodes) == 11
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
        # Shot prompt should use the enhanced version (from PromptEnhancer)
        assert shot.prompt == scene.effective_prompt
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


def test_build_episode_dag_propagates_narration_type():
    """Title-card narration metadata should flow into TTS/subtitle/compose scene data."""
    series = DramaSeries(title="NarrationType Test", model_id="mock")
    ep = Episode(
        number=1,
        scenes=[
            DramaScene(
                scene_id="s01",
                narration="一个月前",
                narration_type="title_card",
                duration_seconds=5.0,
            ),
            DramaScene(
                scene_id="s02",
                narration="旁白",
                narration_type="voiceover",
                duration_seconds=5.0,
            ),
        ],
    )

    dag, _ = build_episode_dag(ep, series)

    assert dag.nodes["tts_s01"].params["scene"]["narration_type"] == "title_card"
    assert dag.nodes["tts_s02"].params["scene"]["narration_type"] == "voiceover"
    assert dag.nodes["subtitle_gen"].params["scenes"][0]["narration_type"] == "title_card"
    assert dag.nodes["compose"].params["scenes"][0]["narration_type"] == "title_card"


# ---------------------------------------------------------------------------
# Scene regeneration DAG (Task 3.4)
# ---------------------------------------------------------------------------


def _make_regen_fixtures():
    """Build a series + episode with 3 scenes for regen tests."""
    series = DramaSeries(
        title="Regen Test",
        model_id="mock",
        characters=[
            Character(
                name="林薇",
                description="活泼少女",
                voice_profile=VoiceProfile(voice_id="Lively_Girl", speed=1.05),
            ),
        ],
    )
    ep = Episode(
        number=1,
        title="Pilot",
        scenes=[
            DramaScene(scene_id="ep01_s01", visual_prompt="shot 1", duration_seconds=5.0,
                        dialogue="你好", speaking_character="林薇"),
            DramaScene(scene_id="ep01_s02", visual_prompt="shot 2", duration_seconds=5.0,
                        dialogue="再见", speaking_character="林薇"),
            DramaScene(scene_id="ep01_s03", visual_prompt="shot 3", duration_seconds=5.0,
                        narration="夜幕降临"),
        ],
    )
    return series, ep


def test_build_scene_regen_dag_basic():
    """Mini-DAG for regen should contain only video_gen + per_scene_tts (2 nodes, no deps)."""
    series, ep = _make_regen_fixtures()
    # First build the full DAG to get a ProjectState
    _, state = build_episode_dag(ep, series)

    dag = build_scene_regen_dag(ep, series, "ep01_s02", state)

    assert len(dag.nodes) == 2
    node_ids = set(dag.nodes.keys())
    assert "video_ep01_s02" in node_ids
    assert "tts_ep01_s02" in node_ids
    # Both nodes should have no dependencies (ready to run immediately)
    for node in dag.nodes.values():
        assert node.depends_on == []


def test_build_scene_regen_dag_with_recompose():
    """With recompose=True, DAG should also have subtitle_gen, compose, render."""
    series, ep = _make_regen_fixtures()
    _, state = build_episode_dag(ep, series)

    dag = build_scene_regen_dag(ep, series, "ep01_s02", state, recompose=True)

    node_ids = set(dag.nodes.keys())
    assert "video_ep01_s02" in node_ids
    assert "tts_ep01_s02" in node_ids
    assert "subtitle_gen" in node_ids
    assert "compose" in node_ids
    assert "render" in node_ids
    assert len(dag.nodes) == 5
    # compose depends on video + subtitle_gen
    assert "video_ep01_s02" in dag.nodes["compose"].depends_on
    assert "subtitle_gen" in dag.nodes["compose"].depends_on


def test_build_scene_regen_dag_invalid_scene_id():
    """Should raise ValueError for a non-existent scene_id."""
    series, ep = _make_regen_fixtures()
    _, state = build_episode_dag(ep, series)

    with pytest.raises(ValueError, match="not found"):
        build_scene_regen_dag(ep, series, "ep01_s99", state)


def test_build_scene_regen_dag_preserves_reference_images():
    """Regen DAG should include character reference images in video_gen params."""
    series, ep = _make_regen_fixtures()
    series.characters[0].reference_image = "/path/to/linwei.png"
    _, state = build_episode_dag(ep, series)

    dag = build_scene_regen_dag(ep, series, "ep01_s01", state)

    video_node = dag.nodes["video_ep01_s01"]
    assert video_node.params["reference_images"] == {"林薇": "/path/to/linwei.png"}


def test_build_scene_regen_dag_propagates_narration_type():
    """Regen DAG should preserve narration_type for downstream TTS handling."""
    series, ep = _make_regen_fixtures()
    ep.scenes[2].narration_type = "title_card"
    _, state = build_episode_dag(ep, series)

    dag = build_scene_regen_dag(ep, series, "ep01_s03", state)

    assert dag.nodes["tts_ep01_s03"].params["scene"]["narration_type"] == "title_card"


def test_scene_and_prop_refs_passed_to_shots(tmp_path):
    """build_episode_dag should pass scene and prop references from ConsistencyManifest to Shots."""
    import pathlib
    from videoclaw.drama.models import ConsistencyManifest

    # Create dummy reference files so verify_references() passes and verified stays True
    ivy_img = tmp_path / "ivy.png"
    pool_img = tmp_path / "pool.png"
    brochure_img = tmp_path / "brochure.png"
    for p in (ivy_img, pool_img, brochure_img):
        p.write_bytes(b"\x89PNG")

    series = DramaSeries(
        title="Ref Test",
        model_id="mock",
        characters=[
            Character(
                name="Ivy",
                visual_prompt="blonde hair",
                reference_image=str(ivy_img),
                reference_image_url="https://x.com/ivy.png",
            ),
        ],
    )
    series.consistency_manifest = ConsistencyManifest(
        character_visuals={"Ivy": "blonde hair"},
        character_references={"Ivy": str(ivy_img)},
        scene_references={"poolside_night": str(pool_img)},
        prop_references={"brochure": str(brochure_img)},
        verified=True,
    )

    ep = Episode(
        number=1,
        title="Test",
        scenes=[
            DramaScene(
                scene_id="ep01_s01",
                visual_prompt="Ivy at poolside",
                characters_present=["Ivy"],
                duration_seconds=5.0,
            ),
        ],
    )

    _, state = build_episode_dag(ep, series)
    shot = state.storyboard[0]

    assert hasattr(shot, "scene_reference_urls")
    assert "poolside_night" in shot.scene_reference_urls
    assert hasattr(shot, "prop_reference_urls")
    assert "brochure" in shot.prop_reference_urls


# ---------------------------------------------------------------------------
# Alignment regen loop
# ---------------------------------------------------------------------------


class TestAlignmentRegenLoop:
    """Test the _alignment_regen_loop in DramaRunner."""

    @pytest.mark.asyncio
    async def test_skips_when_no_alignment_report(self):
        """Should return state unchanged when no alignment_report exists."""
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        from videoclaw.core.state import ProjectState
        from videoclaw.drama.runner import DramaRunner

        runner = DramaRunner.__new__(DramaRunner)
        runner.drama_mgr = MagicMock()
        runner.state_mgr = MagicMock()
        runner.max_concurrency = 1
        runner.budget_usd = None

        state = ProjectState(project_id="test", prompt="test")
        state.status = MagicMock()
        state.status.value = "completed"
        # No alignment_report

        series = DramaSeries(title="Test", model_id="mock")
        ep = Episode(number=1, scenes=[DramaScene(scene_id="s01", visual_prompt="a")])

        result = await runner._alignment_regen_loop(series, ep, state)
        assert result is state

    @pytest.mark.asyncio
    async def test_skips_when_aligned(self):
        """Should return state unchanged when all clips are aligned."""
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        from videoclaw.core.state import ProjectState
        from videoclaw.drama.runner import DramaRunner

        runner = DramaRunner.__new__(DramaRunner)
        runner.drama_mgr = MagicMock()
        runner.state_mgr = MagicMock()
        runner.max_concurrency = 1
        runner.budget_usd = None

        state = ProjectState(project_id="test", prompt="test")
        state.status = MagicMock()
        state.status.value = "completed"
        state.assets["alignment_report"] = json.dumps({
            "is_aligned": True,
            "scenes_needing_regen": [],
            "total_scripted": 10.0,
            "total_actual": 10.2,
        })

        series = DramaSeries(title="Test", model_id="mock")
        ep = Episode(number=1, scenes=[DramaScene(scene_id="s01", visual_prompt="a")])

        result = await runner._alignment_regen_loop(series, ep, state)
        assert result is state

    @pytest.mark.asyncio
    async def test_regens_worst_drifted_scene(self):
        """Should regenerate the worst-drifted scene and recompose."""
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        from videoclaw.core.state import ProjectState, Shot
        from videoclaw.drama.runner import DramaRunner

        runner = DramaRunner.__new__(DramaRunner)
        runner.drama_mgr = MagicMock()
        runner.state_mgr = MagicMock()
        runner.max_concurrency = 1
        runner.budget_usd = None

        state = ProjectState(project_id="test", prompt="test")
        state.status = MagicMock()
        state.status.value = "completed"
        state.storyboard = [Shot(shot_id="s01"), Shot(shot_id="s02")]
        state.assets["alignment_report"] = json.dumps({
            "is_aligned": False,
            "scenes_needing_regen": ["s02", "s01"],
            "total_scripted": 10.0,
            "total_actual": 14.0,
        })

        series = DramaSeries(title="Test", model_id="mock")
        ep = Episode(
            number=1,
            scenes=[
                DramaScene(scene_id="s01", visual_prompt="a", duration_seconds=5.0),
                DramaScene(scene_id="s02", visual_prompt="b", duration_seconds=5.0),
            ],
        )

        # After regen, alignment is OK
        post_regen_state = ProjectState(project_id="test", prompt="test")
        post_regen_state.status = MagicMock()
        post_regen_state.status.value = "completed"
        post_regen_state.assets["alignment_report"] = json.dumps({
            "is_aligned": True,
            "scenes_needing_regen": [],
        })

        with patch("videoclaw.drama.runner.build_scene_regen_dag") as mock_regen_dag, \
             patch("videoclaw.drama.runner.DAGExecutor") as MockExecutor:

            mock_regen_dag.return_value = MagicMock()
            mock_exec_instance = AsyncMock()
            mock_exec_instance.run = AsyncMock(return_value=post_regen_state)
            MockExecutor.return_value = mock_exec_instance

            result = await runner._alignment_regen_loop(series, ep, state)

            # Should have called build_scene_regen_dag for s02 (worst drift)
            mock_regen_dag.assert_called_once()
            call_args = mock_regen_dag.call_args
            assert call_args[0][2] == "s02"  # scene_id
            assert call_args[1]["recompose"] is True
            assert result is post_regen_state
