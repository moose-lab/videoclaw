"""Tests for per-scene transition support in VideoComposer and _handle_compose."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from videoclaw.core.events import EventBus
from videoclaw.core.executor import DAGExecutor
from videoclaw.core.planner import DAG, TaskNode, TaskType
from videoclaw.core.state import ProjectState, Shot, StateManager
from videoclaw.drama.models import DramaScene, DramaSeries, Episode
from videoclaw.drama.runner import build_episode_dag
from videoclaw.generation.compose import (
    AlignedClip,
    AlignmentReport,
    VideoComposer,
    _DURATION_TOLERANCE_SECONDS,
    _SUPPORTED_TRANSITIONS,
    align_clips,
    scenes_needing_regen,
    validate_composed_duration,
)


# ---------------------------------------------------------------------------
# VideoComposer.compose() -- per-boundary transitions
# ---------------------------------------------------------------------------


class TestComposePerBoundaryTransitions:
    """Test that compose() applies different transitions per clip boundary."""

    def test_build_concat_cmd_uses_per_boundary_transitions(self):
        """_build_concat_cmd should use a different transition for each boundary."""
        paths = [Path("/a.mp4"), Path("/b.mp4"), Path("/c.mp4")]
        transitions = ["fade", "wipeleft"]
        durations = [5.0, 5.0, 5.0]

        cmd = VideoComposer._build_concat_cmd(paths, Path("/out.mp4"), transitions, 0.5, durations)
        cmd_str = " ".join(cmd)

        # The filter_complex should contain both transitions
        assert "transition=fade" in cmd_str
        assert "transition=wipeleft" in cmd_str

    def test_build_concat_cmd_two_clips(self):
        """Two-clip case should use the single transition from the list."""
        paths = [Path("/a.mp4"), Path("/b.mp4")]
        transitions = ["wiperight"]
        durations = [5.0, 5.0]

        cmd = VideoComposer._build_concat_cmd(paths, Path("/out.mp4"), transitions, 0.5, durations)
        cmd_str = " ".join(cmd)

        assert "transition=wiperight" in cmd_str

    def test_build_concat_cmd_falls_back_to_dissolve_for_invalid(self):
        """Invalid transitions in the list should fall back to dissolve."""
        paths = [Path("/a.mp4"), Path("/b.mp4")]
        transitions = ["invalid_transition"]
        durations = [5.0, 5.0]

        cmd = VideoComposer._build_concat_cmd(paths, Path("/out.mp4"), transitions, 0.5, durations)
        cmd_str = " ".join(cmd)

        assert "transition=dissolve" in cmd_str
        assert "transition=invalid_transition" not in cmd_str

    @pytest.mark.asyncio
    async def test_compose_with_transitions_list(self, tmp_path):
        """compose() with transitions list should pass per-boundary transitions to _build_concat_cmd."""
        composer = VideoComposer()
        paths = [Path("/a.mp4"), Path("/b.mp4"), Path("/c.mp4")]
        output = tmp_path / "out.mp4"

        with patch.object(composer, "_ensure_ffmpeg", new_callable=AsyncMock), \
             patch.object(composer, "_run_ffmpeg", new_callable=AsyncMock), \
             patch("videoclaw.generation.compose.get_video_duration", new_callable=AsyncMock, return_value=5.0), \
             patch.object(VideoComposer, "_build_concat_cmd", return_value=["ffmpeg"]) as mock_build:

            await composer.compose(
                paths,
                output,
                transition="dissolve",
                transitions=["fade", "wipeleft"],
            )

            mock_build.assert_called_once()
            call_args = mock_build.call_args
            # transitions arg (3rd positional) should be resolved per-boundary list
            resolved = call_args[0][2]
            assert resolved == ["fade", "wipeleft"]

    @pytest.mark.asyncio
    async def test_compose_transitions_empty_strings_fallback(self):
        """Empty strings in transitions list should fall back to the global transition."""
        composer = VideoComposer()
        paths = [Path("/a.mp4"), Path("/b.mp4"), Path("/c.mp4")]

        with patch.object(composer, "_ensure_ffmpeg", new_callable=AsyncMock), \
             patch.object(composer, "_run_ffmpeg", new_callable=AsyncMock), \
             patch("videoclaw.generation.compose.get_video_duration", new_callable=AsyncMock, return_value=5.0), \
             patch.object(VideoComposer, "_build_concat_cmd", return_value=["ffmpeg"]) as mock_build:

            await composer.compose(
                paths,
                Path("/tmp/out.mp4"),
                transition="wipeup",
                transitions=["", "fade"],
            )

            resolved = mock_build.call_args[0][2]
            assert resolved == ["wipeup", "fade"]

    @pytest.mark.asyncio
    async def test_compose_transitions_short_list_padded(self):
        """A transitions list shorter than n_boundaries should be padded with the global transition."""
        composer = VideoComposer()
        paths = [Path("/a.mp4"), Path("/b.mp4"), Path("/c.mp4"), Path("/d.mp4")]

        with patch.object(composer, "_ensure_ffmpeg", new_callable=AsyncMock), \
             patch.object(composer, "_run_ffmpeg", new_callable=AsyncMock), \
             patch("videoclaw.generation.compose.get_video_duration", new_callable=AsyncMock, return_value=5.0), \
             patch.object(VideoComposer, "_build_concat_cmd", return_value=["ffmpeg"]) as mock_build:

            await composer.compose(
                paths,
                Path("/tmp/out.mp4"),
                transition="dissolve",
                transitions=["fade"],  # only 1 entry for 3 boundaries
            )

            resolved = mock_build.call_args[0][2]
            assert resolved == ["fade", "dissolve", "dissolve"]

    @pytest.mark.asyncio
    async def test_compose_transitions_invalid_entries_fallback(self):
        """Invalid transition names in the list should fall back to the global transition."""
        composer = VideoComposer()
        paths = [Path("/a.mp4"), Path("/b.mp4"), Path("/c.mp4")]

        with patch.object(composer, "_ensure_ffmpeg", new_callable=AsyncMock), \
             patch.object(composer, "_run_ffmpeg", new_callable=AsyncMock), \
             patch("videoclaw.generation.compose.get_video_duration", new_callable=AsyncMock, return_value=5.0), \
             patch.object(VideoComposer, "_build_concat_cmd", return_value=["ffmpeg"]) as mock_build:

            await composer.compose(
                paths,
                Path("/tmp/out.mp4"),
                transition="fade",
                transitions=["nonsense", "wipeleft"],
            )

            resolved = mock_build.call_args[0][2]
            assert resolved == ["fade", "wipeleft"]


# ---------------------------------------------------------------------------
# xfade offset calculation
# ---------------------------------------------------------------------------


class TestXfadeOffsetCalculation:
    """Verify that _build_concat_cmd computes correct xfade offsets from clip durations."""

    def test_two_clips_offset(self):
        """For 2 clips: offset = dur[0] - transition_duration."""
        paths = [Path("/a.mp4"), Path("/b.mp4")]
        transitions = ["dissolve"]
        durations = [5.0, 4.0]
        td = 0.5

        cmd = VideoComposer._build_concat_cmd(paths, Path("/out.mp4"), transitions, td, durations)
        cmd_str = " ".join(cmd)

        # offset should be 5.0 - 0.5 = 4.5
        assert "offset=4.5" in cmd_str

    def test_three_clips_offset_chain(self):
        """For 3 clips: offset_0 = dur[0] - td, offset_1 = offset_0 + dur[1] - td."""
        paths = [Path("/a.mp4"), Path("/b.mp4"), Path("/c.mp4")]
        transitions = ["dissolve", "fade"]
        durations = [5.0, 4.0, 3.0]
        td = 0.5

        cmd = VideoComposer._build_concat_cmd(paths, Path("/out.mp4"), transitions, td, durations)
        cmd_str = " ".join(cmd)

        # offset_0 = 5.0 - 0.5 = 4.5
        # offset_1 = 4.5 + 4.0 - 0.5 = 8.0
        assert "offset=4.5" in cmd_str
        assert "offset=8.0" in cmd_str

    def test_different_durations_per_clip(self):
        """Each clip can have a different duration; offsets accumulate correctly."""
        paths = [Path("/a.mp4"), Path("/b.mp4"), Path("/c.mp4"), Path("/d.mp4")]
        transitions = ["dissolve", "fade", "wipeleft"]
        durations = [3.0, 6.0, 2.0, 4.0]
        td = 1.0

        cmd = VideoComposer._build_concat_cmd(paths, Path("/out.mp4"), transitions, td, durations)
        cmd_str = " ".join(cmd)

        # offset_0 = 3.0 - 1.0 = 2.0
        # offset_1 = 2.0 + 6.0 - 1.0 = 7.0
        # offset_2 = 7.0 + 2.0 - 1.0 = 8.0
        assert "offset=2.0" in cmd_str
        assert "offset=7.0" in cmd_str
        assert "offset=8.0" in cmd_str

    def test_offset_clamped_to_zero(self):
        """Offset should never go negative; clamp to 0.0."""
        paths = [Path("/a.mp4"), Path("/b.mp4")]
        transitions = ["dissolve"]
        durations = [0.3, 5.0]  # 0.3 - 0.5 = -0.2 -> clamped to 0.0
        td = 0.5

        cmd = VideoComposer._build_concat_cmd(paths, Path("/out.mp4"), transitions, td, durations)
        cmd_str = " ".join(cmd)

        assert "offset=0.0" in cmd_str

    @pytest.mark.asyncio
    async def test_compose_probes_durations_when_none(self):
        """When clip_durations is None, compose() should probe durations via ffprobe."""
        composer = VideoComposer()
        paths = [Path("/a.mp4"), Path("/b.mp4"), Path("/c.mp4")]

        with patch.object(composer, "_ensure_ffmpeg", new_callable=AsyncMock), \
             patch.object(composer, "_run_ffmpeg", new_callable=AsyncMock), \
             patch("videoclaw.generation.compose.get_video_duration", new_callable=AsyncMock) as mock_probe, \
             patch.object(VideoComposer, "_build_concat_cmd", return_value=["ffmpeg"]) as mock_build:

            mock_probe.side_effect = [5.0, 4.0, 3.0]

            await composer.compose(
                paths,
                Path("/tmp/out.mp4"),
                transition="dissolve",
            )

            # ffprobe should have been called for each clip
            assert mock_probe.call_count == 3
            # _build_concat_cmd should receive the probed durations
            call_args = mock_build.call_args
            assert call_args[0][4] == [5.0, 4.0, 3.0]

    @pytest.mark.asyncio
    async def test_compose_uses_provided_durations(self):
        """When clip_durations is provided, compose() should NOT probe and pass them through."""
        composer = VideoComposer()
        paths = [Path("/a.mp4"), Path("/b.mp4")]

        with patch.object(composer, "_ensure_ffmpeg", new_callable=AsyncMock), \
             patch.object(composer, "_run_ffmpeg", new_callable=AsyncMock), \
             patch("videoclaw.generation.compose.get_video_duration", new_callable=AsyncMock) as mock_probe, \
             patch.object(VideoComposer, "_build_concat_cmd", return_value=["ffmpeg"]) as mock_build:

            await composer.compose(
                paths,
                Path("/tmp/out.mp4"),
                transition="dissolve",
                clip_durations=[6.0, 7.0],
            )

            # ffprobe should NOT have been called
            mock_probe.assert_not_called()
            # _build_concat_cmd should receive the provided durations
            call_args = mock_build.call_args
            assert call_args[0][4] == [6.0, 7.0]


# ---------------------------------------------------------------------------
# Backward compatibility -- no transitions param
# ---------------------------------------------------------------------------


class TestComposeBackwardCompatibility:
    """Verify that compose() without transitions still works as before."""

    @pytest.mark.asyncio
    async def test_compose_without_transitions_uses_global(self):
        """When transitions=None, all boundaries use the global transition."""
        composer = VideoComposer()
        paths = [Path("/a.mp4"), Path("/b.mp4"), Path("/c.mp4")]

        with patch.object(composer, "_ensure_ffmpeg", new_callable=AsyncMock), \
             patch.object(composer, "_run_ffmpeg", new_callable=AsyncMock), \
             patch("videoclaw.generation.compose.get_video_duration", new_callable=AsyncMock, return_value=5.0), \
             patch.object(VideoComposer, "_build_concat_cmd", return_value=["ffmpeg"]) as mock_build:

            await composer.compose(
                paths,
                Path("/tmp/out.mp4"),
                transition="wipeleft",
            )

            resolved = mock_build.call_args[0][2]
            assert resolved == ["wipeleft", "wipeleft"]

    @pytest.mark.asyncio
    async def test_compose_default_dissolve(self):
        """Default transition is dissolve when nothing is specified."""
        composer = VideoComposer()
        paths = [Path("/a.mp4"), Path("/b.mp4")]

        with patch.object(composer, "_ensure_ffmpeg", new_callable=AsyncMock), \
             patch.object(composer, "_run_ffmpeg", new_callable=AsyncMock), \
             patch("videoclaw.generation.compose.get_video_duration", new_callable=AsyncMock, return_value=5.0), \
             patch.object(VideoComposer, "_build_concat_cmd", return_value=["ffmpeg"]) as mock_build:

            await composer.compose(paths, Path("/tmp/out.mp4"))

            resolved = mock_build.call_args[0][2]
            assert resolved == ["dissolve"]

    @pytest.mark.asyncio
    async def test_compose_single_clip_no_transition(self):
        """Single clip should not use any transition."""
        composer = VideoComposer()

        with patch.object(composer, "_ensure_ffmpeg", new_callable=AsyncMock), \
             patch.object(composer, "_run_ffmpeg", new_callable=AsyncMock), \
             patch.object(VideoComposer, "_build_single_copy_cmd", return_value=["ffmpeg"]) as mock_copy:

            await composer.compose([Path("/a.mp4")], Path("/tmp/out.mp4"))

            mock_copy.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_compose extracts per-scene transitions
# ---------------------------------------------------------------------------


class TestHandleComposeTransitions:
    """Test that _handle_compose extracts per-scene transitions from scenes."""

    @pytest.mark.asyncio
    async def test_handle_compose_uses_actual_durations(self, tmp_path):
        """_handle_compose should use actual probed durations via align_clips."""
        sm = StateManager(projects_dir=tmp_path)

        project_dir = tmp_path / "test_project"
        shots_dir = project_dir / "shots"
        shots_dir.mkdir(parents=True)
        (shots_dir / "s01.mp4").write_bytes(b"fake_video_1")
        (shots_dir / "s02.mp4").write_bytes(b"fake_video_2")
        (shots_dir / "s03.mp4").write_bytes(b"fake_video_3")

        # Create fake composed.mp4 so shutil.copy2 fallback works
        (project_dir / "composed.mp4").write_bytes(b"composed")

        state = ProjectState(
            project_id="test_project",
            prompt="test",
            storyboard=[
                Shot(shot_id="s01", asset_path=str(shots_dir / "s01.mp4")),
                Shot(shot_id="s02", asset_path=str(shots_dir / "s02.mp4")),
                Shot(shot_id="s03", asset_path=str(shots_dir / "s03.mp4")),
            ],
        )

        dag = DAG()
        scenes_data = [
            {"scene_id": "s01", "dialogue": "", "duration_seconds": 5.0, "transition": "fade"},
            {"scene_id": "s02", "dialogue": "", "duration_seconds": 5.0, "transition": "wipeleft"},
            {"scene_id": "s03", "dialogue": "", "duration_seconds": 5.0, "transition": ""},
        ]
        node = TaskNode(
            node_id="compose",
            task_type=TaskType.COMPOSE,
            params={"transition": "dissolve", "scenes": scenes_data},
        )
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)
        executor._config = type(executor._config).model_construct(
            **{**executor._config.model_dump(), "projects_dir": tmp_path},
        )

        # Mock align_clips to return actual (slightly different) durations
        mock_report = AlignmentReport(
            clips=[
                AlignedClip("s01", shots_dir / "s01.mp4", 5.0, 5.2, "fade"),
                AlignedClip("s02", shots_dir / "s02.mp4", 5.0, 4.8, "wipeleft"),
                AlignedClip("s03", shots_dir / "s03.mp4", 5.0, 5.1, ""),
            ],
            misaligned_scene_ids=[],
            total_scripted=15.0,
            total_actual=15.1,
        )

        mock_validation = {"ok": True, "expected": 14.1, "actual": 14.1, "drift": 0.0}

        with patch("videoclaw.generation.compose.align_clips", new_callable=AsyncMock, return_value=mock_report), \
             patch("videoclaw.generation.compose.validate_composed_duration", new_callable=AsyncMock, return_value=mock_validation), \
             patch("videoclaw.generation.compose.VideoComposer") as MockComposer:
            mock_instance = MockComposer.return_value
            mock_instance.compose = AsyncMock(return_value=project_dir / "composed.mp4")
            mock_instance.render_final = AsyncMock(return_value=project_dir / "composed_final.mp4")

            await executor._handle_compose(node, state)

            # compose() should use ACTUAL probed durations, not scripted
            compose_call = mock_instance.compose.call_args
            assert compose_call.kwargs.get("transitions") == ["fade", "wipeleft", ""]
            assert compose_call.kwargs.get("clip_durations") == [5.2, 4.8, 5.1]

    @pytest.mark.asyncio
    async def test_handle_compose_stores_alignment_report(self, tmp_path):
        """_handle_compose should store alignment report in state.assets."""
        sm = StateManager(projects_dir=tmp_path)

        project_dir = tmp_path / "test_project"
        shots_dir = project_dir / "shots"
        shots_dir.mkdir(parents=True)
        (shots_dir / "s01.mp4").write_bytes(b"fake_video_1")
        (project_dir / "composed.mp4").write_bytes(b"composed")

        state = ProjectState(
            project_id="test_project",
            prompt="test",
            storyboard=[
                Shot(shot_id="s01", asset_path=str(shots_dir / "s01.mp4")),
            ],
        )

        dag = DAG()
        node = TaskNode(
            node_id="compose",
            task_type=TaskType.COMPOSE,
            params={
                "transition": "dissolve",
                "scenes": [
                    {"scene_id": "s01", "dialogue": "", "duration_seconds": 5.0, "transition": ""},
                ],
            },
        )
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)
        executor._config = type(executor._config).model_construct(
            **{**executor._config.model_dump(), "projects_dir": tmp_path},
        )

        mock_report = AlignmentReport(
            clips=[AlignedClip("s01", shots_dir / "s01.mp4", 5.0, 5.0, "")],
            misaligned_scene_ids=[],
            total_scripted=5.0,
            total_actual=5.0,
        )

        mock_validation = {"ok": True, "expected": 5.0, "actual": 5.0, "drift": 0.0}

        with patch("videoclaw.generation.compose.align_clips", new_callable=AsyncMock, return_value=mock_report), \
             patch("videoclaw.generation.compose.validate_composed_duration", new_callable=AsyncMock, return_value=mock_validation), \
             patch("videoclaw.generation.compose.VideoComposer") as MockComposer:
            mock_instance = MockComposer.return_value
            mock_instance.compose = AsyncMock(return_value=project_dir / "composed.mp4")

            await executor._handle_compose(node, state)

            # alignment_report should be stored in state.assets
            assert "alignment_report" in state.assets
            report_data = json.loads(state.assets["alignment_report"])
            assert report_data["is_aligned"] is True
            assert report_data["compose_validation"]["ok"] is True
            assert report_data["scenes_needing_regen"] == []

    @pytest.mark.asyncio
    async def test_handle_compose_no_scenes_passes_none(self, tmp_path):
        """When there are no scenes in params, transitions should be None."""
        sm = StateManager(projects_dir=tmp_path)

        project_dir = tmp_path / "test_project"
        shots_dir = project_dir / "shots"
        shots_dir.mkdir(parents=True)
        (shots_dir / "s01.mp4").write_bytes(b"fake_video_1")

        # Create fake composed.mp4 so shutil.copy2 fallback works
        (project_dir / "composed.mp4").write_bytes(b"composed")

        state = ProjectState(
            project_id="test_project",
            prompt="test",
            storyboard=[
                Shot(shot_id="s01", asset_path=str(shots_dir / "s01.mp4")),
            ],
        )

        dag = DAG()
        node = TaskNode(
            node_id="compose",
            task_type=TaskType.COMPOSE,
            params={"transition": "dissolve"},
        )
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)
        executor._config = type(executor._config).model_construct(
            **{**executor._config.model_dump(), "projects_dir": tmp_path},
        )

        with patch("videoclaw.generation.compose.VideoComposer") as MockComposer:
            mock_instance = MockComposer.return_value
            mock_instance.compose = AsyncMock(return_value=project_dir / "composed.mp4")

            await executor._handle_compose(node, state)

            compose_call = mock_instance.compose.call_args
            assert compose_call.kwargs.get("transitions") is None


# ---------------------------------------------------------------------------
# Runner includes transition in scenes_data
# ---------------------------------------------------------------------------


class TestRunnerScenesDataTransition:
    """Test that build_episode_dag includes transition in scenes_data."""

    def test_scenes_data_includes_transition(self):
        """The compose node params['scenes'] should contain transition from each DramaScene."""
        series = DramaSeries(title="Test", model_id="mock")
        ep = Episode(
            number=1,
            scenes=[
                DramaScene(scene_id="s01", visual_prompt="a", transition="fade"),
                DramaScene(scene_id="s02", visual_prompt="b", transition="wipeleft"),
                DramaScene(scene_id="s03", visual_prompt="c", transition=""),
            ],
        )

        dag, _ = build_episode_dag(ep, series)
        compose_node = dag.nodes["compose"]
        scenes = compose_node.params["scenes"]

        assert scenes[0]["transition"] == "fade"
        assert scenes[1]["transition"] == "wipeleft"
        assert scenes[2]["transition"] == ""

    def test_scenes_data_default_empty_transition(self):
        """DramaScene default transition is empty string, should be in scenes_data."""
        series = DramaSeries(title="Test", model_id="mock")
        ep = Episode(
            number=1,
            scenes=[DramaScene(scene_id="s01", visual_prompt="a")],
        )

        dag, _ = build_episode_dag(ep, series)
        compose_node = dag.nodes["compose"]
        scenes = compose_node.params["scenes"]

        assert scenes[0]["transition"] == ""


# ---------------------------------------------------------------------------
# align_clips — pre-compose duration alignment
# ---------------------------------------------------------------------------


class TestAlignClips:
    """Test the pre-compose duration alignment function."""

    @pytest.mark.asyncio
    async def test_aligned_clips_within_tolerance(self):
        """Clips within tolerance should report as aligned."""
        paths = [Path("/a.mp4"), Path("/b.mp4"), Path("/c.mp4")]
        scenes = [
            {"scene_id": "s01", "duration_seconds": 5.0, "transition": "fade"},
            {"scene_id": "s02", "duration_seconds": 8.0, "transition": "dissolve"},
            {"scene_id": "s03", "duration_seconds": 6.0, "transition": ""},
        ]

        # Actual durations within tolerance (±1.0s)
        with patch(
            "videoclaw.generation.compose.get_video_duration",
            new_callable=AsyncMock,
            side_effect=[5.3, 7.8, 6.5],
        ):
            report = await align_clips(paths, scenes)

        assert report.is_aligned
        assert report.misaligned_scene_ids == []
        assert len(report.clips) == 3
        assert report.clips[0].actual_duration == 5.3
        assert report.clips[1].actual_duration == 7.8
        assert report.clips[2].actual_duration == 6.5

    @pytest.mark.asyncio
    async def test_misaligned_clips_beyond_tolerance(self):
        """Clips beyond tolerance should be reported as misaligned."""
        paths = [Path("/a.mp4"), Path("/b.mp4")]
        scenes = [
            {"scene_id": "s01", "duration_seconds": 10.0, "transition": "fade"},
            {"scene_id": "s02", "duration_seconds": 8.0, "transition": ""},
        ]

        # s01: actual=7.5, drift=2.5 > 1.0 → misaligned
        # s02: actual=8.3, drift=0.3 < 1.0 → aligned
        with patch(
            "videoclaw.generation.compose.get_video_duration",
            new_callable=AsyncMock,
            side_effect=[7.5, 8.3],
        ):
            report = await align_clips(paths, scenes)

        assert not report.is_aligned
        assert report.misaligned_scene_ids == ["s01"]
        assert report.clips[0].is_misaligned
        assert not report.clips[1].is_misaligned
        assert report.clips[0].drift == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_video_scene_count_mismatch_raises(self):
        """Mismatched video/scene counts should raise ValueError."""
        paths = [Path("/a.mp4"), Path("/b.mp4")]
        scenes = [{"scene_id": "s01", "duration_seconds": 5.0, "transition": ""}]

        with pytest.raises(ValueError, match="Video/scene count mismatch"):
            await align_clips(paths, scenes)

    @pytest.mark.asyncio
    async def test_alignment_report_totals(self):
        """AlignmentReport should compute correct totals."""
        paths = [Path("/a.mp4"), Path("/b.mp4")]
        scenes = [
            {"scene_id": "s01", "duration_seconds": 5.0, "transition": ""},
            {"scene_id": "s02", "duration_seconds": 10.0, "transition": ""},
        ]

        with patch(
            "videoclaw.generation.compose.get_video_duration",
            new_callable=AsyncMock,
            side_effect=[5.0, 10.0],
        ):
            report = await align_clips(paths, scenes)

        assert report.total_scripted == 15.0
        assert report.total_actual == 15.0
        assert report.total_drift == 0.0
        assert report.is_aligned

    @pytest.mark.asyncio
    async def test_alignment_preserves_transition(self):
        """AlignedClip should carry the scene's transition field."""
        paths = [Path("/a.mp4")]
        scenes = [{"scene_id": "s01", "duration_seconds": 5.0, "transition": "wipeleft"}]

        with patch(
            "videoclaw.generation.compose.get_video_duration",
            new_callable=AsyncMock,
            return_value=5.0,
        ):
            report = await align_clips(paths, scenes)

        assert report.clips[0].transition == "wipeleft"


class TestAlignedClipProperties:
    """Test AlignedClip dataclass properties."""

    def test_drift_calculation(self):
        clip = AlignedClip("s01", Path("/a.mp4"), 5.0, 7.3, "")
        assert clip.drift == pytest.approx(2.3)

    def test_is_misaligned_true(self):
        clip = AlignedClip("s01", Path("/a.mp4"), 5.0, 6.5, "")
        assert clip.is_misaligned  # 1.5 > 1.0

    def test_is_misaligned_false(self):
        clip = AlignedClip("s01", Path("/a.mp4"), 5.0, 5.8, "")
        assert not clip.is_misaligned  # 0.8 < 1.0

    def test_is_misaligned_boundary(self):
        clip = AlignedClip("s01", Path("/a.mp4"), 5.0, 6.0, "")
        assert not clip.is_misaligned  # 1.0 == tolerance, not >


class TestAlignmentReport:
    """Test AlignmentReport properties."""

    def test_aligned_report(self):
        clips = [
            AlignedClip("s01", Path("/a.mp4"), 5.0, 5.5, "fade"),
            AlignedClip("s02", Path("/b.mp4"), 8.0, 7.6, ""),
        ]
        report = AlignmentReport(clips=clips, misaligned_scene_ids=[], total_scripted=13.0, total_actual=13.1)
        assert report.is_aligned
        assert report.total_drift == pytest.approx(0.1)

    def test_misaligned_report(self):
        clips = [
            AlignedClip("s01", Path("/a.mp4"), 5.0, 8.0, ""),
        ]
        report = AlignmentReport(clips=clips, misaligned_scene_ids=["s01"], total_scripted=5.0, total_actual=8.0)
        assert not report.is_aligned
        assert report.total_drift == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# validate_composed_duration — post-compose validation
# ---------------------------------------------------------------------------


class TestValidateComposedDuration:
    """Test post-compose duration validation."""

    @pytest.mark.asyncio
    async def test_valid_composed_duration(self):
        """Composed video matching expected duration (with transition overlap) should pass."""
        clips = [
            AlignedClip("s01", Path("/a.mp4"), 5.0, 5.0, "fade"),
            AlignedClip("s02", Path("/b.mp4"), 5.0, 5.0, ""),
        ]
        report = AlignmentReport(clips=clips, misaligned_scene_ids=[], total_scripted=10.0, total_actual=10.0)

        # Expected: 10.0 - 1 * 0.5 = 9.5
        with patch(
            "videoclaw.generation.compose.get_video_duration",
            new_callable=AsyncMock,
            return_value=9.5,
        ):
            result = await validate_composed_duration(Path("/composed.mp4"), report)

        assert result["ok"] is True
        assert result["expected"] == 9.5
        assert result["actual"] == 9.5
        assert result["drift"] == 0.0

    @pytest.mark.asyncio
    async def test_invalid_composed_duration(self):
        """Composed video with significant drift should fail."""
        clips = [
            AlignedClip("s01", Path("/a.mp4"), 5.0, 5.0, "fade"),
            AlignedClip("s02", Path("/b.mp4"), 5.0, 5.0, ""),
        ]
        report = AlignmentReport(clips=clips, misaligned_scene_ids=[], total_scripted=10.0, total_actual=10.0)

        # Expected: 9.5, actual: 7.0 → drift 2.5
        with patch(
            "videoclaw.generation.compose.get_video_duration",
            new_callable=AsyncMock,
            return_value=7.0,
        ):
            result = await validate_composed_duration(Path("/composed.mp4"), report)

        assert result["ok"] is False
        assert result["drift"] == 2.5

    @pytest.mark.asyncio
    async def test_single_clip_no_overlap(self):
        """Single clip should have no transition overlap deduction."""
        clips = [AlignedClip("s01", Path("/a.mp4"), 8.0, 8.0, "")]
        report = AlignmentReport(clips=clips, misaligned_scene_ids=[], total_scripted=8.0, total_actual=8.0)

        with patch(
            "videoclaw.generation.compose.get_video_duration",
            new_callable=AsyncMock,
            return_value=8.0,
        ):
            result = await validate_composed_duration(Path("/composed.mp4"), report)

        assert result["ok"] is True
        assert result["expected"] == 8.0


# ---------------------------------------------------------------------------
# scenes_needing_regen — identify scenes to regenerate
# ---------------------------------------------------------------------------


class TestScenesNeedingRegen:
    """Test scene regen identification from alignment report."""

    def test_no_regen_needed(self):
        clips = [
            AlignedClip("s01", Path("/a.mp4"), 5.0, 5.5, ""),
            AlignedClip("s02", Path("/b.mp4"), 8.0, 8.3, ""),
        ]
        report = AlignmentReport(clips=clips, misaligned_scene_ids=[], total_scripted=13.0, total_actual=13.8)
        assert scenes_needing_regen(report) == []

    def test_regen_sorted_by_drift(self):
        clips = [
            AlignedClip("s01", Path("/a.mp4"), 5.0, 7.0, ""),  # drift 2.0
            AlignedClip("s02", Path("/b.mp4"), 8.0, 5.0, ""),  # drift 3.0
            AlignedClip("s03", Path("/c.mp4"), 6.0, 6.3, ""),  # drift 0.3
        ]
        report = AlignmentReport(
            clips=clips,
            misaligned_scene_ids=["s01", "s02"],
            total_scripted=19.0,
            total_actual=18.3,
        )
        result = scenes_needing_regen(report)
        assert result == ["s02", "s01"]  # sorted by drift descending

    def test_excludes_within_tolerance(self):
        clips = [
            AlignedClip("s01", Path("/a.mp4"), 5.0, 5.8, ""),  # drift 0.8, within tolerance
            AlignedClip("s02", Path("/b.mp4"), 8.0, 10.5, ""),  # drift 2.5, needs regen
        ]
        report = AlignmentReport(
            clips=clips,
            misaligned_scene_ids=["s02"],
            total_scripted=13.0,
            total_actual=16.3,
        )
        result = scenes_needing_regen(report)
        assert result == ["s02"]
