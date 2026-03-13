"""Tests for the VideoRenderer and the _handle_render executor handler."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from videoclaw.core.executor import DAGExecutor
from videoclaw.core.planner import DAG, TaskNode, TaskType
from videoclaw.core.state import ProjectState, StateManager
from videoclaw.generation.render import (
    VideoRenderer,
    RenderProfile,
    _ASPECT_TO_RENDER_RESOLUTION,
)


# ---------------------------------------------------------------------------
# VideoRenderer.build_cmd — FFmpeg command construction
# ---------------------------------------------------------------------------


class TestBuildCmd:
    """Test that build_cmd produces correct FFmpeg arguments."""

    def test_basic_cmd(self, tmp_path):
        """Default arguments produce a valid base command."""
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        cmd = VideoRenderer.build_cmd(input_path=inp, output_path=out)

        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert "-i" in cmd
        assert str(inp) in cmd
        assert str(out) == cmd[-1]
        assert "-c:v" in cmd
        idx = cmd.index("-c:v")
        assert cmd[idx + 1] == "libx264"

    def test_resolution_scaling(self, tmp_path):
        """When resolution is set, a scale filter should appear."""
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        cmd = VideoRenderer.build_cmd(
            input_path=inp,
            output_path=out,
            resolution=(1080, 1920),
        )

        assert "-vf" in cmd
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert "scale=1080:1920" in vf_value

    def test_codec_and_preset(self, tmp_path):
        """Codec and preset are forwarded correctly."""
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        cmd = VideoRenderer.build_cmd(
            input_path=inp,
            output_path=out,
            codec="libx265",
            preset="fast",
        )

        idx_cv = cmd.index("-c:v")
        assert cmd[idx_cv + 1] == "libx265"
        idx_p = cmd.index("-preset")
        assert cmd[idx_p + 1] == "fast"

    def test_crf_present(self, tmp_path):
        """CRF flag should appear when crf is not None."""
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        cmd = VideoRenderer.build_cmd(
            input_path=inp, output_path=out, crf=18,
        )

        assert "-crf" in cmd
        idx = cmd.index("-crf")
        assert cmd[idx + 1] == "18"

    def test_crf_none(self, tmp_path):
        """When crf=None, the -crf flag should not appear."""
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        cmd = VideoRenderer.build_cmd(
            input_path=inp, output_path=out, crf=None,
        )

        assert "-crf" not in cmd

    def test_bitrate_flags(self, tmp_path):
        """Video and audio bitrate flags are set."""
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        cmd = VideoRenderer.build_cmd(
            input_path=inp,
            output_path=out,
            bitrate="10M",
            audio_bitrate="256k",
        )

        idx_bv = cmd.index("-b:v")
        assert cmd[idx_bv + 1] == "10M"
        idx_ba = cmd.index("-b:a")
        assert cmd[idx_ba + 1] == "256k"

    def test_metadata_injection(self, tmp_path):
        """Metadata key=value pairs should appear as -metadata args."""
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        cmd = VideoRenderer.build_cmd(
            input_path=inp,
            output_path=out,
            metadata={"title": "My Episode", "artist": "Studio"},
        )

        # Find all -metadata occurrences
        meta_indices = [i for i, v in enumerate(cmd) if v == "-metadata"]
        assert len(meta_indices) == 2
        meta_values = [cmd[i + 1] for i in meta_indices]
        assert "title=My Episode" in meta_values
        assert "artist=Studio" in meta_values

    def test_no_metadata_when_none(self, tmp_path):
        """No -metadata flags when metadata is None."""
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        cmd = VideoRenderer.build_cmd(
            input_path=inp, output_path=out, metadata=None,
        )

        assert "-metadata" not in cmd

    def test_watermark_adds_input_and_overlay(self, tmp_path):
        """Watermark path adds a second -i and overlay filter."""
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        wm = tmp_path / "watermark.png"
        cmd = VideoRenderer.build_cmd(
            input_path=inp,
            output_path=out,
            watermark_path=wm,
        )

        # Should have two -i flags
        i_indices = [i for i, v in enumerate(cmd) if v == "-i"]
        assert len(i_indices) == 2
        assert cmd[i_indices[1] + 1] == str(wm)

        # Should use filter_complex with overlay
        assert "-filter_complex" in cmd
        fc_idx = cmd.index("-filter_complex")
        assert "overlay" in cmd[fc_idx + 1]

    def test_watermark_with_resolution(self, tmp_path):
        """Watermark + resolution uses filter_complex with scale and overlay."""
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        wm = tmp_path / "watermark.png"
        cmd = VideoRenderer.build_cmd(
            input_path=inp,
            output_path=out,
            resolution=(1080, 1920),
            watermark_path=wm,
        )

        assert "-filter_complex" in cmd
        fc_idx = cmd.index("-filter_complex")
        fc_value = cmd[fc_idx + 1]
        assert "scale=1080:1920" in fc_value
        assert "overlay" in fc_value


# ---------------------------------------------------------------------------
# VideoRenderer.render — async method with mocked FFmpeg
# ---------------------------------------------------------------------------


class TestVideoRendererRender:
    @pytest.mark.asyncio
    async def test_render_calls_ffmpeg(self, tmp_path):
        """render() should call run_ffmpeg and return the output path."""
        inp = tmp_path / "in.mp4"
        inp.write_bytes(b"fake_video")
        out = tmp_path / "out.mp4"

        with (
            patch("videoclaw.generation.render.check_ffmpeg", new_callable=AsyncMock, return_value=True),
            patch("videoclaw.generation.render.run_ffmpeg", new_callable=AsyncMock) as mock_run,
        ):
            renderer = VideoRenderer()
            result = await renderer.render(inp, out, codec="libx264", preset="fast")

        assert result == out
        mock_run.assert_called_once()

        # Verify the args passed to run_ffmpeg (excludes leading 'ffmpeg')
        args = mock_run.call_args[0][0]
        assert "-y" in args
        assert str(inp) in args
        assert str(out) in args

    @pytest.mark.asyncio
    async def test_render_with_metadata(self, tmp_path):
        """render() forwards metadata to the FFmpeg command."""
        inp = tmp_path / "in.mp4"
        inp.write_bytes(b"fake_video")
        out = tmp_path / "out.mp4"

        with (
            patch("videoclaw.generation.render.check_ffmpeg", new_callable=AsyncMock, return_value=True),
            patch("videoclaw.generation.render.run_ffmpeg", new_callable=AsyncMock) as mock_run,
        ):
            renderer = VideoRenderer()
            await renderer.render(
                inp, out,
                metadata={"title": "Test Title", "episode_id": "3"},
            )

        args = mock_run.call_args[0][0]
        # Reconstruct the full command to check metadata
        full_cmd = ["ffmpeg"] + args
        meta_indices = [i for i, v in enumerate(full_cmd) if v == "-metadata"]
        meta_values = [full_cmd[i + 1] for i in meta_indices]
        assert "title=Test Title" in meta_values
        assert "episode_id=3" in meta_values

    @pytest.mark.asyncio
    async def test_render_raises_without_ffmpeg(self):
        """render() raises RuntimeError when FFmpeg is not available."""
        with patch("videoclaw.generation.render.check_ffmpeg", new_callable=AsyncMock, return_value=False):
            renderer = VideoRenderer()
            with pytest.raises(RuntimeError, match="FFmpeg is not installed"):
                await renderer.render(Path("/tmp/in.mp4"), Path("/tmp/out.mp4"))


# ---------------------------------------------------------------------------
# RenderProfile dataclass
# ---------------------------------------------------------------------------


class TestRenderProfile:
    def test_defaults(self):
        profile = RenderProfile()
        assert profile.resolution is None
        assert profile.bitrate == "8M"
        assert profile.audio_bitrate == "192k"
        assert profile.codec == "libx264"
        assert profile.preset == "medium"
        assert profile.crf == 23
        assert profile.metadata == {}
        assert profile.watermark_path is None

    def test_custom_values(self):
        profile = RenderProfile(
            resolution=(1080, 1920),
            codec="libx265",
            crf=18,
            metadata={"title": "Test"},
        )
        assert profile.resolution == (1080, 1920)
        assert profile.codec == "libx265"
        assert profile.crf == 18
        assert profile.metadata == {"title": "Test"}


# ---------------------------------------------------------------------------
# Aspect ratio mapping
# ---------------------------------------------------------------------------


class TestAspectRatioMapping:
    def test_9_16(self):
        assert _ASPECT_TO_RENDER_RESOLUTION["9:16"] == (1080, 1920)

    def test_16_9(self):
        assert _ASPECT_TO_RENDER_RESOLUTION["16:9"] == (1920, 1080)

    def test_1_1(self):
        assert _ASPECT_TO_RENDER_RESOLUTION["1:1"] == (1080, 1080)


# ---------------------------------------------------------------------------
# Handler: _handle_render in DAGExecutor
# ---------------------------------------------------------------------------


class TestHandleRender:
    @pytest.mark.asyncio
    async def test_handle_render_uses_video_renderer(self, tmp_path):
        """_handle_render should use VideoRenderer instead of shutil.copy2."""
        sm = StateManager(projects_dir=tmp_path)

        project_dir = tmp_path / "test_project"
        project_dir.mkdir(parents=True)
        composed = project_dir / "composed_final.mp4"
        composed.write_bytes(b"composed_video_data")

        state = ProjectState(
            project_id="test_project",
            prompt="test",
            assets={"composed_video": str(composed)},
        )

        dag = DAG()
        node = TaskNode(
            node_id="render",
            task_type=TaskType.RENDER,
            params={
                "codec": "libx264",
                "preset": "medium",
                "crf": 23,
                "audio_bitrate": "192k",
            },
        )
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        with patch("videoclaw.generation.render.VideoRenderer") as MockRenderer:
            mock_instance = MockRenderer.return_value
            mock_instance.render = AsyncMock(return_value=project_dir / "final.mp4")
            result = await executor._handle_render(node, state)

        assert "output_path" in result
        assert result["output_path"].endswith("final.mp4")
        assert state.assets["final_video"] == result["output_path"]
        mock_instance.render.assert_called_once()

        # Verify render kwargs
        call_kwargs = mock_instance.render.call_args[1]
        assert call_kwargs["codec"] == "libx264"
        assert call_kwargs["preset"] == "medium"
        assert call_kwargs["crf"] == 23
        assert call_kwargs["audio_bitrate"] == "192k"

    @pytest.mark.asyncio
    async def test_handle_render_falls_back_on_ffmpeg_failure(self, tmp_path):
        """When VideoRenderer raises, handler should fall back to shutil.copy2."""
        sm = StateManager(projects_dir=tmp_path)

        project_dir = tmp_path / "test_project"
        project_dir.mkdir(parents=True)
        composed = project_dir / "composed_final.mp4"
        composed.write_bytes(b"composed_video_data")

        state = ProjectState(
            project_id="test_project",
            prompt="test",
            assets={"composed_video": str(composed)},
        )

        dag = DAG()
        node = TaskNode(node_id="render", task_type=TaskType.RENDER)
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        with patch("videoclaw.generation.render.VideoRenderer") as MockRenderer:
            mock_instance = MockRenderer.return_value
            mock_instance.render = AsyncMock(side_effect=RuntimeError("FFmpeg not found"))
            result = await executor._handle_render(node, state)

        # Should still succeed via fallback
        final_path = Path(result["output_path"])
        assert final_path.exists()
        assert final_path.read_bytes() == b"composed_video_data"
        assert state.assets["final_video"] == result["output_path"]

    @pytest.mark.asyncio
    async def test_handle_render_fails_without_composed(self, tmp_path):
        """_handle_render raises ValueError when no composed video exists."""
        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(project_id="test_project", prompt="test")

        dag = DAG()
        node = TaskNode(node_id="render", task_type=TaskType.RENDER)
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)
        with pytest.raises(ValueError, match="No composed video"):
            await executor._handle_render(node, state)

    @pytest.mark.asyncio
    async def test_handle_render_resolves_aspect_ratio(self, tmp_path):
        """Aspect ratio in state.metadata is resolved to a render resolution."""
        sm = StateManager(projects_dir=tmp_path)

        project_dir = tmp_path / "test_project"
        project_dir.mkdir(parents=True)
        composed = project_dir / "composed_final.mp4"
        composed.write_bytes(b"video")

        state = ProjectState(
            project_id="test_project",
            prompt="test",
            assets={"composed_video": str(composed)},
            metadata={"aspect_ratio": "9:16"},
        )

        dag = DAG()
        node = TaskNode(node_id="render", task_type=TaskType.RENDER)
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        with patch("videoclaw.generation.render.VideoRenderer") as MockRenderer:
            mock_instance = MockRenderer.return_value
            mock_instance.render = AsyncMock(return_value=project_dir / "final.mp4")
            await executor._handle_render(node, state)

        call_kwargs = mock_instance.render.call_args[1]
        assert call_kwargs["resolution"] == (1080, 1920)

    @pytest.mark.asyncio
    async def test_handle_render_injects_metadata_from_state(self, tmp_path):
        """Series and episode metadata are forwarded to the renderer."""
        sm = StateManager(projects_dir=tmp_path)

        project_dir = tmp_path / "test_project"
        project_dir.mkdir(parents=True)
        composed = project_dir / "composed_final.mp4"
        composed.write_bytes(b"video")

        state = ProjectState(
            project_id="test_project",
            prompt="[Drama] Episode 1",
            assets={"composed_video": str(composed)},
            metadata={
                "series_id": "series_001",
                "episode_number": 3,
            },
        )

        dag = DAG()
        node = TaskNode(node_id="render", task_type=TaskType.RENDER)
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        with patch("videoclaw.generation.render.VideoRenderer") as MockRenderer:
            mock_instance = MockRenderer.return_value
            mock_instance.render = AsyncMock(return_value=project_dir / "final.mp4")
            await executor._handle_render(node, state)

        call_kwargs = mock_instance.render.call_args[1]
        assert call_kwargs["metadata"]["title"] == "[Drama] Episode 1"
        assert call_kwargs["metadata"]["episode_id"] == "3"


# ---------------------------------------------------------------------------
# Drama runner render node params
# ---------------------------------------------------------------------------


class TestDramaRunnerRenderNode:
    def test_render_node_has_encoding_params(self):
        """build_episode_dag should produce a render node with codec params."""
        from videoclaw.drama.models import DramaSeries, Episode, DramaScene

        series = DramaSeries(title="Test", model_id="mock")
        ep = Episode(
            number=1,
            scenes=[DramaScene(scene_id="s01", duration_seconds=5.0)],
        )

        from videoclaw.drama.runner import build_episode_dag
        dag, _ = build_episode_dag(ep, series)
        render_node = dag.nodes["render"]

        assert render_node.params["codec"] == "libx264"
        assert render_node.params["preset"] == "medium"
        assert render_node.params["crf"] == 23
        assert render_node.params["audio_bitrate"] == "192k"
