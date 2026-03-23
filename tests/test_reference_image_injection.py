"""Tests for character reference image injection (Task 2.2).

Validates the full data flow: Character.reference_image → Shot.reference_images →
TaskNode.params → _handle_video_gen() → VideoGenerator → GenerationRequest.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from videoclaw.core.planner import TaskNode, TaskType
from videoclaw.core.state import ProjectState, Shot, ShotStatus
from videoclaw.drama.models import Character, Episode, DramaScene, DramaSeries
from videoclaw.drama.runner import build_episode_dag
from videoclaw.generation.video import VideoGenerator
from videoclaw.models.protocol import GenerationRequest, GenerationResult


# ---------------------------------------------------------------------------
# Step 1: Shot.reference_images roundtrip
# ---------------------------------------------------------------------------


class TestShotReferenceImages:
    def test_shot_reference_images_roundtrip(self):
        """Shot.reference_images survives to_dict() / from_dict()."""
        refs = {"林薇": "/tmp/chars/linwei.png", "张明": "/tmp/chars/zhangming.png"}
        shot = Shot(
            shot_id="s001",
            prompt="林薇走进咖啡厅",
            reference_images=refs,
        )

        data = shot.to_dict()
        assert data["reference_images"] == refs

        restored = Shot.from_dict(data)
        assert restored.reference_images == refs

    def test_shot_default_empty_reference_images(self):
        """Backward compat: shots without reference_images get empty dict."""
        shot = Shot(shot_id="s002", prompt="test")
        assert shot.reference_images == {}

    def test_shot_from_dict_missing_reference_images(self):
        """Old serialized data without reference_images still loads."""
        data = {
            "shot_id": "s003",
            "description": "",
            "prompt": "test",
            "duration_seconds": 5.0,
            "model_id": "mock",
            "status": "pending",
            "asset_path": None,
            "cost": 0.0,
            "retries": 0,
        }
        shot = Shot.from_dict(data)
        assert shot.reference_images == {}


# ---------------------------------------------------------------------------
# Step 2: build_episode_dag injects reference images
# ---------------------------------------------------------------------------


class TestBuildEpisodeDag:
    def _make_series_and_episode(self) -> tuple[DramaSeries, Episode]:
        series = DramaSeries(
            series_id="drama001",
            title="测试剧",
            aspect_ratio="9:16",
            characters=[
                Character(name="林薇", description="女主", reference_image="/imgs/linwei.png"),
                Character(name="张明", description="男主", reference_image="/imgs/zhangming.png"),
                Character(name="路人甲", description="路人", reference_image=""),  # no ref
            ],
        )
        episode = Episode(
            episode_id="ep01",
            title="第一集",
            scenes=[
                DramaScene(
                    scene_id="sc01",
                    visual_prompt="林薇走进咖啡厅",
                    duration_seconds=5.0,
                    characters_present=["林薇", "张明"],
                    speaking_character="林薇",
                ),
                DramaScene(
                    scene_id="sc02",
                    visual_prompt="张明独白",
                    duration_seconds=3.0,
                    characters_present=["张明"],
                    speaking_character="张明",
                ),
                DramaScene(
                    scene_id="sc03",
                    visual_prompt="空镜头",
                    duration_seconds=2.0,
                    characters_present=[],
                    speaking_character="",
                ),
            ],
        )
        return series, episode

    def test_build_episode_dag_injects_refs(self):
        """Shots and TaskNode params contain reference image paths."""
        series, episode = self._make_series_and_episode()
        state = ProjectState()

        dag, state = build_episode_dag(episode, series)

        # Storyboard populated
        assert len(state.storyboard) == 3

        # Shot 1: both characters have refs
        shot1 = state.storyboard[0]
        assert shot1.reference_images == {
            "林薇": "/imgs/linwei.png",
            "张明": "/imgs/zhangming.png",
        }

        # Shot 2: only 张明
        shot2 = state.storyboard[1]
        assert shot2.reference_images == {"张明": "/imgs/zhangming.png"}

        # Shot 3: empty scene → no refs
        shot3 = state.storyboard[2]
        assert shot3.reference_images == {}

        # TaskNode params also have reference_images and speaking_character
        vid_node_1 = dag.nodes["video_sc01"]
        assert vid_node_1.params["reference_images"] == shot1.reference_images
        assert vid_node_1.params["speaking_character"] == "林薇"

        vid_node_2 = dag.nodes["video_sc02"]
        assert vid_node_2.params["speaking_character"] == "张明"

    def test_build_episode_dag_skips_missing_refs(self):
        """Characters without reference_image are excluded from the dict."""
        series, episode = self._make_series_and_episode()
        # Add a scene with 路人甲 (no reference image)
        episode.scenes.append(
            DramaScene(
                scene_id="sc04",
                visual_prompt="路人甲出现",
                duration_seconds=2.0,
                characters_present=["路人甲", "林薇"],
                speaking_character="路人甲",
            )
        )
        _, state = build_episode_dag(episode, series)

        shot4 = state.storyboard[3]
        # 路人甲 has no ref, only 林薇
        assert "路人甲" not in shot4.reference_images
        assert "林薇" in shot4.reference_images

    def test_build_episode_dag_unknown_character(self):
        """characters_present with unknown names are silently skipped."""
        series, episode = self._make_series_and_episode()
        episode.scenes[0].characters_present.append("不存在的角色")

        state = ProjectState()
        dag, state = build_episode_dag(episode, series)

        shot1 = state.storyboard[0]
        assert "不存在的角色" not in shot1.reference_images

    def test_dag_structure(self):
        """DAG has expected node types and dependencies."""
        series, episode = self._make_series_and_episode()
        state = ProjectState()
        dag, state = build_episode_dag(episode, series)

        assert "storyboard" in dag.nodes
        assert "music" in dag.nodes
        assert "compose" in dag.nodes
        assert "render" in dag.nodes
        assert "subtitle_gen" in dag.nodes

        # 3 video nodes
        vid_nodes = [n for n in dag.nodes.values() if n.task_type == TaskType.VIDEO_GEN]
        assert len(vid_nodes) == 3

        # 3 per-scene TTS nodes
        tts_nodes = [n for n in dag.nodes.values() if n.task_type == TaskType.PER_SCENE_TTS]
        assert len(tts_nodes) == 3

        # All video nodes depend on scene_validate (scene-first pattern)
        for vn in vid_nodes:
            assert "scene_validate" in vn.depends_on

        # Compose depends on all videos + subtitle_gen + music
        compose = dag.nodes["compose"]
        for vn in vid_nodes:
            assert vn.node_id in compose.depends_on

    def test_aspect_ratio_in_params(self):
        """aspect_ratio from series is passed into TaskNode params."""
        series, episode = self._make_series_and_episode()
        state = ProjectState()
        dag, state = build_episode_dag(episode, series)

        vid_node = dag.nodes["video_sc01"]
        assert vid_node.params["aspect_ratio"] == "9:16"


# ---------------------------------------------------------------------------
# Step 3: Primary character selection logic
# ---------------------------------------------------------------------------


class TestPrimaryCharacterSelection:
    def test_primary_is_speaking_character(self):
        """When speaking_character has a ref, it becomes primary."""
        ref_images = {"林薇": "/a.png", "张明": "/b.png"}
        speaking = "张明"

        primary_name = speaking if speaking in ref_images else None
        if primary_name is None:
            primary_name = next(iter(ref_images))

        assert primary_name == "张明"

    def test_primary_fallback_to_first(self):
        """When speaking_character has no ref, fall back to first."""
        ref_images = {"林薇": "/a.png", "张明": "/b.png"}
        speaking = "路人甲"  # not in ref_images

        primary_name = speaking if speaking in ref_images else None
        if primary_name is None:
            primary_name = next(iter(ref_images))

        assert primary_name == "林薇"

    def test_primary_with_empty_speaking(self):
        """Empty speaking_character falls back to first."""
        ref_images = {"林薇": "/a.png"}
        speaking = ""

        primary_name = speaking if speaking in ref_images else None
        if primary_name is None:
            primary_name = next(iter(ref_images))

        assert primary_name == "林薇"


# ---------------------------------------------------------------------------
# Step 4: _handle_video_gen reads reference files
# ---------------------------------------------------------------------------


class TestHandleVideoGen:
    @pytest.mark.asyncio
    async def test_handle_video_gen_reads_ref_file(self):
        """_handle_video_gen reads PNG files and passes bytes to generator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake reference image
            img_path = Path(tmpdir) / "linwei.png"
            img_data = b"\x89PNG_fake_image_data"
            img_path.write_bytes(img_data)

            shot = Shot(
                shot_id="s001",
                prompt="test prompt",
                model_id="mock",
                reference_images={"林薇": str(img_path)},
            )

            state = ProjectState()
            state.storyboard = [shot]

            node = TaskNode(
                node_id="video_s001",
                task_type=TaskType.VIDEO_GEN,
                params={
                    "shot_id": "s001",
                    "prompt": "test prompt",
                    "reference_images": {"林薇": str(img_path)},
                    "speaking_character": "林薇",
                },
            )

            # Mock the generator and registry
            mock_result = GenerationResult(
                video_data=b"fake_video",
                duration_seconds=5.0,
                cost_usd=0.0,
                model_id="mock",
            )

            with (
                patch("videoclaw.models.registry.get_registry") as mock_get_reg,
                patch("videoclaw.generation.video.VideoGenerator") as MockVG,
                patch("videoclaw.models.router.ModelRouter") as MockRouter,
            ):
                mock_registry = MagicMock()
                mock_registry.list_models.return_value = [{"model_id": "mock"}]
                mock_get_reg.return_value = mock_registry

                mock_gen_instance = AsyncMock()
                mock_gen_instance.generate_shot.return_value = mock_result
                MockVG.return_value = mock_gen_instance

                from videoclaw.core.executor import DAGExecutor
                from videoclaw.core.planner import DAG

                dag = DAG()
                dag.add_node(node)
                executor = DAGExecutor(dag=dag, state=state)

                await executor._handle_video_gen(node, state)

                # Verify generate_shot was called with reference_image bytes
                call_kwargs = mock_gen_instance.generate_shot.call_args
                assert call_kwargs.kwargs["reference_image"] == img_data
                assert call_kwargs.kwargs["extra_references"] is None

    @pytest.mark.asyncio
    async def test_handle_video_gen_missing_file_degrades(self):
        """Missing reference image file → degrades to TEXT_TO_VIDEO (no ref bytes)."""
        shot = Shot(
            shot_id="s001",
            prompt="test prompt",
            model_id="mock",
        )
        state = ProjectState()
        state.storyboard = [shot]

        node = TaskNode(
            node_id="video_s001",
            task_type=TaskType.VIDEO_GEN,
            params={
                "shot_id": "s001",
                "prompt": "test prompt",
                "reference_images": {"林薇": "/nonexistent/path.png"},
                "speaking_character": "林薇",
            },
        )

        mock_result = GenerationResult(
            video_data=b"fake_video",
            duration_seconds=5.0,
            cost_usd=0.0,
            model_id="mock",
        )

        with (
            patch("videoclaw.models.registry.get_registry") as mock_get_reg,
            patch("videoclaw.generation.video.VideoGenerator") as MockVG,
            patch("videoclaw.models.router.ModelRouter"),
        ):
            mock_registry = MagicMock()
            mock_registry.list_models.return_value = [{"model_id": "mock"}]
            mock_get_reg.return_value = mock_registry

            mock_gen_instance = AsyncMock()
            mock_gen_instance.generate_shot.return_value = mock_result
            MockVG.return_value = mock_gen_instance

            from videoclaw.core.executor import DAGExecutor
            from videoclaw.core.planner import DAG

            dag = DAG()
            dag.add_node(node)
            executor = DAGExecutor(dag=dag, state=state)

            await executor._handle_video_gen(node, state)

            # reference_image should be None (file not found)
            call_kwargs = mock_gen_instance.generate_shot.call_args
            assert call_kwargs.kwargs["reference_image"] is None


# ---------------------------------------------------------------------------
# Step 5: VideoGenerator forwards reference image
# ---------------------------------------------------------------------------


class TestGenerateShotForwardsRef:
    @pytest.mark.asyncio
    async def test_generate_shot_forwards_ref_image(self):
        """VideoGenerator passes reference_image bytes into GenerationRequest."""
        ref_bytes = b"\x89PNG_test_image"

        mock_adapter = AsyncMock()
        mock_adapter.model_id = "mock"
        mock_adapter.health_check.return_value = True
        mock_adapter.generate.return_value = GenerationResult(
            video_data=b"video",
            duration_seconds=5.0,
            model_id="mock",
        )

        mock_router = AsyncMock()
        mock_router.select.return_value = mock_adapter

        gen = VideoGenerator(router=mock_router)
        shot = Shot(shot_id="s001", prompt="test", model_id="mock")

        await gen.generate_shot(
            shot,
            reference_image=ref_bytes,
            extra_references={"张明": b"extra_img"},
        )

        # Check the GenerationRequest passed to adapter.generate
        gen_request = mock_adapter.generate.call_args[0][0]
        assert gen_request.reference_image == ref_bytes
        assert gen_request.extra["additional_references"] == {"张明": b"extra_img"}

    @pytest.mark.asyncio
    async def test_generate_shot_no_ref_image(self):
        """Without reference_image, request.reference_image stays None."""
        mock_adapter = AsyncMock()
        mock_adapter.model_id = "mock"
        mock_adapter.health_check.return_value = True
        mock_adapter.generate.return_value = GenerationResult(
            video_data=b"video",
            duration_seconds=5.0,
            model_id="mock",
        )

        mock_router = AsyncMock()
        mock_router.select.return_value = mock_adapter

        gen = VideoGenerator(router=mock_router)
        shot = Shot(shot_id="s001", prompt="test", model_id="mock")

        await gen.generate_shot(shot)

        gen_request = mock_adapter.generate.call_args[0][0]
        assert gen_request.reference_image is None
        assert gen_request.extra == {}
