"""Tests for real DAGExecutor handlers, subtitle generation, and 9:16 aspect ratio."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from videoclaw.core.events import EventBus
from videoclaw.core.executor import DAGExecutor
from videoclaw.core.planner import DAG, TaskNode, TaskType, NodeStatus
from videoclaw.core.state import ProjectState, Shot, ShotStatus, StateManager
from videoclaw.drama.models import (
    Character,
    DramaScene,
    DramaSeries,
    Episode,
    VoiceProfile,
)
from videoclaw.drama.runner import build_episode_dag
from videoclaw.generation.subtitle import generate_srt


# ---------------------------------------------------------------------------
# Subtitle generation
# ---------------------------------------------------------------------------


class TestSubtitleGeneration:
    def test_generate_srt_basic(self, tmp_path):
        scenes = [
            {"dialogue": "你好世界", "duration_seconds": 5.0, "speaking_character": "林薇"},
            {"dialogue": "你确定？", "duration_seconds": 3.0, "speaking_character": "林薇"},
        ]
        output = tmp_path / "test.srt"
        result = generate_srt(scenes, output)

        assert result == output
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "1\n" in content
        assert "2\n" in content
        assert "林薇：你好世界" in content
        assert "林薇：你确定？" in content
        assert "00:00:00,000 --> 00:00:05,000" in content
        assert "00:00:05,000 --> 00:00:08,000" in content

    def test_generate_srt_skips_empty_dialogue(self, tmp_path):
        scenes = [
            {"dialogue": "", "duration_seconds": 5.0},
            {"dialogue": "台词", "duration_seconds": 3.0, "speaking_character": ""},
        ]
        output = tmp_path / "test.srt"
        generate_srt(scenes, output)

        content = output.read_text(encoding="utf-8")
        # Only one entry (second scene) — first has empty dialogue
        assert "1\n" in content
        assert "2\n" not in content
        # Start time should account for the skipped 5s scene
        assert "00:00:05,000 --> 00:00:08,000" in content

    def test_generate_srt_with_narration(self, tmp_path):
        scenes = [
            {"dialogue": "", "narration": "旁白文本", "duration_seconds": 5.0},
        ]
        output = tmp_path / "narration.srt"
        generate_srt(scenes, output, include_narration=True)

        content = output.read_text(encoding="utf-8")
        assert "旁白文本" in content

    def test_generate_srt_no_narration_by_default(self, tmp_path):
        scenes = [
            {"dialogue": "", "narration": "旁白文本", "duration_seconds": 5.0},
        ]
        output = tmp_path / "no_narration.srt"
        generate_srt(scenes, output, include_narration=False)

        content = output.read_text(encoding="utf-8")
        assert content.strip() == ""


# ---------------------------------------------------------------------------
# Enriched DAG params
# ---------------------------------------------------------------------------


class TestEnrichedDAG:
    def test_per_scene_tts_nodes_have_scene_data(self):
        series = DramaSeries(
            title="Test", model_id="mock", language="zh",
            characters=[
                Character(
                    name="林薇",
                    voice_style="calm",
                    voice_profile=VoiceProfile(voice_id="Calm_Woman"),
                ),
            ],
        )
        ep = Episode(
            number=1,
            title="Test",
            scenes=[
                DramaScene(
                    scene_id="s01",
                    dialogue="你好",
                    narration="旁白",
                    speaking_character="林薇",
                    duration_seconds=5.0,
                ),
                DramaScene(
                    scene_id="s02",
                    dialogue="再见",
                    speaking_character="林薇",
                    duration_seconds=3.0,
                ),
            ],
        )

        dag, state = build_episode_dag(ep, series)

        # Per-scene TTS nodes exist
        tts_s01 = dag.nodes["tts_s01"]
        tts_s02 = dag.nodes["tts_s02"]
        assert tts_s01.task_type == TaskType.PER_SCENE_TTS
        assert tts_s02.task_type == TaskType.PER_SCENE_TTS

        # Scene data is correct
        assert tts_s01.params["language"] == "zh"
        assert tts_s01.params["scene"]["dialogue"] == "你好"
        assert tts_s01.params["scene"]["narration"] == "旁白"
        assert tts_s01.params["scene"]["voice"] == "Calm_Woman"
        assert tts_s02.params["scene"]["dialogue"] == "再见"

        # Both depend on scene_validate (scenes validated before generation)
        assert "scene_validate" in tts_s01.depends_on
        assert "scene_validate" in tts_s02.depends_on

    def test_subtitle_gen_node_depends_on_all_tts(self):
        series = DramaSeries(title="Test", model_id="mock")
        ep = Episode(
            number=1,
            scenes=[
                DramaScene(scene_id="s01", dialogue="A", duration_seconds=5.0),
                DramaScene(scene_id="s02", dialogue="B", duration_seconds=3.0),
            ],
        )

        dag, _ = build_episode_dag(ep, series)
        sub_node = dag.nodes["subtitle_gen"]

        assert sub_node.task_type == TaskType.SUBTITLE_GEN
        assert "tts_s01" in sub_node.depends_on
        assert "tts_s02" in sub_node.depends_on
        assert len(sub_node.params["scenes"]) == 2

    def test_compose_depends_on_subtitle_gen(self):
        series = DramaSeries(title="Test", model_id="mock")
        ep = Episode(
            number=1,
            scenes=[
                DramaScene(scene_id="s01", dialogue="台词", duration_seconds=5.0),
            ],
        )

        dag, _ = build_episode_dag(ep, series)
        compose_node = dag.nodes["compose"]

        assert compose_node.params["transition"] == "dissolve"
        assert len(compose_node.params["scenes"]) == 1
        assert "subtitle_gen" in compose_node.depends_on
        assert "music" in compose_node.depends_on

    def test_video_node_has_aspect_ratio(self):
        series = DramaSeries(title="Test", model_id="mock", aspect_ratio="9:16")
        ep = Episode(
            number=1,
            scenes=[
                DramaScene(scene_id="s01", visual_prompt="test", duration_seconds=5.0),
            ],
        )

        dag, _ = build_episode_dag(ep, series)
        video_node = dag.nodes["video_s01"]

        assert video_node.params["aspect_ratio"] == "9:16"

    def test_metadata_includes_language(self):
        series = DramaSeries(title="Test", model_id="mock", language="ja")
        ep = Episode(
            number=1,
            scenes=[DramaScene(scene_id="s01", duration_seconds=5.0)],
        )

        _, state = build_episode_dag(ep, series)

        assert state.metadata["language"] == "ja"


# ---------------------------------------------------------------------------
# 9:16 aspect ratio in VideoGenerator
# ---------------------------------------------------------------------------


class TestAspectRatio:
    def test_aspect_ratio_resolution_mapping(self):
        from videoclaw.generation.video import _ASPECT_TO_RESOLUTION

        assert _ASPECT_TO_RESOLUTION["9:16"] == (720, 1280)
        assert _ASPECT_TO_RESOLUTION["16:9"] == (1280, 720)
        assert _ASPECT_TO_RESOLUTION["1:1"] == (1024, 1024)

    @pytest.mark.asyncio
    async def test_generate_shot_uses_aspect_ratio(self):
        """VideoGenerator.generate_shot should pass correct width/height for 9:16."""
        from videoclaw.generation.video import VideoGenerator
        from videoclaw.models.protocol import GenerationRequest, GenerationResult

        captured_request = None

        class CapturingAdapter:
            model_id = "test"

            async def generate(self, request: GenerationRequest) -> GenerationResult:
                nonlocal captured_request
                captured_request = request
                return GenerationResult(
                    video_data=b"fake",
                    duration_seconds=request.duration_seconds,
                    model_id="test",
                )

        mock_router = AsyncMock()
        mock_router.select = AsyncMock(return_value=CapturingAdapter())

        gen = VideoGenerator(router=mock_router)
        shot = Shot(shot_id="s1", prompt="test", duration_seconds=5.0)

        await gen.generate_shot(shot, aspect_ratio="9:16")

        assert captured_request is not None
        assert captured_request.width == 720
        assert captured_request.height == 1280

    @pytest.mark.asyncio
    async def test_generate_shot_default_aspect_ratio(self):
        """Without aspect_ratio, should default to 16:9 (1280x720)."""
        from videoclaw.generation.video import VideoGenerator
        from videoclaw.models.protocol import GenerationRequest, GenerationResult

        captured_request = None

        class CapturingAdapter:
            model_id = "test"

            async def generate(self, request: GenerationRequest) -> GenerationResult:
                nonlocal captured_request
                captured_request = request
                return GenerationResult(
                    video_data=b"fake",
                    duration_seconds=request.duration_seconds,
                    model_id="test",
                )

        mock_router = AsyncMock()
        mock_router.select = AsyncMock(return_value=CapturingAdapter())

        gen = VideoGenerator(router=mock_router)
        shot = Shot(shot_id="s1", prompt="test", duration_seconds=5.0)

        await gen.generate_shot(shot)

        assert captured_request.width == 1280
        assert captured_request.height == 720


# ---------------------------------------------------------------------------
# Handler: _handle_script_gen
# ---------------------------------------------------------------------------


class TestScriptGenHandler:
    @pytest.mark.asyncio
    async def test_skips_when_script_exists(self, tmp_path):
        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(prompt="test", script="existing script")
        dag = DAG()
        node = TaskNode(node_id="script_gen", task_type=TaskType.SCRIPT_GEN)
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)
        result = await executor._handle_script_gen(node, state)

        assert result["script"] == "existing script"

    @pytest.mark.asyncio
    async def test_calls_generator_when_no_script(self, tmp_path):
        from videoclaw.generation.script import Script, ScriptSection

        mock_script = Script(
            title="Test",
            sections=[ScriptSection(text="Hello", duration_seconds=5.0)],
            voice_over_text="Hello",
            total_duration=5.0,
        )

        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(prompt="test", script=None)
        dag = DAG()
        node = TaskNode(node_id="script_gen", task_type=TaskType.SCRIPT_GEN)
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        with patch("videoclaw.generation.script.ScriptGenerator") as MockGen:
            MockGen.return_value.generate = AsyncMock(return_value=mock_script)
            result = await executor._handle_script_gen(node, state)

        assert state.script == "Hello"
        assert result["sections"] == 1


# ---------------------------------------------------------------------------
# Handler: _handle_storyboard
# ---------------------------------------------------------------------------


class TestStoryboardHandler:
    @pytest.mark.asyncio
    async def test_skips_when_storyboard_exists(self, tmp_path):
        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(
            prompt="test",
            storyboard=[Shot(shot_id="s1"), Shot(shot_id="s2")],
        )
        dag = DAG()
        node = TaskNode(node_id="storyboard", task_type=TaskType.STORYBOARD)
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)
        result = await executor._handle_storyboard(node, state)

        assert result["shot_count"] == 2


# ---------------------------------------------------------------------------
# Handler: _handle_tts
# ---------------------------------------------------------------------------


class TestTTSHandler:
    @pytest.mark.asyncio
    async def test_drama_mode_per_scene_tts(self, tmp_path):
        from videoclaw.drama.models import AudioSegment, AudioType, LineType

        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(prompt="test")
        dag = DAG()
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "language": "zh",
                "scenes": [
                    {"scene_id": "s01", "dialogue": "你好", "narration": "",
                     "voice": None, "speaking_character": "林薇",
                     "emotion": "", "dialogue_line_type": "dialogue"},
                    {"scene_id": "s02", "dialogue": "", "narration": "旁白",
                     "voice": None, "speaking_character": "",
                     "emotion": "", "dialogue_line_type": "dialogue"},
                ],
            },
        )
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        mock_segments = [
            AudioSegment(
                scene_id="s01", audio_type=AudioType.DIALOGUE,
                text="你好", character_name="林薇",
                audio_path=str(tmp_path / "audio" / "line_0000_林薇.mp3"),
                line_type=LineType.DIALOGUE,
            ),
            AudioSegment(
                scene_id="s02", audio_type=AudioType.NARRATION,
                text="旁白", character_name="narrator",
                audio_path=str(tmp_path / "audio" / "line_0001_narrator.mp3"),
                line_type=LineType.NARRATION,
            ),
        ]

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_instance = MockTTS.return_value
            mock_instance.generate_multi_role = AsyncMock(return_value=mock_segments)
            result = await executor._handle_tts(node, state)

        assert result["count"] == 2
        mock_instance.generate_multi_role.assert_called_once()

        # Check assets stored
        audio_entries = json.loads(state.assets["tts_audio"])
        assert len(audio_entries) == 2
        assert audio_entries[0]["type"] == "dialogue"
        assert audio_entries[1]["type"] == "narration"

    @pytest.mark.asyncio
    async def test_generic_mode_single_voiceover(self, tmp_path):
        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(prompt="test", script="Full script text")
        dag = DAG()
        node = TaskNode(node_id="tts", task_type=TaskType.TTS, params={})
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_instance = MockTTS.return_value
            mock_instance.generate_voiceover = AsyncMock(side_effect=lambda text, path, **kw: path)
            result = await executor._handle_tts(node, state)

        assert result["count"] == 1
        audio_entries = json.loads(state.assets["tts_audio"])
        assert audio_entries[0]["type"] == "voiceover"


# ---------------------------------------------------------------------------
# Handler: _handle_per_scene_tts
# ---------------------------------------------------------------------------


class TestPerSceneTTSHandler:
    @pytest.mark.asyncio
    async def test_single_scene_tts(self, tmp_path):
        from videoclaw.drama.models import AudioSegment, AudioType, LineType

        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(prompt="test")
        dag = DAG()
        node = TaskNode(
            node_id="tts_s01",
            task_type=TaskType.PER_SCENE_TTS,
            params={
                "language": "zh",
                "scene": {
                    "scene_id": "s01",
                    "dialogue": "你确定？",
                    "narration": "旁白文本",
                    "speaking_character": "林薇",
                    "emotion": "smug",
                    "dialogue_line_type": "dialogue",
                },
            },
        )
        dag.add_node(node)

        mock_segments = [
            AudioSegment(
                scene_id="s01", audio_type=AudioType.DIALOGUE,
                text="你确定？", character_name="林薇",
                audio_path=str(tmp_path / "audio" / "line_0000_林薇.mp3"),
                line_type=LineType.DIALOGUE,
            ),
            AudioSegment(
                scene_id="s01", audio_type=AudioType.NARRATION,
                text="旁白文本", character_name="narrator",
                audio_path=str(tmp_path / "audio" / "line_0001_narrator.mp3"),
                line_type=LineType.NARRATION,
            ),
        ]

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_instance = MockTTS.return_value
            mock_instance.generate_multi_role = AsyncMock(return_value=mock_segments)
            result = await executor._handle_per_scene_tts(node, state)

        assert result["scene_id"] == "s01"
        assert result["segments"] == 2
        # Per-scene data stored in assets
        assert "tts_scene_s01" in state.assets

    @pytest.mark.asyncio
    async def test_empty_scene_no_audio(self, tmp_path):
        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(prompt="test")
        dag = DAG()
        node = TaskNode(
            node_id="tts_s02",
            task_type=TaskType.PER_SCENE_TTS,
            params={
                "language": "zh",
                "scene": {
                    "scene_id": "s02",
                    "dialogue": "",
                    "narration": "",
                    "speaking_character": "",
                    "emotion": "",
                    "dialogue_line_type": "dialogue",
                },
            },
        )
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_instance = MockTTS.return_value
            mock_instance.generate_multi_role = AsyncMock(return_value=[])
            result = await executor._handle_per_scene_tts(node, state)

        assert result["segments"] == 0


# ---------------------------------------------------------------------------
# Handler: _handle_subtitle_gen
# ---------------------------------------------------------------------------


class TestSubtitleGenHandler:
    @pytest.mark.asyncio
    async def test_aggregates_per_scene_audio_and_generates_subtitles(self, tmp_path):
        sm = StateManager(projects_dir=tmp_path)

        project_dir = tmp_path / "test_project"
        project_dir.mkdir(parents=True)

        state = ProjectState(
            project_id="test_project",
            prompt="test",
            assets={
                "tts_scene_s01": json.dumps([{
                    "segment_id": "a1", "scene_id": "s01",
                    "audio_type": "dialogue", "line_type": "dialogue",
                    "text": "你好", "character_name": "林薇",
                    "audio_path": "/fake/path.mp3",
                    "start_time": 0.0, "duration_seconds": 3.0, "volume": 1.0,
                }]),
                "tts_scene_s02": json.dumps([{
                    "segment_id": "a2", "scene_id": "s02",
                    "audio_type": "narration", "line_type": "narration",
                    "text": "旁白", "character_name": "narrator",
                    "audio_path": "/fake/path2.mp3",
                    "start_time": 0.0, "duration_seconds": 2.0, "volume": 1.0,
                }]),
            },
        )

        dag = DAG()
        node = TaskNode(
            node_id="subtitle_gen",
            task_type=TaskType.SUBTITLE_GEN,
            params={
                "scenes": [
                    {"scene_id": "s01", "dialogue": "你好", "duration_seconds": 5.0,
                     "speaking_character": "林薇"},
                    {"scene_id": "s02", "dialogue": "", "narration": "旁白",
                     "duration_seconds": 3.0, "speaking_character": ""},
                ],
            },
        )
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)
        result = await executor._handle_subtitle_gen(node, state)

        assert result["segments_used"] == 2
        assert "subtitles" in state.assets
        assert "audio_manifest" in state.assets
        assert "tts_audio" in state.assets

    @pytest.mark.asyncio
    async def test_skips_when_no_scenes(self, tmp_path):
        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(prompt="test")
        dag = DAG()
        node = TaskNode(
            node_id="subtitle_gen",
            task_type=TaskType.SUBTITLE_GEN,
            params={},
        )
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)
        result = await executor._handle_subtitle_gen(node, state)

        assert result["status"] == "skipped"


# ---------------------------------------------------------------------------
# Handler: _handle_music
# ---------------------------------------------------------------------------


class TestMusicHandler:
    @pytest.mark.asyncio
    async def test_music_generates_track(self, tmp_path):
        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(prompt="test")
        state.storyboard = [
            Shot(shot_id="s1", prompt="test", duration_seconds=5.0),
        ]
        dag = DAG()
        node = TaskNode(node_id="music", task_type=TaskType.MUSIC)
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        # Mock MusicManager to avoid ffmpeg dependency
        mock_manager = AsyncMock()
        music_file = tmp_path / state.project_id / "audio" / "bgm.aac"
        music_file.parent.mkdir(parents=True, exist_ok=True)
        music_file.write_bytes(b"fake")
        mock_manager.generate_bgm.return_value = music_file

        with patch("videoclaw.generation.audio.music.MusicManager", return_value=mock_manager):
            result = await executor._handle_music(node, state)

        mock_manager.generate_bgm.assert_called_once()
        assert "music_path" in result
        assert result["duration"] == 5.0
        assert state.assets["music"] == str(music_file)


# ---------------------------------------------------------------------------
# Handler: _handle_compose
# ---------------------------------------------------------------------------


class TestComposeHandler:
    @pytest.mark.asyncio
    async def test_compose_with_videos_and_upstream_subtitles(self, tmp_path):
        sm = StateManager(projects_dir=tmp_path)

        # Create fake video files
        project_dir = tmp_path / "test_project"
        shots_dir = project_dir / "shots"
        shots_dir.mkdir(parents=True)
        (shots_dir / "s01.mp4").write_bytes(b"fake_video_1")
        (shots_dir / "s02.mp4").write_bytes(b"fake_video_2")

        # Create fake audio
        audio_dir = project_dir / "audio"
        audio_dir.mkdir(parents=True)
        audio_path = audio_dir / "s01_dialogue.mp3"
        audio_path.write_bytes(b"fake_audio")

        # Create fake subtitles (produced by upstream subtitle_gen node)
        subtitle_path = project_dir / "subtitles.ass"
        subtitle_path.write_text("[Script Info]\nTitle: Test\n", encoding="utf-8")

        state = ProjectState(
            project_id="test_project",
            prompt="test",
            storyboard=[
                Shot(shot_id="s01", asset_path=str(shots_dir / "s01.mp4")),
                Shot(shot_id="s02", asset_path=str(shots_dir / "s02.mp4")),
            ],
            assets={
                "tts_audio": json.dumps([{"path": str(audio_path), "type": "dialogue"}]),
                "subtitles": str(subtitle_path),
            },
        )

        dag = DAG()
        node = TaskNode(
            node_id="compose",
            task_type=TaskType.COMPOSE,
            params={
                "transition": "dissolve",
                "scenes": [
                    {"dialogue": "你好", "duration_seconds": 5.0,
                     "speaking_character": "林薇", "transition": "dissolve"},
                ],
            },
        )
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)

        with patch("videoclaw.generation.compose.VideoComposer") as MockComposer:
            mock_instance = MockComposer.return_value
            mock_instance.compose = AsyncMock(return_value=project_dir / "composed.mp4")
            mock_instance.render_final = AsyncMock(return_value=project_dir / "composed_final.mp4")

            result = await executor._handle_compose(node, state)

        assert "composed_path" in result
        # Subtitles were read from upstream node, not generated inline
        assert state.assets["subtitles"] == str(subtitle_path)
        mock_instance.compose.assert_called_once()
        mock_instance.render_final.assert_called_once()


# ---------------------------------------------------------------------------
# Handler: _handle_render
# ---------------------------------------------------------------------------


class TestRenderHandler:
    @pytest.mark.asyncio
    async def test_render_copies_to_final(self, tmp_path):
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
        result = await executor._handle_render(node, state)

        final_path = Path(result["output_path"])
        assert final_path.name == "final.mp4"
        assert state.assets["final_video"] == result["output_path"]

    @pytest.mark.asyncio
    async def test_render_fails_without_composed(self, tmp_path):
        sm = StateManager(projects_dir=tmp_path)
        state = ProjectState(project_id="test_project", prompt="test")

        dag = DAG()
        node = TaskNode(node_id="render", task_type=TaskType.RENDER)
        dag.add_node(node)

        executor = DAGExecutor(dag=dag, state=state, state_manager=sm)
        with pytest.raises(ValueError, match="No composed video"):
            await executor._handle_render(node, state)
