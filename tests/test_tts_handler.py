"""Tests for the upgraded _handle_tts handler in DAGExecutor.

Verifies that drama mode uses generate_multi_role with proper DialogueLine
construction, VoiceProfile reconstruction, and EpisodeAudioManifest output,
while generic mode still falls back to generate_voiceover.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from videoclaw.core.executor import DAGExecutor
from videoclaw.core.planner import DAG, TaskNode, TaskType
from videoclaw.core.state import ProjectState, StateManager
from videoclaw.drama.models import (
    AudioSegment,
    AudioType,
    DialogueLine,
    EpisodeAudioManifest,
    LineType,
    VoiceProfile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor(state: ProjectState, tmp_path: Path) -> DAGExecutor:
    """Create a DAGExecutor with a minimal DAG and temp project dir."""
    dag = DAG()
    dag.add_node(TaskNode(node_id="tts", task_type=TaskType.TTS, params={}))
    state_mgr = StateManager(projects_dir=tmp_path)

    with patch("videoclaw.core.executor.get_config") as mock_cfg:
        cfg = MagicMock()
        cfg.projects_dir = tmp_path
        cfg.max_retries = 0
        mock_cfg.return_value = cfg
        executor = DAGExecutor(dag=dag, state=state, state_manager=state_mgr)

    return executor


def _make_scenes_data(
    *,
    dialogue: str = "你好",
    narration: str = "",
    scene_id: str = "s01",
    speaking_character: str = "林薇",
    emotion: str = "happy",
    dialogue_line_type: str = "dialogue",
) -> list[dict]:
    return [{
        "scene_id": scene_id,
        "dialogue": dialogue,
        "narration": narration,
        "speaking_character": speaking_character,
        "emotion": emotion,
        "dialogue_line_type": dialogue_line_type,
        "duration_seconds": 5.0,
        "voice": "Lively_Girl",
        "transition": "dissolve",
    }]


def _voice_map_dict() -> dict[str, dict]:
    """Return a raw voice_map as it would appear in state.metadata."""
    return {
        "林薇": VoiceProfile(
            voice_id="Lively_Girl", speed=1.05, pitch=2,
            emotion="happy", role_name="林薇",
        ).to_dict(),
        "narrator": VoiceProfile(
            voice_id="Calm_Woman", speed=0.95,
            role_name="narrator", line_type=LineType.NARRATION,
        ).to_dict(),
    }


# ---------------------------------------------------------------------------
# Tests: DialogueLine construction from scenes_data
# ---------------------------------------------------------------------------

class TestDialogueLineConstruction:
    """Verify that _handle_tts builds DialogueLine list correctly."""

    @pytest.mark.asyncio
    async def test_dialogue_line_from_scene(self, tmp_path):
        """Dialogue text creates a DialogueLine with correct speaker and type."""
        state = ProjectState(metadata={"voice_map": _voice_map_dict()})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(dialogue="你来了", narration=""),
                "language": "zh",
            },
        )

        mock_segments = [
            AudioSegment(
                scene_id="s01", audio_type=AudioType.DIALOGUE,
                text="你来了", character_name="林薇",
                audio_path=str(tmp_path / "audio" / "line_0000_林薇.mp3"),
                line_type=LineType.DIALOGUE,
            ),
        ]

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=mock_segments)

            result = await executor._handle_tts(node, state)

            # Verify generate_multi_role was called
            mock_tts.generate_multi_role.assert_called_once()
            lines_arg = mock_tts.generate_multi_role.call_args[0][0]

            assert len(lines_arg) == 1
            assert lines_arg[0].text == "你来了"
            assert lines_arg[0].speaker == "林薇"
            assert lines_arg[0].line_type == LineType.DIALOGUE
            assert lines_arg[0].scene_id == "s01"
            assert lines_arg[0].emotion_hint == "happy"

    @pytest.mark.asyncio
    async def test_narration_creates_narrator_line(self, tmp_path):
        """Narration text creates a DialogueLine with speaker='narrator' and NARRATION type."""
        state = ProjectState(metadata={"voice_map": _voice_map_dict()})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(dialogue="", narration="夜幕降临"),
                "language": "zh",
            },
        )

        mock_segments = [
            AudioSegment(
                scene_id="s01", audio_type=AudioType.NARRATION,
                text="夜幕降临", character_name="narrator",
                audio_path=str(tmp_path / "audio" / "line_0000_narrator.mp3"),
                line_type=LineType.NARRATION,
            ),
        ]

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=mock_segments)

            await executor._handle_tts(node, state)

            lines_arg = mock_tts.generate_multi_role.call_args[0][0]
            assert len(lines_arg) == 1
            assert lines_arg[0].speaker == "narrator"
            assert lines_arg[0].line_type == LineType.NARRATION
            assert lines_arg[0].emotion_hint is None

    @pytest.mark.asyncio
    async def test_both_dialogue_and_narration(self, tmp_path):
        """Scene with both dialogue and narration produces two DialogueLine entries."""
        state = ProjectState(metadata={"voice_map": _voice_map_dict()})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(
                    dialogue="你来了", narration="夜幕降临",
                ),
                "language": "zh",
            },
        )

        mock_segments = [
            AudioSegment(scene_id="s01", audio_type=AudioType.DIALOGUE,
                         text="你来了", character_name="林薇",
                         line_type=LineType.DIALOGUE),
            AudioSegment(scene_id="s01", audio_type=AudioType.NARRATION,
                         text="夜幕降临", character_name="narrator",
                         line_type=LineType.NARRATION),
        ]

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=mock_segments)

            await executor._handle_tts(node, state)

            lines_arg = mock_tts.generate_multi_role.call_args[0][0]
            assert len(lines_arg) == 2
            assert lines_arg[0].line_type == LineType.DIALOGUE
            assert lines_arg[1].line_type == LineType.NARRATION


# ---------------------------------------------------------------------------
# Tests: VoiceProfile reconstruction
# ---------------------------------------------------------------------------

class TestVoiceMapReconstruction:
    """Verify voice_map is reconstructed from state.metadata or node.params."""

    @pytest.mark.asyncio
    async def test_voice_map_from_metadata(self, tmp_path):
        """voice_map is reconstructed from state.metadata when not in node.params."""
        vm = _voice_map_dict()
        state = ProjectState(metadata={"voice_map": vm})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(),
                "language": "zh",
            },
        )

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=[])

            await executor._handle_tts(node, state)

            voice_map_arg = mock_tts.generate_multi_role.call_args[0][1]
            assert "林薇" in voice_map_arg
            assert isinstance(voice_map_arg["林薇"], VoiceProfile)
            assert voice_map_arg["林薇"].voice_id == "Lively_Girl"
            assert voice_map_arg["林薇"].speed == 1.05

    @pytest.mark.asyncio
    async def test_voice_map_from_node_params(self, tmp_path):
        """voice_map in node.params takes priority over state.metadata."""
        node_vm = {
            "林薇": VoiceProfile(voice_id="CustomVoice", speed=0.8).to_dict(),
        }
        state = ProjectState(metadata={"voice_map": _voice_map_dict()})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(),
                "language": "zh",
                "voice_map": node_vm,
            },
        )

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=[])

            await executor._handle_tts(node, state)

            voice_map_arg = mock_tts.generate_multi_role.call_args[0][1]
            assert voice_map_arg["林薇"].voice_id == "CustomVoice"
            assert voice_map_arg["林薇"].speed == 0.8

    @pytest.mark.asyncio
    async def test_empty_voice_map_fallback(self, tmp_path):
        """When voice_map is missing/empty, an empty dict is passed."""
        state = ProjectState(metadata={})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(),
                "language": "zh",
            },
        )

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=[])

            await executor._handle_tts(node, state)

            voice_map_arg = mock_tts.generate_multi_role.call_args[0][1]
            assert voice_map_arg == {}


# ---------------------------------------------------------------------------
# Tests: generate_multi_role called (not generate_voiceover) in drama mode
# ---------------------------------------------------------------------------

class TestMultiRoleDispatch:
    """Verify that drama mode calls generate_multi_role, not generate_voiceover."""

    @pytest.mark.asyncio
    async def test_drama_mode_calls_multi_role(self, tmp_path):
        """When scenes are present, generate_multi_role is called."""
        state = ProjectState(metadata={"voice_map": _voice_map_dict()})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(),
                "language": "zh",
            },
        )

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=[])
            mock_tts.generate_voiceover = AsyncMock()

            await executor._handle_tts(node, state)

            mock_tts.generate_multi_role.assert_called_once()
            mock_tts.generate_voiceover.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: AudioSegment / EpisodeAudioManifest output
# ---------------------------------------------------------------------------

class TestManifestOutput:
    """Verify that AudioSegment and EpisodeAudioManifest are produced."""

    @pytest.mark.asyncio
    async def test_manifest_stored_in_state(self, tmp_path):
        """audio_manifest is stored as JSON in state.assets."""
        state = ProjectState(
            metadata={
                "voice_map": _voice_map_dict(),
                "episode_id": "ep001",
            },
        )
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(),
                "language": "zh",
            },
        )

        mock_segments = [
            AudioSegment(
                scene_id="s01", audio_type=AudioType.DIALOGUE,
                text="你好", character_name="林薇",
                audio_path="/tmp/audio/line_0000_林薇.mp3",
                line_type=LineType.DIALOGUE,
                duration_seconds=2.5,
            ),
        ]

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=mock_segments)

            result = await executor._handle_tts(node, state)

        # Check audio_manifest is in state.assets
        assert "audio_manifest" in state.assets
        manifest_data = json.loads(state.assets["audio_manifest"])
        assert manifest_data["episode_id"] == "ep001"
        assert len(manifest_data["segments"]) == 1
        assert manifest_data["segments"][0]["audio_type"] == "dialogue"
        assert manifest_data["segments"][0]["character_name"] == "林薇"
        assert manifest_data["total_duration"] == 2.5

    @pytest.mark.asyncio
    async def test_tts_audio_backward_compat(self, tmp_path):
        """tts_audio is still populated for compose handler compatibility."""
        state = ProjectState(metadata={"voice_map": _voice_map_dict()})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(),
                "language": "zh",
            },
        )

        mock_segments = [
            AudioSegment(
                scene_id="s01", audio_type=AudioType.DIALOGUE,
                text="你好", character_name="林薇",
                audio_path="/tmp/audio/line_0000_林薇.mp3",
                line_type=LineType.DIALOGUE,
            ),
        ]

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=mock_segments)

            await executor._handle_tts(node, state)

        assert "tts_audio" in state.assets
        audio_paths = json.loads(state.assets["tts_audio"])
        assert len(audio_paths) == 1
        assert audio_paths[0]["scene_id"] == "s01"
        assert audio_paths[0]["type"] == "dialogue"
        assert audio_paths[0]["path"] == "/tmp/audio/line_0000_林薇.mp3"


# ---------------------------------------------------------------------------
# Tests: dialogue_line_type respected
# ---------------------------------------------------------------------------

class TestDialogueLineType:
    """Verify that dialogue_line_type is correctly mapped to LineType."""

    @pytest.mark.asyncio
    async def test_inner_monologue_type(self, tmp_path):
        """dialogue_line_type='inner_monologue' maps to LineType.INNER_MONOLOGUE."""
        state = ProjectState(metadata={"voice_map": _voice_map_dict()})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(
                    dialogue="他不可能知道",
                    dialogue_line_type="inner_monologue",
                    speaking_character="萧衍",
                ),
                "language": "zh",
            },
        )

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=[])

            await executor._handle_tts(node, state)

            lines_arg = mock_tts.generate_multi_role.call_args[0][0]
            assert lines_arg[0].line_type == LineType.INNER_MONOLOGUE

    @pytest.mark.asyncio
    async def test_invalid_line_type_defaults_to_dialogue(self, tmp_path):
        """Invalid dialogue_line_type falls back to LineType.DIALOGUE."""
        state = ProjectState(metadata={"voice_map": _voice_map_dict()})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(
                    dialogue="你好",
                    dialogue_line_type="unknown_type",
                ),
                "language": "zh",
            },
        )

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=[])

            await executor._handle_tts(node, state)

            lines_arg = mock_tts.generate_multi_role.call_args[0][0]
            assert lines_arg[0].line_type == LineType.DIALOGUE


# ---------------------------------------------------------------------------
# Tests: Generic mode backward compat
# ---------------------------------------------------------------------------

class TestGenericModeBackwardCompat:
    """Verify generic mode still uses generate_voiceover."""

    @pytest.mark.asyncio
    async def test_generic_mode_uses_voiceover(self, tmp_path):
        """When no scenes, generate_voiceover is called with full script."""
        state = ProjectState(script="这是一段旁白文本")
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={"language": "zh"},
        )

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_voiceover = AsyncMock(
                return_value=tmp_path / "audio" / "voiceover.mp3",
            )
            mock_tts.generate_multi_role = AsyncMock()

            result = await executor._handle_tts(node, state)

            mock_tts.generate_voiceover.assert_called_once()
            mock_tts.generate_multi_role.assert_not_called()
            assert result["count"] == 1
            assert "tts_audio" in state.assets

    @pytest.mark.asyncio
    async def test_no_scenes_no_script(self, tmp_path):
        """When neither scenes nor script, handler returns empty result."""
        state = ProjectState()
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={},
        )

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            result = await executor._handle_tts(node, state)

        assert result["count"] == 0
        assert json.loads(state.assets["tts_audio"]) == []


# ---------------------------------------------------------------------------
# Tests: Missing speaker fallback
# ---------------------------------------------------------------------------

class TestMissingSpeakerFallback:
    """Verify behavior when speaking_character is empty."""

    @pytest.mark.asyncio
    async def test_empty_speaker_defaults_to_narrator(self, tmp_path):
        """When speaking_character is empty, speaker defaults to 'narrator'."""
        state = ProjectState(metadata={"voice_map": _voice_map_dict()})
        executor = _make_executor(state, tmp_path)
        node = TaskNode(
            node_id="tts",
            task_type=TaskType.TTS,
            params={
                "scenes": _make_scenes_data(
                    dialogue="你好",
                    speaking_character="",
                ),
                "language": "zh",
            },
        )

        with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS:
            mock_tts = MockTTS.return_value
            mock_tts.generate_multi_role = AsyncMock(return_value=[])

            await executor._handle_tts(node, state)

            lines_arg = mock_tts.generate_multi_role.call_args[0][0]
            assert lines_arg[0].speaker == "narrator"


# ---------------------------------------------------------------------------
# Tests: Runner voice_map in TTS node params
# ---------------------------------------------------------------------------

class TestRunnerVoiceMapInParams:
    """Verify that _build_drama_dag includes voice_map in TTS node params."""

    def test_tts_node_has_voice_map(self):
        """TTS node params should contain voice_map for easy handler access."""
        from videoclaw.drama.models import (
            Character,
            DramaScene,
            DramaSeries,
            Episode,
            VoiceProfile,
        )
        from videoclaw.drama.runner import build_episode_dag

        series = DramaSeries(
            title="Test",
            characters=[
                Character(
                    name="林薇",
                    voice_profile=VoiceProfile(voice_id="Lively_Girl"),
                ),
            ],
            episodes=[],
        )
        episode = Episode(
            number=1,
            title="Ep1",
            scenes=[
                DramaScene(
                    scene_id="s01",
                    dialogue="你好",
                    speaking_character="林薇",
                ),
            ],
        )

        dag, state = build_episode_dag(episode, series)

        tts_node = dag.nodes["tts"]
        assert "voice_map" in tts_node.params
        assert "林薇" in tts_node.params["voice_map"]
        assert tts_node.params["voice_map"]["林薇"]["voice_id"] == "Lively_Girl"
