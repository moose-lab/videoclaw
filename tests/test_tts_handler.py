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
from videoclaw.generation.audio.tts import (
    EMOTION_VOICE_MAP,
    EmotionParams,
    ProsodyHints,
    ResolvedVoice,
    analyze_text_prosody,
    resolve_emotion,
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

        # Per-scene TTS nodes now carry voice_map
        tts_node = dag.nodes["tts_s01"]
        assert "voice_map" in tts_node.params
        assert "林薇" in tts_node.params["voice_map"]
        assert tts_node.params["voice_map"]["林薇"]["voice_id"] == "Lively_Girl"


# ---------------------------------------------------------------------------
# Tests: EMOTION_VOICE_MAP and resolve_emotion
# ---------------------------------------------------------------------------

# The 32 valid scene emotions (from test_drama_e2e.py quality validator)
VALID_EMOTIONS_32 = {
    "tense", "anxious", "dread", "suspense",
    "angry", "furious", "resentful", "defiant",
    "sad", "heartbroken", "grieving", "melancholy",
    "shock", "disbelief", "stunned", "revelation",
    "warm", "tender", "nostalgic", "grateful",
    "sweet", "flirty", "blissful", "intimate",
    "fear", "panic", "horror", "uneasy",
    "triumphant", "smug", "vindicated", "proud",
}

WAVESPEED_EMOTIONS = {"happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"}


class TestEmotionVoiceMap:
    """Verify EMOTION_VOICE_MAP covers all 32 scene emotions."""

    def test_all_32_emotions_have_entries(self):
        """Every valid scene emotion must have a mapping entry."""
        missing = VALID_EMOTIONS_32 - set(EMOTION_VOICE_MAP.keys())
        assert not missing, f"Missing emotion mappings: {missing}"

    def test_all_mapped_emotions_are_wavespeed_compatible(self):
        """Every mapped wavespeed_emotion must be one of WaveSpeed's 7 values."""
        for emotion, params in EMOTION_VOICE_MAP.items():
            assert params.wavespeed_emotion in WAVESPEED_EMOTIONS, (
                f"{emotion} maps to '{params.wavespeed_emotion}' which is not a valid WaveSpeed emotion"
            )

    def test_emotion_groups_map_to_expected_wavespeed(self):
        """Verify each emotion group maps to the expected WaveSpeed emotion."""
        group_expectations = {
            "fearful": {"tense", "anxious", "dread", "suspense", "fear", "panic", "horror", "uneasy"},
            "angry": {"angry", "furious", "resentful", "defiant"},
            "sad": {"sad", "heartbroken", "grieving", "melancholy"},
            "surprised": {"shock", "disbelief", "stunned", "revelation"},
            "happy": {"warm", "tender", "nostalgic", "grateful",
                      "sweet", "flirty", "blissful", "intimate",
                      "triumphant", "smug", "vindicated", "proud"},
        }
        for ws_emotion, scene_emotions in group_expectations.items():
            for se in scene_emotions:
                assert EMOTION_VOICE_MAP[se].wavespeed_emotion == ws_emotion, (
                    f"{se} should map to {ws_emotion}, got {EMOTION_VOICE_MAP[se].wavespeed_emotion}"
                )


class TestResolveEmotion:
    """Verify resolve_emotion applies deltas correctly and returns ResolvedVoice."""

    def test_returns_resolved_voice_type(self):
        """resolve_emotion returns a ResolvedVoice NamedTuple, not a plain tuple."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="neutral")
        result = resolve_emotion("panic", profile)
        assert isinstance(result, ResolvedVoice)

    def test_known_emotion_applies_deltas(self):
        """A known emotion applies speed/pitch/volume deltas to the base profile."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="neutral")
        result = resolve_emotion("panic", profile)

        assert result.emotion == "fearful"
        assert result.speed == pytest.approx(1.0 + 0.15)   # panic: +0.15
        assert result.pitch == 0 + 3                         # panic: +3
        assert result.volume == pytest.approx(1.0 + 0.2)    # panic: +0.2

    def test_deltas_stack_on_character_base(self):
        """Deltas are additive to the character's existing VoiceProfile values."""
        profile = VoiceProfile(speed=1.05, pitch=2, volume=1.0, emotion="happy")
        result = resolve_emotion("dread", profile)

        assert result.emotion == "fearful"
        assert result.speed == pytest.approx(1.05 + (-0.10))  # 0.95
        assert result.pitch == 2 + (-2)                         # 0
        assert result.volume == pytest.approx(1.0 + (-0.1))    # 0.9

    def test_unknown_emotion_passthrough(self):
        """Unknown emotions return the base profile values unchanged."""
        profile = VoiceProfile(speed=0.9, pitch=-1, volume=1.2, emotion="neutral")
        result = resolve_emotion("some_future_emotion", profile)

        assert result.emotion == "some_future_emotion"
        assert result.speed == 0.9
        assert result.pitch == -1
        assert result.volume == 1.2
        assert result.pause_before_ms == 0
        assert result.breathiness == 0.0

    def test_wavespeed_emotion_passthrough(self):
        """WaveSpeed-native emotions not in the 32-set pass through unchanged."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="neutral")
        result = resolve_emotion("disgusted", profile)

        assert result.emotion == "disgusted"
        assert result.speed == 1.0
        assert result.pitch == 0
        assert result.volume == 1.0

    def test_empty_hint_uses_profile_emotion(self):
        """Empty emotion_hint falls back to the profile's base emotion."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="happy")
        result = resolve_emotion("", profile)

        assert result.emotion == "happy"
        assert result.speed == 1.0

    def test_none_coerced_to_empty(self):
        """None emotion_hint also falls back to profile emotion (via empty string)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="sad")
        result = resolve_emotion("", profile)
        assert result.emotion == "sad"

    def test_intensity_gradient_within_group(self):
        """Within a group, more intense emotions should have larger absolute deltas."""
        uneasy = EMOTION_VOICE_MAP["uneasy"]
        fear = EMOTION_VOICE_MAP["fear"]
        panic = EMOTION_VOICE_MAP["panic"]

        assert panic.speed_delta > fear.speed_delta >= uneasy.speed_delta
        assert panic.pitch_delta > fear.pitch_delta >= uneasy.pitch_delta


# ---------------------------------------------------------------------------
# Tests: ResolvedVoice and extended EmotionParams fields (Task 3.1.1)
# ---------------------------------------------------------------------------

class TestEmotionParamsExtended:
    """Verify EmotionParams has intensity, pause_before_ms, breathiness fields."""

    def test_emotion_params_has_intensity_field(self):
        """EmotionParams includes an intensity field with default 1.0."""
        params = EmotionParams("fearful", -0.05, -1, 0.0)
        assert params.intensity == 1.0

    def test_emotion_params_has_pause_before_ms_field(self):
        """EmotionParams includes a pause_before_ms field with default 0."""
        params = EmotionParams("fearful", -0.05, -1, 0.0)
        assert params.pause_before_ms == 0

    def test_emotion_params_has_breathiness_field(self):
        """EmotionParams includes a breathiness field with default 0.0."""
        params = EmotionParams("fearful", -0.05, -1, 0.0)
        assert params.breathiness == 0.0

    def test_emotion_params_custom_values(self):
        """EmotionParams accepts custom intensity, pause, breathiness."""
        params = EmotionParams(
            "fearful", -0.10, -2, -0.1,
            intensity=0.9, pause_before_ms=500, breathiness=0.3,
        )
        assert params.intensity == 0.9
        assert params.pause_before_ms == 500
        assert params.breathiness == 0.3

    def test_all_emotions_have_valid_intensity(self):
        """Every EMOTION_VOICE_MAP entry has intensity in [0.0, 1.0]."""
        for emotion, params in EMOTION_VOICE_MAP.items():
            assert 0.0 <= params.intensity <= 1.0, (
                f"{emotion} intensity {params.intensity} out of range [0.0, 1.0]"
            )

    def test_all_emotions_have_nonneg_pause(self):
        """Every EMOTION_VOICE_MAP entry has non-negative pause_before_ms."""
        for emotion, params in EMOTION_VOICE_MAP.items():
            assert params.pause_before_ms >= 0, (
                f"{emotion} has negative pause_before_ms: {params.pause_before_ms}"
            )

    def test_all_emotions_have_valid_breathiness(self):
        """Every EMOTION_VOICE_MAP entry has breathiness in [-1.0, 1.0]."""
        for emotion, params in EMOTION_VOICE_MAP.items():
            assert -1.0 <= params.breathiness <= 1.0, (
                f"{emotion} breathiness {params.breathiness} out of range [-1.0, 1.0]"
            )


class TestResolvedVoiceType:
    """Verify ResolvedVoice NamedTuple structure."""

    def test_resolved_voice_fields(self):
        """ResolvedVoice has all 6 expected fields."""
        rv = ResolvedVoice(
            emotion="fearful", speed=1.15, pitch=3, volume=1.2,
            pause_before_ms=500, breathiness=0.3,
        )
        assert rv.emotion == "fearful"
        assert rv.speed == 1.15
        assert rv.pitch == 3
        assert rv.volume == 1.2
        assert rv.pause_before_ms == 500
        assert rv.breathiness == 0.3

    def test_resolved_voice_defaults(self):
        """ResolvedVoice pause_before_ms and breathiness default to zero."""
        rv = ResolvedVoice(emotion="happy", speed=1.0, pitch=0, volume=1.0)
        assert rv.pause_before_ms == 0
        assert rv.breathiness == 0.0


class TestIntensityScaling:
    """Verify that intensity_override scales deltas in resolve_emotion."""

    def test_intensity_override_scales_speed_delta(self):
        """intensity_override=0.5 halves the speed delta."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="neutral")
        result = resolve_emotion("panic", profile, intensity_override=0.5)
        # panic speed_delta=0.15, scaled by 0.5 → 0.075
        assert result.speed == pytest.approx(1.0 + 0.15 * 0.5)

    def test_intensity_override_scales_pitch_delta(self):
        """intensity_override=0.5 halves the pitch delta (rounded)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="neutral")
        result = resolve_emotion("panic", profile, intensity_override=0.5)
        # panic pitch_delta=3, scaled by 0.5 → round(1.5) = 2
        assert result.pitch == round(3 * 0.5)

    def test_intensity_override_scales_volume_delta(self):
        """intensity_override=0.5 halves the volume delta."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="neutral")
        result = resolve_emotion("panic", profile, intensity_override=0.5)
        # panic volume_delta=0.2, scaled by 0.5 → 0.1
        assert result.volume == pytest.approx(1.0 + 0.2 * 0.5)

    def test_intensity_zero_means_no_deltas(self):
        """intensity_override=0.0 produces zero deltas (base profile values)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="neutral")
        result = resolve_emotion("furious", profile, intensity_override=0.0)
        # Deltas are zeroed, but emotion is still mapped
        assert result.emotion == "angry"
        assert result.speed == pytest.approx(1.0)
        assert result.pitch == 0
        assert result.volume == pytest.approx(1.0)

    def test_intensity_one_is_full_deltas(self):
        """intensity_override=1.0 gives full deltas (same as no override)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="neutral")
        full = resolve_emotion("furious", profile, intensity_override=1.0)
        default = resolve_emotion("furious", profile)
        assert full.speed == pytest.approx(default.speed)
        assert full.pitch == default.pitch
        assert full.volume == pytest.approx(default.volume)

    def test_intensity_does_not_affect_unknown_emotion(self):
        """intensity_override has no effect on unknown emotions (passthrough)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0, emotion="neutral")
        result = resolve_emotion("some_unknown", profile, intensity_override=0.5)
        assert result.emotion == "some_unknown"
        assert result.speed == 1.0
        assert result.pitch == 0
        assert result.volume == 1.0


class TestPauseAndBreathinessPropagation:
    """Verify pause_before_ms and breathiness flow through resolve_emotion."""

    def test_dread_has_pause(self):
        """Dread should have a notable pause_before_ms (foreboding beat)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0)
        result = resolve_emotion("dread", profile)
        assert result.pause_before_ms > 0, "dread should have a pause before speaking"

    def test_panic_has_no_pause(self):
        """Panic is frantic — no pause."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0)
        result = resolve_emotion("panic", profile)
        assert result.pause_before_ms == 0, "panic should be immediate, no pause"

    def test_horror_has_breathiness(self):
        """Horror should have positive breathiness (scared breathing)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0)
        result = resolve_emotion("horror", profile)
        assert result.breathiness > 0, "horror should be breathy"

    def test_defiant_has_negative_breathiness(self):
        """Defiant should be crisp/strong (negative breathiness)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0)
        result = resolve_emotion("defiant", profile)
        assert result.breathiness < 0, "defiant should be crisp, not breathy"

    def test_unknown_emotion_has_zero_pause_and_breathiness(self):
        """Unknown emotions get zero pause and breathiness."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0)
        result = resolve_emotion("unknown_emotion", profile)
        assert result.pause_before_ms == 0
        assert result.breathiness == 0.0

    def test_stunned_has_longer_pause_than_shock(self):
        """Stunned (frozen) should have a longer pause than shock (immediate)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0)
        stunned = resolve_emotion("stunned", profile)
        shock = resolve_emotion("shock", profile)
        assert stunned.pause_before_ms > shock.pause_before_ms

    def test_grief_has_breathiness(self):
        """Grieving should have breathiness (through tears)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0)
        result = resolve_emotion("grieving", profile)
        assert result.breathiness > 0, "grieving should be breathy (through tears)"

    def test_intimate_has_breathiness(self):
        """Intimate should have positive breathiness (whisper-adjacent)."""
        profile = VoiceProfile(speed=1.0, pitch=0, volume=1.0)
        result = resolve_emotion("intimate", profile)
        assert result.breathiness > 0, "intimate should be breathy"


# ---------------------------------------------------------------------------
# Tests: Text prosody analysis (Task 3.1.2)
# ---------------------------------------------------------------------------

class TestAnalyzeTextProsody:
    """Verify analyze_text_prosody derives adjustments from Chinese text cues."""

    def test_returns_prosody_hints_type(self):
        """analyze_text_prosody returns a ProsodyHints instance."""
        result = analyze_text_prosody("你好")
        assert isinstance(result, ProsodyHints)

    def test_plain_text_returns_zero_adjustments(self):
        """Plain text without punctuation cues returns zero adjustments."""
        result = analyze_text_prosody("你来了")
        assert result.speed_adjust == 0.0
        assert result.pitch_adjust == 0
        assert result.volume_adjust == 0.0

    def test_exclamation_increases_volume_and_speed(self):
        """Chinese exclamation mark (！) boosts volume and speed."""
        result = analyze_text_prosody("住手！")
        assert result.volume_adjust > 0, "！ should increase volume"
        assert result.speed_adjust > 0, "！ should increase speed"

    def test_multiple_exclamations_stronger_effect(self):
        """Multiple exclamation marks have a stronger effect than one."""
        one = analyze_text_prosody("住手！")
        two = analyze_text_prosody("住手！不要！")
        assert two.volume_adjust > one.volume_adjust

    def test_exclamation_effect_is_capped(self):
        """Volume/speed boost is capped even with many exclamation marks."""
        result = analyze_text_prosody("啊！啊！啊！啊！啊！")
        assert result.volume_adjust <= 0.15
        assert result.speed_adjust <= 0.08

    def test_ellipsis_reduces_speed(self):
        """Chinese ellipsis (……) reduces speed for deliberate pacing."""
        result = analyze_text_prosody("你确定……要这样做吗")
        assert result.speed_adjust < 0, "…… should reduce speed"

    def test_question_mark_raises_pitch(self):
        """Chinese question mark (？) raises pitch."""
        result = analyze_text_prosody("你说什么？")
        assert result.pitch_adjust > 0, "？ should raise pitch"

    def test_combined_punctuation(self):
        """Text with multiple punctuation types applies all adjustments."""
        result = analyze_text_prosody("你确定……要踩这双手？！")
        # Has ellipsis (speed-) and excl (speed+/vol+) and question (pitch+)
        assert result.pitch_adjust > 0, "？ should still raise pitch"
        assert result.volume_adjust > 0, "！ should still boost volume"

    def test_western_punctuation_also_recognized(self):
        """Western punctuation (!, ?, ...) also has effects."""
        result_zh = analyze_text_prosody("住手！")
        result_en = analyze_text_prosody("住手!")
        assert result_en.volume_adjust > 0
        assert result_en.speed_adjust > 0

    def test_empty_text_returns_zeros(self):
        """Empty text returns zero adjustments."""
        result = analyze_text_prosody("")
        assert result.speed_adjust == 0.0
        assert result.pitch_adjust == 0
        assert result.volume_adjust == 0.0


class TestProsodyHintsAppliedInMultiRole:
    """Verify generate_multi_role applies text prosody to TTS calls."""

    @pytest.mark.asyncio
    async def test_exclamation_text_gets_boosted_params(self, tmp_path):
        """A line with ！ should get higher speed/volume than plain text."""
        from videoclaw.generation.audio.tts import TTSManager, WaveSpeedTTSProvider

        mock_provider = AsyncMock(spec=WaveSpeedTTSProvider)
        mock_provider.synthesize = AsyncMock(return_value=b"fake_audio")

        tts = TTSManager(provider=mock_provider)

        lines = [
            DialogueLine(text="你来了", speaker="林薇", scene_id="s01",
                         emotion_hint="angry"),
            DialogueLine(text="住手！", speaker="林薇", scene_id="s01",
                         emotion_hint="angry"),
        ]
        voice_map = {
            "林薇": VoiceProfile(voice_id="Lively_Girl", speed=1.0, pitch=0,
                                 volume=1.0, emotion="neutral"),
        }

        await tts.generate_multi_role(lines, voice_map, tmp_path)

        # Both calls should be to synthesize with anger params
        assert mock_provider.synthesize.call_count == 2
        call_plain = mock_provider.synthesize.call_args_list[0]
        call_excl = mock_provider.synthesize.call_args_list[1]

        # The exclamation line should have higher speed and volume
        assert call_excl.kwargs["speed"] > call_plain.kwargs["speed"]
        assert call_excl.kwargs["volume"] > call_plain.kwargs["volume"]
