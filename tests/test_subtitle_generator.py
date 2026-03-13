"""Tests for SubtitleGenerator: SRT/ASS generation, text splitting, manifest timing,
and _handle_compose integration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from videoclaw.generation.subtitle import (
    SubtitleGenerator,
    _format_ass_time,
    _format_srt_time,
    _rgb_to_ass_color,
    generate_srt,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _scenes(
    *,
    count: int = 2,
    dialogue: str = "你好世界",
    narration: str = "",
    duration: float = 5.0,
    character: str = "小明",
) -> list[dict]:
    return [
        {
            "scene_id": f"s{i:02d}",
            "dialogue": dialogue,
            "narration": narration,
            "speaking_character": character,
            "duration_seconds": duration,
        }
        for i in range(1, count + 1)
    ]


def _manifest(scenes: list[dict], *, duration: float = 3.0) -> dict:
    """Build a minimal audio_manifest dict matching the given scenes."""
    segments = []
    t = 0.0
    for sc in scenes:
        segments.append({
            "segment_id": f"seg_{sc['scene_id']}",
            "scene_id": sc["scene_id"],
            "audio_type": "dialogue",
            "line_type": "dialogue",
            "text": sc.get("dialogue", ""),
            "character_name": sc.get("speaking_character", ""),
            "audio_path": f"/tmp/audio/{sc['scene_id']}.mp3",
            "start_time": t,
            "duration_seconds": duration,
            "volume": 1.0,
        })
        t += duration
    return {
        "episode_id": "ep01",
        "segments": segments,
        "total_duration": t,
    }


# ===================================================================
# 1. SRT generation -- backward compatibility
# ===================================================================

class TestSRTGeneration:
    """SRT generation via both the class and the free function."""

    def test_free_function_backward_compat(self, tmp_path: Path):
        """The top-level generate_srt() still works as before."""
        scenes = _scenes()
        out = tmp_path / "out.srt"
        result = generate_srt(scenes, out)

        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        # Should contain SRT index numbers
        assert "1\n" in content
        assert "2\n" in content
        # Should contain character name prefix
        assert "\u5c0f\u660e\uff1a" in content  # 小明：

    def test_srt_timing_without_manifest(self, tmp_path: Path):
        """Without manifest, uses scene duration_seconds."""
        scenes = _scenes(count=2, duration=4.0)
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out)

        content = out.read_text(encoding="utf-8")
        # First scene: 0:00:00,000 -> 0:00:04,000
        assert "00:00:00,000 --> 00:00:04,000" in content
        # Second scene starts at 4s: 0:00:04,000 -> 0:00:08,000
        assert "00:00:04,000 --> 00:00:08,000" in content

    def test_srt_include_narration(self, tmp_path: Path):
        """Narration is included when include_narration=True and no dialogue."""
        scenes = [
            {
                "scene_id": "s01",
                "dialogue": "",
                "narration": "旁白文字",
                "speaking_character": "",
                "duration_seconds": 3.0,
            }
        ]
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()

        # Without flag -- should produce empty
        gen.generate_srt(scenes, out, include_narration=False)
        content = out.read_text(encoding="utf-8")
        assert "旁白文字" not in content

        # With flag
        gen.generate_srt(scenes, out, include_narration=True)
        content = out.read_text(encoding="utf-8")
        assert "旁白文字" in content

    def test_srt_skips_empty_dialogue(self, tmp_path: Path):
        """Scenes without dialogue are skipped (no narration flag)."""
        scenes = [
            {"scene_id": "s01", "dialogue": "有对话", "speaking_character": "", "duration_seconds": 2.0},
            {"scene_id": "s02", "dialogue": "", "speaking_character": "", "duration_seconds": 3.0},
            {"scene_id": "s03", "dialogue": "又有对话", "speaking_character": "", "duration_seconds": 2.0},
        ]
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out)

        content = out.read_text(encoding="utf-8")
        # Should have exactly 2 subtitle entries
        assert content.startswith("1\n")
        assert "2\n" in content
        # Second entry timing starts at 5s (2+3), not at 2s
        assert "00:00:05,000" in content
        # Should NOT have a third entry
        lines = content.strip().split("\n")
        # SRT entries start with a bare number
        entry_numbers = [l for l in lines if l.strip().isdigit()]
        assert len(entry_numbers) == 2

    def test_free_function_passes_audio_manifest(self, tmp_path: Path):
        """The free function forwards audio_manifest to SubtitleGenerator."""
        scenes = _scenes(count=1, duration=10.0)
        manifest = _manifest(scenes, duration=2.5)
        out = tmp_path / "out.srt"

        generate_srt(scenes, out, audio_manifest=manifest)
        content = out.read_text(encoding="utf-8")
        # Should use manifest duration (2.5s), not scene duration (10s)
        assert "00:00:02,500" in content


# ===================================================================
# 2. ASS generation
# ===================================================================

class TestASSGeneration:
    """ASS subtitle generation with styles and character colors."""

    def test_ass_basic_output(self, tmp_path: Path):
        """ASS file has required sections."""
        scenes = _scenes(count=1)
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out)

        content = out.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content
        assert "Dialogue:" in content

    def test_ass_character_colors(self, tmp_path: Path):
        """Per-character styles are created from character_colors."""
        scenes = _scenes(count=1, character="小红")
        colors = {"小红": "#FF0000", "小明": "#00FF00"}
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out, character_colors=colors)

        content = out.read_text(encoding="utf-8")
        # Should have per-character styles
        assert "Style: Char_小红" in content
        assert "Style: Char_小明" in content
        # RGB #FF0000 -> ASS &H000000FF (BGR)
        assert "&H000000FF" in content
        # Dialogue should use the character's style
        assert "Char_小红" in content

    def test_ass_narration_style(self, tmp_path: Path):
        """Narration uses Narration style (top-center, alignment 8)."""
        scenes = [
            {
                "scene_id": "s01",
                "dialogue": "",
                "narration": "这是旁白",
                "speaking_character": "",
                "duration_seconds": 3.0,
            }
        ]
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out, include_narration=True)

        content = out.read_text(encoding="utf-8")
        # Narration style exists with alignment 8
        narration_style_line = [l for l in content.split("\n") if l.startswith("Style: Narration,")]
        assert len(narration_style_line) == 1
        assert ",8," in narration_style_line[0]
        # Event uses Narration style
        events = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        assert any("Narration" in e for e in events)

    def test_ass_default_alignment(self, tmp_path: Path):
        """Default style has alignment 2 (bottom-center)."""
        scenes = _scenes(count=1)
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out)

        content = out.read_text(encoding="utf-8")
        default_style = [l for l in content.split("\n") if l.startswith("Style: Default,")]
        assert len(default_style) == 1
        # Alignment field (19th in Format) should be 2
        parts = default_style[0].split(",")
        # Alignment is the 19th field (0-indexed: 18)
        assert parts[18] == "2"

    def test_ass_timing_with_manifest(self, tmp_path: Path):
        """ASS uses audio manifest durations when provided."""
        scenes = _scenes(count=2, duration=10.0)
        manifest = _manifest(scenes, duration=3.5)
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out, audio_manifest=manifest)

        content = out.read_text(encoding="utf-8")
        events = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        assert len(events) == 2
        # First event: 0:00:00.00 -> 0:00:03.50
        assert "0:00:00.00" in events[0]
        assert "0:00:03.50" in events[0]
        # Second: 0:00:03.50 -> 0:00:07.00
        assert "0:00:03.50" in events[1]
        assert "0:00:07.00" in events[1]

    def test_ass_font_configuration(self, tmp_path: Path):
        """Custom font name and size are reflected in ASS styles."""
        scenes = _scenes(count=1)
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out, font_name="Noto Sans SC", font_size=24)

        content = out.read_text(encoding="utf-8")
        assert "Noto Sans SC" in content
        default_style = [l for l in content.split("\n") if l.startswith("Style: Default,")]
        assert "24" in default_style[0]

    def test_ass_per_character_style_in_events(self, tmp_path: Path):
        """When character_colors provided, dialogue events use Char_X styles."""
        scenes = [
            {"scene_id": "s01", "dialogue": "你好", "narration": "",
             "speaking_character": "小明", "duration_seconds": 3.0},
            {"scene_id": "s02", "dialogue": "你好啊", "narration": "",
             "speaking_character": "小红", "duration_seconds": 3.0},
        ]
        colors = {"小明": "#00FF00", "小红": "#FF0000"}
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out, character_colors=colors)

        content = out.read_text(encoding="utf-8")
        events = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        assert len(events) == 2
        assert "Char_小明" in events[0]
        assert "Char_小红" in events[1]


# ===================================================================
# 3. Long text splitting
# ===================================================================

class TestLongTextSplitting:
    """split_long_text handles Chinese punctuation and max_chars."""

    def test_short_text_unchanged(self):
        result = SubtitleGenerator.split_long_text("你好世界", max_chars=20)
        assert result == "你好世界"

    def test_split_at_chinese_punctuation(self):
        text = "我今天去了超市，买了很多东西，然后回家做饭。"
        result = SubtitleGenerator.split_long_text(text, max_chars=15)
        assert "\\N" in result
        # Punctuation should stay attached to the preceding segment
        parts = result.split("\\N")
        assert any(p.endswith("，") or p.endswith("。") for p in parts)

    def test_force_split_long_segment(self):
        """When no punctuation, force-split at max_chars."""
        text = "这是一段没有任何标点符号的很长的中文文本用来测试"
        result = SubtitleGenerator.split_long_text(text, max_chars=10)
        parts = result.split("\\N")
        assert all(len(p) <= 10 for p in parts)
        # Reassembling should give the original text
        assert "".join(parts) == text

    def test_split_with_srt_line_break(self):
        text = "我今天去了超市，买了很多东西。"
        result = SubtitleGenerator.split_long_text(text, max_chars=10, line_break="\n")
        assert "\n" in result
        assert "\\N" not in result

    def test_exact_boundary(self):
        """Text exactly at max_chars should not be split."""
        text = "十个字的中文文本啊！"  # 10 chars
        result = SubtitleGenerator.split_long_text(text, max_chars=10)
        assert "\\N" not in result

    def test_empty_text(self):
        result = SubtitleGenerator.split_long_text("", max_chars=20)
        assert result == ""


# ===================================================================
# 4. Audio manifest timing integration
# ===================================================================

class TestManifestTiming:
    """Subtitle timing uses manifest durations when available."""

    def test_manifest_overrides_scene_duration(self, tmp_path: Path):
        """Manifest duration takes precedence over scene duration_seconds."""
        scenes = _scenes(count=1, duration=10.0)
        manifest = _manifest(scenes, duration=2.0)
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out, audio_manifest=manifest)

        content = out.read_text(encoding="utf-8")
        # End time should be 2s, not 10s
        assert "00:00:02,000" in content
        assert "00:00:10,000" not in content

    def test_fallback_when_no_manifest(self, tmp_path: Path):
        """Without manifest, scene duration_seconds is used."""
        scenes = _scenes(count=1, duration=7.0)
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out, audio_manifest=None)

        content = out.read_text(encoding="utf-8")
        assert "00:00:07,000" in content

    def test_fallback_when_scene_not_in_manifest(self, tmp_path: Path):
        """If scene_id not found in manifest, use scene duration_seconds."""
        scenes = [
            {"scene_id": "missing_scene", "dialogue": "你好", "speaking_character": "",
             "duration_seconds": 6.0},
        ]
        manifest = {"episode_id": "ep01", "segments": [], "total_duration": 0.0}
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out, audio_manifest=manifest)

        content = out.read_text(encoding="utf-8")
        assert "00:00:06,000" in content

    def test_multiple_segments_per_scene(self, tmp_path: Path):
        """Multiple manifest segments for one scene_id sum their durations."""
        scenes = _scenes(count=1, duration=10.0)
        manifest = {
            "episode_id": "ep01",
            "segments": [
                {"scene_id": "s01", "duration_seconds": 1.5},
                {"scene_id": "s01", "duration_seconds": 2.5},
            ],
            "total_duration": 4.0,
        }
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out, audio_manifest=manifest)

        content = out.read_text(encoding="utf-8")
        # Total: 1.5 + 2.5 = 4.0
        assert "00:00:04,000" in content


# ===================================================================
# 5. Narration positioning (ASS)
# ===================================================================

class TestNarrationPositioning:
    """Narration events use Narration style (top-center)."""

    def test_narration_gets_narration_style(self, tmp_path: Path):
        scenes = [
            {"scene_id": "s01", "dialogue": "", "narration": "旁白",
             "speaking_character": "", "duration_seconds": 3.0},
        ]
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out, include_narration=True)

        content = out.read_text(encoding="utf-8")
        events = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        assert len(events) == 1
        assert ",Narration," in events[0]

    def test_dialogue_gets_default_style(self, tmp_path: Path):
        scenes = _scenes(count=1, character="")
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out)

        content = out.read_text(encoding="utf-8")
        events = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        assert len(events) == 1
        assert ",Default," in events[0]

    def test_narration_shown_when_no_dialogue(self, tmp_path: Path):
        """Even without include_narration, narration appears if dialogue is empty."""
        scenes = [
            {"scene_id": "s01", "dialogue": "", "narration": "旁白",
             "speaking_character": "", "duration_seconds": 3.0},
        ]
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out, include_narration=False)

        content = out.read_text(encoding="utf-8")
        events = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        # Narration shown as fallback when dialogue is empty
        assert len(events) == 1
        assert "旁白" in events[0]


# ===================================================================
# 6. _handle_compose uses SubtitleGenerator
# ===================================================================

class TestHandleComposeIntegration:
    """_handle_compose in DAGExecutor uses SubtitleGenerator."""

    @pytest.fixture()
    def executor_setup(self, tmp_path: Path):
        """Set up a DAGExecutor with patched config for compose testing."""
        from videoclaw.core.executor import DAGExecutor
        from videoclaw.core.planner import DAG, TaskNode, TaskType
        from videoclaw.core.state import ProjectState, StateManager

        dag = DAG()
        dag.add_node(TaskNode(node_id="compose", task_type=TaskType.COMPOSE, params={}))
        state = ProjectState(project_id="test-proj", prompt="test")
        state_mgr = StateManager(projects_dir=tmp_path)

        with patch("videoclaw.core.executor.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.projects_dir = str(tmp_path)
            cfg.max_retries = 0
            mock_cfg.return_value = cfg
            executor = DAGExecutor(dag=dag, state=state, state_manager=state_mgr)

        return executor, state

    @pytest.mark.asyncio
    async def test_compose_uses_subtitle_generator(self, executor_setup, tmp_path: Path):
        """_handle_compose imports and uses SubtitleGenerator, not bare generate_srt."""
        from videoclaw.core.planner import TaskNode, TaskType

        executor, state = executor_setup

        # Create a fake video asset
        project_dir = Path(executor._config.projects_dir) / state.project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        shot_mock = MagicMock()
        shot_mock.asset_path = str(project_dir / "shot.mp4")
        Path(shot_mock.asset_path).write_bytes(b"fake")
        state.storyboard = [shot_mock]

        scenes = [
            {"scene_id": "s01", "dialogue": "你好", "speaking_character": "小明",
             "duration_seconds": 3.0},
        ]
        node = TaskNode(
            node_id="compose",
            task_type=TaskType.COMPOSE,
            params={"scenes": scenes},
        )

        with patch("videoclaw.generation.subtitle.SubtitleGenerator") as MockGen, \
             patch("videoclaw.generation.compose.VideoComposer") as MockComposer:
            mock_instance = MagicMock()
            mock_instance.generate_ass.return_value = project_dir / "subtitles.ass"
            MockGen.return_value = mock_instance

            mock_composer = AsyncMock()
            MockComposer.return_value = mock_composer

            await executor._handle_compose(node, state)

            # SubtitleGenerator was instantiated and generate_ass was called
            MockGen.assert_called_once()
            mock_instance.generate_ass.assert_called_once()

    @pytest.mark.asyncio
    async def test_compose_passes_audio_manifest(self, executor_setup, tmp_path: Path):
        """_handle_compose passes audio_manifest from state.assets."""
        from videoclaw.core.planner import TaskNode, TaskType

        executor, state = executor_setup

        project_dir = Path(executor._config.projects_dir) / state.project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        shot_mock = MagicMock()
        shot_mock.asset_path = str(project_dir / "shot.mp4")
        Path(shot_mock.asset_path).write_bytes(b"fake")
        state.storyboard = [shot_mock]

        manifest_data = {"episode_id": "ep01", "segments": [], "total_duration": 0}
        state.assets["audio_manifest"] = json.dumps(manifest_data)

        scenes = [{"scene_id": "s01", "dialogue": "你好", "speaking_character": "",
                    "duration_seconds": 3.0}]
        node = TaskNode(
            node_id="compose",
            task_type=TaskType.COMPOSE,
            params={"scenes": scenes},
        )

        with patch("videoclaw.generation.subtitle.SubtitleGenerator") as MockGen, \
             patch("videoclaw.generation.compose.VideoComposer") as MockComposer:
            mock_instance = MagicMock()
            mock_instance.generate_ass.return_value = project_dir / "subtitles.ass"
            MockGen.return_value = mock_instance

            mock_composer = AsyncMock()
            MockComposer.return_value = mock_composer

            await executor._handle_compose(node, state)

            # Check audio_manifest was passed
            call_kwargs = mock_instance.generate_ass.call_args
            assert call_kwargs.kwargs.get("audio_manifest") == manifest_data

    @pytest.mark.asyncio
    async def test_compose_falls_back_to_srt(self, executor_setup, tmp_path: Path):
        """If ASS generation fails, _handle_compose falls back to SRT."""
        from videoclaw.core.planner import TaskNode, TaskType

        executor, state = executor_setup

        project_dir = Path(executor._config.projects_dir) / state.project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        shot_mock = MagicMock()
        shot_mock.asset_path = str(project_dir / "shot.mp4")
        Path(shot_mock.asset_path).write_bytes(b"fake")
        state.storyboard = [shot_mock]

        scenes = [{"scene_id": "s01", "dialogue": "你好", "speaking_character": "",
                    "duration_seconds": 3.0}]
        node = TaskNode(
            node_id="compose",
            task_type=TaskType.COMPOSE,
            params={"scenes": scenes},
        )

        with patch("videoclaw.generation.subtitle.SubtitleGenerator") as MockGen, \
             patch("videoclaw.generation.compose.VideoComposer") as MockComposer:
            mock_instance = MagicMock()
            mock_instance.generate_ass.side_effect = RuntimeError("ASS failed")
            mock_instance.generate_srt.return_value = project_dir / "subtitles.srt"
            MockGen.return_value = mock_instance

            mock_composer = AsyncMock()
            MockComposer.return_value = mock_composer

            await executor._handle_compose(node, state)

            # ASS was attempted, then SRT was called as fallback
            mock_instance.generate_ass.assert_called_once()
            mock_instance.generate_srt.assert_called_once()
            assert state.assets["subtitles"].endswith(".srt")


# ===================================================================
# 7. Fallback when no audio_manifest
# ===================================================================

class TestFallbackNoManifest:
    """When no audio_manifest is available, scene durations are used."""

    @pytest.mark.asyncio
    async def test_compose_without_manifest_uses_scene_duration(self, tmp_path: Path):
        """_handle_compose without manifest still generates subtitles."""
        from videoclaw.core.executor import DAGExecutor
        from videoclaw.core.planner import DAG, TaskNode, TaskType
        from videoclaw.core.state import ProjectState, StateManager

        dag = DAG()
        dag.add_node(TaskNode(node_id="compose", task_type=TaskType.COMPOSE, params={}))
        state = ProjectState(project_id="test-proj", prompt="test")
        state_mgr = StateManager(projects_dir=tmp_path)

        with patch("videoclaw.core.executor.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.projects_dir = str(tmp_path)
            cfg.max_retries = 0
            mock_cfg.return_value = cfg
            executor = DAGExecutor(dag=dag, state=state, state_manager=state_mgr)

        project_dir = Path(executor._config.projects_dir) / state.project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        shot_mock = MagicMock()
        shot_mock.asset_path = str(project_dir / "shot.mp4")
        Path(shot_mock.asset_path).write_bytes(b"fake")
        state.storyboard = [shot_mock]

        # No audio_manifest in state.assets
        scenes = [{"scene_id": "s01", "dialogue": "你好", "speaking_character": "",
                    "duration_seconds": 5.0}]
        node = TaskNode(
            node_id="compose",
            task_type=TaskType.COMPOSE,
            params={"scenes": scenes},
        )

        with patch("videoclaw.generation.subtitle.SubtitleGenerator") as MockGen, \
             patch("videoclaw.generation.compose.VideoComposer") as MockComposer:
            mock_instance = MagicMock()
            mock_instance.generate_ass.return_value = project_dir / "subtitles.ass"
            MockGen.return_value = mock_instance

            mock_composer = AsyncMock()
            MockComposer.return_value = mock_composer

            await executor._handle_compose(node, state)

            # audio_manifest should be None
            call_kwargs = mock_instance.generate_ass.call_args
            assert call_kwargs.kwargs.get("audio_manifest") is None


# ===================================================================
# 8. Helper function tests
# ===================================================================

class TestHelpers:
    """Test internal formatting helpers."""

    def test_format_srt_time(self):
        assert _format_srt_time(0.0) == "00:00:00,000"
        assert _format_srt_time(3661.5) == "01:01:01,500"
        assert _format_srt_time(0.123) == "00:00:00,123"

    def test_format_ass_time(self):
        assert _format_ass_time(0.0) == "0:00:00.00"
        assert _format_ass_time(3661.5) == "1:01:01.50"
        assert _format_ass_time(0.12) == "0:00:00.12"

    def test_rgb_to_ass_color(self):
        # Red -> BGR blue
        assert _rgb_to_ass_color("#FF0000") == "&H000000FF"
        # Blue -> BGR blue is first
        assert _rgb_to_ass_color("#0000FF") == "&H00FF0000"
        # White
        assert _rgb_to_ass_color("#FFFFFF") == "&H00FFFFFF"
        # Without hash
        assert _rgb_to_ass_color("00FF00") == "&H0000FF00"

    def test_rgb_to_ass_color_invalid(self):
        # Invalid hex returns white
        assert _rgb_to_ass_color("abc") == "&H00FFFFFF"
