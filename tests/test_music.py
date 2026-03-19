"""Tests for MusicProvider, MusicManager, and _handle_music executor handler (B5)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from videoclaw.generation.audio.music import MusicManager, MusicProvider, SilentMusicProvider


# ---------------------------------------------------------------------------
# SilentMusicProvider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_silent_provider_calls_ffmpeg(tmp_path):
    """SilentMusicProvider calls ffmpeg with the expected -t duration argument."""
    output_path = tmp_path / "bgm.aac"

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        provider = SilentMusicProvider()
        result = await provider.generate(
            mood="orchestral",
            style="",
            duration_seconds=30.0,
            output_path=output_path,
        )

    assert result == output_path
    assert mock_exec.called
    cmd_args = mock_exec.call_args[0]
    assert cmd_args[0] == "ffmpeg"
    assert "-t" in cmd_args
    duration_idx = list(cmd_args).index("-t")
    assert cmd_args[duration_idx + 1] == "30.0"
    assert str(output_path) in cmd_args


@pytest.mark.asyncio
async def test_silent_provider_fallback_on_ffmpeg_failure(tmp_path):
    """SilentMusicProvider writes an empty fallback file when ffmpeg exits non-zero."""
    output_path = tmp_path / "bgm.aac"

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b"ffmpeg error"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        provider = SilentMusicProvider()
        result = await provider.generate(
            mood="",
            style="",
            duration_seconds=10.0,
            output_path=output_path,
        )

    assert result == output_path
    assert output_path.exists()
    assert output_path.read_bytes() == b""


@pytest.mark.asyncio
async def test_silent_provider_creates_parent_dirs(tmp_path):
    """SilentMusicProvider creates missing parent directories before writing."""
    output_path = tmp_path / "nested" / "audio" / "bgm.aac"

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        provider = SilentMusicProvider()
        await provider.generate(
            mood="neutral",
            style="",
            duration_seconds=5.0,
            output_path=output_path,
        )

    assert output_path.parent.exists()


# ---------------------------------------------------------------------------
# MusicManager tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_music_manager_delegates_to_provider():
    """MusicManager.generate_bgm delegates to the underlying provider."""
    mock_provider = MagicMock()
    expected_path = Path("/tmp/bgm.aac")
    mock_provider.generate = AsyncMock(return_value=expected_path)

    manager = MusicManager(provider=mock_provider)
    result = await manager.generate_bgm(
        mood="cinematic",
        style="orchestral",
        duration_seconds=45.0,
        output_path=Path("/tmp/bgm.aac"),
    )

    mock_provider.generate.assert_called_once_with(
        mood="cinematic",
        style="orchestral",
        duration_seconds=45.0,
        output_path=Path("/tmp/bgm.aac"),
    )
    assert result == expected_path


@pytest.mark.asyncio
async def test_music_manager_default_uses_silent_provider():
    """MusicManager without explicit provider defaults to SilentMusicProvider."""
    manager = MusicManager()
    assert isinstance(manager._provider, SilentMusicProvider)


def test_silent_provider_satisfies_protocol():
    """SilentMusicProvider is a runtime-checkable MusicProvider."""
    provider = SilentMusicProvider()
    assert isinstance(provider, MusicProvider)


# ---------------------------------------------------------------------------
# _handle_music executor handler tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_music_stores_path_in_state(tmp_path):
    """_handle_music stores the generated music path in state.assets['music']."""
    from videoclaw.core.executor import DAGExecutor
    from videoclaw.core.planner import DAG, TaskNode, TaskType
    from videoclaw.core.state import ProjectState, Shot

    dag = DAG()
    state = ProjectState(project_id="test_proj")
    state.storyboard = [
        Shot(shot_id="s1", duration_seconds=15.0),
        Shot(shot_id="s2", duration_seconds=10.0),
    ]

    node = TaskNode(node_id="music_1", task_type=TaskType.MUSIC, params={})

    mock_config = MagicMock()
    mock_config.projects_dir = str(tmp_path)

    # Match actual executor output path: <project>/audio/bgm.aac
    music_path = tmp_path / "test_proj" / "audio" / "bgm.aac"
    music_path.parent.mkdir(parents=True, exist_ok=True)
    music_path.write_bytes(b"fake audio")

    with patch("videoclaw.core.executor.get_config", return_value=mock_config):
        executor = DAGExecutor(dag=dag, state=state)
        executor._config = mock_config

        with patch.object(
            MusicManager,
            "generate_bgm",
            new_callable=AsyncMock,
            return_value=music_path,
        ):
            result = await executor._handle_music(node, state)

    assert state.assets["music"] == str(music_path)
    assert "music_path" in result
    assert result["duration"] == 25.0  # 15.0 + 10.0


@pytest.mark.asyncio
async def test_handle_music_default_duration_when_no_storyboard(tmp_path):
    """_handle_music uses 60.0s default duration when storyboard is empty."""
    from videoclaw.core.executor import DAGExecutor
    from videoclaw.core.planner import DAG, TaskNode, TaskType
    from videoclaw.core.state import ProjectState

    dag = DAG()
    state = ProjectState(project_id="test_proj_empty")

    node = TaskNode(node_id="music_1", task_type=TaskType.MUSIC, params={})

    mock_config = MagicMock()
    mock_config.projects_dir = str(tmp_path)

    music_path = tmp_path / "test_proj_empty" / "audio" / "bgm.aac"

    with patch("videoclaw.core.executor.get_config", return_value=mock_config):
        executor = DAGExecutor(dag=dag, state=state)
        executor._config = mock_config

        with patch.object(
            MusicManager,
            "generate_bgm",
            new_callable=AsyncMock,
            return_value=music_path,
        ) as mock_gen:
            music_path.parent.mkdir(parents=True, exist_ok=True)
            music_path.write_bytes(b"")
            result = await executor._handle_music(node, state)

    assert mock_gen.call_args.kwargs["duration_seconds"] == 60.0
    assert result["duration"] == 60.0
