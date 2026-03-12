"""Tests for ClawFlow YAML parser and runner."""

import pytest
import textwrap

import yaml

from videoclaw.flow.parser import FlowDef, FlowStep, FlowValidationError, parse_flow, load_flow
from videoclaw.flow.runner import compile_dag, FlowRunner
from videoclaw.core.planner import TaskType, NodeStatus
from videoclaw.core.state import StateManager


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


MINIMAL_FLOW = {
    "name": "test-flow",
    "steps": [
        {"id": "script", "type": "script_gen", "params": {"prompt": "hello"}},
    ],
}


def test_parse_minimal():
    flow = parse_flow(MINIMAL_FLOW)
    assert flow.name == "test-flow"
    assert len(flow.steps) == 1
    assert flow.steps[0].type == TaskType.SCRIPT_GEN


def test_parse_with_variables():
    raw = {
        "name": "var-flow",
        "variables": {"thing": "widget", "dur": 10},
        "steps": [
            {
                "id": "s1",
                "type": "script_gen",
                "params": {"prompt": "Sell {{thing}}", "length": "{{dur}}"},
            },
        ],
    }
    flow = parse_flow(raw)
    assert flow.steps[0].params["prompt"] == "Sell widget"
    assert flow.steps[0].params["length"] == 10


def test_parse_dependencies():
    raw = {
        "name": "dep-flow",
        "steps": [
            {"id": "a", "type": "script_gen"},
            {"id": "b", "type": "storyboard", "depends_on": ["a"]},
        ],
    }
    flow = parse_flow(raw)
    assert flow.steps[1].depends_on == ["a"]


def test_reject_unknown_type():
    raw = {
        "name": "bad",
        "steps": [{"id": "x", "type": "not_real"}],
    }
    with pytest.raises(FlowValidationError, match="Unknown step type"):
        parse_flow(raw)


def test_reject_duplicate_ids():
    raw = {
        "name": "dup",
        "steps": [
            {"id": "a", "type": "script_gen"},
            {"id": "a", "type": "storyboard"},
        ],
    }
    with pytest.raises(FlowValidationError, match="Duplicate step id"):
        parse_flow(raw)


def test_reject_unknown_dependency():
    raw = {
        "name": "bad-dep",
        "steps": [
            {"id": "a", "type": "script_gen", "depends_on": ["ghost"]},
        ],
    }
    with pytest.raises(FlowValidationError, match="unknown step"):
        parse_flow(raw)


def test_reject_cycle():
    raw = {
        "name": "cycle",
        "steps": [
            {"id": "a", "type": "script_gen", "depends_on": ["b"]},
            {"id": "b", "type": "storyboard", "depends_on": ["a"]},
        ],
    }
    with pytest.raises(FlowValidationError, match="cycle"):
        parse_flow(raw)


def test_reject_undefined_variable():
    raw = {
        "name": "novar",
        "steps": [
            {"id": "a", "type": "script_gen", "params": {"p": "{{missing}}"}},
        ],
    }
    with pytest.raises(ValueError, match="Undefined variable"):
        parse_flow(raw)


def test_load_flow_from_file(tmp_path):
    flow_file = tmp_path / "test.yaml"
    flow_file.write_text(yaml.dump(MINIMAL_FLOW))
    flow = load_flow(flow_file)
    assert flow.name == "test-flow"


def test_flow_roundtrip():
    flow = parse_flow(MINIMAL_FLOW)
    d = flow.to_dict()
    assert d["name"] == "test-flow"
    assert len(d["steps"]) == 1


# ---------------------------------------------------------------------------
# Compile + Runner tests
# ---------------------------------------------------------------------------


def test_compile_dag():
    raw = {
        "name": "pipe",
        "steps": [
            {"id": "script", "type": "script_gen"},
            {"id": "board", "type": "storyboard", "depends_on": ["script"]},
            {"id": "render", "type": "render", "depends_on": ["board"]},
        ],
    }
    flow = parse_flow(raw)
    dag = compile_dag(flow)
    assert len(dag.nodes) == 3
    assert dag.nodes["render"].depends_on == ["board"]


@pytest.mark.asyncio
async def test_flow_runner_e2e(tmp_path):
    """End-to-end test: parse a flow, run it with mock handlers, verify completion."""
    from unittest.mock import AsyncMock, patch
    from videoclaw.config import VideoClawConfig
    from videoclaw.core.state import Shot

    raw = {
        "name": "e2e-test",
        "steps": [
            {"id": "script", "type": "script_gen", "params": {"prompt": "test"}},
            {"id": "storyboard", "type": "storyboard", "depends_on": ["script"]},
            {"id": "tts", "type": "tts", "depends_on": ["script"]},
            {"id": "music", "type": "music", "depends_on": ["storyboard"]},
            {"id": "compose", "type": "compose", "depends_on": ["storyboard", "tts", "music"]},
            {"id": "render", "type": "render", "depends_on": ["compose"]},
        ],
    }
    flow = parse_flow(raw)
    test_config = VideoClawConfig(projects_dir=tmp_path)
    sm = StateManager(projects_dir=tmp_path)
    ps = sm.create_project(prompt="e2e test")

    # Pre-populate so script_gen and storyboard handlers skip LLM calls
    ps.script = "Test script narration."

    # Create a fake video file for the shot so compose handler finds it
    shots_dir = tmp_path / ps.project_id / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    shot_file = shots_dir / "s1.mp4"
    shot_file.write_bytes(b"mock_video_data")

    ps.storyboard = [
        Shot(shot_id="s1", prompt="test shot", duration_seconds=3.0,
             model_id="mock", asset_path=str(shot_file)),
    ]
    sm.save(ps)

    runner = FlowRunner(state_manager=sm, max_concurrency=2)

    with patch("videoclaw.generation.audio.tts.TTSManager") as MockTTS, \
         patch("videoclaw.generation.compose.VideoComposer") as MockComposer, \
         patch("videoclaw.config.get_config", return_value=test_config):
        tts_instance = MockTTS.return_value
        tts_instance.generate_voiceover = AsyncMock(side_effect=lambda text, path, **kw: path)

        composer_instance = MockComposer.return_value

        async def _mock_compose(video_paths, output_path, **kw):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"mock")
            return output_path

        async def _mock_render_final(video_path, audio_tracks, subtitle_path, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"mock")
            return output_path

        composer_instance.compose = AsyncMock(side_effect=_mock_compose)
        composer_instance.render_final = AsyncMock(side_effect=_mock_render_final)

        result = await runner.run(flow, ps)

    assert result.status.value == "completed"
    assert "final_video" in result.assets
