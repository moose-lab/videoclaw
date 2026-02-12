"""Integration tests — run a complete pipeline end-to-end with mock adapters."""

import pytest

from videoclaw.core.events import EventBus
from videoclaw.core.planner import build_dag
from videoclaw.core.executor import DAGExecutor
from videoclaw.core.state import ProjectState, Shot, StateManager
from videoclaw.models.registry import ModelRegistry
from videoclaw.models.adapters.mock import MockVideoAdapter
from videoclaw.models.protocol import GenerationRequest


@pytest.mark.asyncio
async def test_full_pipeline_with_mock(tmp_path):
    """Run the standard pipeline DAG with placeholder handlers.

    Verifies that script → storyboard → [parallel shots + tts + music] →
    compose → render executes without error and produces a completed state.
    """
    sm = StateManager(projects_dir=tmp_path)
    ps = sm.create_project(prompt="Integration test video")

    # Add a couple of shots so the DAG builds video_gen nodes.
    ps.storyboard = [
        Shot(shot_id="s1", prompt="Opening shot", duration_seconds=3.0, model_id="mock"),
        Shot(shot_id="s2", prompt="Closing shot", duration_seconds=3.0, model_id="mock"),
    ]
    sm.save(ps)

    dag = build_dag(ps)
    bus = EventBus()

    # Collect events to verify the lifecycle.
    events: list[dict] = []
    async def _listener(event: str, data: dict) -> None:
        events.append({"event": event, "data": data})
    bus.subscribe("task.started", _listener)
    bus.subscribe("task.completed", _listener)
    bus.subscribe("project.completed", _listener)

    executor = DAGExecutor(dag=dag, state=ps, state_manager=sm, bus=bus, max_concurrency=4)
    result = await executor.run()

    # Pipeline should complete successfully.
    assert result.status.value == "completed"
    assert "final_video" in result.assets

    # All DAG nodes should be completed.
    for node in dag.nodes.values():
        assert node.status.value == "completed", f"Node {node.node_id} status: {node.status}"

    # We should have received events for every node.
    started = [e for e in events if e["event"] == "task.started"]
    completed = [e for e in events if e["event"] == "task.completed"]
    assert len(started) == len(dag.nodes)
    assert len(completed) == len(dag.nodes)

    # Project completed event should fire once.
    proj_done = [e for e in events if e["event"] == "project.completed"]
    assert len(proj_done) == 1


@pytest.mark.asyncio
async def test_model_registry_with_mock():
    """Verify mock adapter registers, generates, and health-checks."""
    registry = ModelRegistry()
    adapter = MockVideoAdapter()
    registry.register(adapter)

    assert "mock" in registry
    assert await adapter.health_check() is True

    req = GenerationRequest(prompt="test prompt", duration_seconds=3.0)
    result = await adapter.generate(req)
    assert result.model_id == "mock"
    assert result.duration_seconds == 3.0
    assert len(result.video_data) > 0
