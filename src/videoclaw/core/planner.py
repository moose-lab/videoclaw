"""Converts a ProjectState into a DAG of executable tasks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from videoclaw.core.state import ProjectState

# ---------------------------------------------------------------------------
# Task types
# ---------------------------------------------------------------------------

class TaskType(StrEnum):
    SCRIPT_GEN = "script_gen"
    STORYBOARD = "storyboard"
    SCENE_VALIDATE = "scene_validate"
    VIDEO_GEN = "video_gen"
    TTS = "tts"
    PER_SCENE_TTS = "per_scene_tts"
    SUBTITLE_GEN = "subtitle_gen"
    MUSIC = "music"
    COMPOSE = "compose"
    RENDER = "render"


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TaskNode:
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_type: TaskType = TaskType.SCRIPT_GEN
    depends_on: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "task_type": self.task_type.value,
            "depends_on": self.depends_on,
            "params": self.params,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskNode:
        data = dict(data)
        data["task_type"] = TaskType(data["task_type"])
        data["status"] = NodeStatus(data.get("status", "pending"))
        return cls(**data)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

class DAG:
    """Directed acyclic graph of :class:`TaskNode` objects."""

    def __init__(self) -> None:
        self.nodes: dict[str, TaskNode] = {}

    # -- Mutation --------------------------------------------------------------

    def add_node(self, node: TaskNode) -> None:
        """Add a task node to the graph."""
        self.nodes[node.node_id] = node

    def mark_running(self, node_id: str) -> None:
        self.nodes[node_id].status = NodeStatus.RUNNING

    def mark_complete(self, node_id: str, result: Any = None) -> None:
        node = self.nodes[node_id]
        node.status = NodeStatus.COMPLETED
        node.result = result

    def mark_failed(self, node_id: str, error: str) -> None:
        node = self.nodes[node_id]
        node.status = NodeStatus.FAILED
        node.error = error

    # -- Queries ---------------------------------------------------------------

    def get_ready_nodes(self) -> list[TaskNode]:
        """Return nodes whose dependencies are all completed and that are still pending."""
        ready: list[TaskNode] = []
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            deps_met = all(
                self.nodes[dep_id].status == NodeStatus.COMPLETED
                for dep_id in node.depends_on
            )
            if deps_met:
                ready.append(node)
        return ready

    @property
    def is_complete(self) -> bool:
        """True when every node has reached a terminal state (completed or failed)."""
        return all(
            n.status in (NodeStatus.COMPLETED, NodeStatus.FAILED)
            for n in self.nodes.values()
        )

    @property
    def has_failures(self) -> bool:
        return any(n.status == NodeStatus.FAILED for n in self.nodes.values())

    # -- Serialisation ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": {nid: n.to_dict() for nid, n in self.nodes.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DAG:
        dag = cls()
        for nid, ndata in data["nodes"].items():
            dag.nodes[nid] = TaskNode.from_dict(ndata)
        return dag


# ---------------------------------------------------------------------------
# Standard pipeline builder
# ---------------------------------------------------------------------------

def build_dag(project_state: ProjectState) -> DAG:
    """Create the standard video pipeline DAG from a :class:`ProjectState`.

    Pipeline shape::

        script_gen
            |
        storyboard
            |
        +---+---+---+      tts      music
        | video shots |      |        |
        +---+---+---+      |        |
            |               |        |
            +-------+-------+--------+
                    |
                 compose
                    |
                  render
    """
    dag = DAG()

    # 1. Script generation
    script_node = TaskNode(
        node_id="script_gen",
        task_type=TaskType.SCRIPT_GEN,
        params={"prompt": project_state.prompt},
    )
    dag.add_node(script_node)

    # 2. Storyboard
    storyboard_node = TaskNode(
        node_id="storyboard",
        task_type=TaskType.STORYBOARD,
        depends_on=["script_gen"],
        params={"prompt": project_state.prompt},
    )
    dag.add_node(storyboard_node)

    # 3. Parallel video generation per shot
    video_node_ids: list[str] = []
    for shot in project_state.storyboard:
        vid_id = f"video_{shot.shot_id}"
        dag.add_node(TaskNode(
            node_id=vid_id,
            task_type=TaskType.VIDEO_GEN,
            depends_on=["storyboard"],
            params={
                "shot_id": shot.shot_id,
                "prompt": shot.prompt,
                "duration": shot.duration_seconds,
                "model_id": shot.model_id,
            },
        ))
        video_node_ids.append(vid_id)

    # 4. TTS (depends on script)
    tts_node = TaskNode(
        node_id="tts",
        task_type=TaskType.TTS,
        depends_on=["storyboard"],
        params={},
    )
    dag.add_node(tts_node)

    # 5. Music (depends on storyboard for mood/timing)
    music_node = TaskNode(
        node_id="music",
        task_type=TaskType.MUSIC,
        depends_on=["storyboard"],
        params={},
    )
    dag.add_node(music_node)

    # 6. Compose -- waits for all video clips + audio tracks
    compose_deps = [*video_node_ids, "tts", "music"]
    compose_node = TaskNode(
        node_id="compose",
        task_type=TaskType.COMPOSE,
        depends_on=compose_deps,
        params={},
    )
    dag.add_node(compose_node)

    # 7. Final render
    render_node = TaskNode(
        node_id="render",
        task_type=TaskType.RENDER,
        depends_on=["compose"],
        params={},
    )
    dag.add_node(render_node)

    return dag
