"""Async DAG executor -- runs tasks respecting dependencies and concurrency limits."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from videoclaw.config import get_config
from videoclaw.core.events import (
    EventBus,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_STARTED,
    PROJECT_COMPLETED,
    event_bus as default_event_bus,
)
from videoclaw.core.planner import DAG, NodeStatus, TaskNode, TaskType
from videoclaw.core.state import ProjectState, ProjectStatus, StateManager

logger = logging.getLogger(__name__)

# Type alias for an async handler that processes a single task node.
NodeHandler = Callable[[TaskNode, ProjectState], Coroutine[Any, Any, Any]]


class DAGExecutor:
    """Execute a :class:`DAG` asynchronously, honouring dependency order.

    Independent nodes run in parallel up to *max_concurrency*.  On completion
    of each node the project state is checkpointed to disk.
    """

    def __init__(
        self,
        dag: DAG,
        state: ProjectState,
        state_manager: StateManager | None = None,
        bus: EventBus | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self.dag = dag
        self.state = state
        self.state_manager = state_manager or StateManager()
        self.bus = bus or default_event_bus
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._config = get_config()

        # Handler dispatch table -- maps TaskType to its async handler.
        # During Phase 1 every entry points at a placeholder.  Later phases
        # will register real generation / composition handlers.
        self._handlers: dict[TaskType, NodeHandler] = {
            TaskType.SCRIPT_GEN: self._handle_script_gen,
            TaskType.STORYBOARD: self._handle_storyboard,
            TaskType.VIDEO_GEN: self._handle_video_gen,
            TaskType.TTS: self._handle_tts,
            TaskType.MUSIC: self._handle_music,
            TaskType.COMPOSE: self._handle_compose,
            TaskType.RENDER: self._handle_render,
        }

    def register_handler(self, task_type: TaskType, handler: NodeHandler) -> None:
        """Override the handler for a specific task type."""
        self._handlers[task_type] = handler

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> ProjectState:
        """Execute the full DAG and return the final project state."""
        self.state.status = ProjectStatus.GENERATING
        self._checkpoint()

        while not self.dag.is_complete:
            ready = self.dag.get_ready_nodes()
            if not ready:
                # Safety valve: if no nodes are ready and the DAG isn't complete
                # it means all remaining nodes are blocked by failures.
                if not any(
                    n.status == NodeStatus.RUNNING for n in self.dag.nodes.values()
                ):
                    logger.error("DAG stalled -- remaining nodes blocked by failures")
                    break
                # Some nodes are still running; wait briefly and re-check.
                await asyncio.sleep(0.05)
                continue

            tasks = [self._run_node(node) for node in ready]
            await asyncio.gather(*tasks)

        # Final status
        if self.dag.has_failures:
            self.state.status = ProjectStatus.FAILED
        else:
            self.state.status = ProjectStatus.COMPLETED
            await self.bus.emit(PROJECT_COMPLETED, {"project_id": self.state.project_id})

        self._checkpoint()
        logger.info(
            "Project %s finished with status %s",
            self.state.project_id,
            self.state.status.value,
        )
        return self.state

    # ------------------------------------------------------------------
    # Per-node execution
    # ------------------------------------------------------------------

    async def _run_node(self, node: TaskNode) -> None:
        """Acquire the semaphore, then execute and finalise a single node."""
        async with self._semaphore:
            await self._execute_node(node)

    async def _execute_node(self, node: TaskNode) -> None:
        """Dispatch *node* to its handler, with retry logic."""
        handler = self._handlers.get(node.task_type)
        if handler is None:
            self.dag.mark_failed(node.node_id, f"No handler for {node.task_type}")
            await self.bus.emit(TASK_FAILED, {
                "node_id": node.node_id,
                "error": f"No handler for {node.task_type}",
            })
            self._checkpoint()
            return

        self.dag.mark_running(node.node_id)
        await self.bus.emit(TASK_STARTED, {
            "node_id": node.node_id,
            "task_type": node.task_type.value,
        })

        max_attempts = self._config.max_retries + 1
        last_error: str = ""

        for attempt in range(1, max_attempts + 1):
            try:
                result = await handler(node, self.state)
                self.dag.mark_complete(node.node_id, result)
                await self.bus.emit(TASK_COMPLETED, {
                    "node_id": node.node_id,
                    "task_type": node.task_type.value,
                    "result": result,
                })
                self._checkpoint()
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Node %s attempt %d/%d failed: %s",
                    node.node_id,
                    attempt,
                    max_attempts,
                    last_error,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(0.1 * attempt)  # simple back-off

        # Exhausted retries
        self.dag.mark_failed(node.node_id, last_error)
        await self.bus.emit(TASK_FAILED, {
            "node_id": node.node_id,
            "task_type": node.task_type.value,
            "error": last_error,
        })
        self._checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint(self) -> None:
        """Persist current project state to disk."""
        try:
            self.state_manager.save(self.state)
        except Exception:
            logger.exception("Failed to checkpoint state for %s", self.state.project_id)

    # ------------------------------------------------------------------
    # Placeholder handlers (Phase 1)
    # ------------------------------------------------------------------
    # Each handler receives the TaskNode and the current ProjectState.
    # Real implementations will be injected via register_handler() or
    # by replacing these methods in later phases.

    async def _handle_script_gen(self, node: TaskNode, state: ProjectState) -> Any:
        logger.info("[placeholder] Generating script for: %s", node.params.get("prompt", ""))
        await asyncio.sleep(0.01)
        script = f"[Generated script for: {state.prompt}]"
        state.script = script
        return {"script": script}

    async def _handle_storyboard(self, node: TaskNode, state: ProjectState) -> Any:
        logger.info("[placeholder] Building storyboard")
        await asyncio.sleep(0.01)
        # In a real pipeline the LLM would decompose the script into shots.
        # For now we leave the storyboard as-is if already populated.
        return {"shot_count": len(state.storyboard)}

    async def _handle_video_gen(self, node: TaskNode, state: ProjectState) -> Any:
        shot_id = node.params.get("shot_id", "unknown")
        logger.info("[placeholder] Generating video for shot %s", shot_id)
        await asyncio.sleep(0.01)
        asset_path = f"shots/{shot_id}.mp4"
        return {"asset_path": asset_path}

    async def _handle_tts(self, node: TaskNode, state: ProjectState) -> Any:
        logger.info("[placeholder] Generating TTS narration")
        await asyncio.sleep(0.01)
        return {"asset_path": "audio/narration.wav"}

    async def _handle_music(self, node: TaskNode, state: ProjectState) -> Any:
        logger.info("[placeholder] Generating background music")
        await asyncio.sleep(0.01)
        return {"asset_path": "audio/music.wav"}

    async def _handle_compose(self, node: TaskNode, state: ProjectState) -> Any:
        logger.info("[placeholder] Composing timeline")
        await asyncio.sleep(0.01)
        return {"timeline": "composed"}

    async def _handle_render(self, node: TaskNode, state: ProjectState) -> Any:
        logger.info("[placeholder] Rendering final video")
        await asyncio.sleep(0.01)
        output_path = f"output/{state.project_id}.mp4"
        state.assets["final_video"] = output_path
        return {"output_path": output_path}
