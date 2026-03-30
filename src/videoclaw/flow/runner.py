"""Compile a FlowDef into a DAG and execute it."""

from __future__ import annotations

import logging

from videoclaw.core.events import EventBus
from videoclaw.core.events import event_bus as default_event_bus
from videoclaw.core.executor import DAGExecutor, NodeHandler
from videoclaw.core.planner import DAG, TaskNode, TaskType
from videoclaw.core.state import ProjectState, StateManager
from videoclaw.flow.parser import FlowDef

logger = logging.getLogger(__name__)


def compile_dag(flow: FlowDef) -> DAG:
    """Convert a :class:`FlowDef` into a :class:`DAG` ready for execution."""
    dag = DAG()
    for step in flow.steps:
        dag.add_node(TaskNode(
            node_id=step.id,
            task_type=step.type,
            depends_on=list(step.depends_on),
            params=dict(step.params),
        ))
    return dag


class FlowRunner:
    """High-level runner: load a flow, build a DAG, execute it.

    Parameters
    ----------
    state_manager:
        Persistence layer for project state.
    bus:
        Event bus for progress notifications.
    max_concurrency:
        Maximum parallel tasks.
    handlers:
        Optional mapping of TaskType → async handler to override defaults.
    """

    def __init__(
        self,
        state_manager: StateManager | None = None,
        bus: EventBus | None = None,
        max_concurrency: int = 4,
        handlers: dict[TaskType, NodeHandler] | None = None,
    ) -> None:
        self.state_manager = state_manager or StateManager()
        self.bus = bus or default_event_bus
        self.max_concurrency = max_concurrency
        self._custom_handlers = handlers or {}

    async def run(self, flow: FlowDef, project_state: ProjectState) -> ProjectState:
        """Compile *flow* to a DAG and execute it against *project_state*.

        Returns the updated :class:`ProjectState` after execution completes.
        """
        logger.info("Running flow %r for project %s", flow.name, project_state.project_id)

        dag = compile_dag(flow)
        executor = DAGExecutor(
            dag=dag,
            state=project_state,
            state_manager=self.state_manager,
            bus=self.bus,
            max_concurrency=self.max_concurrency,
        )

        # Register any custom handlers (e.g. real model adapters).
        for task_type, handler in self._custom_handlers.items():
            executor.register_handler(task_type, handler)

        return await executor.run()
