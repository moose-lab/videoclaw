"""Base Video Agent protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class AgentRole(StrEnum):
    DIRECTOR = "director"
    SCRIPTWRITER = "scriptwriter"
    CAMERAMAN = "cameraman"
    REVIEWER = "reviewer"
    PRODUCER = "producer"
    PUBLISHER = "publisher"


class ReviewVerdict(StrEnum):
    APPROVED = "approved"
    RETRY = "retry"
    MODIFY = "modify"
    REJECT = "reject"


@dataclass
class AgentPlan:
    """An agent's intended actions."""

    agent_role: AgentRole
    steps: list[AgentStep]
    reasoning: str = ""


@dataclass
class AgentStep:
    """A single step in an agent's plan."""

    action: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class AgentResult:
    """Result of an agent's action."""

    agent_role: AgentRole
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cost_usd: float = 0.0


@dataclass
class AgentMessage:
    """Inter-agent communication message."""

    from_role: AgentRole
    to_role: AgentRole
    content: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewResult:
    """Quality review result from the Reviewer agent."""

    verdict: ReviewVerdict
    score: float  # 0.0 - 10.0
    feedback: str
    suggestions: list[str] = field(default_factory=list)


@runtime_checkable
class VideoAgent(Protocol):
    """Protocol for all Video Agents in the system.

    Agents are autonomous units that can think, act, review results,
    and collaborate with other agents.
    """

    @property
    def role(self) -> AgentRole: ...

    @property
    def tools(self) -> list[str]: ...

    async def think(self, context: dict[str, Any]) -> AgentPlan:
        """Analyze current project state and produce an action plan."""
        ...

    async def act(self, plan: AgentPlan) -> AgentResult:
        """Execute the plan using available tools."""
        ...

    async def review(self, result: AgentResult) -> ReviewResult:
        """Review a result and decide whether it passes quality."""
        ...

    async def collaborate(self, message: AgentMessage) -> AgentMessage:
        """Receive and respond to a message from another agent."""
        ...
