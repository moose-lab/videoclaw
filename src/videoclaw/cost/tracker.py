"""Real-time cost tracking engine for VideoClaw projects.

Tracks per-task costs (API, compute, tokens), enforces budgets, and
provides optimisation hints to help users minimise spend.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from videoclaw.core.events import COST_UPDATED, event_bus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CostRecord:
    """Immutable ledger entry for a single executed task."""

    task_id: str
    model_id: str
    execution_mode: str  # "local" | "cloud"
    api_cost_usd: float
    compute_cost_usd: float  # estimated local GPU electricity cost
    duration_seconds: float
    task_type: str = "video_gen"  # e.g. "video_gen", "llm", "tts", "compose", "render"
    input_tokens: int = 0
    output_tokens: int = 0
    video_seconds: float = 0.0
    retries: int = 0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def total_usd(self) -> float:
        """Convenience accessor: combined API + compute cost."""
        return self.api_cost_usd + self.compute_cost_usd


@dataclass(slots=True)
class ProjectCostSummary:
    """Aggregate cost view for a whole project."""

    project_id: str
    total_usd: float
    cloud_usd: float
    local_usd: float
    records: list[CostRecord]
    by_model: dict[str, float]
    by_task_type: dict[str, float]


@dataclass(frozen=True, slots=True)
class CostHint:
    """Actionable recommendation to reduce project spend."""

    message: str
    potential_savings_usd: float
    action: str  # e.g. "switch_to_local", "reduce_retries", "batch_shots"


# ---------------------------------------------------------------------------
# Pricing — delegates to the centralised MODEL_PROFILES registry in
# ``videoclaw.models.router`` for cloud per-second rates.
# ---------------------------------------------------------------------------

_LOCAL_COST_PER_SEC: float = 0.002  # rough electricity estimate


def _cloud_price_per_sec(model_id: str) -> float:
    """Look up cloud price from the authoritative MODEL_PROFILES registry."""
    from videoclaw.models.router import get_price_usd_per_sec

    return get_price_usd_per_sec(model_id)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Accumulates :class:`CostRecord` entries and exposes budget awareness.

    Parameters
    ----------
    project_id:
        Identifier of the project being tracked.
    budget_usd:
        Optional hard budget cap.  If set, :meth:`check_budget` can be used
        to verify the project remains within budget before executing tasks.
    """

    def __init__(self, project_id: str, budget_usd: float | None = None) -> None:
        self.project_id = project_id
        self.budget_usd = budget_usd
        self._records: list[CostRecord] = []
        self._total_usd: float = 0.0  # O(1) running total accumulator
        self._background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks

    # -- Mutation -----------------------------------------------------------

    def record(self, cost: CostRecord) -> None:
        """Append *cost* to the ledger and emit a :data:`COST_UPDATED` event.

        The event is scheduled onto the running event loop.  If no loop is
        running the event is silently skipped (e.g. during unit tests).
        """
        self._records.append(cost)
        self._total_usd += cost.total_usd
        logger.debug(
            "Recorded cost for task=%s model=%s total=$%.4f cumulative=$%.4f",
            cost.task_id,
            cost.model_id,
            cost.total_usd,
            self._total_usd,
        )
        # Fire-and-forget event emission with full billing payload.
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                event_bus.emit(
                    COST_UPDATED,
                    {
                        "project_id": self.project_id,
                        "task_id": cost.task_id,
                        "task_type": cost.task_type,
                        "model_id": cost.model_id,
                        "record_usd": cost.total_usd,
                        "total_usd": self._total_usd,
                        "budget_usd": self.budget_usd,
                        "within_budget": (
                            self._total_usd <= self.budget_usd
                            if self.budget_usd is not None
                            else True
                        ),
                        "record_count": len(self._records),
                    },
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            # No running event loop -- skip event emission.
            pass

    # -- Queries ------------------------------------------------------------

    def get_summary(self) -> ProjectCostSummary:
        """Build a full :class:`ProjectCostSummary` from recorded entries."""
        cloud_usd = 0.0
        local_usd = 0.0
        by_model: dict[str, float] = defaultdict(float)
        by_task: dict[str, float] = defaultdict(float)

        for r in self._records:
            total = r.total_usd
            if r.execution_mode == "cloud":
                cloud_usd += total
            else:
                local_usd += total
            by_model[r.model_id] += total
            by_task[r.task_type] += total

        return ProjectCostSummary(
            project_id=self.project_id,
            total_usd=cloud_usd + local_usd,
            cloud_usd=cloud_usd,
            local_usd=local_usd,
            records=list(self._records),
            by_model=dict(by_model),
            by_task_type=dict(by_task),
        )

    def get_optimization_hints(self) -> list[CostHint]:
        """Analyse recorded costs and return actionable savings suggestions."""
        hints: list[CostHint] = []

        # 1. Detect expensive cloud shots that could run locally.
        cloud_shots: dict[str, list[CostRecord]] = defaultdict(list)
        for r in self._records:
            if r.execution_mode == "cloud" and r.video_seconds > 0:
                cloud_shots[r.model_id].append(r)

        for model_id, records in cloud_shots.items():
            cloud_total = sum(r.total_usd for r in records)
            local_estimate = sum(r.video_seconds * _LOCAL_COST_PER_SEC for r in records)
            savings = cloud_total - local_estimate
            if savings > 0.01:
                shot_range = f"Shots {records[0].task_id}..{records[-1].task_id}"
                hints.append(
                    CostHint(
                        message=(
                            f"{shot_range} used {model_id} at "
                            f"${cloud_total:.2f} total. "
                            f"Using a local model would save ~${savings:.2f}."
                        ),
                        potential_savings_usd=round(savings, 4),
                        action="switch_to_local",
                    )
                )

        # 2. Flag tasks with high retry counts.
        retry_waste = sum(
            r.total_usd * (r.retries / max(r.retries + 1, 1))
            for r in self._records
            if r.retries >= 2
        )
        if retry_waste > 0.01:
            hints.append(
                CostHint(
                    message=(
                        f"Retries across tasks wasted ~${retry_waste:.2f}. "
                        "Consider using a more reliable model or adjusting prompts."
                    ),
                    potential_savings_usd=round(retry_waste, 4),
                    action="reduce_retries",
                )
            )

        # 3. Suggest batching if many small shots went to the same cloud model.
        for model_id, records in cloud_shots.items():
            if len(records) >= 4:
                hints.append(
                    CostHint(
                        message=(
                            f"{len(records)} individual shots sent to {model_id}. "
                            "Batching could reduce per-request overhead."
                        ),
                        potential_savings_usd=round(len(records) * 0.005, 4),
                        action="batch_shots",
                    )
                )

        return hints

    def estimate_remaining(self, pending_tasks: list[dict[str, Any]]) -> float:
        """Rough USD estimate for *pending_tasks*.

        Each dict should contain at minimum ``model_id`` and
        ``duration_seconds`` keys.
        """
        total = 0.0
        for task in pending_tasks:
            model_id: str = task.get("model_id", "mock")
            duration: float = float(task.get("duration_seconds", 5.0))
            mode: str = task.get("execution_mode", "cloud")
            if mode == "cloud":
                rate = _cloud_price_per_sec(model_id)
                total += duration * rate
            else:
                total += duration * _LOCAL_COST_PER_SEC
        return round(total, 4)

    def check_budget(self) -> tuple[bool, float]:
        """Return ``(within_budget, remaining_usd)``.

        If no budget was set the project is always considered within budget
        and ``remaining`` is reported as ``float('inf')``.
        """
        spent = self._running_total()
        if self.budget_usd is None:
            return True, float("inf")
        remaining = self.budget_usd - spent
        return remaining >= 0, round(remaining, 4)

    # -- Display ------------------------------------------------------------

    def format_table(self) -> str:
        """Render the cost summary as a Rich-compatible table string.

        Returns a string that can be printed directly or wrapped in a
        :class:`rich.panel.Panel`.
        """
        from rich.console import Console
        from rich.table import Table

        summary = self.get_summary()

        table = Table(
            title=f"Cost Summary -- {self.project_id}",
            show_header=True,
            header_style="bold cyan",
            title_style="bold white",
        )
        table.add_column("Task", style="white", min_width=14)
        table.add_column("Model", style="magenta")
        table.add_column("Mode", style="blue")
        table.add_column("API $", justify="right", style="yellow")
        table.add_column("Compute $", justify="right", style="yellow")
        table.add_column("Total $", justify="right", style="bold green")
        table.add_column("Duration", justify="right", style="dim")
        table.add_column("Tokens", justify="right", style="dim")

        for r in summary.records:
            table.add_row(
                r.task_id,
                r.model_id,
                r.execution_mode,
                f"${r.api_cost_usd:.4f}",
                f"${r.compute_cost_usd:.4f}",
                f"${r.total_usd:.4f}",
                f"{r.duration_seconds:.1f}s",
                (
                    f"{r.input_tokens + r.output_tokens:,}"
                    if r.input_tokens + r.output_tokens
                    else "-"
                ),
            )

        # Totals row.
        table.add_section()
        table.add_row(
            "TOTAL",
            "",
            "",
            "",
            "",
            f"[bold]${summary.total_usd:.4f}[/bold]",
            "",
            "",
        )

        # Budget row when applicable.
        if self.budget_usd is not None:
            within, remaining = self.check_budget()
            status_style = "green" if within else "bold red"
            table.add_row(
                "BUDGET",
                "",
                "",
                "",
                "",
                f"[{status_style}]${remaining:.4f} remaining[/{status_style}]",
                "",
                "",
            )

        # Render to string via an in-memory console.
        console = Console(width=120, force_terminal=True)
        with console.capture() as capture:
            console.print(table)
        return capture.get()

    # -- Persistence --------------------------------------------------------

    def save_ledger(self, path: Path) -> None:
        """Persist all records to a JSON file for later replay.

        The file is written atomically (write-to-temp then rename) to
        avoid corruption if the process is interrupted mid-write.
        """
        records_data = []
        for r in self._records:
            d = asdict(r)
            d["timestamp"] = r.timestamp.isoformat()
            records_data.append(d)

        payload = {
            "project_id": self.project_id,
            "budget_usd": self.budget_usd,
            "total_usd": self._total_usd,
            "record_count": len(records_data),
            "records": records_data,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        logger.debug("Saved cost ledger (%d records) to %s", len(records_data), path)

    @classmethod
    def load_ledger(cls, path: Path) -> CostTracker:
        """Restore a tracker from a previously saved ledger file.

        Raises:
            FileNotFoundError: if *path* does not exist.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        tracker = cls(
            project_id=data["project_id"],
            budget_usd=data.get("budget_usd"),
        )
        for rd in data.get("records", []):
            rd["timestamp"] = datetime.fromisoformat(rd["timestamp"])
            record = CostRecord(**rd)
            tracker._records.append(record)
            tracker._total_usd += record.total_usd
        logger.debug("Loaded cost ledger (%d records) from %s", len(tracker._records), path)
        return tracker

    # -- Internals ----------------------------------------------------------

    def _running_total(self) -> float:
        return self._total_usd
