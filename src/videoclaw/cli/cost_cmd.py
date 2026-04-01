"""CLI commands for cost tracking — replay, query, and summary.

Registered as ``claw cost`` sub-commands via the Typer ``cost_app``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from videoclaw.cli._app import app
from videoclaw.cli._output import get_console, get_output
from videoclaw.config import get_config

cost_app = typer.Typer(help="Cost tracking and budget analysis.", no_args_is_help=True)
app.add_typer(cost_app, name="cost")


@cost_app.command("show")
def cost_show(
    project_id: Annotated[str, typer.Argument(help="Project or series ID.")],
) -> None:
    """Display the cost ledger for a project with Rich table and hints."""
    console = get_console()
    out = get_output()
    out._command = "cost show"
    cfg = get_config()

    # Try project-level cost.json first, then drama series
    cost_path = cfg.projects_dir / project_id / "cost.json"
    if not cost_path.exists():
        # Try drama deliverables path
        drama_cost = _find_drama_cost(project_id)
        if drama_cost:
            cost_path = drama_cost
        else:
            console.print(f"[red]No cost ledger found for {project_id!r}[/red]")
            console.print(
                f"[dim]Looked in: {cfg.projects_dir / project_id / 'cost.json'}[/dim]"
            )
            out.set_error(f"No cost ledger for {project_id}")
            out.emit()
            raise typer.Exit(1)

    from videoclaw.cost.tracker import CostTracker

    tracker = CostTracker.load_ledger(cost_path)
    summary = tracker.get_summary()

    # Rich table display
    console.print(tracker.format_table())

    # By-model breakdown
    if summary.by_model:
        console.print("\n[bold]Cost by Model:[/bold]")
        for model, cost in sorted(summary.by_model.items(), key=lambda x: -x[1]):
            bar = _bar(cost, summary.total_usd)
            console.print(f"  {model:<24} ${cost:.4f}  {bar}")

    # By-task-type breakdown
    if summary.by_task_type:
        console.print("\n[bold]Cost by Task Type:[/bold]")
        for ttype, cost in sorted(summary.by_task_type.items(), key=lambda x: -x[1]):
            bar = _bar(cost, summary.total_usd)
            console.print(f"  {ttype:<24} ${cost:.4f}  {bar}")

    # Optimization hints
    hints = tracker.get_optimization_hints()
    if hints:
        console.print("\n[bold yellow]Optimization Hints:[/bold yellow]")
        for hint in hints:
            console.print(
                f"  [yellow]*[/yellow] {hint.message} "
                f"[dim](~${hint.potential_savings_usd:.2f} savings)[/dim]"
            )

    out.set_result({
        "project_id": summary.project_id,
        "total_usd": summary.total_usd,
        "cloud_usd": summary.cloud_usd,
        "local_usd": summary.local_usd,
        "record_count": len(summary.records),
        "by_model": summary.by_model,
        "by_task_type": summary.by_task_type,
        "hints": [
            {"message": h.message, "savings": h.potential_savings_usd, "action": h.action}
            for h in hints
        ],
    })
    out.emit()


@cost_app.command("list")
def cost_list() -> None:
    """List all projects that have cost ledgers."""
    console = get_console()
    out = get_output()
    out._command = "cost list"
    cfg = get_config()

    from videoclaw.cost.tracker import CostTracker

    results: list[dict] = []

    if cfg.projects_dir.exists():
        for project_dir in sorted(cfg.projects_dir.iterdir()):
            cost_path = project_dir / "cost.json"
            if cost_path.exists():
                tracker = CostTracker.load_ledger(cost_path)
                summary = tracker.get_summary()
                results.append({
                    "project_id": summary.project_id,
                    "total_usd": summary.total_usd,
                    "records": len(summary.records),
                })
                console.print(
                    f"  {summary.project_id:<20} "
                    f"${summary.total_usd:.4f}  "
                    f"({len(summary.records)} records)"
                )

    if not results:
        console.print("[dim]No cost ledgers found.[/dim]")

    out.set_result({"projects": results})
    out.emit()


@cost_app.command("compare")
def cost_compare(
    project_ids: Annotated[list[str], typer.Argument(help="Two or more project IDs to compare.")],
) -> None:
    """Compare costs across multiple projects side by side."""
    console = get_console()
    out = get_output()
    out._command = "cost compare"
    cfg = get_config()

    from rich.table import Table

    from videoclaw.cost.tracker import CostTracker

    table = Table(title="Cost Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Project", style="white")
    table.add_column("Total $", justify="right", style="bold green")
    table.add_column("Cloud $", justify="right", style="yellow")
    table.add_column("Records", justify="right", style="dim")
    table.add_column("Top Model", style="magenta")

    comparisons = []
    for pid in project_ids:
        cost_path = cfg.projects_dir / pid / "cost.json"
        if not cost_path.exists():
            table.add_row(pid, "[red]N/A[/red]", "", "", "")
            continue
        tracker = CostTracker.load_ledger(cost_path)
        summary = tracker.get_summary()
        top_model = max(summary.by_model, key=summary.by_model.get) if summary.by_model else "-"
        table.add_row(
            pid,
            f"${summary.total_usd:.4f}",
            f"${summary.cloud_usd:.4f}",
            str(len(summary.records)),
            top_model,
        )
        comparisons.append({
            "project_id": pid,
            "total_usd": summary.total_usd,
            "cloud_usd": summary.cloud_usd,
            "records": len(summary.records),
        })

    console.print(table)
    out.set_result({"comparisons": comparisons})
    out.emit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_drama_cost(series_id: str) -> Path | None:
    """Search for a drama series cost.json in deliverables."""
    cfg = get_config()
    # Look in all episode directories under deliverables
    deliverables = Path("docs/deliverables")
    if deliverables.exists():
        for ep_dir in deliverables.iterdir():
            cost_path = ep_dir / "cost.json"
            if cost_path.exists():
                try:
                    import json
                    data = json.loads(cost_path.read_text(encoding="utf-8"))
                    if data.get("project_id", "").startswith(series_id[:8]):
                        return cost_path
                except Exception:
                    pass
    # Also check projects dir with series_id
    cost_path = cfg.projects_dir / series_id / "cost.json"
    if cost_path.exists():
        return cost_path
    return None


def _bar(value: float, total: float, width: int = 20) -> str:
    """Render a proportional bar chart segment."""
    if total <= 0:
        return ""
    ratio = min(value / total, 1.0)
    filled = int(ratio * width)
    return f"[cyan]{'█' * filled}{'░' * (width - filled)}[/cyan] {ratio:.0%}"
