"""``claw project`` -- project management commands."""

from __future__ import annotations

import shutil
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from videoclaw.cli._app import project_app
from videoclaw.cli._output import get_console, get_output
from videoclaw.config import get_config


@project_app.command("list")
def project_list() -> None:
    """List all saved projects."""
    console = get_console()
    out = get_output()
    out._command = "project.list"

    from videoclaw.core.state import StateManager

    sm = StateManager()
    project_ids = sm.list_projects()

    if not project_ids:
        console.print("[yellow]No projects found.[/yellow]")
        out.set_result({"projects": []})
        out.emit()
        raise typer.Exit()

    table = Table(
        title="Projects",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Project ID", style="cyan", min_width=20)
    table.add_column("Status", style="magenta")
    table.add_column("Shots", justify="right")
    table.add_column("Prompt", style="white")
    table.add_column("Created", style="dim")

    projects = []
    for pid in sorted(project_ids):
        try:
            state = sm.load(pid)
            table.add_row(
                pid,
                state.status.value,
                str(len(state.storyboard)),
                state.prompt[:50] + ("..." if len(state.prompt) > 50 else ""),
                state.created_at[:19],
            )
            projects.append({
                "project_id": pid,
                "status": state.status.value,
                "shots": len(state.storyboard),
                "prompt": state.prompt,
                "created_at": state.created_at,
            })
        except Exception:
            table.add_row(pid, "[red]error[/red]", "-", "-", "-")
            projects.append({"project_id": pid, "status": "error"})

    console.print(table)

    out.set_result({"projects": projects})
    out.emit()


@project_app.command("show")
def project_show(
    project_id: Annotated[str, typer.Argument(help="Project identifier.")],
) -> None:
    """Display detailed information about a project."""
    console = get_console()
    out = get_output()
    out._command = "project.show"

    from videoclaw.core.state import StateManager

    sm = StateManager()
    try:
        state = sm.load(project_id)
    except FileNotFoundError:
        console.print(f"[red]Project {project_id!r} not found.[/red]")
        out.set_error(f"Project {project_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]ID:[/bold]      {state.project_id}\n"
            f"[bold]Status:[/bold]  {state.status.value}\n"
            f"[bold]Prompt:[/bold]  {state.prompt}\n"
            f"[bold]Created:[/bold] {state.created_at}\n"
            f"[bold]Updated:[/bold] {state.updated_at}\n"
            f"[bold]Cost:[/bold]    ${state.cost_total:.4f}",
            title="[bold green]Project Details[/bold green]",
            border_style="green",
        )
    )

    shots_data = []
    if state.storyboard:
        shot_table = Table(
            title="Storyboard",
            show_header=True,
            header_style="bold magenta",
        )
        shot_table.add_column("#", width=4, style="dim")
        shot_table.add_column("Shot ID", style="cyan")
        shot_table.add_column("Status", style="magenta")
        shot_table.add_column("Model", style="white")
        shot_table.add_column("Duration", justify="right", style="green")
        shot_table.add_column("Cost", justify="right", style="yellow")
        shot_table.add_column("Description", style="dim")

        for idx, shot in enumerate(state.storyboard, 1):
            shot_table.add_row(
                str(idx),
                shot.shot_id,
                shot.status.value,
                shot.model_id,
                f"{shot.duration_seconds:.1f}s",
                f"${shot.cost:.4f}",
                shot.description[:40],
            )
            shots_data.append({
                "shot_id": shot.shot_id,
                "status": shot.status.value,
                "model_id": shot.model_id,
                "duration_seconds": shot.duration_seconds,
                "cost": shot.cost,
                "description": shot.description,
            })
        console.print(shot_table)

    cost_path = get_config().projects_dir / project_id / "cost.json"
    if cost_path.exists():
        console.print(f"\n[dim]Cost details: {cost_path}[/dim]")

    out.set_result({
        "project_id": state.project_id,
        "status": state.status.value,
        "prompt": state.prompt,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "cost_total": state.cost_total,
        "shots": shots_data,
    })
    out.emit()


@project_app.command("delete")
def project_delete(
    project_id: Annotated[str, typer.Argument(help="Project ID to delete.")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation prompt.")] = False,
) -> None:
    """Delete a project and all its generated assets."""
    console = get_console()
    out = get_output()
    out._command = "project.delete"

    cfg = get_config()
    project_dir = cfg.projects_dir / project_id

    if not project_dir.exists():
        console.print(f"[red]Project {project_id!r} not found.[/red]")
        out.set_error(f"Project {project_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    if not force and not out.json_mode:
        confirm = typer.confirm(
            f"Delete project {project_id!r} and all its assets?",
            default=False,
        )
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit()

    shutil.rmtree(project_dir)
    console.print(f"[green]Project {project_id!r} deleted.[/green]")

    out.set_result({"project_id": project_id, "deleted": True})
    out.emit()
