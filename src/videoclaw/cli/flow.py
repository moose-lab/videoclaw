"""``claw flow`` -- ClawFlow YAML pipeline execution and validation."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from videoclaw.cli._app import configure_logging, flow_app, show_banner
from videoclaw.cli._output import get_console, get_output
from videoclaw.config import get_config


@flow_app.command("run")
def flow_run(
    path: Annotated[str, typer.Argument(help="Path to a ClawFlow YAML file.")],
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="Override the script prompt."),
    ] = None,
    budget: Annotated[
        float | None,
        typer.Option("--budget", "-b", help="Maximum budget in USD."),
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", help="Max parallel tasks.")
    ] = 4,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Execute a video pipeline defined in a ClawFlow YAML file."""
    configure_logging(verbose)
    show_banner()
    console = get_console()
    out = get_output()
    out._command = "flow.run"

    from videoclaw.core.state import StateManager
    from videoclaw.flow.parser import FlowValidationError, load_flow
    from videoclaw.flow.runner import FlowRunner

    try:
        flow = load_flow(path)
    except (FileNotFoundError, FlowValidationError) as exc:
        console.print(f"[red]Error loading flow: {exc}[/red]")
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Flow:[/bold]    {flow.name}\n"
            f"[bold]Desc:[/bold]    {flow.description}\n"
            f"[bold]Steps:[/bold]   {len(flow.steps)}\n"
            f"[bold]Version:[/bold] {flow.version}",
            title="[bold cyan]ClawFlow Pipeline[/bold cyan]",
            border_style="cyan",
        )
    )

    # Override script prompt if provided.
    if prompt:
        for step in flow.steps:
            if "prompt" in step.params:
                step.params["prompt"] = prompt
                break

    cfg = get_config()
    cfg.ensure_dirs()
    sm = StateManager()
    state = sm.create_project(prompt=prompt or flow.name)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running flow...", total=len(flow.steps))

            async def _run() -> None:
                runner = FlowRunner(state_manager=sm, max_concurrency=concurrency)
                await runner.run(flow, state)

            asyncio.run(_run())
            progress.update(task, completed=len(flow.steps))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Project:[/bold] {state.project_id}\n"
            f"[bold]Status:[/bold]  {state.status.value}",
            title="[bold green]Flow Complete[/bold green]",
            border_style="green",
        )
    )

    out.set_result({
        "project_id": state.project_id,
        "status": state.status.value,
        "flow": flow.name,
        "steps": len(flow.steps),
    })
    out.emit()


@flow_app.command("validate")
def flow_validate(
    path: Annotated[str, typer.Argument(help="Path to a ClawFlow YAML file.")],
) -> None:
    """Validate a ClawFlow YAML file without executing it."""
    console = get_console()
    out = get_output()
    out._command = "flow.validate"

    from videoclaw.flow.parser import FlowValidationError, load_flow

    try:
        flow = load_flow(path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)
    except FlowValidationError as exc:
        console.print(f"[red]Validation failed: {exc}[/red]")
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    # Display validated flow summary.
    table = Table(title=f"Flow: {flow.name}", show_header=True, header_style="bold cyan")
    table.add_column("Step ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Depends On", style="dim")
    table.add_column("Params", style="white")

    steps_data = []
    for step in flow.steps:
        table.add_row(
            step.id,
            step.type.value,
            ", ".join(step.depends_on) or "-",
            ", ".join(f"{k}={v}" for k, v in step.params.items()) or "-",
        )
        steps_data.append({
            "id": step.id,
            "type": step.type.value,
            "depends_on": step.depends_on,
            "params": step.params,
        })

    console.print(table)
    console.print("[bold green]Flow is valid.[/bold green]")

    out.set_result({"valid": True, "name": flow.name, "steps": steps_data})
    out.emit()
