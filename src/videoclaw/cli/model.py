"""``claw model`` -- model adapter management."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from videoclaw.cli._app import model_app, status_icon
from videoclaw.cli._output import get_console, get_output


@model_app.command("list")
def model_list() -> None:
    """List all registered model adapters and their health status."""
    console = get_console()
    out = get_output()
    out._command = "model.list"

    from videoclaw.models.registry import get_registry

    registry = get_registry()
    registry.discover()

    models = registry.list_models()
    if not models:
        console.print("[yellow]No model adapters registered.[/yellow]")
        console.print("Adapters are auto-discovered via the [cyan]videoclaw.adapters[/cyan] entry-point group.")
        out.set_result({"models": []})
        out.emit()
        return

    # Run health checks asynchronously.
    try:
        health = asyncio.run(registry.health_check_all())
    except Exception:
        health = {}  # Graceful degradation: show models without health status

    table = Table(
        title="Registered Model Adapters",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Model ID", style="cyan", min_width=16)
    table.add_column("Capabilities", style="white")
    table.add_column("Mode", style="magenta")
    table.add_column("Health", justify="center")

    result_models = []
    for m in models:
        mid = m["model_id"]
        caps = ", ".join(m["capabilities"])
        mode = m["execution_mode"]
        ok = health.get(mid, False)
        table.add_row(mid, caps, mode, status_icon(ok))
        result_models.append({**m, "healthy": ok})

    console.print(table)

    out.set_result({"models": result_models})
    out.emit()


@model_app.command("pull")
def model_pull(
    model_id: Annotated[str, typer.Argument(help="Identifier of the model to pull.")],
) -> None:
    """Download / prepare a local model for offline generation."""
    console = get_console()
    out = get_output()
    out._command = "model.pull"

    console.print(
        Panel(
            f"[bold]Would download model:[/bold] {model_id}\n\n"
            "Local model management is coming in [cyan]v0.2[/cyan].\n"
            "This will pull weights into the models cache directory and\n"
            "register the adapter for subsequent runs.",
            title="[bold yellow]Model Pull (placeholder)[/bold yellow]",
            border_style="yellow",
        )
    )

    out.set_result({"model_id": model_id, "status": "not_implemented"})
    out.emit()
