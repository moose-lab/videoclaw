"""``claw template`` -- flow template management."""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from videoclaw.cli._app import template_app, resolve_templates_dir
from videoclaw.cli._output import get_console, get_output


@template_app.command("list")
def template_list() -> None:
    """List available flow templates."""
    console = get_console()
    out = get_output()
    out._command = "template.list"

    tpl_dir = resolve_templates_dir()

    if not tpl_dir.exists():
        console.print("[yellow]No templates directory found.[/yellow]")
        out.set_result({"templates": []})
        out.emit()
        raise typer.Exit()

    templates = sorted(tpl_dir.glob("*.claw.yaml"))
    if not templates:
        console.print("[yellow]No .claw.yaml templates found.[/yellow]")
        out.set_result({"templates": []})
        out.emit()
        raise typer.Exit()

    table = Table(
        title="Flow Templates",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="cyan", min_width=20)
    table.add_column("Description", style="white")
    table.add_column("Path", style="dim")

    import yaml  # type: ignore[import-untyped]

    result_templates = []
    for tpl_path in templates:
        try:
            data = yaml.safe_load(tpl_path.read_text(encoding="utf-8"))
            name = data.get("name", tpl_path.stem)
            desc = data.get("description", "-")
        except Exception:
            name = tpl_path.stem
            desc = "[red]parse error[/red]"

        table.add_row(name, desc, str(tpl_path))
        result_templates.append({"name": name, "description": desc, "path": str(tpl_path)})

    console.print(table)

    out.set_result({"templates": result_templates})
    out.emit()


@template_app.command("use")
def template_use(
    name: Annotated[str, typer.Argument(help="Template name (without extension).")],
    prompt: Annotated[Optional[str], typer.Option("--prompt", "-p", help="Override the default prompt.")] = None,
) -> None:
    """Generate a video using a predefined flow template."""
    console = get_console()
    out = get_output()
    out._command = "template.use"

    tpl_dir = resolve_templates_dir()
    tpl_path = tpl_dir / f"{name}.claw.yaml"

    if not tpl_path.exists():
        console.print(f"[red]Template {name!r} not found at {tpl_path}[/red]")
        out.set_error(f"Template {name!r} not found at {tpl_path}")
        out.emit()
        raise typer.Exit(code=1)

    import yaml  # type: ignore[import-untyped]

    data = yaml.safe_load(tpl_path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})

    console.print(
        Panel(
            f"[bold]Template:[/bold]    {data.get('name', name)}\n"
            f"[bold]Description:[/bold] {data.get('description', '-')}\n"
            f"[bold]Duration:[/bold]    {defaults.get('duration', 30)}s\n"
            f"[bold]Style:[/bold]       {defaults.get('style', 'default')}\n"
            f"[bold]Pipeline:[/bold]    {' -> '.join(data.get('pipeline', {}).keys())}",
            title="[bold cyan]Template Loaded[/bold cyan]",
            border_style="cyan",
        )
    )

    effective_prompt = prompt or f"Generate a {data.get('name', name)} video"
    console.print(f"\n[dim]Launching generation with prompt: {effective_prompt!r}[/dim]\n")

    # Delegate to the generate command (its JSON output will be emitted by generate itself).
    from videoclaw.cli.generate import generate

    generate(
        prompt=effective_prompt,
        duration=defaults.get("duration", 30),
        style=defaults.get("style"),
        aspect_ratio=defaults.get("aspect_ratio", "16:9"),
        strategy=defaults.get("strategy", "auto"),
        output=None,
        budget=None,
        model=None,
        concurrency=4,
        dry_run=False,
        verbose=False,
    )
