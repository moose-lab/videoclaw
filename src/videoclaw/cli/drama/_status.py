"""``claw drama list`` and ``claw drama show`` commands."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from videoclaw.cli._app import drama_app
from videoclaw.cli._output import get_console, get_output


# ---------------------------------------------------------------------------
# claw drama list
# ---------------------------------------------------------------------------

@drama_app.command("list")
def drama_list() -> None:
    """List all drama series."""
    console = get_console()
    out = get_output()
    out._command = "drama.list"

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    series_ids = mgr.list_series()

    if not series_ids:
        console.print("[yellow]No drama series found.[/yellow]")
        out.set_result({"series": []})
        out.emit()
        return

    table = Table(title="Drama Series", show_header=True, header_style="bold cyan")
    table.add_column("Series ID", style="cyan", min_width=18)
    table.add_column("Title", style="white")
    table.add_column("Status", style="magenta")
    table.add_column("Episodes", justify="right")
    table.add_column("Cost", justify="right", style="yellow")

    result_series = []
    for sid in sorted(series_ids):
        try:
            s = mgr.load(sid)
            completed = sum(1 for ep in s.episodes if ep.status == "completed")
            table.add_row(
                sid,
                s.title or s.synopsis[:30],
                s.status.value,
                f"{completed}/{len(s.episodes)}",
                f"${s.cost_total:.4f}",
            )
            result_series.append({
                "series_id": sid,
                "title": s.title,
                "status": s.status.value,
                "episodes_completed": completed,
                "episodes_total": len(s.episodes),
                "cost_total": s.cost_total,
            })
        except Exception:
            table.add_row(sid, "[red]error[/red]", "-", "-", "-")
            result_series.append({"series_id": sid, "status": "error"})

    console.print(table)

    out.set_result({"series": result_series})
    out.emit()


# ---------------------------------------------------------------------------
# claw drama show
# ---------------------------------------------------------------------------

@drama_app.command("show")
def drama_show(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
) -> None:
    """Show detailed info about a drama series."""
    console = get_console()
    out = get_output()
    out._command = "drama.show"

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        out.set_error(f"Series {series_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]ID:[/bold]       {series.series_id}\n"
            f"[bold]Title:[/bold]    {series.title}\n"
            f"[bold]Genre:[/bold]    {series.genre}\n"
            f"[bold]Status:[/bold]   {series.status.value}\n"
            f"[bold]Style:[/bold]    {series.style}\n"
            f"[bold]Language:[/bold] {series.language}\n"
            f"[bold]Model:[/bold]    {series.model_id}\n"
            f"[bold]Cost:[/bold]     ${series.cost_total:.4f}\n"
            f"[bold]Synopsis:[/bold] {series.synopsis[:100]}",
            title="[bold green]Drama Series[/bold green]",
            border_style="green",
        )
    )

    characters_data = []
    if series.characters:
        char_table = Table(title="Characters", show_header=True, header_style="bold magenta")
        char_table.add_column("Name", style="cyan")
        char_table.add_column("Description", style="white")
        char_table.add_column("Voice", style="dim")
        for c in series.characters:
            char_table.add_row(c.name, c.description[:50], c.voice_style)
            characters_data.append({
                "name": c.name,
                "description": c.description,
                "voice_style": c.voice_style,
            })
        console.print(char_table)

    episodes_data = []
    if series.episodes:
        ep_table = Table(title="Episodes", show_header=True, header_style="bold cyan")
        ep_table.add_column("#", width=4, style="dim")
        ep_table.add_column("Title", style="cyan")
        ep_table.add_column("Status", style="magenta")
        ep_table.add_column("Scenes", justify="right")
        ep_table.add_column("Cost", justify="right", style="yellow")
        ep_table.add_column("Synopsis", style="dim", max_width=40)
        for ep in series.episodes:
            ep_table.add_row(
                str(ep.number),
                ep.title,
                ep.status.value,
                str(len(ep.scenes)),
                f"${ep.cost:.4f}",
                ep.synopsis[:40],
            )
            episodes_data.append({
                "number": ep.number,
                "title": ep.title,
                "status": ep.status.value,
                "scenes": len(ep.scenes),
                "cost": ep.cost,
            })
        console.print(ep_table)

    out.set_result({
        "series_id": series.series_id,
        "title": series.title,
        "status": series.status.value,
        "characters": characters_data,
        "episodes": episodes_data,
    })
    out.emit()
