"""``claw drama plan`` and ``claw drama script`` commands."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from videoclaw.drama.models import DramaManager, DramaSeries, Episode

from rich.table import Table

from videoclaw.cli._app import (
    drama_app,
    configure_logging,
)
from videoclaw.cli._output import get_console, get_output


# ---------------------------------------------------------------------------
# claw drama plan
# ---------------------------------------------------------------------------

@drama_app.command("plan")
def drama_plan(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Plan episodes for an existing drama series using LLM."""
    configure_logging(verbose)
    console = get_console()
    out = get_output()
    out._command = "drama.plan"

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        out.set_error(f"Series {series_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    try:
        asyncio.run(_drama_plan_async(series, mgr))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result({
        "series_id": series.series_id,
        "episodes": len(series.episodes),
        "characters": len(series.characters),
    })
    out.emit()


async def _drama_plan_async(series: DramaSeries, mgr: DramaManager) -> None:
    console = get_console()

    from videoclaw.drama.planner import DramaPlanner

    planner = DramaPlanner()
    with console.status("[cyan]Director is planning the series...", spinner="dots"):
        series = await planner.plan_series(series)
    mgr.save(series)

    if series.title:
        console.print(f"\n[bold]Title:[/bold] {series.title}")

    if series.characters:
        char_table = Table(title="Characters", show_header=True, header_style="bold magenta")
        char_table.add_column("Name", style="cyan")
        char_table.add_column("Description", style="white")
        char_table.add_column("Visual", style="dim", max_width=40)
        for c in series.characters:
            char_table.add_row(c.name, c.description[:50], c.visual_prompt[:40])
        console.print(char_table)

    if series.episodes:
        ep_table = Table(title="Episodes", show_header=True, header_style="bold cyan")
        ep_table.add_column("#", width=4, style="dim")
        ep_table.add_column("Title", style="cyan")
        ep_table.add_column("Synopsis", style="white")
        ep_table.add_column("Duration", justify="right", style="green")
        for ep in series.episodes:
            ep_table.add_row(
                str(ep.number),
                ep.title,
                ep.synopsis[:60] + ("..." if len(ep.synopsis) > 60 else ""),
                f"{ep.duration_seconds:.0f}s",
            )
        console.print(ep_table)

    console.print(f"\n[bold green]Series planned: {series.series_id}[/bold green]")


# ---------------------------------------------------------------------------
# claw drama script
# ---------------------------------------------------------------------------

@drama_app.command("script")
def drama_script(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number to script.")] = 1,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Script a specific episode -- generate scene breakdown via LLM."""
    configure_logging(verbose)
    console = get_console()
    out = get_output()
    out._command = "drama.script"

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        out.set_error(f"Series {series_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    if not series.episodes:
        console.print("[yellow]No episodes planned. Run `claw drama plan` first.[/yellow]")
        out.set_error("No episodes planned. Run `claw drama plan` first.")
        out.emit()
        raise typer.Exit(code=1)

    ep = next((e for e in series.episodes if e.number == episode), None)
    if ep is None:
        console.print(f"[red]Episode {episode} not found in series.[/red]")
        out.set_error(f"Episode {episode} not found.")
        out.emit()
        raise typer.Exit(code=1)

    try:
        asyncio.run(_drama_script_async(series, ep, mgr))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result({
        "series_id": series.series_id,
        "episode": ep.number,
        "scenes": len(ep.scenes),
        "duration_seconds": ep.duration_seconds,
    })
    out.emit()


async def _drama_script_async(series: DramaSeries, ep: Episode, mgr: DramaManager) -> None:
    console = get_console()

    from videoclaw.drama.planner import DramaPlanner

    planner = DramaPlanner()

    # Retrieve cliffhanger from previous episode if available
    prev_cliffhanger: str | None = None
    for prev_ep in series.episodes:
        if prev_ep.number == ep.number - 1 and prev_ep.script:
            try:
                prev_script = json.loads(prev_ep.script)
                prev_cliffhanger = prev_script.get("cliffhanger")
            except (json.JSONDecodeError, TypeError):
                pass
            break

    with console.status(f"[cyan]Scripting episode {ep.number}: {ep.title}...", spinner="dots"):
        await planner.script_episode(series, ep, prev_cliffhanger)

    mgr.save(series)

    # Shot-scale color mapping
    _SCALE_COLOR = {
        "close_up": "red",
        "medium_close": "yellow",
        "medium": "yellow",
        "wide": "green",
        "extreme_wide": "green",
    }

    table = Table(
        title=f"Episode {ep.number}: {ep.title} -- Scene Breakdown",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Scene", style="dim", width=8)
    table.add_column("Shot Scale", width=14)
    table.add_column("Camera", style="white", width=12)
    table.add_column("Characters", style="magenta", max_width=20)
    table.add_column("Dialogue", style="white", max_width=30)
    table.add_column("Duration", justify="right", style="green", width=8)

    total_duration = 0.0
    for scene in ep.scenes:
        scale_str = scene.shot_scale.value if scene.shot_scale else "-"
        scale_color = _SCALE_COLOR.get(scale_str, "white")
        dialogue_text = (scene.dialogue or scene.narration or "")[:28]
        if len(scene.dialogue or scene.narration or "") > 28:
            dialogue_text += ".."
        chars = ", ".join(scene.characters_present) if scene.characters_present else "-"
        if len(chars) > 18:
            chars = chars[:16] + ".."
        total_duration += scene.duration_seconds

        table.add_row(
            scene.scene_id,
            f"[{scale_color}]{scale_str}[/{scale_color}]",
            scene.camera_movement,
            chars,
            dialogue_text,
            f"{scene.duration_seconds:.1f}s",
        )

    console.print(table)
    console.print(f"\n[bold]Total duration:[/bold] [green]{total_duration:.1f}s[/green]  |  "
                  f"[bold]Scenes:[/bold] {len(ep.scenes)}")
    console.print(f"[bold green]Episode {ep.number} scripted and saved.[/bold green]")
