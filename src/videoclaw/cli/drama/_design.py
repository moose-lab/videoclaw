"""``claw drama design-characters``, ``refresh-urls``, ``design-scenes``, ``assign-voices`` commands."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from videoclaw.drama.models import DramaManager, DramaSeries

from rich.panel import Panel
from rich.table import Table

from videoclaw.cli._app import (
    drama_app,
    configure_logging,
)
from videoclaw.cli._output import get_console, get_output


# ---------------------------------------------------------------------------
# claw drama design-characters
# ---------------------------------------------------------------------------

@drama_app.command("design-characters")
def drama_design_characters(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Regenerate existing images.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate reference images for characters in a drama series."""
    configure_logging(verbose)
    console = get_console()
    out = get_output()
    out._command = "drama.design-characters"

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        out.set_error(f"Series {series_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    if not series.characters:
        console.print("[yellow]No characters found. Run `claw drama plan` first.[/yellow]")
        out.set_error("No characters found. Run `claw drama plan` first.")
        out.emit()
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Series:[/bold]     {series.title}\n"
            f"[bold]Characters:[/bold] {len(series.characters)}\n"
            f"[bold]Force:[/bold]      {force}",
            title="[bold cyan]Character Design[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        asyncio.run(_design_characters_async(series, mgr, force))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result({
        "series_id": series.series_id,
        "characters": [
            {"name": c.name, "reference_image": c.reference_image}
            for c in series.characters
        ],
    })
    out.emit()


async def _design_characters_async(series: DramaSeries, mgr: DramaManager, force: bool) -> None:
    console = get_console()

    from videoclaw.drama.character_designer import CharacterDesigner

    designer = CharacterDesigner(drama_manager=mgr)

    with console.status("[cyan]Generating character reference images...", spinner="dots"):
        series = await designer.design_characters(series, force=force)

    table = Table(title="Character Reference Images", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Reference Image", style="green")
    for c in series.characters:
        table.add_row(c.name, c.reference_image or "[dim]none[/dim]")
    console.print(table)

    console.print(f"\n[bold green]Character designs complete for {series.series_id}[/bold green]")


# ---------------------------------------------------------------------------
# claw drama refresh-urls
# ---------------------------------------------------------------------------

@drama_app.command("refresh-urls")
def drama_refresh_urls(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Force refresh all URLs, even valid ones.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Refresh character reference image HTTPS URLs.

    Image provider URLs (Evolink/BytePlus) expire after ~24 hours.
    This command regenerates turnaround sheets to obtain fresh HTTPS URLs
    that Seedance 2.0 can accept as reference images.

    By default, only refreshes characters whose URLs are missing.
    Use --force to regenerate all character images.
    """
    configure_logging(verbose)
    console = get_console()
    out = get_output()
    out._command = "drama.refresh-urls"

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        out.set_error(f"Series {series_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    if not series.characters:
        console.print("[yellow]No characters found. Run `claw drama design-characters` first.[/yellow]")
        out.set_error("No characters found.")
        out.emit()
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Series:[/bold]     {series.title}\n"
            f"[bold]Characters:[/bold] {len(series.characters)}\n"
            f"[bold]Force:[/bold]      {force}",
            title="[bold cyan]Refresh Character URLs[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        refreshed = asyncio.run(_refresh_urls_async(series, mgr, force))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result({
        "series_id": series.series_id,
        "refreshed": {
            name: url[:80] + "..." if url and len(url) > 80 else url
            for name, url in refreshed.items()
        },
    })
    out.emit()


async def _refresh_urls_async(
    series: DramaSeries, mgr: DramaManager, force: bool,
) -> dict[str, str]:
    console = get_console()

    from videoclaw.drama.character_designer import CharacterDesigner

    designer = CharacterDesigner(drama_manager=mgr)

    with console.status("[cyan]Refreshing character reference URLs...", spinner="dots"):
        refreshed = await designer.refresh_urls(series, force=force)

    table = Table(title="Character URLs", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("URL", style="green")
    for name, url in refreshed.items():
        display_url = url[:60] + "..." if url and len(url) > 60 else url or "[dim]none[/dim]"
        table.add_row(name, display_url)
    console.print(table)

    ok_count = sum(1 for v in refreshed.values() if v)
    console.print(
        f"\n[bold green]URL refresh complete: "
        f"{ok_count}/{len(refreshed)} characters have valid URLs[/bold green]"
    )
    return refreshed


# ---------------------------------------------------------------------------
# claw drama design-scenes
# ---------------------------------------------------------------------------

@drama_app.command("design-scenes")
def drama_design_scenes(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Regenerate existing images.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate reference images for unique scene locations in a drama series."""
    configure_logging(verbose)
    console = get_console()
    out = get_output()
    out._command = "drama.design-scenes"

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        out.set_error(f"Series {series_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    if not series.episodes or not any(ep.scenes for ep in series.episodes):
        console.print("[yellow]No scenes found. Run `claw drama script` first.[/yellow]")
        out.set_error("No scenes found. Run `claw drama script` first.")
        out.emit()
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Series:[/bold]   {series.title}\n"
            f"[bold]Episodes:[/bold] {len(series.episodes)}\n"
            f"[bold]Force:[/bold]    {force}",
            title="[bold cyan]Scene Design[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        asyncio.run(_design_scenes_async(series, mgr, force))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result({"series_id": series.series_id})
    out.emit()


async def _design_scenes_async(series: DramaSeries, mgr: DramaManager, force: bool) -> None:
    console = get_console()

    from videoclaw.drama.scene_designer import SceneDesigner

    designer = SceneDesigner(drama_manager=mgr)

    # --- Scene locations ---
    with console.status("[cyan]Generating scene reference images...", spinner="dots"):
        locations = await designer.design_scenes(series, force=force)

    if locations:
        table = Table(title="Scene Reference Images", show_header=True, header_style="bold magenta")
        table.add_column("Location", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Reference Image", style="green")
        for loc in locations:
            table.add_row(
                loc.name,
                (loc.description[:50] + "...") if len(loc.description) > 50 else loc.description,
                loc.reference_image or "[dim]none[/dim]",
            )
        console.print(table)
    else:
        console.print("[yellow]No unique locations extracted from scenes.[/yellow]")

    # --- Props / items ---
    with console.status("[cyan]Analyzing props for consistency...", spinner="dots"):
        props = await designer.design_props(series, force=force)

    if props:
        ptable = Table(title="Prop Reference Images", show_header=True, header_style="bold magenta")
        ptable.add_column("Prop", style="cyan")
        ptable.add_column("Used In", style="white")
        ptable.add_column("Reference Image", style="green")
        for prop in props:
            ptable.add_row(
                prop.name,
                ", ".join(prop.scenes_used[:3]),
                prop.reference_image or "[dim]none[/dim]",
            )
        console.print(ptable)

    total = len(locations) + len(props)
    console.print(f"\n[bold green]Asset design complete: {len(locations)} locations, {len(props)} props[/bold green]")


# ---------------------------------------------------------------------------
# claw drama assign-voices
# ---------------------------------------------------------------------------

@drama_app.command("assign-voices")
def drama_assign_voices(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-assign voices even if already set.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Assign TTS voice profiles to all characters in a drama series."""
    configure_logging(verbose)
    console = get_console()
    out = get_output()
    out._command = "drama.assign-voices"

    from videoclaw.drama.models import DramaManager, assign_voice_profile

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        out.set_error(f"Series {series_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    if not series.characters:
        console.print("[yellow]No characters found. Run `claw drama plan` first.[/yellow]")
        out.set_error("No characters found. Run `claw drama plan` first.")
        out.emit()
        raise typer.Exit(code=1)

    if force:
        for c in series.characters:
            c.voice_profile = None

    for c in series.characters:
        assign_voice_profile(c)

    mgr.save(series)

    table = Table(title="Voice Assignments", show_header=True, header_style="bold magenta")
    table.add_column("Character", style="cyan")
    table.add_column("Voice Style", style="white")
    table.add_column("TTS Voice ID", style="green")

    voice_ids_seen: set[str] = set()
    voices_data = []
    for c in series.characters:
        vp = c.voice_profile
        voice_id = vp.voice_id if vp else "-"
        style = c.voice_style or "-"
        duplicate = voice_id in voice_ids_seen and voice_id != "-"
        voice_ids_seen.add(voice_id)
        id_display = f"[yellow]{voice_id} (dup)[/yellow]" if duplicate else voice_id
        table.add_row(c.name, style, id_display)
        voices_data.append({
            "character": c.name,
            "voice_style": style,
            "voice_id": voice_id,
        })

    console.print(table)
    console.print(f"\n[bold green]Voices assigned for {len(series.characters)} characters in {series_id}[/bold green]")

    out.set_result({"series_id": series_id, "voices": voices_data})
    out.emit()
