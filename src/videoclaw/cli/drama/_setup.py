"""``claw drama new`` and ``claw drama import`` commands."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Optional

import typer

if TYPE_CHECKING:
    from videoclaw.drama.models import DramaManager, DramaSeries

from rich.panel import Panel
from rich.table import Table

from videoclaw.cli._app import (
    drama_app,
    configure_logging,
    show_banner,
    validate_aspect_ratio,
    validate_language,
    validate_prompt,
)
from videoclaw.cli._output import get_console, get_output


# ---------------------------------------------------------------------------
# claw drama new
# ---------------------------------------------------------------------------

@drama_app.command("new")
def drama_new(
    synopsis: Annotated[str, typer.Argument(help="High-level story concept for the drama series.", callback=validate_prompt)],
    title: Annotated[Optional[str], typer.Option("--title", "-t", help="Series title.")] = None,
    genre: Annotated[str, typer.Option("--genre", "-g", help="Genre.")] = "drama",
    episodes: Annotated[int, typer.Option("--episodes", "-n", help="Number of episodes.")] = 5,
    duration: Annotated[float, typer.Option("--duration", "-d", help="Target seconds per episode.")] = 60.0,
    style: Annotated[str, typer.Option("--style", "-s", help="Visual style.")] = "cinematic",
    language: Annotated[str, typer.Option("--lang", "-l", help="Script language (zh/en).", callback=validate_language)] = "zh",
    aspect_ratio: Annotated[str, typer.Option("--aspect-ratio", "-a", help="Aspect ratio.", callback=validate_aspect_ratio)] = "9:16",
    model: Annotated[str, typer.Option("--model", "-m", help="Video model id.")] = "seedance-2.0",
    plan: Annotated[bool, typer.Option("--plan/--no-plan", help="Immediately plan episodes via LLM.")] = False,
    design_characters: Annotated[bool, typer.Option("--design-characters", help="Generate character reference images after planning.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Create a new AI short drama series from a concept.

    \b
    The LLM generates the script creatively. For importing a COMPLETE,
    pre-written script without modifications, use 'claw drama import' instead.

    \b
    Default model: Seedance 2.0 (4-15s clips, audio co-generation, 9:16 vertical).
    """
    configure_logging(verbose)
    show_banner()
    console = get_console()
    out = get_output()
    out._command = "drama.new"

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    series = mgr.create(
        title=title or "",
        synopsis=synopsis,
        genre=genre,
        total_episodes=episodes,
        target_episode_duration=duration,
        style=style,
        language=language,
        aspect_ratio=aspect_ratio,
        model_id=model,
    )

    console.print(
        Panel(
            f"[bold]Series ID:[/bold]  {series.series_id}\n"
            f"[bold]Synopsis:[/bold]   {synopsis[:80]}\n"
            f"[bold]Genre:[/bold]      {genre}\n"
            f"[bold]Episodes:[/bold]   {episodes}\n"
            f"[bold]Duration:[/bold]   {duration}s/episode\n"
            f"[bold]Style:[/bold]      {style}\n"
            f"[bold]Model:[/bold]      {model}",
            title="[bold green]New Drama Series[/bold green]",
            border_style="green",
        )
    )

    try:
        if plan:
            console.print("\n[bold cyan]Planning episodes via LLM...[/bold cyan]")
            from videoclaw.cli.drama._plan import _drama_plan_async
            asyncio.run(_drama_plan_async(series, mgr))

        if design_characters and plan:
            console.print("\n[bold cyan]Generating character reference images...[/bold cyan]")
            from videoclaw.cli.drama._design import _design_characters_async
            asyncio.run(_design_characters_async(series, mgr, force=False))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result({
        "series_id": series.series_id,
        "title": series.title,
        "genre": genre,
        "episodes": episodes,
    })
    out.emit()


# ---------------------------------------------------------------------------
# claw drama import
# ---------------------------------------------------------------------------

@drama_app.command("import")
def drama_import(
    script_file: Annotated[str, typer.Argument(help="Path to complete script file (.docx or .txt).")],
    title: Annotated[Optional[str], typer.Option("--title", "-t", help="Series title.")] = None,
    genre: Annotated[str, typer.Option("--genre", "-g", help="Genre.")] = "drama",
    language: Annotated[str, typer.Option("--lang", "-l", help="Script language (zh/en).", callback=validate_language)] = "en",
    style: Annotated[str, typer.Option("--style", "-s", help="Visual style.")] = "cinematic",
    aspect_ratio: Annotated[str, typer.Option("--aspect-ratio", "-a", help="Aspect ratio.", callback=validate_aspect_ratio)] = "9:16",
    model: Annotated[str, typer.Option("--model", "-m", help="Video model (default: seedance-2.0).")] = "seedance-2.0",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Import a complete script and decompose into storyboard shots.

    \b
    Imports a FINALIZED script from a .docx or .txt file. The script is
    treated as LOCKED -- no creative modifications are allowed. The system
    only decomposes scenes into Seedance 2.0-compatible shots (4-15s each).

    \b
    Default video model: Seedance 2.0
      - 4-15 seconds per clip
      - Audio + dialogue co-generation (no separate TTS needed)
      - 9:16 vertical format (720p) for TikTok
      - Universal Reference for character consistency

    \b
    Example:
      claw drama import script.docx --title "Satan in a Suit" --lang en
    """
    configure_logging(verbose)
    show_banner()

    try:
        asyncio.run(_drama_import_async(
            script_file, title, genre, language, style, aspect_ratio, model,
        ))
    except typer.Exit:
        raise
    except Exception as exc:
        out = get_output()
        out._command = "drama.import"
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)


async def _drama_import_async(
    script_file: str,
    title: str | None,
    genre: str,
    language: str,
    style: str,
    aspect_ratio: str,
    model: str,
) -> None:
    console = get_console()
    out = get_output()
    out._command = "drama.import"

    from videoclaw.drama.models import DramaManager, ScriptModification
    from videoclaw.drama.planner import DramaPlanner

    planner = DramaPlanner()

    # 1. Read the script file
    console.print(f"[cyan]Reading script:[/cyan] {script_file}")
    try:
        script_text = planner.read_script_file(script_file)
    except FileNotFoundError:
        console.print(f"[red]File not found: {script_file}[/red]")
        out.set_error(f"File not found: {script_file}")
        out.emit()
        raise typer.Exit(code=1)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        out.set_error(str(e))
        out.emit()
        raise typer.Exit(code=1)

    console.print(f"  Script length: {len(script_text)} characters")

    # 2. Create the series
    mgr = DramaManager()
    series = mgr.create(
        title=title or "",
        synopsis="",
        genre=genre,
        total_episodes=1,
        target_episode_duration=120.0,
        style=style,
        language=language,
        aspect_ratio=aspect_ratio,
        model_id=model,
    )

    # 3. Human confirmation callback for detected gaps
    def _confirm_gaps(modifications: list[ScriptModification]) -> list[ScriptModification]:
        if not modifications:
            return []

        # In JSON mode, auto-approve all modifications (no interactive prompt)
        if out.json_mode:
            for mod in modifications:
                mod.approved = True
            return modifications

        console.print(
            f"\n[bold yellow]Detected {len(modifications)} gap(s) "
            f"in the imported script:[/bold yellow]\n"
        )

        approved: list[ScriptModification] = []
        for i, mod in enumerate(modifications, 1):
            console.print(
                f"  [bold]{i}.[/bold] "
                f"[cyan]{mod.scene_id or 'global'}[/cyan] -- "
                f"[yellow]{mod.field_name}[/yellow]: {mod.reason}"
            )
            if mod.proposed_value:
                console.print(f"     Proposed fix: {mod.proposed_value}")

            if typer.confirm(f"     Approve this modification?", default=False):
                mod.approved = True
                approved.append(mod)
            else:
                console.print("     [dim]Skipped -- original script preserved.[/dim]")

        return approved

    # 4. Import and decompose
    with console.status(
        "[cyan]Decomposing script into storyboard (script is LOCKED)...",
        spinner="dots",
    ):
        series = await planner.import_complete_script(
            series,
            script_text,
            confirm_callback=_confirm_gaps,
        )

    mgr.save(series)

    # 5. Display results
    console.print(
        Panel(
            f"[bold]Series ID:[/bold]     {series.series_id}\n"
            f"[bold]Title:[/bold]         {series.title}\n"
            f"[bold]Script Lock:[/bold]   [bold red]LOCKED[/bold red] (no creative modifications)\n"
            f"[bold]Source:[/bold]        imported\n"
            f"[bold]Episodes:[/bold]      {len(series.episodes)}\n"
            f"[bold]Total Scenes:[/bold]  {sum(len(ep.scenes) for ep in series.episodes)}\n"
            f"[bold]Model:[/bold]         {series.model_id}\n"
            f"[bold]Consistency:[/bold]   {'verified' if series.consistency_manifest and series.consistency_manifest.verified else 'pending (run design-characters first)'}",
            title="[bold green]Script Imported[/bold green]",
            border_style="green",
        )
    )

    if series.characters:
        char_table = Table(title="Characters (from script)", show_header=True, header_style="bold magenta")
        char_table.add_column("Name", style="cyan")
        char_table.add_column("Description", style="white", max_width=50)
        char_table.add_column("Voice", style="dim")
        for c in series.characters:
            char_table.add_row(c.name, c.description[:50], c.voice_style)
        console.print(char_table)

    for ep in series.episodes:
        ep_table = Table(
            title=f"Episode {ep.number}: {ep.title}",
            show_header=True,
            header_style="bold cyan",
        )
        ep_table.add_column("Shot", width=10, style="dim")
        ep_table.add_column("Duration", width=8, justify="right", style="green")
        ep_table.add_column("Scale", width=12, style="yellow")
        ep_table.add_column("Dialogue", max_width=40, style="white")
        ep_table.add_column("Characters", max_width=20, style="cyan")

        for scene in ep.scenes:
            ep_table.add_row(
                scene.scene_id,
                f"{scene.duration_seconds:.0f}s",
                str(scene.shot_scale.value if scene.shot_scale else "-"),
                (scene.dialogue[:37] + "...") if len(scene.dialogue) > 40 else scene.dialogue,
                ", ".join(scene.characters_present[:3]),
            )
        console.print(ep_table)
        console.print(f"  Total duration: {ep.duration_seconds:.0f}s / {len(ep.scenes)} shots")

    if series.pending_modifications:
        console.print(
            f"\n[yellow]Note: {len(series.pending_modifications)} unapproved "
            f"modification(s) pending. Review with 'claw drama show'.[/yellow]"
        )

    console.print(
        f"\n[bold]Next steps:[/bold]\n"
        f"  1. claw drama design-characters {series.series_id}\n"
        f"  2. claw drama design-scenes {series.series_id}\n"
        f"  3. claw drama assign-voices {series.series_id}\n"
        f"  4. claw drama run {series.series_id}"
    )

    out.set_result({
        "series_id": series.series_id,
        "title": series.title,
        "episodes": len(series.episodes),
        "total_scenes": sum(len(ep.scenes) for ep in series.episodes),
    })
    out.emit()
