"""``claw drama`` -- AI short drama series orchestration commands."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Annotated, Optional

import typer

if TYPE_CHECKING:
    from videoclaw.drama.models import DramaManager, DramaSeries, Episode
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

from videoclaw.cli._app import (
    drama_app,
    configure_logging,
    show_banner,
    validate_aspect_ratio,
    validate_language,
    validate_prompt,
)
from videoclaw.cli._output import get_console, get_output
from videoclaw.config import get_config


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
            asyncio.run(_drama_plan_async(series, mgr))

        if design_characters and plan:
            console.print("\n[bold cyan]Generating character reference images...[/bold cyan]")
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


# ---------------------------------------------------------------------------
# claw drama list / show
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


# ---------------------------------------------------------------------------
# claw drama plan / script
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


# ---------------------------------------------------------------------------
# claw drama design-characters / design-scenes / assign-voices
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

    with console.status("[cyan]Generating scene reference images...", spinner="dots"):
        locations = await designer.design_scenes(series, force=force)

    if not locations:
        console.print("[yellow]No unique locations extracted from scenes.[/yellow]")
        return

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

    console.print(f"\n[bold green]Scene designs complete: {len(locations)} locations[/bold green]")


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


# ---------------------------------------------------------------------------
# claw drama run
# ---------------------------------------------------------------------------

@drama_app.command("run")
def drama_run(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    episode: Annotated[Optional[int], typer.Option("--episode", "-e", help="Run a specific episode number.")] = None,
    start: Annotated[int, typer.Option("--start", help="Start from episode number.")] = 1,
    end: Annotated[Optional[int], typer.Option("--end", help="End at episode number.")] = None,
    budget: Annotated[Optional[float], typer.Option("--budget", "-b", help="Max budget in USD.")] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", help="Max parallel tasks.")] = 4,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show execution plan without running.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the generation pipeline for drama episodes.

    \b
    Uses Seedance 2.0 by default (4-15s per clip). Each clip generates
    video + audio + dialogue in a single pass. Character consistency is
    enforced via Universal Reference and a pre-built ConsistencyManifest.
    """
    configure_logging(verbose)
    show_banner()
    console = get_console()
    out = get_output()
    out._command = "drama.run"

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

    if episode is not None:
        start = episode
        end = episode

    console.print(
        Panel(
            f"[bold]Series:[/bold]   {series.title}\n"
            f"[bold]Episodes:[/bold] {start} to {end or len(series.episodes)}\n"
            f"[bold]Model:[/bold]    {series.model_id}",
            title="[bold cyan]Drama Run[/bold cyan]",
            border_style="cyan",
        )
    )

    if dry_run:
        episodes_to_run = [
            ep for ep in series.episodes
            if start <= ep.number <= (end or len(series.episodes))
            and ep.status != "completed"
        ]
        plan_data = {
            "dry_run": True,
            "series_id": series.series_id,
            "episodes": [
                {"number": ep.number, "title": ep.title, "scenes": len(ep.scenes)}
                for ep in episodes_to_run
            ],
        }
        console.print(
            Panel(
                f"[bold]Episodes to run:[/bold] {len(episodes_to_run)}\n"
                f"[dim]Dry run -- no generation performed.[/dim]",
                title="[bold yellow]Execution Plan (dry-run)[/bold yellow]",
                border_style="yellow",
            )
        )
        out.set_result(plan_data)
        out.emit()
        return

    try:
        asyncio.run(_drama_run_async(series, mgr, start, end, budget, concurrency))
    except typer.Exit:
        raise
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result({
        "series_id": series.series_id,
        "status": series.status.value,
        "cost_total": series.cost_total,
    })
    out.emit()


async def _drama_run_async(
    series: DramaSeries, mgr: DramaManager, start: int, end: int | None,
    budget_usd: float | None = None, max_concurrency: int = 4,
) -> None:
    console = get_console()

    from videoclaw.drama.planner import DramaPlanner
    from videoclaw.drama.runner import DramaRunner

    planner = DramaPlanner()
    runner = DramaRunner(drama_manager=mgr)
    effective_budget = budget_usd or get_config().budget_default_usd

    episodes_to_run = [
        ep for ep in series.episodes
        if start <= ep.number <= (end or len(series.episodes))
        and ep.status != "completed"
    ]

    # Retrieve cliffhanger from the episode before the first one we're running
    prev_cliffhanger: str | None = None
    if episodes_to_run:
        prev_num = episodes_to_run[0].number - 1
        for ep in series.episodes:
            if ep.number == prev_num and ep.script:
                try:
                    prev_script = json.loads(ep.script)
                    prev_cliffhanger = prev_script.get("cliffhanger")
                except (json.JSONDecodeError, TypeError):
                    pass
                break

    for ep in episodes_to_run:
        console.print(f"\n[bold cyan]Episode {ep.number}: {ep.title}[/bold cyan]")

        if not ep.scenes:
            if series.script_locked:
                console.print(
                    f"[bold red]Episode {ep.number} has no scenes but script "
                    f"is LOCKED (imported). Cannot auto-generate.[/bold red]\n"
                    f"[yellow]Re-run 'claw drama import' with the complete script.[/yellow]"
                )
                raise typer.Exit(code=1)
            with console.status(f"[cyan]Scripting episode {ep.number}...", spinner="dots"):
                script_data = await planner.script_episode(series, ep, prev_cliffhanger)
            prev_cliffhanger = script_data.get("cliffhanger")
            mgr.save(series)
            console.print(f"  Scenes: {len(ep.scenes)}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Episode {ep.number}...",
                total=len(ep.scenes) + 4,
            )
            state = await runner.run_episode(series, ep)
            progress.update(task, completed=len(ep.scenes) + 4)

        status_style = "green" if ep.status == "completed" else "red"
        console.print(f"  Status: [{status_style}]{ep.status}[/{status_style}]  Cost: ${ep.cost:.4f}")

        if series.cost_total >= effective_budget:
            console.print(
                f"[bold red]Budget limit reached: ${series.cost_total:.2f} >= "
                f"${effective_budget:.2f}. Stopping.[/bold red]"
            )
            break

    console.print(
        Panel(
            f"[bold]Series:[/bold]  {series.title}\n"
            f"[bold]Status:[/bold]  {series.status.value}\n"
            f"[bold]Cost:[/bold]    ${series.cost_total:.4f}",
            title="[bold green]Drama Complete[/bold green]",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# claw drama regen-shot
# ---------------------------------------------------------------------------

@drama_app.command("regen-shot")
def drama_regen_shot(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    scene: Annotated[str, typer.Option("--scene", "-s", help="Scene ID to regenerate (e.g. ep01_s03).")],
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number.")] = 1,
    recompose: Annotated[bool, typer.Option("--recompose", help="Re-compose and re-render the full episode after regenerating.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Regenerate a single scene's video and audio assets.

    \b
    Use `claw drama show <series_id>` to find scene IDs, then re-run only the
    scene that needs fixing.  With --recompose, the entire episode is
    re-assembled after the scene is regenerated.

    \b
    Examples:
        claw drama regen-shot abc123 -e 2 -s ep02_s03
        claw drama regen-shot abc123 -e 1 -s ep01_s01 --recompose
    """
    configure_logging(verbose)
    show_banner()
    console = get_console()
    out = get_output()
    out._command = "drama.regen-shot"

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        out.set_error(f"Series {series_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    ep = next((e for e in series.episodes if e.number == episode), None)
    if ep is None:
        console.print(f"[red]Episode {episode} not found in series.[/red]")
        out.set_error(f"Episode {episode} not found.")
        out.emit()
        raise typer.Exit(code=1)

    if not ep.scenes:
        console.print("[yellow]Episode has no scenes. Run `claw drama script` first.[/yellow]")
        out.set_error("Episode has no scenes. Run `claw drama script` first.")
        out.emit()
        raise typer.Exit(code=1)

    scene_ids = [s.scene_id for s in ep.scenes]
    if scene not in scene_ids:
        console.print(f"[red]Scene {scene!r} not found in episode {episode}.[/red]")
        console.print(f"[dim]Available scenes: {', '.join(scene_ids)}[/dim]")
        out.set_error(f"Scene {scene!r} not found. Available: {', '.join(scene_ids)}")
        out.emit()
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Series:[/bold]    {series.title}\n"
            f"[bold]Episode:[/bold]   {episode}\n"
            f"[bold]Scene:[/bold]     {scene}\n"
            f"[bold]Recompose:[/bold] {'yes' if recompose else 'no'}",
            title="[bold cyan]Regen Shot[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        asyncio.run(_drama_regen_shot_async(series, mgr, ep, scene, recompose))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result({
        "series_id": series_id,
        "episode": episode,
        "scene": scene,
        "recompose": recompose,
    })
    out.emit()


# ---------------------------------------------------------------------------
# claw drama audit
# ---------------------------------------------------------------------------

@drama_app.command("audit")
def drama_audit(
    clip_dir: Annotated[str, typer.Argument(help="Directory containing generated MP4 clips.")],
    scenes_json: Annotated[str, typer.Option("--scenes", "-s", help="Path to series_data.json or 05_redecomposition_60s.json.")] = "",
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number.")] = 1,
    clip_prefix: Annotated[str, typer.Option("--prefix", help="Filename prefix for session5 clips.")] = "session5_",
    session6: Annotated[str, typer.Option("--session6", help="Comma-separated scene IDs re-generated in session6.")] = "",
    output: Annotated[str, typer.Option("--output", "-o", help="Write JSON audit report to this file.")] = "",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run Claude Vision audit on generated video clips.

    \b
    Extracts keyframes from each clip and sends them to Claude Vision for
    per-shot quality inspection: time-of-day consistency, character presence,
    subtitle spelling errors, generation artifacts, and dramatic tension.

    \b
    Examples:
        claw drama audit docs/deliverables/ep01_satan_in_a_suit/video_clips \\
            --scenes docs/deliverables/05_redecomposition_60s.json \\
            --session6 ep01_s06 --output audit_session6.json
    """
    configure_logging(verbose)
    show_banner()
    console = get_console()
    out = get_output()
    out._command = "drama.audit"

    import json as _json
    from pathlib import Path
    from videoclaw.drama.models import DramaScene
    from videoclaw.drama.vision_auditor import VisionAuditor

    clip_path = Path(clip_dir)
    if not clip_path.is_dir():
        console.print(f"[red]clip_dir {clip_dir!r} is not a directory.[/red]")
        out.set_error(f"clip_dir {clip_dir!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    # Load scenes
    if not scenes_json:
        console.print("[red]--scenes is required.[/red]")
        raise typer.Exit(code=1)

    scenes_path = Path(scenes_json)
    if not scenes_path.exists():
        console.print(f"[red]scenes file {scenes_json!r} not found.[/red]")
        raise typer.Exit(code=1)

    raw = _json.loads(scenes_path.read_text())
    # Support both series.json (has 'episodes') and flat {'scenes': [...]}
    if "episodes" in raw:
        ep_data = next(
            (e for e in raw["episodes"] if e.get("number", 1) == episode),
            raw["episodes"][0] if raw["episodes"] else {},
        )
        scenes_data = ep_data.get("scenes", [])
    elif "scenes" in raw:
        scenes_data = raw["scenes"]
    else:
        console.print("[red]Cannot find 'scenes' in the provided JSON.[/red]")
        raise typer.Exit(code=1)

    scenes = [DramaScene.from_dict(s) for s in scenes_data]
    session6_scenes = [s.strip() for s in session6.split(",") if s.strip()] if session6 else []

    console.print(f"[bold]Auditing {len(scenes)} scenes[/bold] in [cyan]{clip_dir}[/cyan]")
    if session6_scenes:
        console.print(f"Session 6 clips: {session6_scenes}")

    auditor = VisionAuditor()

    async def _run() -> None:
        report = await auditor.audit_episode(
            scenes,
            clip_path,
            clip_prefix=clip_prefix,
            session6_scenes=session6_scenes,
        )
        console.print(report.summary())

        if output:
            out_path = Path(output)
            out_path.write_text(_json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            console.print(f"[green]Report written: {output}[/green]")

        out.set_result({
            "total": report.total_shots,
            "passed": report.passed_shots,
            "regen_required": report.regen_required,
        })
        out.emit()

    asyncio.run(_run())


async def _drama_regen_shot_async(
    series: DramaSeries, mgr: DramaManager, episode: Episode, scene_id: str, recompose: bool,
) -> None:
    console = get_console()

    from videoclaw.drama.runner import DramaRunner

    runner = DramaRunner(drama_manager=mgr)

    node_count = 2 + (3 if recompose else 0)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Regenerating {scene_id}...", total=node_count)
        state = await runner.regenerate_scene(series, episode, scene_id, recompose)
        progress.update(task, completed=node_count)

    status_style = "green" if state.status.value == "completed" else "red"
    console.print(
        Panel(
            f"[bold]Scene:[/bold]  {scene_id}\n"
            f"[bold]Status:[/bold] [{status_style}]{state.status.value}[/{status_style}]\n"
            f"[bold]Cost:[/bold]   ${state.cost_total:.4f}",
            title="[bold green]Regen Complete[/bold green]",
            border_style="green",
        )
    )
