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
# claw drama preview-prompts — 预览增强后 Seedance 提示词
# ---------------------------------------------------------------------------

@drama_app.command("preview-prompts")
def drama_preview_prompts(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number.")] = 1,
    scene: Annotated[Optional[str], typer.Option("--scene", "-s", help="Show only this scene ID.")] = None,
    output: Annotated[str, typer.Option("--output", "-o", help="Write prompts to JSON file.")] = "",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Preview the enhanced Seedance 2.0 prompts for each scene.

    \b
    Shows the final prompt that would be sent to Seedance, including:
    - Realism header + CHARACTER IDENTITY (Western drama)
    - Camera (shot scale + movement)
    - Visual scene description
    - Text directives (subtitles, name cards, title cards)
    - Style anchor + constraints

    \b
    Use this to debug prompt quality before spending API credits on generation.

    \b
    Examples:
        claw drama preview-prompts abc123
        claw drama preview-prompts abc123 -e 1 -s ep01_s03
        claw drama preview-prompts abc123 -o prompts.json
    """
    configure_logging(verbose)
    console = get_console()
    out_ctx = get_output()
    out_ctx._command = "drama.preview-prompts"

    from videoclaw.drama.models import DramaManager
    from videoclaw.drama.prompt_enhancer import PromptEnhancer

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        out_ctx.set_error(f"Series {series_id!r} not found.")
        out_ctx.emit()
        raise typer.Exit(code=1)

    ep = next((e for e in series.episodes if e.number == episode), None)
    if ep is None:
        console.print(f"[red]Episode {episode} not found in series.[/red]")
        out_ctx.set_error(f"Episode {episode} not found.")
        out_ctx.emit()
        raise typer.Exit(code=1)

    if not ep.scenes:
        console.print("[yellow]Episode has no scenes.[/yellow]")
        out_ctx.set_error("Episode has no scenes.")
        out_ctx.emit()
        raise typer.Exit(code=1)

    enhancer = PromptEnhancer()

    # Enhance all scenes (populates name card tracking across episode)
    enhancer.enhance_all_scenes(ep, series)

    prompts_data: list[dict] = []
    for sc in ep.scenes:
        if scene and sc.scene_id != scene:
            continue
        entry = {
            "scene_id": sc.scene_id,
            "duration": sc.duration_seconds,
            "shot_scale": sc.shot_scale.value if sc.shot_scale else "",
            "camera": sc.camera_movement,
            "characters": sc.characters_present,
            "dialogue": sc.dialogue,
            "narration": sc.narration,
            "prompt": sc.visual_prompt,
        }
        prompts_data.append(entry)

        # Rich display
        console.print(f"\n[bold cyan]── {sc.scene_id} ──[/bold cyan]")
        console.print(f"[dim]Duration: {sc.duration_seconds}s | "
                      f"Scale: {sc.shot_scale.value if sc.shot_scale else 'n/a'} | "
                      f"Camera: {sc.camera_movement}[/dim]")
        if sc.characters_present:
            console.print(f"[dim]Characters: {', '.join(sc.characters_present)}[/dim]")
        if sc.dialogue:
            console.print(f'[dim]Dialogue: "{sc.dialogue[:80]}{"…" if len(sc.dialogue) > 80 else ""}"[/dim]')
        console.print(f"\n[green]{sc.visual_prompt}[/green]")

    if output:
        import json as _json
        from pathlib import Path as _Path
        _Path(output).write_text(_json.dumps(prompts_data, indent=2, ensure_ascii=False))
        console.print(f"\n[green]Prompts written to {output}[/green]")

    console.print(f"\n[bold]Total: {len(prompts_data)} scene(s)[/bold]")

    out_ctx.set_result({
        "series_id": series_id,
        "episode": episode,
        "scene_count": len(prompts_data),
    })
    out_ctx.emit()


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
    refresh_urls: Annotated[bool, typer.Option("--refresh-urls/--no-refresh-urls", help="Auto-validate and refresh expired character reference URLs before generation.")] = True,
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
        asyncio.run(_drama_run_async(series, mgr, start, end, budget, concurrency, refresh_urls))
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
    auto_refresh_urls: bool = True,
) -> None:
    console = get_console()

    from videoclaw.drama.planner import DramaPlanner
    from videoclaw.drama.runner import DramaRunner

    planner = DramaPlanner()
    runner = DramaRunner(drama_manager=mgr, auto_refresh_urls=auto_refresh_urls)
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
    scene: Annotated[str, typer.Option("--scene", "-s", help="Scene ID(s) to regenerate, comma-separated (e.g. ep01_s01,ep01_s03,ep01_s05).")],
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number.")] = 1,
    recompose: Annotated[bool, typer.Option("--recompose", help="Re-compose and re-render the full episode after regenerating.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Regenerate one or more scene's video and audio assets.

    \b
    Supports batch regeneration: pass multiple scene IDs separated by commas.
    With --recompose, the entire episode is re-assembled after all scenes
    are regenerated.

    \b
    Examples:
        claw drama regen-shot abc123 -e 2 -s ep02_s03
        claw drama regen-shot abc123 -e 1 -s ep01_s01,ep01_s03,ep01_s05
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

    # Parse comma-separated scene IDs for batch regeneration
    regen_scene_ids = [s.strip() for s in scene.split(",") if s.strip()]
    available_scene_ids = [s.scene_id for s in ep.scenes]

    invalid = [sid for sid in regen_scene_ids if sid not in available_scene_ids]
    if invalid:
        console.print(f"[red]Scene(s) not found: {', '.join(invalid)}[/red]")
        console.print(f"[dim]Available scenes: {', '.join(available_scene_ids)}[/dim]")
        out.set_error(f"Scenes not found: {', '.join(invalid)}")
        out.emit()
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Series:[/bold]    {series.title}\n"
            f"[bold]Episode:[/bold]   {episode}\n"
            f"[bold]Scene(s):[/bold]  {', '.join(regen_scene_ids)}\n"
            f"[bold]Recompose:[/bold] {'yes' if recompose else 'no'}",
            title="[bold cyan]Regen Shot[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        asyncio.run(_drama_regen_shot_async(series, mgr, ep, regen_scene_ids, recompose))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result({
        "series_id": series_id,
        "episode": episode,
        "scenes": regen_scene_ids,
        "recompose": recompose,
    })
    out.emit()


# ---------------------------------------------------------------------------
# claw drama audit
# ---------------------------------------------------------------------------

@drama_app.command("audit")
def drama_audit(
    target: Annotated[Optional[str], typer.Argument(
        help="Series ID (series-aware mode) or omit for standalone mode."
    )] = None,
    clip_dir: Annotated[str, typer.Option("--clip-dir", "-c", help="Directory containing generated MP4 clips.")] = "",
    scenes_json: Annotated[str, typer.Option("--scenes", "-s", help="Path to scenes JSON (standalone mode).")] = "",
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number.")] = 1,
    output: Annotated[str, typer.Option("--output", "-o", help="Write JSON audit report to this file.")] = "",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run Claude Vision audit on generated video clips.

    \b
    Extracts keyframes via FFmpeg and sends them to Claude Vision for per-shot
    QA: time-of-day consistency, character presence, subtitle spelling errors,
    generation artifacts, and dramatic tension.

    \b
    Audit results and repair directions are printed for your review.
    No automatic regeneration is triggered — re-generation decisions are yours.

    \b
    SERIES-AWARE MODE (recommended — loads scenes from drama manager):
        claw drama audit <series_id> [--clip-dir DIR] [--episode N]
        Persists audit results back into the series state.

    \b
    STANDALONE MODE (for externally-generated clips):
        claw drama audit --clip-dir DIR --scenes SCENES_JSON [--episode N]

    \b
    Examples:
        claw drama audit 97e8424712d24fb2 --clip-dir docs/deliverables/ep01/video_clips
        claw drama audit --clip-dir video_clips --scenes series_data.json --output report.json
    """
    configure_logging(verbose)
    show_banner()
    console = get_console()
    out = get_output()
    out._command = "drama.audit"

    import json as _json
    from pathlib import Path
    from videoclaw.drama.vision_auditor import VisionAuditor

    # Determine mode: series-aware if target looks like a series_id (not a path)
    series_mode = target is not None and not Path(target).is_dir()

    def _finish(report) -> None:  # type: ignore[no-untyped-def]
        console.print(report.summary())

        failed = [r for r in report.shot_results if not r.passed]
        if failed:
            console.print("\n[bold yellow]Repair directions:[/bold yellow]")
            for r in failed:
                regen_flag = " [bold red][REGEN RECOMMENDED][/bold red]" if r.regen_required else ""
                console.print(f"  [cyan]{r.shot_id}[/cyan]{regen_flag}")
                for issue in r.issues:
                    console.print(f"    • {issue}")
                if r.regen_note:
                    console.print(f"    → {r.regen_note}")
        else:
            console.print("\n[bold green]All shots passed — no repairs needed.[/bold green]")

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

    if series_mode:
        # ---- Series-aware mode ----------------------------------------
        from videoclaw.drama.models import DramaManager

        series_id = target
        mgr = DramaManager()
        series = mgr.load(series_id)
        if series is None:
            console.print(f"[red]Series {series_id!r} not found. Use 'claw drama list' to see available series.[/red]")
            out.set_error(f"series {series_id!r} not found")
            out.emit()
            raise typer.Exit(code=1)

        clip_path: Path | None = Path(clip_dir) if clip_dir else None
        if clip_path is not None and not clip_path.is_dir():
            console.print(f"[red]--clip-dir {clip_dir!r} is not a directory.[/red]")
            out.set_error(f"clip_dir {clip_dir!r} not found")
            out.emit()
            raise typer.Exit(code=1)

        ep_obj = next((e for e in series.episodes if e.number == episode), None)
        if ep_obj is None:
            console.print(f"[red]Episode {episode} not found in series {series_id!r}.[/red]")
            raise typer.Exit(code=1)

        console.print(
            f"[bold]Series-aware audit:[/bold] [cyan]{series_id}[/cyan] EP{episode:02d} "
            f"— {len(ep_obj.scenes)} shots"
        )

        auditor = VisionAuditor()

        async def _run_series() -> None:
            report = await auditor.audit_series_episode(
                series,
                episode_number=episode,
                clip_dir=clip_path,
                drama_manager=mgr,
                persist_results=True,
            )
            _finish(report)

        asyncio.run(_run_series())

    else:
        # ---- Standalone mode ------------------------------------------
        from videoclaw.drama.models import DramaScene

        # target may be a legacy positional clip_dir (backward compat)
        effective_clip_dir = clip_dir or (target if target and Path(target).is_dir() else "")
        if not effective_clip_dir:
            console.print("[red]Provide a series_id as argument (series-aware mode) or --clip-dir (standalone mode).[/red]")
            raise typer.Exit(code=1)

        clip_path = Path(effective_clip_dir)
        if not clip_path.is_dir():
            console.print(f"[red]clip-dir {effective_clip_dir!r} is not a directory.[/red]")
            out.set_error(f"clip_dir {effective_clip_dir!r} not found")
            out.emit()
            raise typer.Exit(code=1)

        if not scenes_json:
            console.print("[red]--scenes is required in standalone mode.[/red]")
            raise typer.Exit(code=1)

        scenes_path = Path(scenes_json)
        if not scenes_path.exists():
            console.print(f"[red]scenes file {scenes_json!r} not found.[/red]")
            raise typer.Exit(code=1)

        raw = _json.loads(scenes_path.read_text())
        if "episodes" in raw:
            ep_data = next(
                (e for e in raw["episodes"] if e.get("number", 1) == episode),
                raw["episodes"][0] if raw["episodes"] else {},
            )
            scenes_data = ep_data.get("scenes", [])
        elif "scenes" in raw:
            scenes_data = raw["scenes"]
        else:
            console.print("[red]Cannot find 'scenes' key in the provided JSON.[/red]")
            raise typer.Exit(code=1)

        scenes = [DramaScene.from_dict(s) for s in scenes_data]
        console.print(f"[bold]Standalone audit:[/bold] {len(scenes)} scenes in [cyan]{effective_clip_dir}[/cyan]")

        auditor = VisionAuditor()

        async def _run_standalone() -> None:
            report = await auditor.audit_clip_dir(scenes, clip_path)
            _finish(report)

        asyncio.run(_run_standalone())


async def _drama_regen_shot_async(
    series: DramaSeries,
    mgr: DramaManager,
    episode: Episode,
    scene_ids: list[str],
    recompose: bool,
) -> None:
    console = get_console()

    from videoclaw.drama.runner import DramaRunner

    runner = DramaRunner(drama_manager=mgr)

    # Regenerate each scene sequentially; only recompose on the last one
    total_cost = 0.0
    results: list[tuple[str, str]] = []

    for i, scene_id in enumerate(scene_ids):
        is_last = i == len(scene_ids) - 1
        do_recompose = recompose and is_last

        node_count = 2 + (3 if do_recompose else 0)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            label = f"Regenerating {scene_id} ({i+1}/{len(scene_ids)})"
            if do_recompose:
                label += " + recompose"
            task = progress.add_task(label, total=node_count)
            state = await runner.regenerate_scene(series, episode, scene_id, do_recompose)
            progress.update(task, completed=node_count)

        total_cost += state.cost_total
        results.append((scene_id, state.status.value))

    # Summary
    all_ok = all(s == "completed" for _, s in results)
    status_style = "green" if all_ok else "red"
    scenes_summary = "\n".join(
        f"  {sid}: [{'green' if s == 'completed' else 'red'}]{s}[/{'green' if s == 'completed' else 'red'}]"
        for sid, s in results
    )
    console.print(
        Panel(
            f"[bold]Scenes:[/bold]\n{scenes_summary}\n"
            f"[bold]Total cost:[/bold] ${total_cost:.4f}",
            title=f"[bold {status_style}]Regen Complete[/bold {status_style}]",
            border_style=status_style,
        )
    )


# ---------------------------------------------------------------------------
# claw drama audit-regen — 审计-重生成自动循环
# ---------------------------------------------------------------------------

@drama_app.command("audit-regen")
def drama_audit_regen(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number.")] = 1,
    clip_dir: Annotated[str, typer.Option("--clip-dir", "-c", help="Directory containing generated MP4 clips.")] = "",
    max_rounds: Annotated[int, typer.Option("--max-rounds", "-n", help="Maximum audit-regen iterations.")] = 3,
    recompose: Annotated[bool, typer.Option("--recompose", help="Re-compose after final round.")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run vision audit → regenerate failing scenes → re-audit loop.

    \b
    Automated QA loop that:
    1. Audits all scenes via Claude Vision
    2. Identifies scenes with regen_required=true
    3. Regenerates those scenes
    4. Re-audits to verify fixes
    5. Repeats until all pass or max rounds reached

    \b
    With --recompose (default), the full episode is re-composed after the
    final round of regeneration.

    \b
    Examples:
        claw drama audit-regen 97e8424712d24fb2
        claw drama audit-regen abc123 -e 1 -n 5 --clip-dir video_clips/
    """
    configure_logging(verbose)
    show_banner()
    console = get_console()
    out = get_output()
    out._command = "drama.audit-regen"

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
        console.print("[yellow]Episode has no scenes.[/yellow]")
        out.set_error("Episode has no scenes.")
        out.emit()
        raise typer.Exit(code=1)

    from pathlib import Path as _Path

    clip_path: _Path | None = _Path(clip_dir) if clip_dir else None

    console.print(
        Panel(
            f"[bold]Series:[/bold]     {series.title}\n"
            f"[bold]Episode:[/bold]    {episode} ({len(ep.scenes)} scenes)\n"
            f"[bold]Max rounds:[/bold] {max_rounds}\n"
            f"[bold]Recompose:[/bold]  {'yes' if recompose else 'no'}",
            title="[bold cyan]Audit-Regen Loop[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        result = asyncio.run(
            _drama_audit_regen_async(
                series, mgr, ep, clip_path, max_rounds, recompose,
            )
        )
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result(result)
    out.emit()


async def _drama_audit_regen_async(
    series: DramaSeries,
    mgr: DramaManager,
    episode: Episode,
    clip_dir: Path | None,
    max_rounds: int,
    recompose: bool,
) -> dict:
    """Execute the audit → regen → re-audit loop."""
    from pathlib import Path as _Path

    from videoclaw.drama.runner import DramaRunner
    from videoclaw.drama.vision_auditor import VisionAuditor

    console = get_console()
    auditor = VisionAuditor()
    runner = DramaRunner(drama_manager=mgr)

    total_cost = 0.0
    history: list[dict] = []

    for round_num in range(1, max_rounds + 1):
        console.print(f"\n[bold cyan]═══ Round {round_num}/{max_rounds} — Audit ═══[/bold cyan]")

        # --- Audit ---
        report = await auditor.audit_series_episode(
            series,
            episode_number=episode.number,
            clip_dir=clip_dir,
            drama_manager=mgr,
            persist_results=True,
        )
        console.print(report.summary())

        regen_ids = report.regen_required
        round_record = {
            "round": round_num,
            "total": report.total_shots,
            "passed": report.passed_shots,
            "regen_required": list(regen_ids),
        }
        history.append(round_record)

        if not regen_ids:
            console.print(
                f"\n[bold green]All {report.total_shots} shots passed "
                f"after {round_num} round(s).[/bold green]"
            )
            break

        console.print(
            f"\n[bold yellow]Regenerating {len(regen_ids)} scene(s): "
            f"{', '.join(regen_ids)}[/bold yellow]"
        )

        # --- Regen failing scenes ---
        for i, scene_id in enumerate(regen_ids):
            is_last_scene = i == len(regen_ids) - 1
            is_last_round = round_num == max_rounds
            # Only recompose on the very last scene of the very last round
            do_recompose = recompose and is_last_scene and is_last_round

            console.print(
                f"  Regenerating {scene_id} ({i+1}/{len(regen_ids)})..."
            )
            state = await runner.regenerate_scene(
                series, episode, scene_id, do_recompose,
            )
            total_cost += state.cost_total

            status = state.status.value
            style = "green" if status == "completed" else "red"
            console.print(f"    [{style}]{status}[/{style}] — ${state.cost_total:.4f}")
    else:
        # Exhausted max rounds — do final recompose if requested
        if recompose and regen_ids:
            console.print(
                f"\n[bold yellow]Max rounds reached. "
                f"Remaining issues: {', '.join(regen_ids)}[/bold yellow]"
            )
            # Final recompose with the last regen
            console.print("[dim]Running final recompose...[/dim]")
            last_scene = regen_ids[-1]
            state = await runner.regenerate_scene(
                series, episode, last_scene, recompose=True,
            )
            total_cost += state.cost_total

    summary = {
        "series_id": series.series_id,
        "episode": episode.number,
        "rounds": len(history),
        "total_cost": round(total_cost, 4),
        "final_passed": history[-1]["passed"] if history else 0,
        "final_total": history[-1]["total"] if history else 0,
        "remaining_issues": history[-1].get("regen_required", []) if history else [],
        "history": history,
    }

    all_passed = summary["final_passed"] == summary["final_total"]
    style = "green" if all_passed else "yellow"
    console.print(
        Panel(
            f"[bold]Rounds:[/bold]    {summary['rounds']}\n"
            f"[bold]Passed:[/bold]    {summary['final_passed']}/{summary['final_total']}\n"
            f"[bold]Total cost:[/bold] ${summary['total_cost']:.4f}",
            title=f"[bold {style}]Audit-Regen Complete[/bold {style}]",
            border_style=style,
        )
    )

    return summary


# ---------------------------------------------------------------------------
# claw drama pipeline — 全流程一键制作
# ---------------------------------------------------------------------------

@drama_app.command("pipeline")
def drama_pipeline(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number.")] = 1,
    skip_design: Annotated[bool, typer.Option("--skip-design", help="Skip character design (already done).")] = False,
    skip_refresh: Annotated[bool, typer.Option("--skip-refresh", help="Skip URL refresh (URLs still valid).")] = False,
    skip_run: Annotated[bool, typer.Option("--skip-run", help="Skip video generation.")] = False,
    skip_audit: Annotated[bool, typer.Option("--skip-audit", help="Skip audit-regen loop.")] = False,
    audit_rounds: Annotated[int, typer.Option("--audit-rounds", "-n", help="Max audit-regen iterations.")] = 3,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", help="Max parallel tasks.")] = 4,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the full production pipeline: design → refresh → generate → audit.

    \b
    One command from prepared script to final video:
      1. design-characters — generate turnaround sheets (skip with --skip-design)
      2. refresh-urls — ensure fresh HTTPS URLs (skip with --skip-refresh)
      3. run — execute video generation pipeline (skip with --skip-run)
      4. audit-regen — vision QA + auto-regenerate failing shots (skip with --skip-audit)

    \b
    Prerequisites: series must exist with planned episodes and scenes.
    Use `claw drama import` or `claw drama plan` + `claw drama script` first.

    \b
    Examples:
        claw drama pipeline 97e8424712d24fb2
        claw drama pipeline abc123 -e 1 --skip-design
        claw drama pipeline abc123 -e 2 -n 5 -c 2
    """
    configure_logging(verbose)
    show_banner()
    console = get_console()
    out = get_output()
    out._command = "drama.pipeline"

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
        console.print("[yellow]Episode has no scenes. Run `claw drama script` or `claw drama import` first.[/yellow]")
        out.set_error("Episode has no scenes.")
        out.emit()
        raise typer.Exit(code=1)

    stages = []
    if not skip_design:
        stages.append("design-characters")
    if not skip_refresh:
        stages.append("refresh-urls")
    if not skip_run:
        stages.append("run")
    if not skip_audit:
        stages.append("audit-regen")

    console.print(
        Panel(
            f"[bold]Series:[/bold]      {series.title}\n"
            f"[bold]Episode:[/bold]     {episode} ({len(ep.scenes)} scenes)\n"
            f"[bold]Stages:[/bold]      {' → '.join(stages)}\n"
            f"[bold]Audit rounds:[/bold] {audit_rounds}",
            title="[bold cyan]Full Pipeline[/bold cyan]",
            border_style="cyan",
        )
    )

    try:
        result = asyncio.run(
            _drama_pipeline_async(
                series, mgr, ep, skip_design, skip_refresh,
                skip_run, skip_audit, audit_rounds, concurrency,
            )
        )
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)

    out.set_result(result)
    out.emit()


async def _drama_pipeline_async(
    series: DramaSeries,
    mgr: DramaManager,
    episode: Episode,
    skip_design: bool,
    skip_refresh: bool,
    skip_run: bool,
    skip_audit: bool,
    audit_rounds: int,
    concurrency: int,
) -> dict:
    """Execute the full production pipeline stages sequentially."""
    console = get_console()
    result: dict = {
        "series_id": series.series_id,
        "episode": episode.number,
        "stages": {},
        "total_cost": 0.0,
    }

    # --- Stage 1: Design characters ---
    if not skip_design:
        console.print("\n[bold cyan]═══ Stage 1/4: Design Characters ═══[/bold cyan]")
        from videoclaw.drama.character_designer import CharacterDesigner

        designer = CharacterDesigner(drama_manager=mgr)
        with console.status("[cyan]Generating turnaround sheets...", spinner="dots"):
            await designer.design_characters(series, force=False)

        chars_with_img = sum(1 for c in series.characters if c.reference_image)
        console.print(
            f"  [green]{chars_with_img}/{len(series.characters)} characters have reference images[/green]"
        )
        result["stages"]["design_characters"] = {"characters": chars_with_img}
    else:
        console.print("\n[dim]Stage 1/4: Design Characters — skipped[/dim]")

    # --- Stage 2: Refresh URLs ---
    if not skip_refresh:
        console.print("\n[bold cyan]═══ Stage 2/4: Refresh URLs ═══[/bold cyan]")
        from videoclaw.drama.runner import ensure_fresh_urls

        with console.status("[cyan]Validating character URLs...", spinner="dots"):
            refreshed = await ensure_fresh_urls(series, drama_manager=mgr)

        ok_count = sum(1 for v in refreshed.values() if v)
        console.print(f"  [green]{ok_count}/{len(refreshed)} characters have valid URLs[/green]")
        result["stages"]["refresh_urls"] = {"valid": ok_count, "total": len(refreshed)}
    else:
        console.print("\n[dim]Stage 2/4: Refresh URLs — skipped[/dim]")

    # --- Stage 3: Run generation ---
    if not skip_run:
        console.print("\n[bold cyan]═══ Stage 3/4: Generate Episode ═══[/bold cyan]")
        from videoclaw.drama.runner import DramaRunner

        runner = DramaRunner(
            drama_manager=mgr,
            max_concurrency=concurrency,
            auto_refresh_urls=False,  # Already refreshed in stage 2
        )

        state = await runner.run_episode(series, episode)
        gen_cost = state.cost_total
        result["total_cost"] += gen_cost

        status_style = "green" if state.status.value == "completed" else "red"
        console.print(
            f"  [{status_style}]Generation {state.status.value}[/{status_style}] — "
            f"${gen_cost:.4f}"
        )
        result["stages"]["run"] = {
            "status": state.status.value,
            "cost": round(gen_cost, 4),
        }
    else:
        console.print("\n[dim]Stage 3/4: Generate Episode — skipped[/dim]")

    # --- Stage 4: Audit-regen loop ---
    if not skip_audit:
        console.print("\n[bold cyan]═══ Stage 4/4: Audit & Regen ═══[/bold cyan]")
        from videoclaw.drama.runner import DramaRunner
        from videoclaw.drama.vision_auditor import VisionAuditor

        auditor = VisionAuditor()
        runner = DramaRunner(drama_manager=mgr, auto_refresh_urls=False)

        audit_cost = 0.0
        final_passed = 0
        final_total = len(episode.scenes)

        for round_num in range(1, audit_rounds + 1):
            console.print(f"  [cyan]Audit round {round_num}/{audit_rounds}...[/cyan]")

            report = await auditor.audit_series_episode(
                series,
                episode_number=episode.number,
                drama_manager=mgr,
                persist_results=True,
            )

            final_passed = report.passed_shots
            final_total = report.total_shots

            if not report.regen_required:
                console.print(
                    f"  [green]All {final_total} shots passed after round {round_num}[/green]"
                )
                break

            console.print(
                f"  [yellow]Regenerating {len(report.regen_required)} scene(s): "
                f"{', '.join(report.regen_required)}[/yellow]"
            )

            for scene_id in report.regen_required:
                state = await runner.regenerate_scene(
                    series, episode, scene_id, recompose=False,
                )
                audit_cost += state.cost_total

        result["total_cost"] += audit_cost
        result["stages"]["audit_regen"] = {
            "passed": final_passed,
            "total": final_total,
            "cost": round(audit_cost, 4),
        }
    else:
        console.print("\n[dim]Stage 4/4: Audit & Regen — skipped[/dim]")

    # --- Summary ---
    all_stages_ok = all(
        stage.get("status", "completed") != "failed"
        for stage in result["stages"].values()
        if isinstance(stage, dict) and "status" in stage
    )
    style = "green" if all_stages_ok else "yellow"
    console.print(
        Panel(
            f"[bold]Stages completed:[/bold] {len(result['stages'])}\n"
            f"[bold]Total cost:[/bold]       ${result['total_cost']:.4f}\n"
            + (
                f"[bold]Audit result:[/bold]     "
                f"{result['stages'].get('audit_regen', {}).get('passed', '?')}/"
                f"{result['stages'].get('audit_regen', {}).get('total', '?')} passed"
                if "audit_regen" in result["stages"] else ""
            ),
            title=f"[bold {style}]Pipeline Complete[/bold {style}]",
            border_style=style,
        )
    )

    return result
