"""``claw drama preview-prompts``, ``run``, and ``regen-shot`` commands."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Annotated

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

from videoclaw.cli._app import (
    configure_logging,
    drama_app,
    show_banner,
)
from videoclaw.cli._output import get_console, get_output
from videoclaw.config import get_config

# ---------------------------------------------------------------------------
# claw drama preview-prompts
# ---------------------------------------------------------------------------

@drama_app.command("preview-prompts")
def drama_preview_prompts(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number.")] = 1,
    scene: Annotated[
        str | None,
        typer.Option("--scene", "-s", help="Show only this scene ID."),
    ] = None,
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
    # Persist enhanced prompts to series.json for reproducibility
    mgr.save(series)

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
            "prompt": sc.effective_prompt,
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
            dlg_trunc = sc.dialogue[:80]
            ellipsis = "\u2026" if len(sc.dialogue) > 80 else ""
            console.print(
                f'[dim]Dialogue: "{dlg_trunc}{ellipsis}"[/dim]'
            )
        console.print(f"\n[green]{sc.effective_prompt}[/green]")

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
    episode: Annotated[
        int | None,
        typer.Option("--episode", "-e", help="Run a specific episode number."),
    ] = None,
    start: Annotated[
        int, typer.Option("--start", help="Start from episode number.")
    ] = 1,
    end: Annotated[
        int | None, typer.Option("--end", help="End at episode number.")
    ] = None,
    budget: Annotated[
        float | None, typer.Option("--budget", "-b", help="Max budget in USD.")
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", help="Max parallel tasks.")
    ] = 4,
    refresh_urls: Annotated[
        bool,
        typer.Option(
            "--refresh-urls/--no-refresh-urls",
            help="Auto-validate and refresh expired character reference URLs before generation.",
        ),
    ] = True,
    max_shots: Annotated[
        int | None,
        typer.Option(
            "--max-shots",
            help="Limit video/TTS generation to the first N shots (useful for test runs).",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show execution plan without running.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the generation pipeline for drama episodes.

    \b
    Uses Seedance 2.0 by default (4-15s per clip). Each clip generates
    video + audio + dialogue in a single pass. Character consistency is
    enforced via Universal Reference and a pre-built ConsistencyManifest.

    \b
    Use --max-shots N to limit generation to the first N shots (e.g. for test runs).
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
        asyncio.run(_drama_run_async(
            series, mgr, start, end, budget,
            concurrency, refresh_urls, max_shots=max_shots,
        ))
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
    *,
    max_shots: int | None = None,
) -> None:
    console = get_console()

    from videoclaw.drama.planner import DramaPlanner
    from videoclaw.drama.runner import DramaRunner

    planner = DramaPlanner()
    effective_budget = budget_usd or get_config().budget_default_usd
    runner = DramaRunner(
        drama_manager=mgr,
        auto_refresh_urls=auto_refresh_urls,
        budget_usd=effective_budget,
    )

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
            await runner.run_episode(series, ep, max_shots=max_shots)
            progress.update(task, completed=len(ep.scenes) + 4)

        status_style = "green" if ep.status == "completed" else "red"
        console.print(
            f"  Status: [{status_style}]{ep.status}[/{status_style}]"
            f"  Cost: ${ep.cost:.4f}"
        )

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
    scene: Annotated[
        str,
        typer.Option(
            "--scene", "-s",
            help="Scene ID(s) to regenerate, comma-separated"
            " (e.g. ep01_s01,ep01_s03,ep01_s05).",
        ),
    ],
    episode: Annotated[
        int, typer.Option("--episode", "-e", help="Episode number.")
    ] = 1,
    recompose: Annotated[
        bool,
        typer.Option(
            "--recompose",
            help="Re-compose and re-render the full episode after regenerating.",
        ),
    ] = False,
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
        f"  {sid}: [{'green' if s == 'completed' else 'red'}]"
        f"{s}[/{'green' if s == 'completed' else 'red'}]"
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
