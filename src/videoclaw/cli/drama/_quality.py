"""``claw drama audit``, ``audit-regen``, and ``pipeline`` commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import typer

if TYPE_CHECKING:
    from videoclaw.drama.models import DramaManager, DramaSeries, Episode

from rich.panel import Panel

from videoclaw.cli._app import (
    drama_app,
    configure_logging,
    show_banner,
)
from videoclaw.cli._output import get_console, get_output


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
    No automatic regeneration is triggered -- re-generation decisions are yours.

    \b
    SERIES-AWARE MODE (recommended -- loads scenes from drama manager):
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


# ---------------------------------------------------------------------------
# claw drama audit-regen
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
    """Run vision audit -> regenerate failing scenes -> re-audit loop.

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

    clip_path: Path | None = Path(clip_dir) if clip_dir else None

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
    """Execute the audit -> regen -> re-audit loop."""
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
        # Exhausted max rounds -- do final recompose if requested
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
# claw drama pipeline
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
    """Run the full production pipeline: design -> refresh -> generate -> audit.

    \b
    One command from prepared script to final video:
      1. design-characters -- generate turnaround sheets (skip with --skip-design)
      2. refresh-urls -- ensure fresh HTTPS URLs (skip with --skip-refresh)
      3. run -- execute video generation pipeline (skip with --skip-run)
      4. audit-regen -- vision QA + auto-regenerate failing shots (skip with --skip-audit)

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
