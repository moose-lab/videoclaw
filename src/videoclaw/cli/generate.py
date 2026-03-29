"""``claw generate`` -- full pipeline video generation from a text prompt."""

from __future__ import annotations

import asyncio
from typing import Annotated, Optional

import typer
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
    app,
    configure_logging,
    show_banner,
    validate_aspect_ratio,
    validate_strategy,
)
from videoclaw.cli._output import get_console, get_output
from videoclaw.config import get_config


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="Creative prompt for the video.")],
    duration: Annotated[int, typer.Option("--duration", "-d", help="Target duration in seconds.")] = 30,
    style: Annotated[Optional[str], typer.Option("--style", "-s", help="Visual style hint.")] = None,
    aspect_ratio: Annotated[str, typer.Option("--aspect-ratio", "-a", help="Aspect ratio.", callback=validate_aspect_ratio)] = "16:9",
    strategy: Annotated[str, typer.Option("--strategy", help="Routing strategy: quality / cost / speed / auto.", callback=validate_strategy)] = "auto",
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Output file path.")] = None,
    budget: Annotated[Optional[float], typer.Option("--budget", "-b", help="Maximum budget in USD.")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Preferred model id.")] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", help="Max parallel tasks.")] = 4,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Build DAG and show plan without executing.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Generate a video from a text prompt.

    Orchestrates the full pipeline: plan, generate shots, compose, and render.
    """
    configure_logging(verbose)
    show_banner()

    try:
        asyncio.run(
            _generate_async(
                prompt=prompt,
                duration=duration,
                style=style,
                aspect_ratio=aspect_ratio,
                strategy=strategy,
                output_path=output,
                budget_usd=budget,
                preferred_model=model,
                max_concurrency=concurrency,
                dry_run=dry_run,
            )
        )
    except Exception as exc:
        from videoclaw.cli._output import get_output
        out = get_output()
        out._command = "generate"
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)


async def _generate_async(
    *,
    prompt: str,
    duration: int,
    style: str | None,
    aspect_ratio: str,
    strategy: str,
    output_path: str | None,
    budget_usd: float | None,
    preferred_model: str | None,
    max_concurrency: int,
    dry_run: bool,
) -> None:
    """Async orchestration behind ``claw generate``."""
    from videoclaw.core.state import StateManager, ProjectState
    from videoclaw.core.planner import build_dag
    from videoclaw.cost.tracker import CostTracker

    console = get_console()
    out = get_output()
    out._command = "generate"
    cfg = get_config()
    cfg.ensure_dirs()

    # ---- 1. Create project ------------------------------------------------
    sm = StateManager()
    state = sm.create_project(prompt)

    console.print(
        Panel(
            f"[bold]Project:[/bold] {state.project_id}\n"
            f"[bold]Prompt:[/bold]  {prompt}\n"
            f"[bold]Duration:[/bold] {duration}s  |  "
            f"[bold]Aspect:[/bold] {aspect_ratio}  |  "
            f"[bold]Strategy:[/bold] {strategy}",
            title="[bold green]New Project[/bold green]",
            border_style="green",
        )
    )

    # ---- 2. Plan (Director) -----------------------------------------------
    console.print("\n[bold cyan]Phase 1: Planning[/bold cyan]")

    from videoclaw.core.director import Director
    from videoclaw.core.state import Shot

    with console.status("[cyan]Director is planning shots...", spinner="dots"):
        director = Director()
        state = await director.plan(
            state,
            duration=duration,
            style=style,
            aspect_ratio=aspect_ratio,
            preferred_model=preferred_model or cfg.default_video_model,
        )
        sm.save(state)

    # ---- 3. Show planned shots -------------------------------------------
    shot_table = Table(
        title="Storyboard",
        show_header=True,
        header_style="bold magenta",
    )
    shot_table.add_column("#", style="dim", width=4)
    shot_table.add_column("Shot ID", style="cyan", min_width=14)
    shot_table.add_column("Description", style="white")
    shot_table.add_column("Model", style="magenta")
    shot_table.add_column("Duration", justify="right", style="green")

    for idx, shot in enumerate(state.storyboard, 1):
        shot_table.add_row(
            str(idx),
            shot.shot_id,
            shot.description[:60] + ("..." if len(shot.description) > 60 else ""),
            shot.model_id,
            f"{shot.duration_seconds:.1f}s",
        )

    console.print(shot_table)

    # ---- 4. Build DAG ----------------------------------------------------
    dag = build_dag(state)
    tracker = CostTracker(project_id=state.project_id, budget_usd=budget_usd)
    total_nodes = len(dag.nodes)

    # ---- dry-run: show plan and exit -------------------------------------
    if dry_run:
        dag_summary = {
            "project_id": state.project_id,
            "dry_run": True,
            "total_tasks": total_nodes,
            "shots": len(state.storyboard),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "task_type": n.task_type.value,
                    "depends_on": n.depends_on,
                }
                for n in dag.nodes.values()
            ],
        }
        console.print(
            Panel(
                f"[bold]Tasks:[/bold] {total_nodes}\n"
                f"[bold]Shots:[/bold] {len(state.storyboard)}\n"
                f"[dim]Dry run — no generation performed.[/dim]",
                title="[bold yellow]Execution Plan (dry-run)[/bold yellow]",
                border_style="yellow",
            )
        )
        out.set_result(dag_summary)
        out.emit()
        return

    # ---- 5. Execute DAG --------------------------------------------------
    console.print("\n[bold cyan]Phase 2: Execution[/bold cyan]")

    from videoclaw.core.executor import DAGExecutor
    from videoclaw.core.events import event_bus, TASK_COMPLETED

    executor = DAGExecutor(dag=dag, state=state, max_concurrency=max_concurrency)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Executing pipeline...", total=total_nodes)

        completed_count = 0

        async def on_task_completed(event_type: str, event_data: dict) -> None:
            nonlocal completed_count
            completed_count += 1
            task_type = event_data.get("task_type", "unknown")
            progress.update(task, description=f"Completed {task_type} ({completed_count}/{total_nodes})")
            progress.advance(task)

        event_bus.subscribe(TASK_COMPLETED, on_task_completed)
        try:
            await executor.run()
        finally:
            event_bus.unsubscribe(TASK_COMPLETED, on_task_completed)

    # ---- 6. Cost summary -------------------------------------------------
    console.print()
    console.print(tracker.format_table())

    hints = tracker.get_optimization_hints()
    if hints:
        hint_panel_lines = [f"  -> {h.message}" for h in hints]
        console.print(
            Panel(
                "\n".join(hint_panel_lines),
                title="[bold yellow]Optimisation Hints[/bold yellow]",
                border_style="yellow",
            )
        )

    # ---- 7. Output path --------------------------------------------------
    output_file = output_path or str(cfg.projects_dir / state.project_id / "output.mp4")
    console.print(
        Panel(
            f"[bold green]{output_file}[/bold green]",
            title="Output",
            border_style="green",
        )
    )

    within, remaining = tracker.check_budget()
    if not within:
        console.print(f"[bold red]WARNING: Over budget by ${abs(remaining):.2f}[/bold red]")

    console.print("[bold green]Done![/bold green]\n")

    # ---- JSON output -----------------------------------------------------
    out.set_result({
        "project_id": state.project_id,
        "status": state.status.value,
        "output": output_file,
        "shots": len(state.storyboard),
        "cost_total": state.cost_total,
    })
    out.emit()
