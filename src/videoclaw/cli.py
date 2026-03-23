"""VideoClaw CLI -- beautiful terminal interface for AI video generation.

Entry point is the :data:`app` Typer instance, registered as ``claw`` in
``pyproject.toml``::

    claw generate "a cat riding a skateboard"
    claw doctor
    claw model list
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
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
from rich.text import Text

import videoclaw
from videoclaw.config import get_config

# Alias for the import path ``from videoclaw.config import Settings``.
Settings = type(get_config())

logger = logging.getLogger("videoclaw")

console = Console()

# ---------------------------------------------------------------------------
# Typer app hierarchy
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="claw",
    help="VideoClaw -- The Agent OS for AI Video Generation.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

model_app = typer.Typer(help="Manage model adapters.", no_args_is_help=True)
project_app = typer.Typer(help="Manage VideoClaw projects.", no_args_is_help=True)
template_app = typer.Typer(help="Flow templates for common video types.", no_args_is_help=True)
flow_app = typer.Typer(help="Run and validate ClawFlow YAML pipelines.", no_args_is_help=True)
drama_app = typer.Typer(help="AI short drama series orchestration.", no_args_is_help=True)

app.add_typer(model_app, name="model")
app.add_typer(project_app, name="project")
app.add_typer(template_app, name="template")
app.add_typer(flow_app, name="flow")
app.add_typer(drama_app, name="drama")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BANNER = r"""
[bold cyan] _   _ _     _             _____ _
| | | (_)   | |           /  __ \ |
| | | |_  __| | ___  ___  | /  \/ | __ ___      __
| | | | |/ _` |/ _ \/ _ \ | |   | |/ _` \ \ /\ / /
\ \_/ / | (_| |  __/ (_) || \__/\ | (_| |\ V  V /
 \___/|_|\__,_|\___|\___/  \____/_|\__,_| \_/\_/
[/bold cyan]
[dim]The Agent OS for AI Video Generation  v{version}[/dim]
"""


def _show_banner() -> None:
    console.print(_BANNER.format(version=videoclaw.__version__))


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def _resolve_templates_dir() -> Path:
    """Return the ``templates/`` directory shipped with the project."""
    # Walk up from this file to the repo root.
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        repo_root / "templates",
        Path.cwd() / "templates",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


# ---------------------------------------------------------------------------
# claw generate
# ---------------------------------------------------------------------------


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="Creative prompt for the video.")],
    duration: Annotated[int, typer.Option("--duration", "-d", help="Target duration in seconds.")] = 30,
    style: Annotated[Optional[str], typer.Option("--style", "-s", help="Visual style hint.")] = None,
    aspect_ratio: Annotated[str, typer.Option("--aspect-ratio", "-a", help="Aspect ratio.")] = "16:9",
    strategy: Annotated[str, typer.Option("--strategy", help="Routing strategy: quality / cost / speed / auto.")] = "auto",
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Output file path.")] = None,
    budget: Annotated[Optional[float], typer.Option("--budget", "-b", help="Maximum budget in USD.")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Preferred model id.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Generate a video from a text prompt.

    Orchestrates the full pipeline: plan, generate shots, compose, and render.
    """
    _configure_logging(verbose)
    _show_banner()

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
        )
    )


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
) -> None:
    """Async orchestration behind ``claw generate``."""
    from videoclaw.core.state import StateManager, ProjectState
    from videoclaw.core.planner import build_dag, DAG
    from videoclaw.cost.tracker import CostTracker

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

    try:
        from videoclaw.core.director import Director

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
    except ImportError:
        # Director not yet implemented -- generate placeholder shots.
        console.print("[yellow]Director module not available yet; using placeholder storyboard.[/yellow]")
        from videoclaw.core.state import Shot

        num_shots = max(1, duration // 5)
        model_id = preferred_model or cfg.default_video_model
        for i in range(num_shots):
            state.storyboard.append(
                Shot(
                    description=f"Shot {i + 1}: {prompt}",
                    prompt=f"({style + ', ' if style else ''}{prompt}) shot {i + 1}/{num_shots}",
                    duration_seconds=duration / num_shots,
                    model_id=model_id,
                )
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

    # ---- 4. Build DAG and execute ----------------------------------------
    console.print("\n[bold cyan]Phase 2: Execution[/bold cyan]")

    dag = build_dag(state)
    tracker = CostTracker(project_id=state.project_id, budget_usd=budget_usd)

    total_nodes = len(dag.nodes)

    try:
        from videoclaw.core.executor import DAGExecutor
        from videoclaw.core.events import event_bus, TASK_COMPLETED

        executor = DAGExecutor(dag=dag, state=state)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Executing pipeline...", total=total_nodes)

            # Subscribe to task completion events to update progress
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

    except ImportError:
        # Executor not yet implemented -- simulate progress.
        console.print("[yellow]DAGExecutor module not available yet; simulating execution.[/yellow]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Simulating pipeline...", total=total_nodes)
            for node_id, node in dag.nodes.items():
                progress.update(task, description=f"Running {node.task_type.value}...")
                await asyncio.sleep(0.15)
                dag.mark_complete(node_id, result="simulated")
                progress.advance(task)

        # Populate mock cost records so the summary is meaningful.
        from videoclaw.cost.tracker import CostRecord

        for node in dag.nodes.values():
            tracker.record(
                CostRecord(
                    task_id=node.node_id,
                    model_id=preferred_model or "mock",
                    execution_mode="cloud" if preferred_model else "local",
                    api_cost_usd=0.0,
                    compute_cost_usd=0.0,
                    duration_seconds=0.15,
                )
            )

    # ---- 5. Cost summary -------------------------------------------------
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

    # ---- 6. Output path --------------------------------------------------
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


# ---------------------------------------------------------------------------
# claw doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Run system health checks and report status."""
    _show_banner()
    console.print("[bold]Running system diagnostics...[/bold]\n")

    table = Table(
        title="System Health",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Check", style="white", min_width=30)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Details", style="dim")

    # -- Python version -----------------------------------------------------
    py_ver = platform.python_version()
    py_ok = sys.version_info >= (3, 12)
    table.add_row(
        "Python >= 3.12",
        _status_icon(py_ok),
        f"Python {py_ver}",
    )

    # -- FFmpeg -------------------------------------------------------------
    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_ok = ffmpeg_path is not None
    ffmpeg_detail = ffmpeg_path or "not found in PATH"
    if ffmpeg_ok:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            first_line = result.stdout.split("\n", 1)[0]
            ffmpeg_detail = first_line[:60]
        except Exception:
            pass
    table.add_row("FFmpeg", _status_icon(ffmpeg_ok), ffmpeg_detail)

    # -- API keys -----------------------------------------------------------
    cfg = get_config()
    openai_ok = bool(cfg.openai_api_key or os.environ.get("OPENAI_API_KEY"))
    table.add_row(
        "OpenAI API key",
        _status_icon(openai_ok),
        "configured" if openai_ok else "missing (set OPENAI_API_KEY or VIDEOCLAW_OPENAI_API_KEY)",
    )

    anthropic_ok = bool(cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))
    table.add_row(
        "Anthropic API key",
        _status_icon(anthropic_ok),
        "configured" if anthropic_ok else "missing (optional)",
    )

    # -- Registered models --------------------------------------------------
    from videoclaw.models.registry import get_registry

    registry = get_registry()
    model_count = len(registry)
    table.add_row(
        "Registered models",
        _status_icon(model_count > 0),
        f"{model_count} adapter(s)" if model_count else "none -- run `claw model list` after setup",
    )

    # -- Disk space ---------------------------------------------------------
    try:
        stat = shutil.disk_usage(cfg.projects_dir.resolve())
        free_gb = stat.free / (1024**3)
        disk_ok = free_gb > 1.0
        table.add_row(
            "Disk space (projects dir)",
            _status_icon(disk_ok),
            f"{free_gb:.1f} GB free",
        )
    except Exception:
        table.add_row("Disk space", "[yellow]?[/yellow]", "unable to determine")

    # -- GPU / Apple Silicon Metal ------------------------------------------
    gpu_detail = "not detected"
    gpu_ok = False
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        gpu_ok = True
        gpu_detail = "Apple Silicon (Metal) available"
    else:
        try:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                gpu_ok = True
                gpu_detail = f"CUDA -- {torch.cuda.get_device_name(0)}"
        except ImportError:
            gpu_detail = "torch not installed (install videoclaw[local])"
    table.add_row("GPU", _status_icon(gpu_ok), gpu_detail)

    console.print(table)
    console.print()


def _status_icon(ok: bool) -> str:
    return "[green]OK[/green]" if ok else "[red]FAIL[/red]"


# ---------------------------------------------------------------------------
# claw model list / pull
# ---------------------------------------------------------------------------


@model_app.command("list")
def model_list() -> None:
    """List all registered model adapters and their health status."""
    from videoclaw.models.registry import get_registry

    registry = get_registry()
    registry.discover()

    models = registry.list_models()
    if not models:
        console.print("[yellow]No model adapters registered.[/yellow]")
        console.print("Adapters are auto-discovered via the [cyan]videoclaw.adapters[/cyan] entry-point group.")
        raise typer.Exit()

    # Run health checks asynchronously.
    health = asyncio.run(registry.health_check_all())

    table = Table(
        title="Registered Model Adapters",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Model ID", style="cyan", min_width=16)
    table.add_column("Capabilities", style="white")
    table.add_column("Mode", style="magenta")
    table.add_column("Health", justify="center")

    for m in models:
        mid = m["model_id"]
        caps = ", ".join(m["capabilities"])
        mode = m["execution_mode"]
        ok = health.get(mid, False)
        table.add_row(mid, caps, mode, _status_icon(ok))

    console.print(table)


@model_app.command("pull")
def model_pull(
    model_id: Annotated[str, typer.Argument(help="Identifier of the model to pull.")],
) -> None:
    """Download / prepare a local model for offline generation."""
    console.print(
        Panel(
            f"[bold]Would download model:[/bold] {model_id}\n\n"
            "Local model management is coming in [cyan]v0.2[/cyan].\n"
            "This will pull weights into the models cache directory and\n"
            "register the adapter for subsequent runs.",
            title="[bold yellow]Model Pull (placeholder)[/bold yellow]",
            border_style="yellow",
        )
    )


# ---------------------------------------------------------------------------
# claw project list / show
# ---------------------------------------------------------------------------


@project_app.command("list")
def project_list() -> None:
    """List all saved projects."""
    from videoclaw.core.state import StateManager

    sm = StateManager()
    project_ids = sm.list_projects()

    if not project_ids:
        console.print("[yellow]No projects found.[/yellow]")
        raise typer.Exit()

    table = Table(
        title="Projects",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Project ID", style="cyan", min_width=20)
    table.add_column("Status", style="magenta")
    table.add_column("Shots", justify="right")
    table.add_column("Prompt", style="white")
    table.add_column("Created", style="dim")

    for pid in sorted(project_ids):
        try:
            state = sm.load(pid)
            table.add_row(
                pid,
                state.status.value,
                str(len(state.storyboard)),
                state.prompt[:50] + ("..." if len(state.prompt) > 50 else ""),
                state.created_at[:19],
            )
        except Exception:
            table.add_row(pid, "[red]error[/red]", "-", "-", "-")

    console.print(table)


@project_app.command("show")
def project_show(
    project_id: Annotated[str, typer.Argument(help="Project identifier.")],
) -> None:
    """Display detailed information about a project."""
    from videoclaw.core.state import StateManager

    sm = StateManager()
    try:
        state = sm.load(project_id)
    except FileNotFoundError:
        console.print(f"[red]Project {project_id!r} not found.[/red]")
        raise typer.Exit(code=1)

    # Header panel.
    console.print(
        Panel(
            f"[bold]ID:[/bold]      {state.project_id}\n"
            f"[bold]Status:[/bold]  {state.status.value}\n"
            f"[bold]Prompt:[/bold]  {state.prompt}\n"
            f"[bold]Created:[/bold] {state.created_at}\n"
            f"[bold]Updated:[/bold] {state.updated_at}\n"
            f"[bold]Cost:[/bold]    ${state.cost_total:.4f}",
            title="[bold green]Project Details[/bold green]",
            border_style="green",
        )
    )

    # Shots table.
    if state.storyboard:
        shot_table = Table(
            title="Storyboard",
            show_header=True,
            header_style="bold magenta",
        )
        shot_table.add_column("#", width=4, style="dim")
        shot_table.add_column("Shot ID", style="cyan")
        shot_table.add_column("Status", style="magenta")
        shot_table.add_column("Model", style="white")
        shot_table.add_column("Duration", justify="right", style="green")
        shot_table.add_column("Cost", justify="right", style="yellow")
        shot_table.add_column("Description", style="dim")

        for idx, shot in enumerate(state.storyboard, 1):
            shot_table.add_row(
                str(idx),
                shot.shot_id,
                shot.status.value,
                shot.model_id,
                f"{shot.duration_seconds:.1f}s",
                f"${shot.cost:.4f}",
                shot.description[:40],
            )
        console.print(shot_table)

    # Cost file (if tracker saved one).
    cost_path = get_config().projects_dir / project_id / "cost.json"
    if cost_path.exists():
        console.print(f"\n[dim]Cost details: {cost_path}[/dim]")


# ---------------------------------------------------------------------------
# claw template list / use
# ---------------------------------------------------------------------------


@template_app.command("list")
def template_list() -> None:
    """List available flow templates."""
    tpl_dir = _resolve_templates_dir()

    if not tpl_dir.exists():
        console.print("[yellow]No templates directory found.[/yellow]")
        raise typer.Exit()

    templates = sorted(tpl_dir.glob("*.claw.yaml"))
    if not templates:
        console.print("[yellow]No .claw.yaml templates found.[/yellow]")
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

    for tpl_path in templates:
        try:
            data = yaml.safe_load(tpl_path.read_text(encoding="utf-8"))
            name = data.get("name", tpl_path.stem)
            desc = data.get("description", "-")
        except Exception:
            name = tpl_path.stem
            desc = "[red]parse error[/red]"

        table.add_row(name, desc, str(tpl_path))

    console.print(table)


@template_app.command("use")
def template_use(
    name: Annotated[str, typer.Argument(help="Template name (without extension).")],
    prompt: Annotated[Optional[str], typer.Option("--prompt", "-p", help="Override the default prompt.")] = None,
) -> None:
    """Generate a video using a predefined flow template."""
    tpl_dir = _resolve_templates_dir()
    tpl_path = tpl_dir / f"{name}.claw.yaml"

    if not tpl_path.exists():
        console.print(f"[red]Template {name!r} not found at {tpl_path}[/red]")
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

    # Delegate to the generate command.
    generate(
        prompt=effective_prompt,
        duration=defaults.get("duration", 30),
        style=defaults.get("style"),
        aspect_ratio=defaults.get("aspect_ratio", "16:9"),
        strategy=defaults.get("strategy", "auto"),
        output=None,
        budget=None,
        model=None,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# claw flow run / validate
# ---------------------------------------------------------------------------


@flow_app.command("run")
def flow_run(
    path: Annotated[str, typer.Argument(help="Path to a ClawFlow YAML file.")],
    prompt: Annotated[Optional[str], typer.Option("--prompt", "-p", help="Override the script prompt.")] = None,
    budget: Annotated[Optional[float], typer.Option("--budget", "-b", help="Maximum budget in USD.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Execute a video pipeline defined in a ClawFlow YAML file."""
    _configure_logging(verbose)
    _show_banner()

    from videoclaw.flow.parser import load_flow, FlowValidationError
    from videoclaw.flow.runner import FlowRunner
    from videoclaw.core.state import StateManager

    try:
        flow = load_flow(path)
    except (FileNotFoundError, FlowValidationError) as exc:
        console.print(f"[red]Error loading flow: {exc}[/red]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Flow:[/bold]    {flow.name}\n"
            f"[bold]Desc:[/bold]    {flow.description}\n"
            f"[bold]Steps:[/bold]   {len(flow.steps)}\n"
            f"[bold]Version:[/bold] {flow.version}",
            title="[bold cyan]ClawFlow Pipeline[/bold cyan]",
            border_style="cyan",
        )
    )

    # Override script prompt if provided.
    if prompt:
        for step in flow.steps:
            if "prompt" in step.params:
                step.params["prompt"] = prompt
                break

    cfg = get_config()
    cfg.ensure_dirs()
    sm = StateManager()
    state = sm.create_project(prompt=prompt or flow.name)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running flow...", total=len(flow.steps))

        async def _run() -> None:
            runner = FlowRunner(state_manager=sm, max_concurrency=4)
            await runner.run(flow, state)

        asyncio.run(_run())
        progress.update(task, completed=len(flow.steps))

    console.print(
        Panel(
            f"[bold]Project:[/bold] {state.project_id}\n"
            f"[bold]Status:[/bold]  {state.status.value}",
            title="[bold green]Flow Complete[/bold green]",
            border_style="green",
        )
    )


@flow_app.command("validate")
def flow_validate(
    path: Annotated[str, typer.Argument(help="Path to a ClawFlow YAML file.")],
) -> None:
    """Validate a ClawFlow YAML file without executing it."""
    from videoclaw.flow.parser import load_flow, FlowValidationError

    try:
        flow = load_flow(path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except FlowValidationError as exc:
        console.print(f"[red]Validation failed: {exc}[/red]")
        raise typer.Exit(code=1)

    # Display validated flow summary.
    table = Table(title=f"Flow: {flow.name}", show_header=True, header_style="bold cyan")
    table.add_column("Step ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Depends On", style="dim")
    table.add_column("Params", style="white")

    for step in flow.steps:
        table.add_row(
            step.id,
            step.type.value,
            ", ".join(step.depends_on) or "-",
            ", ".join(f"{k}={v}" for k, v in step.params.items()) or "-",
        )

    console.print(table)
    console.print("[bold green]Flow is valid.[/bold green]")


# ---------------------------------------------------------------------------
# claw drama new / list / show / plan / script / assign-voices / run
# ---------------------------------------------------------------------------


@drama_app.command("new")
def drama_new(
    synopsis: Annotated[str, typer.Argument(help="High-level story concept for the drama series.")],
    title: Annotated[Optional[str], typer.Option("--title", "-t", help="Series title.")] = None,
    genre: Annotated[str, typer.Option("--genre", "-g", help="Genre.")] = "drama",
    episodes: Annotated[int, typer.Option("--episodes", "-n", help="Number of episodes.")] = 5,
    duration: Annotated[float, typer.Option("--duration", "-d", help="Target seconds per episode.")] = 60.0,
    style: Annotated[str, typer.Option("--style", "-s", help="Visual style.")] = "cinematic",
    language: Annotated[str, typer.Option("--lang", "-l", help="Script language (zh/en).")] = "zh",
    aspect_ratio: Annotated[str, typer.Option("--aspect-ratio", "-a", help="Aspect ratio.")] = "9:16",
    model: Annotated[str, typer.Option("--model", "-m", help="Video model id.")] = "mock",
    plan: Annotated[bool, typer.Option("--plan/--no-plan", help="Immediately plan episodes via LLM.")] = False,
    design_characters: Annotated[bool, typer.Option("--design-characters", help="Generate character reference images after planning.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Create a new AI short drama series."""
    _configure_logging(verbose)
    _show_banner()

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

    if plan:
        console.print("\n[bold cyan]Planning episodes via LLM...[/bold cyan]")
        asyncio.run(_drama_plan_async(series, mgr))

    if design_characters and plan:
        console.print("\n[bold cyan]Generating character reference images...[/bold cyan]")
        asyncio.run(_design_characters_async(series, mgr, force=False))


async def _drama_plan_async(series, mgr) -> None:
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


@drama_app.command("list")
def drama_list() -> None:
    """List all drama series."""
    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    series_ids = mgr.list_series()

    if not series_ids:
        console.print("[yellow]No drama series found.[/yellow]")
        raise typer.Exit()

    table = Table(title="Drama Series", show_header=True, header_style="bold cyan")
    table.add_column("Series ID", style="cyan", min_width=18)
    table.add_column("Title", style="white")
    table.add_column("Status", style="magenta")
    table.add_column("Episodes", justify="right")
    table.add_column("Cost", justify="right", style="yellow")

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
        except Exception:
            table.add_row(sid, "[red]error[/red]", "-", "-", "-")

    console.print(table)


@drama_app.command("show")
def drama_show(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
) -> None:
    """Show detailed info about a drama series."""
    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
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

    if series.characters:
        char_table = Table(title="Characters", show_header=True, header_style="bold magenta")
        char_table.add_column("Name", style="cyan")
        char_table.add_column("Description", style="white")
        char_table.add_column("Voice", style="dim")
        for c in series.characters:
            char_table.add_row(c.name, c.description[:50], c.voice_style)
        console.print(char_table)

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
        console.print(ep_table)


@drama_app.command("plan")
def drama_plan(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Plan episodes for an existing drama series using LLM."""
    _configure_logging(verbose)

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        raise typer.Exit(code=1)

    asyncio.run(_drama_plan_async(series, mgr))


@drama_app.command("design-characters")
def drama_design_characters(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Regenerate existing images.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate reference images for characters in a drama series."""
    _configure_logging(verbose)

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        raise typer.Exit(code=1)

    if not series.characters:
        console.print("[yellow]No characters found. Run `claw drama plan` first.[/yellow]")
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

    asyncio.run(_design_characters_async(series, mgr, force))


async def _design_characters_async(series, mgr, force: bool) -> None:
    from videoclaw.drama.character_designer import CharacterDesigner

    designer = CharacterDesigner(drama_manager=mgr)

    with console.status("[cyan]Generating character reference images...", spinner="dots"):
        series = await designer.design_characters(series, force=force)

    # Show results
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
    _configure_logging(verbose)

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        raise typer.Exit(code=1)

    if not series.episodes or not any(ep.scenes for ep in series.episodes):
        console.print("[yellow]No scenes found. Run `claw drama script` first.[/yellow]")
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

    asyncio.run(_design_scenes_async(series, mgr, force))


async def _design_scenes_async(series, mgr, force: bool) -> None:
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


@drama_app.command("run")
def drama_run(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    episode: Annotated[Optional[int], typer.Option("--episode", "-e", help="Run a specific episode number.")] = None,
    start: Annotated[int, typer.Option("--start", help="Start from episode number.")] = 1,
    end: Annotated[Optional[int], typer.Option("--end", help="End at episode number.")] = None,
    budget: Annotated[Optional[float], typer.Option("--budget", "-b", help="Max budget in USD.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the generation pipeline for drama episodes."""
    _configure_logging(verbose)
    _show_banner()

    from videoclaw.drama.models import DramaManager
    from videoclaw.drama.planner import DramaPlanner

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        raise typer.Exit(code=1)

    if not series.episodes:
        console.print("[yellow]No episodes planned. Run `claw drama plan` first.[/yellow]")
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

    asyncio.run(_drama_run_async(series, mgr, start, end, budget))


async def _drama_run_async(
    series, mgr, start: int, end: int | None, budget_usd: float | None = None,
) -> None:
    from videoclaw.drama.planner import DramaPlanner
    from videoclaw.drama.runner import DramaRunner

    from videoclaw.config import get_config

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

        # Script the episode if not already scripted
        if not ep.scenes:
            with console.status(f"[cyan]Scripting episode {ep.number}...", spinner="dots"):
                script_data = await planner.script_episode(series, ep, prev_cliffhanger)
            prev_cliffhanger = script_data.get("cliffhanger")
            mgr.save(series)
            console.print(f"  Scenes: {len(ep.scenes)}")

        # Run the generation pipeline
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
                total=len(ep.scenes) + 4,  # scenes + script/storyboard/compose/render
            )
            state = await runner.run_episode(series, ep)
            progress.update(task, completed=len(ep.scenes) + 4)

        status_style = "green" if ep.status == "completed" else "red"
        console.print(f"  Status: [{status_style}]{ep.status}[/{status_style}]  Cost: ${ep.cost:.4f}")

        # Budget guard: check cumulative cost against limit
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


@drama_app.command("script")
def drama_script(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number to script.")] = 1,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Script a specific episode — generate scene breakdown via LLM."""
    _configure_logging(verbose)

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        raise typer.Exit(code=1)

    if not series.episodes:
        console.print("[yellow]No episodes planned. Run `claw drama plan` first.[/yellow]")
        raise typer.Exit(code=1)

    ep = next((e for e in series.episodes if e.number == episode), None)
    if ep is None:
        console.print(f"[red]Episode {episode} not found in series.[/red]")
        raise typer.Exit(code=1)

    asyncio.run(_drama_script_async(series, ep, mgr))


async def _drama_script_async(series, ep, mgr) -> None:
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
        script_data = await planner.script_episode(series, ep, prev_cliffhanger)

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
        title=f"Episode {ep.number}: {ep.title} — Scene Breakdown",
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


@drama_app.command("assign-voices")
def drama_assign_voices(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-assign voices even if already set.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Assign TTS voice profiles to all characters in a drama series."""
    _configure_logging(verbose)

    from videoclaw.drama.models import DramaManager, assign_voice_profile

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        raise typer.Exit(code=1)

    if not series.characters:
        console.print("[yellow]No characters found. Run `claw drama plan` first.[/yellow]")
        raise typer.Exit(code=1)

    # Assign voice profiles
    if force:
        for c in series.characters:
            c.voice_profile = None

    for c in series.characters:
        assign_voice_profile(c)

    mgr.save(series)

    # Display voice assignment table
    table = Table(title="Voice Assignments", show_header=True, header_style="bold magenta")
    table.add_column("Character", style="cyan")
    table.add_column("Voice Style", style="white")
    table.add_column("TTS Voice ID", style="green")

    voice_ids_seen: set[str] = set()
    for c in series.characters:
        vp = c.voice_profile
        voice_id = vp.voice_id if vp else "-"
        style = c.voice_style or "-"
        duplicate = voice_id in voice_ids_seen and voice_id != "-"
        voice_ids_seen.add(voice_id)
        id_display = f"[yellow]{voice_id} (dup)[/yellow]" if duplicate else voice_id
        table.add_row(c.name, style, id_display)

    console.print(table)
    console.print(f"\n[bold green]Voices assigned for {len(series.characters)} characters in {series_id}[/bold green]")


@drama_app.command("regen-shot")
def drama_regen_shot(
    series_id: Annotated[str, typer.Argument(help="Drama series ID.")],
    scene: Annotated[str, typer.Option("--scene", "-s", help="Scene ID to regenerate (e.g. ep01_s03).")],
    episode: Annotated[int, typer.Option("--episode", "-e", help="Episode number.")] = 1,
    recompose: Annotated[bool, typer.Option("--recompose", help="Re-compose and re-render the full episode after regenerating.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Regenerate a single scene's video and audio assets.

    Use `claw drama show <series_id>` to find scene IDs, then re-run only the
    scene that needs fixing.  With --recompose, the entire episode is
    re-assembled after the scene is regenerated.

    Examples:

        claw drama regen-shot abc123 -e 2 -s ep02_s03

        claw drama regen-shot abc123 -e 1 -s ep01_s01 --recompose
    """
    _configure_logging(verbose)
    _show_banner()

    from videoclaw.drama.models import DramaManager

    mgr = DramaManager()
    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        raise typer.Exit(code=1)

    # Find the target episode
    ep = next((e for e in series.episodes if e.number == episode), None)
    if ep is None:
        console.print(f"[red]Episode {episode} not found in series.[/red]")
        raise typer.Exit(code=1)

    if not ep.scenes:
        console.print("[yellow]Episode has no scenes. Run `claw drama script` first.[/yellow]")
        raise typer.Exit(code=1)

    # Verify scene_id exists
    scene_ids = [s.scene_id for s in ep.scenes]
    if scene not in scene_ids:
        console.print(f"[red]Scene {scene!r} not found in episode {episode}.[/red]")
        console.print(f"[dim]Available scenes: {', '.join(scene_ids)}[/dim]")
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

    asyncio.run(_drama_regen_shot_async(series, mgr, ep, scene, recompose))


async def _drama_regen_shot_async(
    series, mgr, episode, scene_id: str, recompose: bool,
) -> None:
    from videoclaw.drama.runner import DramaRunner

    runner = DramaRunner(drama_manager=mgr)

    node_count = 2 + (3 if recompose else 0)  # video + tts + optional subtitle/compose/render
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


# ---------------------------------------------------------------------------
# claw version
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the VideoClaw version."""
    console.print(f"[bold cyan]VideoClaw[/bold cyan] v{videoclaw.__version__}")
