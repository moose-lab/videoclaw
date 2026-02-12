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

app.add_typer(model_app, name="model")
app.add_typer(project_app, name="project")
app.add_typer(template_app, name="template")
app.add_typer(flow_app, name="flow")


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
                preferred_model=preferred_model,
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

        executor = DAGExecutor(dag=dag, state=state, tracker=tracker)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Executing pipeline...", total=total_nodes)
            await executor.run(on_node_complete=lambda _nid: progress.advance(task))

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
# claw ui
# ---------------------------------------------------------------------------


@app.command()
def ui(
    port: Annotated[int, typer.Option("--port", "-p", help="Dev server port.")] = 3000,
) -> None:
    """Launch the VideoClaw web UI (Next.js dev server)."""
    web_dir = Path(__file__).resolve().parent.parent.parent / "web"
    if not (web_dir / "package.json").exists():
        console.print("[red]Web UI not found. Expected at:[/red]")
        console.print(f"  {web_dir}")
        console.print("\n[dim]Run: cd web && npm install && npm run dev[/dim]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Starting VideoClaw Web UI[/bold]\n\n"
            f"  URL:  [cyan]http://localhost:{port}[/cyan]\n"
            f"  API:  [cyan]http://localhost:8000[/cyan]\n\n"
            "[dim]Make sure the API server is also running:\n"
            "  uvicorn videoclaw.server.app:create_app --factory[/dim]",
            title="[bold cyan]Web UI[/bold cyan]",
            border_style="cyan",
        )
    )

    os.execvp("npm", ["npm", "run", "dev", "--prefix", str(web_dir), "--", "-p", str(port)])


# ---------------------------------------------------------------------------
# claw version
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the VideoClaw version."""
    console.print(f"[bold cyan]VideoClaw[/bold cyan] v{videoclaw.__version__}")
