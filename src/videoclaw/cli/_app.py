"""VideoClaw CLI -- main Typer app definition and shared helpers.

The :data:`app` Typer instance is the root CLI registered as ``claw`` in
``pyproject.toml``.  Sub-apps (model, project, template, flow, drama) and
top-level commands are registered by importing command modules in
``cli/__init__.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

import videoclaw
from videoclaw.cli._output import get_console, get_output

logger = logging.getLogger("videoclaw")

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
drama_app = typer.Typer(
    help=(
        "AI short drama series orchestration.\n\n"
        "Default video model: [bold]Seedance 2.0[/bold]\n"
        "  - 4-15 seconds per clip (hard limit)\n"
        "  - Video + audio + dialogue co-generation in one pass\n"
        "  - 9:16 vertical format (720p) for TikTok\n"
        "  - Universal Reference for cross-clip character consistency\n\n"
        "Workflows:\n"
        "  [cyan]claw drama new[/cyan]    Create from concept (LLM writes script)\n"
        "  [cyan]claw drama import[/cyan] Import complete script (LOCKED, no creative changes)\n"
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)
config_app = typer.Typer(help="View and check VideoClaw configuration.", no_args_is_help=True)

app.add_typer(model_app, name="model")
app.add_typer(project_app, name="project")
app.add_typer(template_app, name="template")
app.add_typer(flow_app, name="flow")
app.add_typer(drama_app, name="drama")
app.add_typer(config_app, name="config")


# ---------------------------------------------------------------------------
# Global callback (--json, --verbose)
# ---------------------------------------------------------------------------

@app.callback()
def main_callback(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json", "-j",
            help="Output structured JSON instead of rich formatting (agent-friendly).",
        ),
    ] = False,
) -> None:
    """VideoClaw -- The Agent OS for AI Video Generation."""
    ctx = get_output()
    ctx.json_mode = json_output


# ---------------------------------------------------------------------------
# Shared helpers
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


def show_banner() -> None:
    """Print the ASCII banner (suppressed in JSON mode)."""
    get_console().print(_BANNER.format(version=videoclaw.__version__))


def configure_logging(verbose: bool = False) -> None:
    """Set up Rich-based logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            console=get_console(),
            rich_tracebacks=True,
            show_path=False,
        )],
    )


def resolve_templates_dir() -> Path:
    """Return the ``templates/`` directory shipped with the project."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    candidates = [
        repo_root / "templates",
        Path.cwd() / "templates",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


def status_icon(ok: bool) -> str:
    """Return a coloured OK/FAIL status string."""
    return "[green]OK[/green]" if ok else "[red]FAIL[/red]"


# ---------------------------------------------------------------------------
# Input validators
# ---------------------------------------------------------------------------

VALID_ASPECT_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"}
VALID_STRATEGIES = {"quality", "cost", "speed", "auto"}
VALID_LANGUAGES = {"zh", "en"}


def validate_aspect_ratio(value: str) -> str:
    """Typer callback to validate ``--aspect-ratio``."""
    if value not in VALID_ASPECT_RATIOS:
        raise typer.BadParameter(
            f"Invalid aspect ratio {value!r}. "
            f"Valid: {', '.join(sorted(VALID_ASPECT_RATIOS))}"
        )
    return value


def validate_strategy(value: str) -> str:
    """Typer callback to validate ``--strategy``."""
    if value not in VALID_STRATEGIES:
        raise typer.BadParameter(
            f"Invalid strategy {value!r}. "
            f"Valid: {', '.join(sorted(VALID_STRATEGIES))}"
        )
    return value


def validate_language(value: str) -> str:
    """Typer callback to validate ``--lang``."""
    if value not in VALID_LANGUAGES:
        raise typer.BadParameter(
            f"Invalid language {value!r}. Valid: {', '.join(sorted(VALID_LANGUAGES))}"
        )
    return value
