"""``claw info`` -- agent-friendly system overview.

Outputs a comprehensive snapshot of available commands, registered models,
configuration status, and workflow guidance. Designed as the bootstrap
entry point for AI agents interacting with VideoClaw.
"""

from __future__ import annotations

import videoclaw
from videoclaw.cli._app import app
from videoclaw.cli._output import get_console, get_output
from videoclaw.config import get_config

# Structured command catalog — single source of truth for discoverability.
_COMMANDS = {
    "core": [
        {"command": "claw generate", "description": "Full pipeline: prompt → video"},
        {"command": "claw doctor", "description": "System health checks"},
        {"command": "claw info", "description": "Agent bootstrap — this command"},
        {"command": "claw version", "description": "Print version"},
    ],
    "drama": [
        {"command": "claw drama new", "description": "Create series from concept"},
        {"command": "claw drama import", "description": "Import complete script (.docx/.txt)"},
        {"command": "claw drama list", "description": "List all drama series"},
        {"command": "claw drama show <id>", "description": "Show series details"},
        {"command": "claw drama plan <id>", "description": "Plan episodes via LLM"},
        {"command": "claw drama script <id>", "description": "Script an episode (scene breakdown)"},
        {"command": "claw drama design-characters <id>", "description": "Generate turnarounds"},
        {"command": "claw drama refresh-urls <id>", "description": "Refresh expired image URLs"},
        {"command": "claw drama design-scenes <id>", "description": "Generate scene/prop refs"},
        {"command": "claw drama assign-voices <id>", "description": "Assign TTS voice profiles"},
        {"command": "claw drama preview-prompts <id>", "description": "Preview Seedance prompts"},
        {"command": "claw drama run <id>", "description": "Run video generation pipeline"},
        {"command": "claw drama regen-shot <id>", "description": "Regenerate specific scene(s)"},
        {"command": "claw drama audit <id>", "description": "Vision QA on generated clips"},
        {"command": "claw drama audit-regen <id>", "description": "Audit → regen loop"},
        {"command": "claw drama pipeline <id>", "description": "Full pipeline: design→gen→audit"},
        {"command": "claw drama export <id>", "description": "Export assets to deliverables dir"},
    ],
    "stage": [
        {"command": "claw video <prompt>", "description": "Generate a single video clip"},
        {"command": "claw image <prompt>", "description": "Generate a single image"},
        {"command": "claw tts <text>", "description": "Text-to-speech synthesis"},
        {"command": "claw storyboard <prompt>", "description": "Decompose prompt into shots"},
        {"command": "claw compose <videos>", "description": "Compose multiple videos"},
        {"command": "claw subtitle <json>", "description": "Generate subtitles"},
    ],
    "config": [
        {"command": "claw model list", "description": "List registered video model adapters"},
        {"command": "claw config show", "description": "Show all configuration"},
        {"command": "claw config check", "description": "Validate config completeness"},
        {"command": "claw project list", "description": "List projects"},
        {"command": "claw flow run <yaml>", "description": "Run a ClawFlow YAML pipeline"},
    ],
}

_WORKFLOWS = {
    "western_tiktok_drama": {
        "description": "Western photorealistic TikTok short drama (Seedance 2.0, 9:16, en)",
        "steps": [
            "claw drama import script.docx --lang en --title 'My Drama'",
            "claw drama design-characters <series_id>",
            "claw drama design-scenes <series_id>",
            "claw drama assign-voices <series_id>",
            "claw drama pipeline <series_id> --episode 1",
        ],
    },
    "chinese_ai_drama": {
        "description": "Chinese AI short drama (Seedance 2.0, 9:16, zh)",
        "steps": [
            "claw drama new '剧情概念' --lang zh --genre drama",
            "claw drama plan <series_id>",
            "claw drama script <series_id> --episode 1",
            "claw drama pipeline <series_id> --episode 1",
        ],
    },
    "single_video": {
        "description": "Generate a single video clip from a text prompt",
        "steps": [
            "claw video 'A cat playing piano' --duration 5 --model seedance-2.0",
        ],
    },
}


@app.command()
def info() -> None:
    """Show system overview — commands, models, config, workflows.

    \b
    This is the recommended first command for agents. It provides
    a complete structured overview of VideoClaw's capabilities,
    available commands, registered models, and suggested workflows.

    \b
    Use with --json for machine-readable output:
        claw --json info
    """
    console = get_console()
    out = get_output()
    out._command = "info"

    from rich.panel import Panel
    from rich.table import Table

    cfg = get_config()

    # --- Models ---
    from videoclaw.models.registry import get_registry
    registry = get_registry()
    model_names = sorted(registry.list_models()) if hasattr(registry, "list_models") else []

    # --- Drama series ---
    try:
        from videoclaw.drama.models import DramaManager
        mgr = DramaManager()
        series_ids = mgr.list_series()
    except Exception:
        series_ids = []

    # --- Display ---
    console.print(
        Panel(
            f"[bold]Version:[/bold]     {videoclaw.__version__}\n"
            f"[bold]Models:[/bold]      {len(model_names)} registered"
            f" ({', '.join(model_names[:5]) or 'none'})\n"
            f"[bold]Series:[/bold]      {len(series_ids)} drama series\n"
            f"[bold]Video model:[/bold] {cfg.default_video_model}\n"
            f"[bold]LLM:[/bold]         {cfg.default_llm}\n"
            f"[bold]Language:[/bold]    {cfg.default_language}",
            title="[bold cyan]VideoClaw Info[/bold cyan]",
            border_style="cyan",
        )
    )

    for group_name, commands in _COMMANDS.items():
        table = Table(
            title=f"Commands: {group_name}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Command", style="green", min_width=35)
        table.add_column("Description", style="white")
        for cmd in commands:
            table.add_row(cmd["command"], cmd["description"])
        console.print(table)

    console.print(
        "\n[bold]Global flags:[/bold]"
        " --json (structured output), --verbose (debug logs)"
    )
    console.print(
        "[bold]Tip:[/bold] Use [cyan]claw --json <command>[/cyan]"
        " for agent-friendly JSON output.\n"
    )

    # --- JSON result ---
    out.set_result({
        "version": videoclaw.__version__,
        "default_video_model": cfg.default_video_model,
        "default_llm": cfg.default_llm,
        "default_language": cfg.default_language,
        "registered_models": model_names,
        "drama_series_count": len(series_ids),
        "commands": _COMMANDS,
        "workflows": _WORKFLOWS,
    })
    out.emit()
