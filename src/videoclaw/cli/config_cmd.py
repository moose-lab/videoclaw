"""``claw config`` -- configuration management commands."""

from __future__ import annotations

import os
import shutil

from rich.panel import Panel
from rich.table import Table

from videoclaw.cli._app import config_app, status_icon
from videoclaw.cli._output import get_console, get_output
from videoclaw.config import get_config


def _mask_key(value: str | None) -> str:
    """Mask an API key for display: show first 4 chars + ****."""
    if not value:
        return "[dim]not set[/dim]"
    if len(value) <= 6:
        return value[:2] + "****"
    return value[:4] + "****"


@config_app.command("show")
def config_show() -> None:
    """Display all current configuration values (API keys masked)."""
    console = get_console()
    out = get_output()
    out._command = "config.show"

    cfg = get_config()

    # Categorise fields
    api_keys = {
        "openai_api_key": cfg.openai_api_key,
        "anthropic_api_key": cfg.anthropic_api_key,
        "moonshot_api_key": cfg.moonshot_api_key,
        "evolink_api_key": cfg.evolink_api_key,
        "kling_access_key": cfg.kling_access_key,
        "kling_secret_key": cfg.kling_secret_key,
        "minimax_api_key": cfg.minimax_api_key,
        "zhipu_api_key": cfg.zhipu_api_key,
        "wavespeed_api_key": cfg.wavespeed_api_key,
        "ark_api_key": cfg.ark_api_key,
        "byteplus_api_key": cfg.byteplus_api_key,
        "google_api_key": cfg.google_api_key,
    }

    table = Table(title="VideoClaw Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="cyan", min_width=28)
    table.add_column("Value", style="white")

    # Directories
    table.add_row("projects_dir", str(cfg.projects_dir.resolve()))
    table.add_row("models_dir", str(cfg.models_dir.resolve()))
    table.add_row("", "")

    # Defaults
    table.add_row("default_llm", cfg.default_llm)
    table.add_row("default_video_model", cfg.default_video_model)
    table.add_row("default_language", cfg.default_language)
    table.add_row("log_level", cfg.log_level)
    table.add_row("", "")

    # Budget & resilience
    table.add_row("budget_default_usd", f"${cfg.budget_default_usd:.2f}")
    table.add_row("max_retries", str(cfg.max_retries))
    table.add_row("", "")

    # API base URLs
    table.add_row("seedance_base_url", cfg.seedance_base_url)
    table.add_row("moonshot_api_base", cfg.moonshot_api_base)
    table.add_row("evolink_api_base", cfg.evolink_api_base)
    table.add_row("byteplus_api_base", cfg.byteplus_api_base)
    table.add_row("", "")

    # API keys (masked)
    for key, value in api_keys.items():
        table.add_row(key, _mask_key(value))

    console.print(table)

    # JSON result (keys masked)
    result = {
        "projects_dir": str(cfg.projects_dir.resolve()),
        "models_dir": str(cfg.models_dir.resolve()),
        "default_llm": cfg.default_llm,
        "default_video_model": cfg.default_video_model,
        "default_language": cfg.default_language,
        "budget_default_usd": cfg.budget_default_usd,
        "max_retries": cfg.max_retries,
        "seedance_base_url": cfg.seedance_base_url,
        "api_keys": {k: ("configured" if v else "not_set") for k, v in api_keys.items()},
    }
    out.set_result(result)
    out.emit()


@config_app.command("check")
def config_check() -> None:
    """Validate configuration completeness for video generation."""
    console = get_console()
    out = get_output()
    out._command = "config.check"

    cfg = get_config()
    checks: dict[str, dict] = {}

    table = Table(title="Configuration Check", show_header=True, header_style="bold cyan")
    table.add_column("Check", style="white", min_width=30)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Details", style="dim")

    # Projects directory writable
    try:
        cfg.ensure_dirs()
        dir_ok = cfg.projects_dir.exists() and os.access(cfg.projects_dir, os.W_OK)
    except Exception:
        dir_ok = False
    table.add_row("Projects directory writable", status_icon(dir_ok), str(cfg.projects_dir.resolve()))
    checks["projects_dir"] = {"ok": dir_ok}

    # FFmpeg
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    table.add_row("FFmpeg in PATH", status_icon(ffmpeg_ok), shutil.which("ffmpeg") or "not found")
    checks["ffmpeg"] = {"ok": ffmpeg_ok}

    # At least one video model API key
    video_keys = {
        "ark_api_key (Seedance)": cfg.ark_api_key,
        "byteplus_api_key": cfg.byteplus_api_key,
        "openai_api_key (Sora)": cfg.openai_api_key,
        "kling_access_key": cfg.kling_access_key,
        "minimax_api_key": cfg.minimax_api_key,
        "zhipu_api_key": cfg.zhipu_api_key,
    }
    video_key_ok = any(v for v in video_keys.values())
    configured = [k for k, v in video_keys.items() if v]
    table.add_row(
        "Video model API key",
        status_icon(video_key_ok),
        ", ".join(configured) if configured else "none configured",
    )
    checks["video_api_key"] = {"ok": video_key_ok, "configured": configured}

    # LLM API key (for script/storyboard generation)
    llm_ok = bool(
        cfg.openai_api_key
        or os.environ.get("OPENAI_API_KEY")
        or cfg.anthropic_api_key
        or cfg.moonshot_api_key
        or cfg.evolink_api_key
    )
    table.add_row("LLM API key", status_icon(llm_ok), "needed for script/storyboard generation")
    checks["llm_api_key"] = {"ok": llm_ok}

    # Image generation key (for character/scene references)
    img_ok = bool(cfg.google_api_key or cfg.evolink_api_key or cfg.byteplus_api_key)
    table.add_row("Image generation API key", status_icon(img_ok), "needed for character/scene references")
    checks["image_api_key"] = {"ok": img_ok}

    console.print(table)

    all_ok = all(c["ok"] for c in checks.values())
    if all_ok:
        console.print("\n[bold green]All checks passed. Ready for video generation.[/bold green]")
    else:
        console.print("\n[yellow]Some checks failed. Set missing API keys via environment variables (VIDEOCLAW_*) or .env file.[/yellow]")

    out.set_result({"checks": checks, "all_ok": all_ok})
    out.emit()
