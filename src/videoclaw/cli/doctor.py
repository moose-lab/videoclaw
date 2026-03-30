"""``claw doctor`` -- system health checks."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

from rich.table import Table

from videoclaw.cli._app import app, show_banner, status_icon
from videoclaw.cli._output import get_console, get_output
from videoclaw.config import get_config



@app.command()
def doctor() -> None:
    """Run system health checks and report status."""
    console = get_console()
    out = get_output()
    out._command = "doctor"
    show_banner()
    console.print("[bold]Running system diagnostics...[/bold]\n")

    checks: dict[str, dict] = {}

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
    table.add_row("Python >= 3.12", status_icon(py_ok), f"Python {py_ver}")
    checks["python"] = {"ok": py_ok, "version": py_ver}

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
    table.add_row("FFmpeg", status_icon(ffmpeg_ok), ffmpeg_detail)
    checks["ffmpeg"] = {"ok": ffmpeg_ok, "path": ffmpeg_path}

    # -- API keys -----------------------------------------------------------
    cfg = get_config()
    openai_ok = bool(cfg.openai_api_key or os.environ.get("OPENAI_API_KEY"))
    table.add_row(
        "OpenAI API key",
        status_icon(openai_ok),
        "configured" if openai_ok else "missing (set OPENAI_API_KEY or VIDEOCLAW_OPENAI_API_KEY)",
    )
    checks["openai_key"] = {"ok": openai_ok}

    anthropic_ok = bool(cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))
    table.add_row(
        "Anthropic API key",
        status_icon(anthropic_ok),
        "configured" if anthropic_ok else "missing (optional)",
    )
    checks["anthropic_key"] = {"ok": anthropic_ok}

    # -- Registered models --------------------------------------------------
    from videoclaw.models.registry import get_registry

    registry = get_registry()
    model_count = len(registry)
    table.add_row(
        "Registered models",
        status_icon(model_count > 0),
        f"{model_count} adapter(s)" if model_count else "none -- run `claw model list` after setup",
    )
    checks["models"] = {"ok": model_count > 0, "count": model_count}

    # -- Disk space ---------------------------------------------------------
    try:
        stat = shutil.disk_usage(cfg.projects_dir.resolve())
        free_gb = stat.free / (1024**3)
        disk_ok = free_gb > 1.0
        table.add_row(
            "Disk space (projects dir)",
            status_icon(disk_ok),
            f"{free_gb:.1f} GB free",
        )
        checks["disk"] = {"ok": disk_ok, "free_gb": round(free_gb, 1)}
    except Exception:
        table.add_row("Disk space", "[yellow]?[/yellow]", "unable to determine")
        checks["disk"] = {"ok": False, "error": "unable to determine"}

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
    table.add_row("GPU", status_icon(gpu_ok), gpu_detail)
    checks["gpu"] = {"ok": gpu_ok, "detail": gpu_detail}

    console.print(table)
    console.print()

    out.set_result({"checks": checks, "all_ok": all(c["ok"] for c in checks.values())})
    out.emit()
