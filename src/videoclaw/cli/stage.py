"""Single-stage CLI commands for independent pipeline step execution.

These commands allow agents and humans to run individual generation stages
without going through the full DAG pipeline.  Each command wraps the
corresponding generation module directly.

Commands::

    claw video <prompt>           Generate a single video clip
    claw image <prompt>           Generate a single image
    claw tts <text>               Text-to-speech synthesis
    claw storyboard <prompt>      Decompose a prompt into shot list
    claw compose <videos...>      Compose multiple videos together
    claw render <input>           Encode/render a final video
    claw subtitle <scenes.json>   Generate subtitles from scene data
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from videoclaw.cli._app import (
    app,
    configure_logging,
    validate_aspect_ratio,
    validate_language,
    validate_prompt,
    validate_strategy,
)
from videoclaw.cli._output import get_console, get_output
from videoclaw.config import get_config

# Input safety limits
_MAX_JSON_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# claw video
# ---------------------------------------------------------------------------

@app.command()
def video(
    prompt: Annotated[
        str,
        typer.Argument(
            help="Generation prompt for the video clip.",
            callback=validate_prompt,
        ),
    ],
    duration: Annotated[
        float, typer.Option("--duration", "-d", help="Clip duration in seconds.")
    ] = 5.0,
    aspect_ratio: Annotated[
        str,
        typer.Option(
            "--aspect-ratio", "-a",
            help="Aspect ratio.",
            callback=validate_aspect_ratio,
        ),
    ] = "16:9",
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Video model id.")
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help="Routing strategy: quality/cost/speed/auto.",
            callback=validate_strategy,
        ),
    ] = "auto",
    reference_image: Annotated[
        str | None,
        typer.Option("--ref-image", "-r", help="Reference image path or URL."),
    ] = None,
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Output file path.")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging.")
    ] = False,
) -> None:
    """Generate a single video clip from a text prompt.

    \b
    Examples:
      claw video "a cat riding a skateboard" -d 5 -o cat.mp4
      claw video "sunset over ocean" --model seedance-2.0 -a 9:16
      claw video "product showcase" --ref-image product.png -o promo.mp4
    """
    configure_logging(verbose)
    out = get_output()
    out._command = "video"

    try:
        asyncio.run(_video_async(
            prompt=prompt,
            duration=duration,
            aspect_ratio=aspect_ratio,
            model_id=model,
            strategy=strategy,
            reference_image=reference_image,
            output_path=output,
        ))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)


async def _video_async(
    *,
    prompt: str,
    duration: float,
    aspect_ratio: str,
    model_id: str | None,
    strategy: str,
    reference_image: str | None,
    output_path: str | None,
) -> None:
    console = get_console()
    out = get_output()
    cfg = get_config()

    from videoclaw.core.state import Shot
    from videoclaw.generation.video import VideoGenerator
    from videoclaw.models.registry import get_registry
    from videoclaw.models.router import ModelRouter, RoutingStrategy

    strategy_map = {
        "quality": RoutingStrategy.QUALITY_FIRST,
        "cost": RoutingStrategy.COST_FIRST,
        "speed": RoutingStrategy.SPEED_FIRST,
        "auto": RoutingStrategy.AUTO,
    }

    effective_model = model_id or cfg.default_video_model
    shot = Shot(
        description=prompt,
        prompt=prompt,
        duration_seconds=duration,
        model_id=effective_model,
    )

    registry = get_registry()
    registry.discover()
    router = ModelRouter(registry)
    generator = VideoGenerator(router=router)

    extra: dict = {}
    if reference_image:
        if reference_image.startswith("http"):
            extra["image_urls"] = [{"url": reference_image, "role": "reference_image"}]
        else:
            extra["image_paths"] = [{"path": reference_image, "role": "reference_image"}]

    console.print(f"[cyan]Generating video:[/cyan] {prompt[:80]}")
    console.print(
        f"  Model: {effective_model}  |  Duration: {duration}s"
        f"  |  Aspect: {aspect_ratio}"
    )

    with console.status("[cyan]Generating...", spinner="dots"):
        result = await generator.generate_shot(
            shot,
            strategy=strategy_map[strategy],
            aspect_ratio=aspect_ratio,
            extra=extra if extra else None,
        )

    if result.video_data:
        out_path = Path(output_path) if output_path else Path(f"output_{shot.shot_id}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(result.video_data)
        console.print(f"[bold green]Video saved:[/bold green] {out_path}")
        console.print(f"  Cost: ${result.cost_usd:.4f}  |  Model: {result.model_id}")
        out.set_result({
            "path": str(out_path.resolve()),
            "model_id": result.model_id,
            "cost_usd": result.cost_usd,
            "duration_seconds": duration,
        })
    else:
        console.print("[red]Video generation returned no data.[/red]")
        out.set_error("Video generation returned no data.")

    out.emit()


# ---------------------------------------------------------------------------
# claw image
# ---------------------------------------------------------------------------

@app.command()
def image(
    prompt: Annotated[
        str,
        typer.Argument(
            help="Generation prompt for the image.",
            callback=validate_prompt,
        ),
    ],
    provider: Annotated[
        str,
        typer.Option(
            "--provider", "-p",
            help="Image provider: gemini / evolink / byteplus.",
        ),
    ] = "gemini",
    size: Annotated[
        str,
        typer.Option(
            "--size", "-s",
            help="Image aspect ratio (e.g. 3:4, 1:1, 16:9).",
        ),
    ] = "3:4",
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Output file path.")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging.")
    ] = False,
) -> None:
    """Generate a single image from a text prompt.

    \b
    Examples:
      claw image "a beautiful sunset" -o sunset.png
      claw image "character portrait" --provider gemini --size 3:4
      claw image "product photo" --provider evolink -o product.png
    """
    configure_logging(verbose)
    out = get_output()
    out._command = "image"

    try:
        asyncio.run(_image_async(
            prompt=prompt,
            provider=provider,
            size=size,
            output_path=output,
        ))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)


async def _image_async(
    *, prompt: str, provider: str, size: str, output_path: str | None,
) -> None:
    console = get_console()
    out = get_output()

    console.print(f"[cyan]Generating image:[/cyan] {prompt[:80]}")
    console.print(f"  Provider: {provider}  |  Size: {size}")

    with console.status("[cyan]Generating...", spinner="dots"):
        if provider == "gemini":
            from videoclaw.generation.gemini_image import GeminiImageGenerator
            gen = GeminiImageGenerator()
            image_path = await gen.generate(prompt=prompt, aspect_ratio=size, output_dir=Path("."))
        elif provider == "evolink":
            from videoclaw.generation.evolink_image import EvolinkImageGenerator
            gen = EvolinkImageGenerator()
            image_path = await gen.generate(prompt=prompt, aspect_ratio=size, output_dir=Path("."))
        elif provider == "byteplus":
            from videoclaw.generation.byteplus_image import BytePlusImageGenerator
            gen = BytePlusImageGenerator()
            image_path = await gen.generate(prompt=prompt, aspect_ratio=size, output_dir=Path("."))
        else:
            console.print(
                f"[red]Unknown provider {provider!r}."
                " Valid: gemini, evolink, byteplus[/red]"
            )
            out.set_error(f"Unknown provider: {provider}")
            out.emit()
            raise typer.Exit(code=1)

    if output_path and image_path:
        import shutil
        shutil.move(str(image_path), output_path)
        image_path = Path(output_path)

    if image_path and image_path.exists():
        console.print(f"[bold green]Image saved:[/bold green] {image_path}")
        out.set_result({
            "path": str(image_path.resolve()),
            "provider": provider,
            "size": size,
        })
    else:
        console.print("[red]Image generation failed.[/red]")
        out.set_error("Image generation failed.")

    out.emit()


# ---------------------------------------------------------------------------
# claw tts
# ---------------------------------------------------------------------------

@app.command()
def tts(
    text: Annotated[
        str | None,
        typer.Argument(help="Text to synthesize (omit to read from stdin)."),
    ] = None,
    voice: Annotated[
        str | None, typer.Option("--voice", help="TTS voice ID.")
    ] = None,
    language: Annotated[
        str,
        typer.Option(
            "--lang", "-l",
            help="Language (zh/en).",
            callback=validate_language,
        ),
    ] = "zh",
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Output audio file path."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Synthesize speech from text.

    \b
    If no text argument is given, reads from stdin (useful for piping).

    \b
    Examples:
      claw tts "Hello world" -o hello.mp3
      claw tts "Welcome to our show" --voice en-US-AriaNeural --lang en
      echo "pipe this text" | claw tts -o output.mp3
    """
    configure_logging(verbose)
    console = get_console()
    out = get_output()
    out._command = "tts"

    # Read from stdin if no text argument
    effective_text = text
    if not effective_text:
        if not sys.stdin.isatty():
            effective_text = sys.stdin.read().strip()
        if not effective_text:
            console.print("[red]No text provided. Pass text as argument or pipe via stdin.[/red]")
            out.set_error("No text provided.")
            out.emit()
            raise typer.Exit(code=1)

    try:
        asyncio.run(_tts_async(
            text=effective_text,
            voice=voice,
            language=language,
            output_path=output,
        ))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)


async def _tts_async(
    *, text: str, voice: str | None, language: str, output_path: str | None,
) -> None:
    console = get_console()
    out = get_output()

    from videoclaw.generation.audio.tts import TTSManager

    tts_mgr = TTSManager()
    out_path = Path(output_path) if output_path else Path("tts_output.mp3")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Synthesizing TTS:[/cyan] {text[:60]}{'...' if len(text) > 60 else ''}")
    console.print(f"  Language: {language}  |  Voice: {voice or 'default'}")

    with console.status("[cyan]Synthesizing...", spinner="dots"):
        result_path = await tts_mgr.generate_voiceover(
            text, out_path, voice=voice, language=language,
        )

    console.print(f"[bold green]Audio saved:[/bold green] {result_path}")
    out.set_result({
        "path": str(result_path.resolve()),
        "language": language,
        "voice": voice,
        "text_length": len(text),
    })
    out.emit()


# ---------------------------------------------------------------------------
# claw storyboard
# ---------------------------------------------------------------------------

@app.command()
def storyboard(
    prompt: Annotated[
        str,
        typer.Argument(
            help="Creative prompt or script text to decompose.",
            callback=validate_prompt,
        ),
    ],
    duration: Annotated[
        float,
        typer.Option("--duration", "-d", help="Target total duration in seconds."),
    ] = 60.0,
    style: Annotated[
        str, typer.Option("--style", "-s", help="Visual style hint.")
    ] = "cinematic",
    aspect_ratio: Annotated[
        str,
        typer.Option(
            "--aspect-ratio", "-a",
            help="Aspect ratio.",
            callback=validate_aspect_ratio,
        ),
    ] = "16:9",
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Output JSON file for shots."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Decompose a prompt or script into a shot-by-shot storyboard.

    \b
    Uses LLM to plan individual shots with descriptions, durations, and
    visual prompts. Outputs a storyboard (shot list) that can be used
    as input for subsequent video generation.

    \b
    Examples:
      claw storyboard "A man walks into a bar" -d 30
      claw storyboard "Product unboxing sequence" -a 9:16 -o shots.json
    """
    configure_logging(verbose)
    out = get_output()
    out._command = "storyboard"

    try:
        asyncio.run(_storyboard_async(
            prompt=prompt,
            duration=duration,
            style=style,
            aspect_ratio=aspect_ratio,
            output_path=output,
        ))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)


async def _storyboard_async(
    *, prompt: str, duration: float, style: str, aspect_ratio: str, output_path: str | None,
) -> None:
    from rich.table import Table

    from videoclaw.core.director import Director
    from videoclaw.core.state import StateManager

    console = get_console()
    out = get_output()
    cfg = get_config()

    sm = StateManager()
    state = sm.create_project(prompt)
    state.metadata["aspect_ratio"] = aspect_ratio
    state.metadata["style"] = style

    console.print(f"[cyan]Planning storyboard:[/cyan] {prompt[:80]}")
    console.print(f"  Duration: {duration}s  |  Style: {style}  |  Aspect: {aspect_ratio}")

    with console.status("[cyan]Director is planning shots...", spinner="dots"):
        director = Director()
        state = await director.plan(
            state,
            duration=int(duration),
            style=style,
            aspect_ratio=aspect_ratio,
            preferred_model=cfg.default_video_model,
        )

    # Display shot table
    table = Table(title="Storyboard", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Shot ID", style="cyan", min_width=14)
    table.add_column("Description", style="white")
    table.add_column("Model", style="magenta")
    table.add_column("Duration", justify="right", style="green")

    shots_data = []
    for idx, shot in enumerate(state.storyboard, 1):
        table.add_row(
            str(idx),
            shot.shot_id,
            shot.description[:60] + ("..." if len(shot.description) > 60 else ""),
            shot.model_id,
            f"{shot.duration_seconds:.1f}s",
        )
        shots_data.append({
            "shot_id": shot.shot_id,
            "description": shot.description,
            "prompt": shot.prompt,
            "model_id": shot.model_id,
            "duration_seconds": shot.duration_seconds,
        })

    console.print(table)
    total_dur = sum(s.duration_seconds for s in state.storyboard)
    console.print(f"\n[bold]Total:[/bold] {len(state.storyboard)} shots, {total_dur:.1f}s")

    # Optionally save to JSON
    if output_path:
        out_path = Path(output_path)
        out_path.write_text(
            json.dumps({"prompt": prompt, "shots": shots_data}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"[bold green]Storyboard saved:[/bold green] {out_path}")

    out.set_result({
        "project_id": state.project_id,
        "shots": shots_data,
        "total_duration": total_dur,
    })
    out.emit()


# ---------------------------------------------------------------------------
# claw compose
# ---------------------------------------------------------------------------

@app.command()
def compose(
    videos: Annotated[list[str], typer.Argument(help="Video files to compose together.")],
    transition: Annotated[
        str,
        typer.Option(
            "--transition", "-t",
            help="Transition type: dissolve / cut / fade.",
        ),
    ] = "dissolve",
    transition_duration: Annotated[
        float,
        typer.Option(
            "--transition-duration", help="Transition duration in seconds."
        ),
    ] = 0.5,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output file path.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Compose multiple video clips into a single video.

    \b
    Examples:
      claw compose shot1.mp4 shot2.mp4 shot3.mp4 -o final.mp4
      claw compose clip_*.mp4 --transition fade -o composed.mp4
    """
    configure_logging(verbose)
    console = get_console()
    out = get_output()
    out._command = "compose"

    # Validate input files
    video_paths: list[Path] = []
    for v in videos:
        p = Path(v)
        if not p.exists():
            console.print(f"[red]File not found: {v}[/red]")
            out.set_error(f"File not found: {v}")
            out.emit()
            raise typer.Exit(code=1)
        video_paths.append(p)

    if len(video_paths) < 2:
        console.print("[red]At least 2 video files are required for composition.[/red]")
        out.set_error("At least 2 video files required.")
        out.emit()
        raise typer.Exit(code=1)

    try:
        asyncio.run(_compose_async(
            video_paths=video_paths,
            transition=transition,
            transition_duration=transition_duration,
            output_path=output,
        ))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)


async def _compose_async(
    *,
    video_paths: list[Path],
    transition: str,
    transition_duration: float,
    output_path: str | None,
) -> None:
    console = get_console()
    out = get_output()

    from videoclaw.generation.compose import VideoComposer

    composer = VideoComposer()
    out_path = Path(output_path) if output_path else Path("composed.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Composing {len(video_paths)} clips:[/cyan]")
    for p in video_paths:
        console.print(f"  {p}")
    console.print(f"  Transition: {transition}")

    with console.status("[cyan]Composing...", spinner="dots"):
        await composer.compose(
            video_paths, out_path,
            transition=transition, transition_duration=transition_duration,
        )

    console.print(f"[bold green]Composed video saved:[/bold green] {out_path}")
    out.set_result({
        "path": str(out_path.resolve()),
        "input_count": len(video_paths),
        "transition": transition,
    })
    out.emit()


# ---------------------------------------------------------------------------
# claw render
# ---------------------------------------------------------------------------

@app.command()
def render(
    input_file: Annotated[str, typer.Argument(help="Input video file to render.")],
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output file path.")] = None,
    resolution: Annotated[
        str | None,
        typer.Option(
            "--resolution", "-r",
            help="Output resolution (e.g. 1920x1080, 1080p,"
            " or aspect ratio like 9:16).",
        ),
    ] = None,
    codec: Annotated[
        str, typer.Option("--codec", help="Video codec.")
    ] = "libx264",
    bitrate: Annotated[
        str, typer.Option("--bitrate", help="Video bitrate.")
    ] = "8M",
    audio_bitrate: Annotated[
        str, typer.Option("--audio-bitrate", help="Audio bitrate.")
    ] = "192k",
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help="Encoding preset (ultrafast/fast/medium/slow).",
        ),
    ] = "medium",
    crf: Annotated[
        int,
        typer.Option(
            "--crf",
            help="Constant rate factor (0-51, lower = better quality).",
        ),
    ] = 23,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Render/encode a video file with specified parameters.

    \b
    Examples:
      claw render input.mp4 -o output.mp4 -r 1080p
      claw render raw.mp4 --codec libx265 --crf 18 -o final.mp4
      claw render composed.mp4 -r 9:16 --bitrate 12M -o tiktok.mp4
    """
    configure_logging(verbose)
    console = get_console()
    out = get_output()
    out._command = "render"

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        out.set_error(f"File not found: {input_file}")
        out.emit()
        raise typer.Exit(code=1)

    try:
        asyncio.run(_render_async(
            input_path=input_path,
            output_path=output,
            resolution=resolution,
            codec=codec,
            bitrate=bitrate,
            audio_bitrate=audio_bitrate,
            preset=preset,
            crf=crf,
        ))
    except Exception as exc:
        out.set_error(str(exc))
        out.emit()
        raise typer.Exit(code=1)


async def _render_async(
    *,
    input_path: Path,
    output_path: str | None,
    resolution: str | None,
    codec: str,
    bitrate: str,
    audio_bitrate: str,
    preset: str,
    crf: int,
) -> None:
    console = get_console()
    out = get_output()

    from videoclaw.generation.render import _ASPECT_TO_RENDER_RESOLUTION, VideoRenderer

    out_path = (
        Path(output_path) if output_path
        else input_path.with_name(f"{input_path.stem}_rendered.mp4")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse resolution
    res_tuple: tuple[int, int] | None = None
    if resolution:
        # Try aspect ratio mapping first
        if resolution in _ASPECT_TO_RENDER_RESOLUTION:
            res_tuple = _ASPECT_TO_RENDER_RESOLUTION[resolution]
        elif "x" in resolution:
            parts = resolution.split("x")
            res_tuple = (int(parts[0]), int(parts[1]))
        elif resolution.endswith("p"):
            h = int(resolution[:-1])
            res_tuple = (h * 16 // 9, h)  # assume 16:9

    console.print(f"[cyan]Rendering:[/cyan] {input_path}")
    console.print(f"  Codec: {codec}  |  Bitrate: {bitrate}  |  CRF: {crf}")
    if res_tuple:
        console.print(f"  Resolution: {res_tuple[0]}x{res_tuple[1]}")

    renderer = VideoRenderer()
    with console.status("[cyan]Rendering...", spinner="dots"):
        await renderer.render(
            input_path=input_path,
            output_path=out_path,
            resolution=res_tuple,
            bitrate=bitrate,
            audio_bitrate=audio_bitrate,
            codec=codec,
            preset=preset,
            crf=crf,
        )

    console.print(f"[bold green]Rendered video saved:[/bold green] {out_path}")
    out.set_result({
        "path": str(out_path.resolve()),
        "codec": codec,
        "bitrate": bitrate,
        "crf": crf,
        "resolution": list(res_tuple) if res_tuple else None,
    })
    out.emit()


# ---------------------------------------------------------------------------
# claw subtitle
# ---------------------------------------------------------------------------

@app.command()
def subtitle(
    input_file: Annotated[str, typer.Argument(help="Scene data JSON file.")],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Subtitle format: srt / ass."),
    ] = "srt",
    language: Annotated[
        str,
        typer.Option(
            "--lang", "-l",
            help="Language (zh/en).",
            callback=validate_language,
        ),
    ] = "zh",
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Output subtitle file path."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Generate subtitles from scene data JSON.

    \b
    The input JSON should contain a list of scene objects with fields:
    scene_id, dialogue, narration, duration_seconds, characters_present.

    \b
    Examples:
      claw subtitle scenes.json -f srt -o output.srt
      claw subtitle scenes.json --format ass --lang en -o output.ass
    """
    configure_logging(verbose)
    console = get_console()
    out = get_output()
    out._command = "subtitle"

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        out.set_error(f"File not found: {input_file}")
        out.emit()
        raise typer.Exit(code=1)

    # Guard against excessively large JSON files (DoS prevention)
    file_size = input_path.stat().st_size
    if file_size > _MAX_JSON_FILE_SIZE:
        size_mb = file_size / 1024 / 1024
        console.print(
            f"[red]JSON file too large: {size_mb:.0f} MB (max 100 MB)[/red]"
        )
        out.set_error(f"JSON file too large: {file_size} bytes")
        out.emit()
        raise typer.Exit(code=1)

    try:
        scenes = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(scenes, dict) and "scenes" in scenes:
            scenes = scenes["scenes"]
    except (json.JSONDecodeError, KeyError) as exc:
        console.print(f"[red]Invalid JSON: {exc}[/red]")
        out.set_error(f"Invalid JSON: {exc}")
        out.emit()
        raise typer.Exit(code=1)

    suffix = ".ass" if format == "ass" else ".srt"
    out_path = Path(output) if output else input_path.with_suffix(suffix)

    from videoclaw.generation.subtitle import SubtitleGenerator

    sub_gen = SubtitleGenerator()

    console.print(f"[cyan]Generating {format.upper()} subtitles:[/cyan] {len(scenes)} scenes")

    if format == "ass":
        sub_gen.generate_ass(scenes, out_path, language=language)
    else:
        sub_gen.generate_srt(scenes, out_path, language=language)

    console.print(f"[bold green]Subtitles saved:[/bold green] {out_path}")
    out.set_result({
        "path": str(out_path.resolve()),
        "format": format,
        "scenes": len(scenes),
        "language": language,
    })
    out.emit()
