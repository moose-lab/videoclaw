"""``claw drama export`` — export structured deliverables for human review.

Collects all intermediate and final assets from a drama series into a
single structured directory under ``docs/deliverables/``, organized by
pipeline stage for easy manual inspection.

Output structure::

    docs/deliverables/{series_slug}/
    ├── 00_metadata/
    │   ├── series.json
    │   └── state.json
    ├── 01_script/
    │   └── ep{N}_scenes.json
    ├── 02_characters/
    │   ├── {name}_turnaround.png
    │   └── characters.json
    ├── 03_scenes/
    │   ├── scene_{location}.png
    │   ├── prop_{name}.png
    │   └── assets.json
    ├── 04_prompts/
    │   └── ep{N}_prompts.json
    ├── 05_video_clips/
    │   ├── ep{N}_{scene_id}.mp4
    │   └── manifest.json
    ├── 06_audio/
    │   ├── ep{N}_{scene_id}_dialogue.wav
    │   └── manifest.json
    ├── 07_subtitles/
    │   └── ep{N}_subtitles.{ass,srt}
    ├── 08_composition/
    │   └── ep{N}_composed_final.mp4
    ├── 09_final/
    │   └── ep{N}_final.mp4
    └── 10_audit/
        └── ep{N}_audit_report.json
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Annotated

import typer

from videoclaw.cli._app import (
    configure_logging,
    drama_app,
    show_banner,
)
from videoclaw.cli._output import get_console, get_output


def _slugify(text: str) -> str:
    """Convert a title to a filesystem-safe slug."""
    text = re.sub(r"[^\w\s-]", "", text.strip().lower())
    return re.sub(r"[\s_]+", "_", text).strip("_") or "untitled"


def _copy_if_exists(src: str | Path | None, dst: Path) -> bool:
    """Copy a file if the source exists. Returns True on success."""
    if not src:
        return False
    src_path = Path(src)
    if not src_path.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return True


@drama_app.command("export")
def drama_export(
    series_id: Annotated[
        str, typer.Argument(help="Drama series ID.")
    ],
    episode: Annotated[
        int,
        typer.Option("--episode", "-e", help="Episode number (default: all)."),
    ] = 0,
    output_dir: Annotated[
        str,
        typer.Option(
            "--output", "-o",
            help="Output directory (default: docs/deliverables/{title}).",
        ),
    ] = "",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v")
    ] = False,
) -> None:
    """Export all intermediate assets to a structured deliverables directory.

    \b
    Collects metadata, scripts, character/scene images, enhanced prompts,
    video clips, audio, subtitles, compositions, final video, and audit
    reports into a single directory for human review.

    \b
    Examples:
        claw drama export 97e8424712d24fb2
        claw drama export abc123 -e 1 -o ./review/
    """
    configure_logging(verbose)
    show_banner()
    console = get_console()
    out = get_output()
    out._command = "drama.export"

    from videoclaw.config import get_config
    from videoclaw.core.state import StateManager
    from videoclaw.drama.models import DramaManager

    cfg = get_config()
    mgr = DramaManager()

    try:
        series = mgr.load(series_id)
    except FileNotFoundError:
        console.print(f"[red]Series {series_id!r} not found.[/red]")
        out.set_error(f"Series {series_id!r} not found.")
        out.emit()
        raise typer.Exit(code=1)

    # Determine output directory
    if output_dir:
        base = Path(output_dir)
    else:
        slug = _slugify(series.title) if series.title else series_id[:12]
        base = Path("docs/deliverables") / slug

    console.print(
        f"[bold cyan]Exporting deliverables:[/bold cyan] "
        f"{series.title or series_id} → {base}"
    )

    # Select episodes
    if episode > 0:
        episodes = [ep for ep in series.episodes if ep.number == episode]
        if not episodes:
            console.print(f"[red]Episode {episode} not found.[/red]")
            raise typer.Exit(code=1)
    else:
        episodes = series.episodes

    stats: dict[str, int] = {}

    # ── 00_metadata ─────────────────────────────────────────────
    meta_dir = base / "00_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    series_json = meta_dir / "series.json"
    series_json.write_text(
        json.dumps(series.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    stats["metadata"] = 1
    console.print("  [green]00_metadata/series.json[/green]")

    # Export project state for each episode
    state_mgr = StateManager(projects_dir=cfg.projects_dir)
    for ep in episodes:
        if ep.project_id:
            try:
                state = state_mgr.load(ep.project_id)
                state_path = meta_dir / f"ep{ep.number:02d}_state.json"
                state_path.write_text(
                    json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                stats["metadata"] = stats.get("metadata", 0) + 1
                console.print(
                    f"  [green]00_metadata/ep{ep.number:02d}_state.json[/green]"
                )
            except FileNotFoundError:
                pass

    # ── 01_script ───────────────────────────────────────────────
    script_dir = base / "01_script"
    for ep in episodes:
        if not ep.scenes:
            continue
        scenes_data = [sc.to_dict() for sc in ep.scenes]
        out_path = script_dir / f"ep{ep.number:02d}_scenes.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(scenes_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        stats["script"] = stats.get("script", 0) + 1
        console.print(
            f"  [green]01_script/ep{ep.number:02d}_scenes.json[/green] "
            f"({len(ep.scenes)} scenes)"
        )

    # ── 02_characters ───────────────────────────────────────────
    char_dir = base / "02_characters"
    char_count = 0
    char_data = []
    for c in series.characters:
        entry = {
            "name": c.name,
            "description": c.description,
            "visual_prompt": c.visual_prompt,
            "voice_style": c.voice_style,
            "reference_image_url": c.reference_image_url,
        }
        if c.voice_profile:
            entry["voice_profile"] = c.voice_profile.to_dict()

        # Copy turnaround image
        if c.reference_image:
            safe = _slugify(c.name)
            dst = char_dir / f"{safe}_turnaround.png"
            if _copy_if_exists(c.reference_image, dst):
                entry["exported_image"] = str(dst.relative_to(base))
                char_count += 1

        char_data.append(entry)

    if char_data:
        char_dir.mkdir(parents=True, exist_ok=True)
        (char_dir / "characters.json").write_text(
            json.dumps(char_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        stats["characters"] = char_count
        console.print(
            f"  [green]02_characters/[/green] "
            f"{len(char_data)} characters, {char_count} images"
        )

    # ── 03_scenes ───────────────────────────────────────────────
    scene_dir = base / "03_scenes"
    scene_count = 0
    scene_assets = {"locations": [], "props": []}

    # Scene/location images from consistency manifest
    if series.consistency_manifest:
        for loc_key, img_path in (
            series.consistency_manifest.scene_references.items()
        ):
            dst = scene_dir / f"scene_{_slugify(loc_key)}.png"
            if _copy_if_exists(img_path, dst):
                scene_count += 1
                scene_assets["locations"].append({
                    "key": loc_key,
                    "exported_image": str(dst.relative_to(base)),
                })

        for prop_name, img_path in (
            series.consistency_manifest.prop_references.items()
        ):
            dst = scene_dir / f"prop_{_slugify(prop_name)}.png"
            if _copy_if_exists(img_path, dst):
                scene_count += 1
                scene_assets["props"].append({
                    "name": prop_name,
                    "exported_image": str(dst.relative_to(base)),
                })

    # Also check metadata for locations/props
    for loc in series.metadata.get("locations", []):
        img = loc.get("reference_image")
        if img and not any(
            a["key"] == loc.get("name", "") for a in scene_assets["locations"]
        ):
            dst = scene_dir / f"scene_{_slugify(loc.get('name', 'unknown'))}.png"
            if _copy_if_exists(img, dst):
                scene_count += 1
                scene_assets["locations"].append({
                    "key": loc.get("name", ""),
                    "description": loc.get("description", ""),
                    "exported_image": str(dst.relative_to(base)),
                })

    for prop in series.metadata.get("props", []):
        img = prop.get("reference_image")
        if img and not any(
            a["name"] == prop.get("name", "") for a in scene_assets["props"]
        ):
            dst = scene_dir / f"prop_{_slugify(prop.get('name', 'unknown'))}.png"
            if _copy_if_exists(img, dst):
                scene_count += 1
                scene_assets["props"].append({
                    "name": prop.get("name", ""),
                    "exported_image": str(dst.relative_to(base)),
                })

    if scene_count > 0:
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "assets.json").write_text(
            json.dumps(scene_assets, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        stats["scenes"] = scene_count
        console.print(
            f"  [green]03_scenes/[/green] {scene_count} images"
        )

    # ── 04_prompts ──────────────────────────────────────────────
    prompt_dir = base / "04_prompts"
    for ep in episodes:
        if not ep.scenes:
            continue
        prompts = []
        for sc in ep.scenes:
            prompts.append({
                "scene_id": sc.scene_id,
                "duration": sc.duration_seconds,
                "shot_scale": sc.shot_scale.value if sc.shot_scale else "",
                "camera": sc.camera_movement,
                "characters": sc.characters_present,
                "dialogue": sc.dialogue,
                "narration": sc.narration,
                "original_prompt": sc.description,
                "enhanced_prompt": sc.effective_prompt,
            })
        out_path = prompt_dir / f"ep{ep.number:02d}_prompts.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(prompts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        stats["prompts"] = stats.get("prompts", 0) + len(prompts)
        console.print(
            f"  [green]04_prompts/ep{ep.number:02d}_prompts.json[/green] "
            f"({len(prompts)} scenes)"
        )

    # ── 05–09: Execution assets (per episode) ───────────────────
    for ep in episodes:
        if not ep.project_id:
            continue
        proj_dir = cfg.projects_dir / ep.project_id
        if not proj_dir.exists():
            continue

        ep_tag = f"ep{ep.number:02d}"

        # 05_video_clips
        clips_dir = base / "05_video_clips"
        shots_dir = proj_dir / "shots"
        clip_manifest = []
        if shots_dir.exists():
            for clip in sorted(shots_dir.glob("*.mp4")):
                # Rename to friendly name: ep01_scene_id.mp4
                # Extract scene_id from filename pattern session*_{scene_id}_{hash}.mp4
                parts = clip.stem.split("_")
                # Try to find scene_id in parts
                scene_id = "_".join(parts[1:-1]) if len(parts) >= 3 else clip.stem
                dst = clips_dir / f"{ep_tag}_{scene_id}.mp4"
                if _copy_if_exists(clip, dst):
                    clip_manifest.append({
                        "scene_id": scene_id,
                        "original": clip.name,
                        "exported": str(dst.relative_to(base)),
                        "size_bytes": clip.stat().st_size,
                    })

            if clip_manifest:
                clips_dir.mkdir(parents=True, exist_ok=True)
                (clips_dir / "manifest.json").write_text(
                    json.dumps(clip_manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                stats["clips"] = stats.get("clips", 0) + len(clip_manifest)
                console.print(
                    f"  [green]05_video_clips/[/green] "
                    f"{len(clip_manifest)} clips ({ep_tag})"
                )

        # 06_audio
        audio_src = proj_dir / "audio"
        if audio_src.exists():
            audio_dir = base / "06_audio"
            audio_manifest = []
            for af in sorted(audio_src.glob("*")):
                if af.is_file() and af.suffix in (".wav", ".mp3", ".aac"):
                    dst = audio_dir / f"{ep_tag}_{af.name}"
                    if _copy_if_exists(af, dst):
                        audio_manifest.append({
                            "original": af.name,
                            "exported": str(dst.relative_to(base)),
                            "size_bytes": af.stat().st_size,
                        })

            if audio_manifest:
                audio_dir.mkdir(parents=True, exist_ok=True)
                (audio_dir / "manifest.json").write_text(
                    json.dumps(audio_manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                stats["audio"] = stats.get("audio", 0) + len(audio_manifest)
                console.print(
                    f"  [green]06_audio/[/green] "
                    f"{len(audio_manifest)} files ({ep_tag})"
                )

        # 07_subtitles
        for ext in ("ass", "srt", "vtt"):
            sub_src = proj_dir / f"subtitles.{ext}"
            if sub_src.exists():
                sub_dir = base / "07_subtitles"
                dst = sub_dir / f"{ep_tag}_subtitles.{ext}"
                if _copy_if_exists(sub_src, dst):
                    stats["subtitles"] = stats.get("subtitles", 0) + 1
                    console.print(
                        f"  [green]07_subtitles/{ep_tag}_subtitles.{ext}[/green]"
                    )

        # 08_composition
        for comp_name in ("composed.mp4", "composed_final.mp4"):
            comp_src = proj_dir / comp_name
            if comp_src.exists():
                comp_dir = base / "08_composition"
                dst = comp_dir / f"{ep_tag}_{comp_name}"
                if _copy_if_exists(comp_src, dst):
                    stats["composition"] = stats.get("composition", 0) + 1
                    console.print(
                        f"  [green]08_composition/{ep_tag}_{comp_name}[/green]"
                    )

        # 09_final
        final_src = proj_dir / "final.mp4"
        if final_src.exists():
            final_dir = base / "09_final"
            dst = final_dir / f"{ep_tag}_final.mp4"
            if _copy_if_exists(final_src, dst):
                stats["final"] = stats.get("final", 0) + 1
                console.print(
                    f"  [green]09_final/{ep_tag}_final.mp4[/green]"
                )

    # ── 10_audit ────────────────────────────────────────────────
    audit_dir = base / "10_audit"
    for ep in episodes:
        if not ep.scenes:
            continue
        audit_data = []
        has_audit = False
        for sc in ep.scenes:
            if sc.audit_result:
                has_audit = True
                audit_data.append({
                    "scene_id": sc.scene_id,
                    **sc.audit_result,
                })
            else:
                audit_data.append({
                    "scene_id": sc.scene_id,
                    "status": "not_audited",
                })

        if has_audit:
            out_path = audit_dir / f"ep{ep.number:02d}_audit_report.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(audit_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            stats["audit"] = stats.get("audit", 0) + len(audit_data)
            console.print(
                f"  [green]10_audit/ep{ep.number:02d}_audit_report.json[/green]"
            )

    # ── Summary ─────────────────────────────────────────────────
    total = sum(stats.values())
    console.print(
        f"\n[bold green]Export complete:[/bold green] "
        f"{total} assets → {base}"
    )
    for category, count in sorted(stats.items()):
        console.print(f"  {category}: {count}")

    out.set_result({
        "series_id": series_id,
        "output_dir": str(base),
        "stats": stats,
        "total_assets": total,
    })
    out.emit()
