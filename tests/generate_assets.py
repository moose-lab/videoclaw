"""Generate all required assets for Wind-Chaser smoke test.

Executes Phase 1 (images) and Phase 2 (TTS) in sequence, then
attempts Phase 3 (video) and Phase 4 (subtitle + compose + render).
Reports results per asset.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_assets")

DELIVERABLES = _PROJECT_ROOT / "docs" / "deliverables" / "wind_chaser_satan_in_a_suit"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class AssetResult:
    name: str
    category: str  # character_image, scene_image, tts, subtitle, video, compose
    model: str = ""
    mode: str = ""
    status: str = "pending"  # success, failed, skipped
    path: str = ""
    error: str = ""
    duration_sec: float = 0.0


results: list[AssetResult] = []


# ---------------------------------------------------------------------------
# Phase 1: Character reference images
# ---------------------------------------------------------------------------

async def generate_character_images(series) -> list[AssetResult]:
    """Generate character reference images via Evolink Seedream 5.0."""
    from videoclaw.drama.character_designer import CharacterDesigner
    from videoclaw.drama.models import DramaManager

    phase_results = []
    mgr = DramaManager()
    mgr.save(series)

    try:
        designer = CharacterDesigner(drama_manager=mgr)
        series_out = await designer.design_characters(series, force=True)

        for char in series_out.characters:
            for i, img_path in enumerate(char.reference_images):
                poses = ["front", "three_quarter", "full_body"]
                pose = poses[i] if i < len(poses) else f"pose_{i}"
                exists = Path(img_path).exists() if img_path else False
                phase_results.append(AssetResult(
                    name=f"{char.name} ({pose})",
                    category="character_image",
                    model="doubao-seedream-5.0-lite",
                    mode="Evolink API → multi-angle reference",
                    status="success" if exists else "failed",
                    path=img_path or "",
                    error="" if exists else "file not found after generation",
                ))

            if char.reference_image:
                # primary = front view, already counted above
                pass

        # Update series with new paths
        series.characters = series_out.characters
        mgr.save(series)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Character image generation failed: %s", e)
        for char in series.characters:
            for pose in ["front", "three_quarter", "full_body"]:
                phase_results.append(AssetResult(
                    name=f"{char.name} ({pose})",
                    category="character_image",
                    model="doubao-seedream-5.0-lite",
                    mode="Evolink API → multi-angle reference",
                    status="failed",
                    error=str(e),
                ))

    return phase_results


# ---------------------------------------------------------------------------
# Phase 1b: Scene reference images
# ---------------------------------------------------------------------------

async def generate_scene_images(series) -> list[AssetResult]:
    """Generate scene reference images via Evolink Seedream 5.0."""
    from videoclaw.drama.scene_designer import SceneDesigner
    from videoclaw.drama.models import DramaManager

    phase_results = []
    mgr = DramaManager()

    try:
        designer = SceneDesigner(drama_manager=mgr)
        locations = await designer.design_scenes(series, force=True)

        for loc in locations:
            exists = Path(loc.reference_image).exists() if loc.reference_image else False
            phase_results.append(AssetResult(
                name=f"scene: {loc.name}",
                category="scene_image",
                model="doubao-seedream-5.0-lite",
                mode="Evolink API → 16:9 establishing shot",
                status="success" if exists else "failed",
                path=loc.reference_image or "",
                error="" if exists else "file not found after generation",
            ))

    except Exception as e:
        logger.error("Scene image generation failed: %s", e)
        specs_path = DELIVERABLES / "images" / "scenes" / "scene_reference_specs.json"
        if specs_path.exists():
            specs = json.loads(specs_path.read_text())
            for loc in specs["locations"]:
                phase_results.append(AssetResult(
                    name=f"scene: {loc['name']}",
                    category="scene_image",
                    model="doubao-seedream-5.0-lite",
                    mode="Evolink API → 16:9 establishing shot",
                    status="failed",
                    error=str(e),
                ))
        else:
            phase_results.append(AssetResult(
                name="scene images",
                category="scene_image",
                model="doubao-seedream-5.0-lite",
                mode="Evolink API",
                status="failed",
                error=str(e),
            ))

    return phase_results


# ---------------------------------------------------------------------------
# Phase 2: TTS audio
# ---------------------------------------------------------------------------

async def generate_tts_audio(series) -> list[AssetResult]:
    """Generate per-scene TTS audio via EdgeTTS (free)."""
    phase_results = []

    script_path = DELIVERABLES / "ep01_the_devils_pool_script.json"
    script_data = json.loads(script_path.read_text())

    ep = series.episodes[0]
    audio_dir = DELIVERABLES / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Build voice map from characters
    voice_map = {}
    for char in series.characters:
        if char.voice_profile:
            voice_map[char.name] = char.voice_profile

    # Try EdgeTTS for each scene with dialogue
    try:
        import edge_tts
    except ImportError:
        for scene in script_data["scenes"]:
            if scene.get("dialogue") or scene.get("narration"):
                phase_results.append(AssetResult(
                    name=f"tts: {scene['scene_id']}",
                    category="tts",
                    model="edge-tts",
                    mode="EdgeTTS Neural",
                    status="failed",
                    error="edge-tts package not installed (pip install edge-tts)",
                ))
        return phase_results

    for scene in script_data["scenes"]:
        text = scene.get("dialogue", "") or scene.get("narration", "")
        if not text:
            continue

        scene_id = scene["scene_id"]
        speaker = scene.get("speaking_character", "")
        line_type = scene.get("dialogue_line_type", "dialogue")

        # Determine voice
        voice_id = "en-US-JennyNeural"  # default
        if speaker and speaker in voice_map:
            vp = voice_map[speaker]
            voice_id = vp.voice_id if hasattr(vp, 'voice_id') else "en-US-JennyNeural"

        output_file = audio_dir / f"dialogue_{scene_id}.mp3"

        try:
            communicate = edge_tts.Communicate(text, voice_id)
            await communicate.save(str(output_file))

            exists = output_file.exists() and output_file.stat().st_size > 0
            phase_results.append(AssetResult(
                name=f"tts: {scene_id} ({speaker or 'narrator'})",
                category="tts",
                model=voice_id,
                mode=f"EdgeTTS Neural / {line_type}",
                status="success" if exists else "failed",
                path=str(output_file),
                error="" if exists else "empty output file",
            ))
        except Exception as e:
            logger.error("TTS failed for %s: %s", scene_id, e)
            phase_results.append(AssetResult(
                name=f"tts: {scene_id} ({speaker or 'narrator'})",
                category="tts",
                model=voice_id,
                mode=f"EdgeTTS Neural / {line_type}",
                status="failed",
                error=str(e),
            ))

    return phase_results


# ---------------------------------------------------------------------------
# Phase 3: Video generation (Seedance 2.0)
# ---------------------------------------------------------------------------

async def generate_video_clips(series) -> list[AssetResult]:
    """Attempt video generation via Seedance 2.0."""
    phase_results = []
    ep = series.episodes[0]

    ark_key = os.environ.get("VIDEOCLAW_ARK_API_KEY") or os.environ.get("ARK_API_KEY")
    if not ark_key:
        for scene in ep.scenes:
            phase_results.append(AssetResult(
                name=f"video: {scene.scene_id}",
                category="video",
                model="doubao-seedance-2-0-260128",
                mode="Seedance 2.0 / image-to-video + Universal Reference",
                status="skipped",
                error="VIDEOCLAW_ARK_API_KEY not set",
            ))
        return phase_results

    # If key is set, attempt generation via DAG runner
    try:
        from videoclaw.drama.runner import DramaRunner
        from videoclaw.drama.models import DramaManager

        mgr = DramaManager()
        runner = DramaRunner(drama_manager=mgr)
        state = await runner.run_episode(series, ep)

        for shot in state.storyboard:
            exists = bool(shot.asset_path) and Path(shot.asset_path).exists()
            phase_results.append(AssetResult(
                name=f"video: {shot.shot_id}",
                category="video",
                model="doubao-seedance-2-0-260128",
                mode="Seedance 2.0 / image-to-video + Universal Reference",
                status="success" if exists else "failed",
                path=shot.asset_path or "",
                error="" if exists else "asset not generated",
            ))

    except Exception as e:
        logger.error("Video generation failed: %s", e)
        for scene in ep.scenes:
            phase_results.append(AssetResult(
                name=f"video: {scene.scene_id}",
                category="video",
                model="doubao-seedance-2-0-260128",
                mode="Seedance 2.0 / image-to-video + Universal Reference",
                status="failed",
                error=str(e),
            ))

    return phase_results


# ---------------------------------------------------------------------------
# Phase 4: Subtitle generation
# ---------------------------------------------------------------------------

async def generate_subtitles(series) -> list[AssetResult]:
    """Generate subtitles from TTS timing data."""
    phase_results = []

    script_path = DELIVERABLES / "ep01_the_devils_pool_script.json"
    script_data = json.loads(script_path.read_text())
    audio_dir = DELIVERABLES / "audio"
    subtitle_path = DELIVERABLES / "subtitles.srt"

    try:
        # Build SRT from scene data + actual audio file durations
        srt_lines = []
        idx = 1
        current_time = 0.0

        for scene in script_data["scenes"]:
            text = scene.get("dialogue", "")
            if not text:
                current_time += scene.get("duration_seconds", 0)
                continue

            speaker = scene.get("speaking_character", "")
            duration = scene.get("duration_seconds", 3.0)

            # Check if we have actual audio to get real duration
            audio_file = audio_dir / f"dialogue_{scene['scene_id']}.mp3"
            if audio_file.exists():
                try:
                    import subprocess
                    result = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries",
                         "format=duration", "-of", "csv=p=0", str(audio_file)],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        audio_dur = float(result.stdout.strip())
                        duration = max(duration, audio_dur)
                except Exception:
                    pass

            start = current_time
            end = current_time + duration

            start_ts = _format_srt_time(start)
            end_ts = _format_srt_time(end)

            display = f"{speaker}: {text}" if speaker else text
            srt_lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{display}\n")
            idx += 1
            current_time = end

        subtitle_path.write_text("\n".join(srt_lines), encoding="utf-8")

        exists = subtitle_path.exists() and subtitle_path.stat().st_size > 0
        phase_results.append(AssetResult(
            name="subtitles.srt",
            category="subtitle",
            model="local SRT generator",
            mode="word-based line break / Arial 22pt / en locale",
            status="success" if exists else "failed",
            path=str(subtitle_path),
            error="" if exists else "empty file",
        ))

    except Exception as e:
        logger.error("Subtitle generation failed: %s", e)
        phase_results.append(AssetResult(
            name="subtitles.srt",
            category="subtitle",
            model="local SRT generator",
            mode="word-based line break / en locale",
            status="failed",
            error=str(e),
        ))

    return phase_results


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    from videoclaw.drama.models import (
        Character, DramaManager, DramaScene, DramaSeries,
        Episode, EpisodeStatus, assign_voice_profile,
    )

    print("=" * 70)
    print("  ASSET GENERATION — Wind-Chaser: Satan in a Suit")
    print("=" * 70)

    # Load series
    chars_path = DELIVERABLES / "executor" / "characters.json"
    script_path = DELIVERABLES / "ep01_the_devils_pool_script.json"
    series_path = DELIVERABLES / "final_series.json"

    series_data = json.loads(series_path.read_text())
    series = DramaSeries.from_dict(series_data)
    chars_data = json.loads(chars_path.read_text())
    series.characters = [Character.from_dict(c) for c in chars_data]
    for char in series.characters:
        assign_voice_profile(char, language="en")

    script_data = json.loads(script_path.read_text())
    scenes = [DramaScene.from_dict(s) for s in script_data["scenes"]]
    ep = Episode(
        episode_id="ep01_the_devils_pool",
        number=1,
        title=script_data["episode_title"],
        synopsis=script_data.get("cliffhanger", ""),
        opening_hook="A server girl sprints through a glittering pool party.",
        duration_seconds=sum(s.duration_seconds for s in scenes),
        script=json.dumps(script_data, ensure_ascii=False),
        scenes=scenes,
        status=EpisodeStatus.PENDING,
    )
    series.episodes = [ep]

    print(f"\n  Series: {series.title}")
    print(f"  Characters: {len(series.characters)}")
    print(f"  Scenes: {len(ep.scenes)}, Duration: {ep.duration_seconds}s")

    global results

    # --- Phase 1: Character images ---
    print(f"\n{'─'*70}")
    print("  Phase 1a: Character Reference Images (Evolink Seedream 5.0)")
    print(f"{'─'*70}")
    t0 = time.time()
    char_results = await generate_character_images(series)
    dt = time.time() - t0
    for r in char_results:
        r.duration_sec = dt / max(len(char_results), 1)
    results.extend(char_results)
    ok = sum(1 for r in char_results if r.status == "success")
    print(f"  Result: {ok}/{len(char_results)} generated ({dt:.1f}s)")

    # --- Phase 1b: Scene images ---
    print(f"\n{'─'*70}")
    print("  Phase 1b: Scene Reference Images (Evolink Seedream 5.0)")
    print(f"{'─'*70}")
    t0 = time.time()
    scene_results = await generate_scene_images(series)
    dt = time.time() - t0
    for r in scene_results:
        r.duration_sec = dt / max(len(scene_results), 1)
    results.extend(scene_results)
    ok = sum(1 for r in scene_results if r.status == "success")
    print(f"  Result: {ok}/{len(scene_results)} generated ({dt:.1f}s)")

    # --- Phase 2: TTS ---
    print(f"\n{'─'*70}")
    print("  Phase 2: Per-Scene TTS Audio (EdgeTTS)")
    print(f"{'─'*70}")
    t0 = time.time()
    tts_results = await generate_tts_audio(series)
    dt = time.time() - t0
    for r in tts_results:
        r.duration_sec = dt / max(len(tts_results), 1)
    results.extend(tts_results)
    ok = sum(1 for r in tts_results if r.status == "success")
    print(f"  Result: {ok}/{len(tts_results)} generated ({dt:.1f}s)")

    # --- Phase 2b: Subtitles ---
    print(f"\n{'─'*70}")
    print("  Phase 2b: Subtitle Generation")
    print(f"{'─'*70}")
    sub_results = await generate_subtitles(series)
    results.extend(sub_results)
    ok = sum(1 for r in sub_results if r.status == "success")
    print(f"  Result: {ok}/{len(sub_results)} generated")

    # --- Phase 3: Video ---
    print(f"\n{'─'*70}")
    print("  Phase 3: Video Clips (Seedance 2.0)")
    print(f"{'─'*70}")
    t0 = time.time()
    video_results = await generate_video_clips(series)
    dt = time.time() - t0
    for r in video_results:
        r.duration_sec = dt / max(len(video_results), 1)
    results.extend(video_results)
    ok = sum(1 for r in video_results if r.status == "success")
    skipped = sum(1 for r in video_results if r.status == "skipped")
    print(f"  Result: {ok}/{len(video_results)} generated, {skipped} skipped ({dt:.1f}s)")

    # --- Final report ---
    print(f"\n{'='*70}")
    print("  FINAL ASSET REPORT")
    print(f"{'='*70}\n")

    # Group by category
    categories = ["character_image", "scene_image", "tts", "subtitle", "video"]
    cat_labels = {
        "character_image": "Character Ref Images",
        "scene_image": "Scene Ref Images",
        "tts": "TTS Audio",
        "subtitle": "Subtitles",
        "video": "Video Clips",
    }

    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue

        ok = sum(1 for r in cat_results if r.status == "success")
        fail = sum(1 for r in cat_results if r.status == "failed")
        skip = sum(1 for r in cat_results if r.status == "skipped")

        print(f"  [{cat_labels[cat]}] {ok} success / {fail} failed / {skip} skipped")
        for r in cat_results:
            icon = {"success": "OK", "failed": "FAIL", "skipped": "SKIP"}[r.status]
            err = f" — {r.error}" if r.error else ""
            print(f"    {icon:4s}  {r.name:40s}  model={r.model}")
            if err:
                print(f"          error: {r.error}")
        print()

    # Summary table
    print(f"{'='*70}")
    print("  ASSET GENERATION SUMMARY TABLE")
    print(f"{'='*70}\n")
    print(f"  {'Asset':<40s} {'Status':<8s} {'Model':<30s} {'Mode'}")
    print(f"  {'─'*40} {'─'*8} {'─'*30} {'─'*40}")
    for r in results:
        print(f"  {r.name:<40s} {r.status:<8s} {r.model:<30s} {r.mode}")

    total_ok = sum(1 for r in results if r.status == "success")
    total_fail = sum(1 for r in results if r.status == "failed")
    total_skip = sum(1 for r in results if r.status == "skipped")
    print(f"\n  Total: {total_ok} success / {total_fail} failed / {total_skip} skipped / {len(results)} total")


if __name__ == "__main__":
    asyncio.run(main())
