"""Retry: generate TTS audio + scene images for Wind-Chaser smoke test."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%H:%M:%S")

DELIVERABLES = _PROJECT_ROOT / "docs" / "deliverables" / "wind_chaser_satan_in_a_suit"


async def generate_tts():
    """Generate TTS audio for all 9 speaking scenes."""
    import edge_tts

    script_path = DELIVERABLES / "ep01_the_devils_pool_script.json"
    script_data = json.loads(script_path.read_text())
    audio_dir = DELIVERABLES / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Voice map
    voice_map = {
        "Ivy Angel": "en-US-AriaNeural",
        "Colton Black": "en-US-DavisNeural",
        "Chloe Green": "en-GB-RyanNeural",
    }

    results = []
    for scene in script_data["scenes"]:
        text = scene.get("dialogue", "") or scene.get("narration", "")
        if not text:
            continue

        scene_id = scene["scene_id"]
        speaker = scene.get("speaking_character", "")
        voice_id = voice_map.get(speaker, "en-US-JennyNeural")
        output_file = audio_dir / f"dialogue_{scene_id}.mp3"

        try:
            communicate = edge_tts.Communicate(text, voice_id)
            await communicate.save(str(output_file))
            size = output_file.stat().st_size if output_file.exists() else 0
            status = "OK" if size > 0 else "FAIL"
            results.append((scene_id, speaker, voice_id, status, size))
            print(f"  {status:4s}  {scene_id:12s}  {speaker:15s}  {voice_id:25s}  {size:>8d} bytes")
        except Exception as e:
            results.append((scene_id, speaker, voice_id, "FAIL", 0))
            print(f"  FAIL  {scene_id:12s}  {speaker:15s}  {voice_id:25s}  error: {e}")

    ok = sum(1 for r in results if r[3] == "OK")
    print(f"\n  TTS Result: {ok}/{len(results)} files generated")
    return results


async def generate_scene_images():
    """Generate 3 curated scene reference images (not per-visual_prompt)."""
    from videoclaw.generation.image import EvolinkImageGenerator

    scene_dir = Path("projects/dramas/wind_chaser_ranch_001/scenes")
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Use the curated scene specs instead of auto-extraction
    specs_path = DELIVERABLES / "images" / "scenes" / "scene_reference_specs.json"
    specs = json.loads(specs_path.read_text())

    try:
        gen = EvolinkImageGenerator()
    except Exception as e:
        print(f"  FAIL  Cannot initialize Evolink: {e}")
        return []

    results = []
    for loc in specs["locations"]:
        filename = loc["file"]
        prompt = loc["prompt"]
        output_path = scene_dir / filename

        try:
            print(f"  Generating: {loc['name']}...")
            path = await gen.generate(
                prompt,
                output_dir=scene_dir,
                filename=filename,
                size="16:9",
            )
            exists = path.exists() and path.stat().st_size > 0
            status = "OK" if exists else "FAIL"
            results.append((loc["name"], filename, status, path.stat().st_size if exists else 0))
            print(f"  {status:4s}  {loc['name']:30s}  {path.stat().st_size if exists else 0:>8d} bytes")

            # Also copy to deliverables
            del_path = DELIVERABLES / "images" / "scenes" / filename
            if exists:
                import shutil
                shutil.copy2(str(path), str(del_path))

        except Exception as e:
            results.append((loc["name"], filename, "FAIL", 0))
            print(f"  FAIL  {loc['name']:30s}  error: {e}")

    ok = sum(1 for r in results if r[2] == "OK")
    print(f"\n  Scene Images Result: {ok}/{len(results)} files generated")
    return results


async def main():
    print("=" * 60)
    print("  RETRY: TTS Audio + Scene Images")
    print("=" * 60)

    print(f"\n--- TTS Audio (EdgeTTS) ---")
    print(f"  Note: Production should use WaveSpeed API (VIDEOCLAW_WAVESPEED_API_KEY)")
    print(f"  Current fallback: EdgeTTS (free)\n")
    await generate_tts()

    print(f"\n--- Scene Reference Images (Evolink Seedream 5.0) ---\n")
    await generate_scene_images()


if __name__ == "__main__":
    asyncio.run(main())
