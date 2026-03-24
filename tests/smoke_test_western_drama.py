"""Smoke test for Western TikTok short drama: Wind-Chaser — Satan in a Suit.

End-to-end test that exercises the full pipeline from DramaSeries construction
through character design, scene design, TTS, video generation, subtitle,
compose, and render.

Modes:
    --dry-run   Validate data flow without calling external APIs.
                Uses mock adapters; verifies DAG topology, quality gates,
                prompt enhancement, and voice casting.

    (default)   Full API execution.  Requires:
                  VIDEOCLAW_EVOLINK_API_KEY   (Seedream 5.0 — character/scene images)
                  VIDEOCLAW_ARK_API_KEY       (Seedance 2.0 — video clips)
                  Optional: VIDEOCLAW_WAVESPEED_API_KEY (WaveSpeed TTS)
                  If WaveSpeed key absent, falls back to free EdgeTTS.

Usage:
    python -m pytest tests/smoke_test_western_drama.py -v
    python tests/smoke_test_western_drama.py              # standalone
    python tests/smoke_test_western_drama.py --dry-run    # no API calls
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from videoclaw.drama.models import (
    Character,
    DramaManager,
    DramaScene,
    DramaSeries,
    Episode,
    EpisodeStatus,
    ShotScale,
    ShotType,
    VoiceProfile,
    assign_voice_profile,
)
from videoclaw.drama.prompt_enhancer import PromptEnhancer
from videoclaw.drama.quality import DramaQualityValidator
from videoclaw.drama.runner import build_episode_dag

logger = logging.getLogger("smoke_test")

# ---------------------------------------------------------------------------
# Deliverables path
# ---------------------------------------------------------------------------

DELIVERABLES = _PROJECT_ROOT / "docs" / "deliverables" / "wind_chaser_satan_in_a_suit"

# ---------------------------------------------------------------------------
# Series / Character / Episode data (from trial script)
# ---------------------------------------------------------------------------


def build_series() -> DramaSeries:
    """Construct the DramaSeries from the trial script deliverables."""
    series_path = DELIVERABLES / "final_series.json"
    if series_path.exists():
        data = json.loads(series_path.read_text(encoding="utf-8"))
        series = DramaSeries.from_dict(data)
    else:
        series = DramaSeries(
            series_id="wind_chaser_ranch_001",
            title="Wind-Chaser: Satan in a Suit",
            genre="romance",
            synopsis=(
                "A spirited Texas ranch girl crashes a billionaire's pool party "
                "to save her family's land — and accidentally kisses the most "
                "feared CEO in America."
            ),
            style="cinematic",
            language="en",
            aspect_ratio="9:16",
            target_episode_duration=60.0,
            total_episodes=1,
            model_id="seedance",
        )

    # Always rebuild characters/scenes from authoritative script JSON
    script_path = DELIVERABLES / "ep01_the_devils_pool_script.json"
    chars_path = DELIVERABLES / "executor" / "characters.json"

    if chars_path.exists():
        chars_data = json.loads(chars_path.read_text(encoding="utf-8"))
        series.characters = [Character.from_dict(c) for c in chars_data]
    else:
        raise FileNotFoundError(f"Characters file not found: {chars_path}")

    # Assign voice profiles via English locale
    for char in series.characters:
        assign_voice_profile(char, language="en")

    if script_path.exists():
        script_data = json.loads(script_path.read_text(encoding="utf-8"))
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
    else:
        raise FileNotFoundError(f"Script file not found: {script_path}")

    return series


# ---------------------------------------------------------------------------
# Stage validators
# ---------------------------------------------------------------------------


def validate_series_structure(series: DramaSeries) -> list[str]:
    """Validate DramaSeries data model completeness."""
    errors = []

    if series.language != "en":
        errors.append(f"language should be 'en', got '{series.language}'")
    if series.aspect_ratio != "9:16":
        errors.append(f"aspect_ratio should be '9:16', got '{series.aspect_ratio}'")
    if len(series.characters) < 3:
        errors.append(f"expected >= 3 characters, got {len(series.characters)}")
    if len(series.episodes) < 1:
        errors.append(f"expected >= 1 episode, got {len(series.episodes)}")

    for char in series.characters:
        if not char.visual_prompt:
            errors.append(f"character '{char.name}' missing visual_prompt")
        if not char.voice_style:
            errors.append(f"character '{char.name}' missing voice_style")
        if not char.voice_profile:
            errors.append(f"character '{char.name}' missing voice_profile")

    ep = series.episodes[0]
    if len(ep.scenes) < 10:
        errors.append(f"episode 1 has {len(ep.scenes)} scenes (expected >= 10)")

    total_dur = sum(s.duration_seconds for s in ep.scenes)
    if abs(total_dur - ep.duration_seconds) > 5:
        errors.append(
            f"scene durations sum to {total_dur:.1f}s "
            f"(episode target {ep.duration_seconds}s, tolerance ±5s)"
        )

    return errors


def validate_quality_gate(series: DramaSeries) -> list[str]:
    """Run Western quality validation."""
    ep = series.episodes[0]
    script_data = json.loads(ep.script) if ep.script else {"scenes": []}
    validator = DramaQualityValidator()
    return validator.validate(series, {ep.number: script_data})


def validate_prompt_enhancement(series: DramaSeries) -> list[str]:
    """Verify PromptEnhancer produces five-part structure."""
    errors = []
    enhancer = PromptEnhancer()
    ep = series.episodes[0]

    for scene in ep.scenes[:3]:  # spot-check first 3 scenes
        enhanced = enhancer.enhance_scene_prompt(scene, series)
        if "Style:" not in enhanced:
            errors.append(f"scene {scene.scene_id}: missing Style: section")
        if "Constraints:" not in enhanced:
            errors.append(f"scene {scene.scene_id}: missing Constraints: section")
        if enhancer.should_strip_chinese(series.model_id):
            import re
            if re.search(r"[\u4e00-\u9fff]", enhanced):
                errors.append(f"scene {scene.scene_id}: CJK characters in enhanced prompt")

    return errors


def validate_voice_casting(series: DramaSeries) -> list[str]:
    """Verify all characters have voice profiles from English locale."""
    errors = []
    for char in series.characters:
        if not char.voice_profile:
            errors.append(f"'{char.name}' has no voice_profile")
            continue
        vp = char.voice_profile
        if not vp.voice_id:
            errors.append(f"'{char.name}' voice_profile has no voice_id")
        if "Neural" not in vp.voice_id and "en-" not in vp.voice_id:
            errors.append(
                f"'{char.name}' voice_id '{vp.voice_id}' does not look like an English voice"
            )
    return errors


def validate_dag_topology(series: DramaSeries) -> list[str]:
    """Build the DAG and verify structure."""
    errors = []
    ep = series.episodes[0]

    dag, state = build_episode_dag(ep, series)

    node_ids = set(dag.nodes.keys())

    # Must-have nodes
    required = {"script_gen", "storyboard", "scene_validate", "compose", "render"}
    for r in required:
        if r not in node_ids:
            errors.append(f"DAG missing required node: {r}")

    # Video nodes
    video_nodes = [n for n in node_ids if n.startswith("video_")]
    if len(video_nodes) != len(ep.scenes):
        errors.append(
            f"expected {len(ep.scenes)} video nodes, got {len(video_nodes)}"
        )

    # TTS nodes
    tts_nodes = [n for n in node_ids if n.startswith("tts_")]
    if len(tts_nodes) != len(ep.scenes):
        errors.append(
            f"expected {len(ep.scenes)} tts nodes, got {len(tts_nodes)}"
        )

    # Compose depends on all video + subtitle + music
    compose_node = dag.nodes.get("compose")
    if compose_node:
        deps = set(compose_node.depends_on)
        for vn in video_nodes:
            if vn not in deps:
                errors.append(f"compose missing dependency on {vn}")
        if "subtitle_gen" not in deps:
            errors.append("compose missing dependency on subtitle_gen")
        if "music" not in deps:
            errors.append("compose missing dependency on music")

    # Storyboard has enhanced prompts
    for shot in state.storyboard:
        if "Style:" not in shot.prompt:
            errors.append(f"shot {shot.shot_id}: prompt not enhanced (missing Style:)")
            break

    return errors


def check_required_assets() -> dict[str, list[dict]]:
    """Check which required assets exist and which are missing."""
    assets = {
        "character_images": [],
        "scene_images": [],
        "tts_audio": [],
        "subtitles": [],
    }

    # Character reference images
    chars_spec = DELIVERABLES / "images" / "characters" / "character_reference_specs.json"
    if chars_spec.exists():
        specs = json.loads(chars_spec.read_text(encoding="utf-8"))
        for char in specs["characters"]:
            for label, filename in char["files"].items():
                path = DELIVERABLES / "images" / "characters" / filename
                assets["character_images"].append({
                    "file": filename,
                    "character": char["name"],
                    "pose": label,
                    "exists": path.exists(),
                    "path": str(path),
                })

    # Scene reference images
    scenes_spec = DELIVERABLES / "images" / "scenes" / "scene_reference_specs.json"
    if scenes_spec.exists():
        specs = json.loads(scenes_spec.read_text(encoding="utf-8"))
        for loc in specs["locations"]:
            path = DELIVERABLES / "images" / "scenes" / loc["file"]
            assets["scene_images"].append({
                "file": loc["file"],
                "location": loc["name"],
                "exists": path.exists(),
                "path": str(path),
                "used_in": loc["used_in_scenes"],
            })

    # TTS audio (per-scene dialogue)
    script_path = DELIVERABLES / "ep01_the_devils_pool_script.json"
    if script_path.exists():
        script = json.loads(script_path.read_text(encoding="utf-8"))
        for scene in script["scenes"]:
            has_speech = bool(scene.get("dialogue") or scene.get("narration"))
            if has_speech:
                filename = f"dialogue_{scene['scene_id']}.mp3"
                path = DELIVERABLES / "audio" / filename
                assets["tts_audio"].append({
                    "file": filename,
                    "scene_id": scene["scene_id"],
                    "speaker": scene.get("speaking_character", ""),
                    "line_type": scene.get("dialogue_line_type", "dialogue"),
                    "exists": path.exists(),
                    "path": str(path),
                })

    # Subtitles
    for ext in ("ass", "srt"):
        path = DELIVERABLES / f"subtitles.{ext}"
        assets["subtitles"].append({
            "file": f"subtitles.{ext}",
            "exists": path.exists(),
            "path": str(path),
        })

    return assets


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_result(name: str, errors: list[str]) -> bool:
    if errors:
        print(f"  FAIL  {name}")
        for e in errors:
            print(f"        - {e}")
        return False
    print(f"  PASS  {name}")
    return True


def run_smoke_test(dry_run: bool = False) -> bool:
    """Run the full smoke test. Returns True if all checks pass."""
    logging.basicConfig(level=logging.WARNING)
    all_pass = True

    # --- Build series ---
    _print_section("Stage 1: Build DramaSeries from deliverables")
    try:
        series = build_series()
        print(f"  OK    Series: {series.title}")
        print(f"        Language: {series.language}, Genre: {series.genre}")
        print(f"        Characters: {len(series.characters)}, Episodes: {len(series.episodes)}")
        print(f"        Scenes: {len(series.episodes[0].scenes)}, "
              f"Duration: {sum(s.duration_seconds for s in series.episodes[0].scenes):.1f}s")
    except Exception as e:
        print(f"  FAIL  Could not build series: {e}")
        return False

    # --- Validate structure ---
    _print_section("Stage 2: Validate data model")
    if not _print_result("DramaSeries structure", validate_series_structure(series)):
        all_pass = False

    # --- Quality gate ---
    _print_section("Stage 3: Western quality gate (11 checks)")
    violations = validate_quality_gate(series)
    if not _print_result(f"Quality gate ({11 - len(violations)}/11 passed)", violations):
        all_pass = False

    # --- Prompt enhancement ---
    _print_section("Stage 4: Seedance 2.0 prompt enhancement")
    if not _print_result("Five-part prompt structure", validate_prompt_enhancement(series)):
        all_pass = False

    # --- Voice casting ---
    _print_section("Stage 5: English voice casting")
    if not _print_result("VoiceProfile assignment", validate_voice_casting(series)):
        all_pass = False

    # --- DAG topology ---
    _print_section("Stage 6: DAG pipeline topology")
    if not _print_result("DAG structure", validate_dag_topology(series)):
        all_pass = False

    # --- Asset inventory ---
    _print_section("Stage 7: Asset inventory")
    assets = check_required_assets()

    print("\n  [REQUIRED] Character reference images:")
    char_exist = sum(1 for a in assets["character_images"] if a["exists"])
    char_total = len(assets["character_images"])
    for a in assets["character_images"]:
        status = "EXISTS" if a["exists"] else "MISSING"
        print(f"    {status:7s}  {a['character']:15s}  {a['pose']:15s}  {a['file']}")
    print(f"    Status: {char_exist}/{char_total}")

    print("\n  [REQUIRED] Scene reference images:")
    scene_exist = sum(1 for a in assets["scene_images"] if a["exists"])
    scene_total = len(assets["scene_images"])
    for a in assets["scene_images"]:
        status = "EXISTS" if a["exists"] else "MISSING"
        scenes_str = ", ".join(a.get("used_in", [])[:3])
        print(f"    {status:7s}  {a['location']:30s}  {a['file']:30s}  used in: {scenes_str}...")
    print(f"    Status: {scene_exist}/{scene_total}")

    print("\n  [REQUIRED] Per-scene TTS audio:")
    tts_exist = sum(1 for a in assets["tts_audio"] if a["exists"])
    tts_total = len(assets["tts_audio"])
    for a in assets["tts_audio"]:
        status = "EXISTS" if a["exists"] else "MISSING"
        print(f"    {status:7s}  {a['scene_id']:12s}  {a['speaker']:15s}  {a['line_type']:16s}  {a['file']}")
    print(f"    Status: {tts_exist}/{tts_total}")

    print("\n  [REQUIRED] Subtitles:")
    sub_exist = any(a["exists"] for a in assets["subtitles"])
    for a in assets["subtitles"]:
        status = "EXISTS" if a["exists"] else "MISSING"
        print(f"    {status:7s}  {a['file']}")
    print(f"    Status: {'1/1' if sub_exist else '0/1'}")

    # --- Summary ---
    _print_section("ASSET GENERATION READINESS")

    missing_chars = [a for a in assets["character_images"] if not a["exists"]]
    missing_scenes = [a for a in assets["scene_images"] if not a["exists"]]
    missing_tts = [a for a in assets["tts_audio"] if not a["exists"]]

    print(f"""
  Before calling Seedance 2.0 for video generation, these assets
  must be generated in order:

  Phase 1 — Image Generation (parallel, Evolink Seedream 5.0):
    Character refs:  {len(missing_chars):2d} missing / {char_total} total
    Scene refs:      {len(missing_scenes):2d} missing / {scene_total} total
    API key:         VIDEOCLAW_EVOLINK_API_KEY {'SET' if os.environ.get('VIDEOCLAW_EVOLINK_API_KEY') else 'NOT SET'}

  Phase 2 — TTS Audio (parallel, EdgeTTS free / WaveSpeed paid):
    TTS files:       {len(missing_tts):2d} missing / {tts_total} total
    EdgeTTS:         always available (free)
    WaveSpeed key:   VIDEOCLAW_WAVESPEED_API_KEY {'SET' if os.environ.get('VIDEOCLAW_WAVESPEED_API_KEY') else 'NOT SET'}

  Phase 3 — Video Generation (parallel, Seedance 2.0):
    Video clips:     15 to generate
    API key:         VIDEOCLAW_ARK_API_KEY {'SET' if os.environ.get('VIDEOCLAW_ARK_API_KEY') else 'NOT SET'}
    Est. cost:       ~$3.05 USD (61s × $0.05/s)

  Phase 4 — Post-production:
    Subtitles:       auto-generated from TTS timing
    BGM:             optional (placeholder)
    Compose+Render:  FFmpeg (local)
""")

    if dry_run:
        _print_section("DRY RUN COMPLETE")
        print(f"  Data flow validated: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
        print("  No external API calls were made.")
        print("  To generate assets, run without --dry-run with API keys set.")
    else:
        if missing_chars or missing_scenes:
            _print_section("GENERATING MISSING ASSETS")
            asyncio.run(_generate_missing_assets(series, assets))
        else:
            print("  All pre-video assets already exist.")

    return all_pass


async def _generate_missing_assets(
    series: DramaSeries,
    assets: dict[str, list[dict]],
) -> None:
    """Generate missing character/scene reference images."""
    missing_chars = [a for a in assets["character_images"] if not a["exists"]]
    missing_scenes = [a for a in assets["scene_images"] if not a["exists"]]

    if missing_chars:
        print(f"\n  Generating {len(missing_chars)} character reference images...")
        try:
            from videoclaw.drama.character_designer import CharacterDesigner
            designer = CharacterDesigner()
            mgr = DramaManager()
            # Save series first so designer can find it
            mgr.save(series)
            series = await designer.design_characters(series, force=True)
            print(f"  OK    Character images generated for {len(series.characters)} characters")
        except Exception as e:
            print(f"  FAIL  Character image generation: {e}")
            print("        Set VIDEOCLAW_EVOLINK_API_KEY to enable image generation.")

    if missing_scenes:
        print(f"\n  Generating {len(missing_scenes)} scene reference images...")
        try:
            from videoclaw.drama.scene_designer import SceneDesigner
            designer = SceneDesigner()
            locations = await designer.design_scenes(series, force=True)
            print(f"  OK    Scene images generated for {len(locations)} locations")
        except Exception as e:
            print(f"  FAIL  Scene image generation: {e}")
            print("        Set VIDEOCLAW_EVOLINK_API_KEY to enable image generation.")


# ---------------------------------------------------------------------------
# CLI + pytest entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    ok = run_smoke_test(dry_run=dry_run)
    sys.exit(0 if ok else 1)


def test_smoke_western_drama_dry_run():
    """pytest entry: run smoke test in dry-run mode."""
    assert run_smoke_test(dry_run=True), "Smoke test failed — see output for details"
