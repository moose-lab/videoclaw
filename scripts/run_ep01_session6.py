"""Session 6 — EP01 "Satan in a Suit"

Steps:
  1. Regenerate ep01_s06 (pool-deck shock scene — failed in Session 5)
  2. FFmpeg-concat all 7 Session 5/6 clips → ep01_v1.mp4

Usage:
    .venv/bin/python scripts/run_ep01_session6.py [--dry-run] [--skip-s06] [--skip-concat]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from videoclaw.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DELIVERABLES = PROJECT_ROOT / "docs" / "deliverables" / "ep01_satan_in_a_suit"
SERIES_DATA = DELIVERABLES / "series_data.json"
IMAGE_URLS_FILE = DELIVERABLES / "image_urls.json"
VIDEO_DIR = DELIVERABLES / "video_clips"
CHECKPOINT_DIR = DELIVERABLES / "checkpoints"

SESSION5_CLIPS = [
    VIDEO_DIR / "session5_ep01_s01.mp4",
    VIDEO_DIR / "session5_ep01_s02.mp4",
    VIDEO_DIR / "session5_ep01_s03.mp4",
    VIDEO_DIR / "session5_ep01_s04.mp4",
    VIDEO_DIR / "session5_ep01_s05.mp4",
    # s06 → session6_ep01_s06.mp4 injected here
    VIDEO_DIR / "session5_ep01_s07.mp4",
]
S06_OUTPUT = VIDEO_DIR / "session6_ep01_s06.mp4"
EP01_V1 = VIDEO_DIR / "ep01_v1.mp4"

MODEL = "doubao-seedance-2.0-fast-260128"
POLL_INTERVAL_S = 8.0
POLL_TIMEOUT_S = 600.0
COST_PER_SECOND = 0.05

_INTRO_SCALES = {"close_up", "medium_close"}
_CHARACTER_INTROS = {
    "Ivy": "IVY ANGEL, 26 — Ranch Girl",
    "Ivy Angel": "IVY ANGEL, 26 — Ranch Girl",
    "Colton": 'COLTON BLACK, 30 — "Satan in a Suit"',
    "Colton Black": 'COLTON BLACK, 30 — "Satan in a Suit"',
    "Chloe": "CHLOE GREEN, 28 — Heiress",
    "Chloe Green": "CHLOE GREEN, 28 — Heiress",
}
_CHAR_NAME_TO_PREFIX = {
    "Ivy": "Ivy_Angel", "Ivy Angel": "Ivy_Angel",
    "Colton": "Colton_Black", "Colton Black": "Colton_Black",
    "Chloe": "Chloe_Green", "Chloe Green": "Chloe_Green",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_config() -> tuple[str, dict[str, str]]:
    import os
    config = get_config()
    api_key = os.environ.get("ARK_API_KEY") or config.ark_api_key
    base_url = (config.seedance_base_url or "https://sd2.vectorspace.cn").rstrip("/")
    return base_url, {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


async def _api_post(url: str, payload: dict, headers: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 429:
                    await asyncio.sleep(15.0 * (attempt + 1))
                    continue
                return resp.json()
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(3)
            else:
                raise
    raise RuntimeError("API call failed after retries")


async def _create_task(base_url: str, headers: dict, payload: dict) -> str:
    data = await _api_post(f"{base_url}/api/v1/doubao/create", payload, headers)
    task_id = data.get("id") or data.get("task_id")
    if not task_id:
        raise RuntimeError(f"No task_id in response: {data}")
    return task_id


async def _poll_task(base_url: str, headers: dict, task_id: str) -> str:
    elapsed = 0.0
    while elapsed < POLL_TIMEOUT_S:
        await asyncio.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S
        try:
            data = await _api_post(
                f"{base_url}/api/v1/doubao/get_result", {"id": task_id}, headers
            )
        except Exception as e:
            logger.warning("Poll error (%.0fs): %s", elapsed, e)
            continue
        status = (data.get("status") or "").lower()
        if status in ("succeeded", "success", "completed", "done"):
            content = data.get("content", {})
            url = (
                (content.get("video_url") if isinstance(content, dict) else None)
                or data.get("video_url")
                or data.get("output", {}).get("video_url")
            )
            if url:
                return url
            raise RuntimeError(f"Task {task_id} done but no video URL")
        if status in ("failed", "error", "cancelled"):
            raise RuntimeError(f"Task {task_id} failed: {data.get('error', 'unknown')}")
        if int(elapsed) % 30 == 0:
            logger.info("  polling %s: %s (%.0fs)", task_id, status, elapsed)
    raise TimeoutError(f"Timeout after {POLL_TIMEOUT_S}s (task={task_id})")


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def _build_prompt(scene: dict, chars_introduced: set[str]) -> str:
    visual = scene.get("visual_prompt", "")
    if scene.get("characters_present"):
        visual = (
            "IMPORTANT: Generate as live-action film with REAL HUMAN ACTORS. "
            "Photorealistic skin, real hair, realistic fabric and lighting. "
            "NOT cartoon, NOT anime, NOT illustration. "
            "Netflix drama cinematography. "
            + visual
        )
    parts = [visual]

    # Character name card on first close-up appearance
    shot_scale = scene.get("shot_scale", "")
    if shot_scale in _INTRO_SCALES:
        speaker = scene.get("speaking_character", "").strip()
        chars = scene.get("characters_present", [])
        focal = None
        if speaker and speaker in _CHARACTER_INTROS:
            focal = speaker
        elif len(chars) == 1 and chars[0] in _CHARACTER_INTROS:
            focal = chars[0]
        if focal and focal not in chars_introduced:
            parts.append(f'[Show name card at bottom: {_CHARACTER_INTROS[focal]}]')
            chars_introduced.add(focal)

    # Narration
    narration = scene.get("narration", "").strip()
    narration_type = scene.get("narration_type", "voiceover")
    if narration:
        if narration_type == "title_card":
            parts.append(f'[Show large centered title text: "{narration}"]')
        else:
            parts.append(
                f'[Narrator speaks: "{narration}". Show subtitle at bottom: "{narration}"]'
            )

    # Dialogue
    dialogue = scene.get("dialogue", "").strip()
    if dialogue:
        speaker = scene.get("speaking_character", "Character")
        line_type = scene.get("dialogue_line_type", "dialogue")
        if line_type == "inner_monologue":
            parts.append(
                f'[{speaker} thinks (inner monologue): "{dialogue}". '
                f'Show subtitle: "{dialogue}"]'
            )
        else:
            parts.append(
                f'[{speaker} speaks: "{dialogue}". '
                f'Show subtitle at bottom: "{dialogue}"]'
            )
    return "\n".join(parts)


def _get_char_urls(chars: list[str], char_url_map: dict) -> list[dict]:
    items = []
    seen: set[str] = set()
    for name in chars:
        prefix = _CHAR_NAME_TO_PREFIX.get(name)
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)
        for suffix in ("stylized", "turnaround", "orthographic", "front", "three_quarter", "full_body"):
            key = f"{prefix}_{suffix}"
            if key in char_url_map and len(items) < 9:
                items.append({"url": char_url_map[key], "role": "reference_image"})
                break
    return items


# ---------------------------------------------------------------------------
# Step 1: Generate s06
# ---------------------------------------------------------------------------

async def generate_s06(dry_run: bool = False) -> dict:
    """Generate ep01_s06 and save to session6_ep01_s06.mp4."""
    with open(SERIES_DATA) as f:
        series = json.load(f)
    scenes = series["episodes"][0]["scenes"]
    scene = next((s for s in scenes if s["scene_id"] == "ep01_s06"), None)
    if not scene:
        raise ValueError("ep01_s06 not found in series_data.json")

    image_data = json.load(open(IMAGE_URLS_FILE))
    char_url_map = image_data.get("characters", {})

    dur = max(5, min(15, int(float(scene.get("duration_seconds", 8)))))
    chars_introduced: set[str] = {"Ivy Angel", "Colton Black", "Chloe Green"}  # already introduced in s01-s05
    prompt = _build_prompt(scene, chars_introduced)

    content: list[dict] = [{"type": "text", "text": prompt}]
    char_imgs = _get_char_urls(scene.get("characters_present", []), char_url_map)
    for ci in char_imgs:
        content.append({"type": "image_url", "image_url": {"url": ci["url"]}, "role": ci["role"]})

    img_count = len(char_imgs)
    logger.info("ep01_s06: %ds, %d refs, prompt[:120]=%r", dur, img_count, prompt[:120])
    logger.info("Characters: %s", scene.get("characters_present"))

    if dry_run:
        logger.info("[DRY RUN] Would submit to Seedance 2.0")
        return {"scene_id": "ep01_s06", "status": "dry_run"}

    payload = {
        "model": MODEL,
        "content": content,
        "generate_audio": True,
        "ratio": "9:16",
        "resolution": "720p",
        "duration": dur,
        "watermark": False,
    }

    base_url, headers = _get_api_config()
    t0 = time.time()
    logger.info("Submitting ep01_s06 to Seedance 2.0...")
    task_id = await _create_task(base_url, headers, payload)
    logger.info("task_id=%s, polling...", task_id)

    video_url = await _poll_task(base_url, headers, task_id)
    video_bytes = await _download(video_url)
    elapsed = time.time() - t0
    cost = dur * COST_PER_SECOND

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    S06_OUTPUT.write_bytes(video_bytes)

    result = {
        "scene_id": "ep01_s06",
        "status": "success",
        "output_path": str(S06_OUTPUT),
        "cost_usd": round(cost, 2),
        "elapsed_s": round(elapsed, 1),
        "size_mb": round(len(video_bytes) / 1024 / 1024, 2),
        "task_id": task_id,
        "ref_images": img_count,
    }
    logger.info(
        "ep01_s06 OK — %.0fs, $%.2f, %.1fMB",
        elapsed, cost, result["size_mb"],
    )
    return result


# ---------------------------------------------------------------------------
# Step 2: Concat EP01 v1
# ---------------------------------------------------------------------------

def concat_ep01_v1() -> bool:
    """Concat s01-s07 into ep01_v1.mp4 using FFmpeg stream copy."""
    clips = [
        VIDEO_DIR / "session5_ep01_s01.mp4",
        VIDEO_DIR / "session5_ep01_s02.mp4",
        VIDEO_DIR / "session5_ep01_s03.mp4",
        VIDEO_DIR / "session5_ep01_s04.mp4",
        VIDEO_DIR / "session5_ep01_s05.mp4",
        S06_OUTPUT,
        VIDEO_DIR / "session5_ep01_s07.mp4",
    ]

    missing = [c for c in clips if not c.exists()]
    if missing:
        logger.error("Missing clips: %s", [str(m) for m in missing])
        return False

    # Write concat list
    concat_list = VIDEO_DIR / "concat_list.txt"
    concat_list.write_text(
        "\n".join(f"file '{c.resolve()}'" for c in clips)
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(EP01_V1),
    ]
    logger.info("Concatenating 7 clips → ep01_v1.mp4")
    logger.info("  %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg error:\n%s", result.stderr[-2000:])
        return False

    size_mb = EP01_V1.stat().st_size / 1024 / 1024
    logger.info("ep01_v1.mp4 — %.1fMB", size_mb)

    # Duration check
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(EP01_V1)],
        capture_output=True, text=True,
    )
    if probe.returncode == 0 and probe.stdout.strip():
        dur = float(probe.stdout.strip())
        logger.info("Total duration: %.1fs", dur)

    concat_list.unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(dry_run: bool, skip_s06: bool, skip_concat: bool) -> None:
    checkpoint: dict = {
        "session": 6,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: s06
    if not skip_s06:
        logger.info("=" * 50)
        logger.info("STEP 1: Generate ep01_s06")
        logger.info("=" * 50)
        if S06_OUTPUT.exists() and S06_OUTPUT.stat().st_size > 10_000:
            logger.info("session6_ep01_s06.mp4 already exists (%.1fMB) — skipping",
                        S06_OUTPUT.stat().st_size / 1024 / 1024)
            checkpoint["s06"] = {"status": "skipped_existing", "output_path": str(S06_OUTPUT)}
        else:
            s06_result = await generate_s06(dry_run=dry_run)
            checkpoint["s06"] = s06_result
            if s06_result.get("status") != "success" and not dry_run:
                logger.error("s06 generation failed — aborting concat step")
                _save_checkpoint(checkpoint)
                return
    else:
        logger.info("Skipping s06 generation (--skip-s06)")
        checkpoint["s06"] = {"status": "skipped_by_flag"}

    # Step 2: Concat
    if not skip_concat and not dry_run:
        logger.info("=" * 50)
        logger.info("STEP 2: Concat EP01 v1")
        logger.info("=" * 50)
        ok = concat_ep01_v1()
        checkpoint["concat"] = {"status": "success" if ok else "failed", "output": str(EP01_V1)}
        if ok:
            logger.info("EP01 v1 delivered: %s", EP01_V1)
        else:
            logger.error("Concat failed")
    else:
        logger.info("Skipping concat (%s)", "--dry-run" if dry_run else "--skip-concat")

    _save_checkpoint(checkpoint)
    logger.info("Session 6 complete.")


def _save_checkpoint(data: dict) -> None:
    ts = data.get("timestamp", time.strftime("%Y%m%d_%H%M%S"))
    path = CHECKPOINT_DIR / f"checkpoint_session6_{ts}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Checkpoint: %s", path)


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Session 6 — EP01 s06 regen + concat")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-s06", action="store_true", help="Skip s06 generation")
    parser.add_argument("--skip-concat", action="store_true", help="Skip final concat")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, skip_s06=args.skip_s06, skip_concat=args.skip_concat))


if __name__ == "__main__":
    cli_main()
