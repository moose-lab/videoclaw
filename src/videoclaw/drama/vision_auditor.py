"""Vision auditor — Claude Vision QA for generated video clips.

Extracts keyframes from each generated clip and sends them to Claude Vision
for per-shot quality inspection.  Checks performed:

- ``time_of_day``: lighting matches script spec (day / evening / night)
- ``characters``: expected characters are visually present and recognisable
- ``subtitle_spelling``: on-screen text is free of OCR-style errors
- ``artifacts``: anatomical distortions, flickering, blurring
- ``dramatic_tension``: hook / cliffhanger quality for first / last shots

Two operating modes
-------------------
**Series-aware mode** (preferred) — works with the drama management system::

    auditor = VisionAuditor()
    report = await auditor.audit_series_episode(
        series, episode_number=1, drama_manager=mgr
    )

**Standalone mode** — for externally generated clips::

    auditor = VisionAuditor()
    report = await auditor.audit_clip_dir(scenes, clip_dir=Path("video_clips/"))

In both modes, ``VisionAuditor.resolve_clip`` automatically discovers the best
matching file for each ``scene_id`` in *clip_dir*, preferring the most recently
modified clip when multiple session-prefixed files exist.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from videoclaw.drama.models import DramaScene, DramaSeries
from videoclaw.models.llm.litellm_wrapper import LLMClient
from videoclaw.utils.ffmpeg import get_video_duration, get_video_info

if TYPE_CHECKING:
    from videoclaw.drama.models import DramaManager

logger = logging.getLogger(__name__)

# Vision model — sonnet is multimodal capable and cost-efficient
VISION_MODEL = "claude-sonnet-4-6"

# V1 template — retained for backward compatibility, not used in production.
_AUDIT_SYSTEM = """\
You are a professional AI video quality auditor for TikTok-style short dramas.
You will be shown keyframes from a video clip alongside the scene specification.
Evaluate the clip against the spec and return a structured JSON audit result.

Be concise and specific. Focus on production-blocking issues, not minor imperfections.
Return ONLY valid JSON matching the schema given in the user message.
"""

# V1 template — retained for backward compatibility, not used in production.
_AUDIT_USER_TEMPLATE = """\
Scene specification:
  scene_id: {scene_id}
  description: {description}
  time_of_day: {time_of_day}
  expected_characters: {characters}
  dialogue: {dialogue}
  shot_role: {shot_role}  (hook=first shot, cliffhanger=last shot, normal=middle)
  emotion: {emotion}

The {frame_count} keyframes above are from this generated clip (first / mid / last frame).

Audit checklist:
1. time_of_day: Does the lighting match "{time_of_day}"?
   - night → dark sky, artificial lighting, shadows
   - day → bright natural lighting, daylight
   - evening → golden hour, warm tones, dimming light
   - unspecified → skip this check
2. characters: Are {characters} visually present and recognisable?
3. subtitle_spelling: List any on-screen text with obvious spelling errors.
   (Common Seedance OCR errors: dropped letters, merged words, wrong characters)
4. artifacts: Any anatomical distortion, flickering, blurring, or generation artifacts?
5. dramatic_tension: For shot_role="{shot_role}":
   - hook → strong opening impact (≥3 sec to draw viewer in)
   - cliffhanger → ends on unresolved tension / cut-off moment
   - normal → acceptable for mid-episode

Return JSON exactly matching this schema:
{{
  "shot_id": "{scene_id}",
  "passed": true,
  "checks": {{
    "time_of_day": {{"ok": true, "note": ""}},
    "characters": {{"ok": true, "note": ""}},
    "subtitle_spelling": {{"ok": true, "errors": []}},
    "artifacts": {{"ok": true, "note": ""}},
    "dramatic_tension": {{"ok": true, "note": ""}}
  }},
  "issues": [],
  "regen_required": false,
  "regen_note": ""
}}

Set passed=false and regen_required=true only for production-blocking issues:
- Wrong time of day when clearly inconsistent (e.g. bright daylight for a night scene)
- Wrong / missing main characters
- Severe generation artifacts making the clip unusable
Subtitle spelling errors and minor tension issues → passed=true, regen_required=false.
"""

# ---------------------------------------------------------------------------
# V2 pragmatic grading thresholds & templates
# ---------------------------------------------------------------------------

_FATAL_THRESHOLD = 0.75
_TOLERABLE_THRESHOLD = 0.85

_AUDIT_SYSTEM_V2 = """\
You are a pragmatic AI video quality auditor for Western live-action TikTok short dramas.
You will see keyframes from a generated video clip.
Your job: identify ONLY defects that would make a viewer swipe away.
Accept minor imperfections — 80% of AI videos have 1-2 small defects; that is normal.

Return ONLY valid JSON. No markdown, no explanation."""

_COMPOSITION_AUDIT_PROMPT = """\
You are auditing a FINAL COMPOSED episode of a Western TikTok short drama.
The {frame_count} keyframes are evenly sampled from the full {duration:.0f}s video.

Check for episode-level issues ONLY (not per-shot details):

1. VISUAL COHERENCE — Do shots look like they belong to the same episode?
   (consistent color grading, lighting style, character appearances)
2. TRANSITION QUALITY — Any jarring visual jumps between adjacent frames?
3. PACING — Does the frame sequence suggest good dramatic rhythm?
   (variety of shot scales, building tension)

Return JSON:
{{
  "fatals": ["episode-level fatal issues"],
  "tolerables": ["minor coherence issues"]
}}

Be generous — minor style variations between shots are normal for AI-generated content."""

_AUDIT_USER_TEMPLATE_V2 = """\
Scene spec:
  scene_id: {scene_id}
  description: {description}
  time_of_day: {time_of_day}
  expected_characters: {characters}
  dialogue: {dialogue}
  shot_role: {shot_role}
  emotion: {emotion}
  shot_scale: {shot_scale}

The {frame_count} keyframes above are from this clip (first / mid / last).

Check these 4 dimensions and classify each defect as "fatal" or "tolerable":

1. ANATOMY — Extra fingers/limbs, facial collapse, body morphing visible on main characters.
   - Fatal if: visible in close_up/medium shot on a main character.
   - Tolerable if: in wide shot or on background character.

2. CHARACTER PRESENCE — Are the expected characters ({characters}) in the frame?
   - Fatal if: a main character is missing entirely.
   - Tolerable if: a secondary character is unclear but present.

3. SCENE MATCH — Does the scene match "{description}" and time_of_day="{time_of_day}"?
   - Fatal if: completely wrong location or time_of_day.
   - Tolerable if: minor detail differences.

4. DRAMATIC TENSION (only for hook/cliffhanger shots, skip for normal):
   - shot_role={shot_role}
   - Fatal if: hook shot has zero conflict/tension; cliffhanger has full resolution.
   - Tolerable if: tension exists but is weak.

Return JSON:
{{
  "fatals": ["short description of each fatal defect"],
  "tolerables": ["short description of each tolerable defect"]
}}

Empty lists = clean shot. Be pragmatic: if a viewer on a phone wouldn't notice it, skip it."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ShotAuditResult:
    shot_id: str
    passed: bool
    regen_required: bool = False
    regen_note: str = ""
    issues: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    clip_path: str = ""
    raw_response: str = ""
    fatals: list[str] = field(default_factory=list)
    tolerables: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict[str, Any], shot_id: str, clip_path: str = "") -> ShotAuditResult:
        return cls(
            shot_id=data.get("shot_id", shot_id),
            passed=bool(data.get("passed", True)),
            regen_required=bool(data.get("regen_required", False)),
            regen_note=data.get("regen_note", ""),
            issues=list(data.get("issues", [])),
            checks=dict(data.get("checks", {})),
            clip_path=clip_path,
            fatals=list(data.get("fatals", [])),
            tolerables=list(data.get("tolerables", [])),
        )

    @classmethod
    def error_result(cls, shot_id: str, error: str) -> ShotAuditResult:
        return cls(
            shot_id=shot_id,
            passed=False,
            regen_required=False,
            regen_note=f"Audit error: {error}",
            issues=[f"Audit failed: {error}"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "passed": self.passed,
            "regen_required": self.regen_required,
            "regen_note": self.regen_note,
            "issues": self.issues,
            "checks": self.checks,
            "clip_path": self.clip_path,
            "fatals": self.fatals,
            "tolerables": self.tolerables,
        }


@dataclass
class EpisodeAuditReport:
    series_id: str
    episode_number: int
    shot_results: list[ShotAuditResult] = field(default_factory=list)
    total_shots: int = 0
    passed_shots: int = 0
    regen_required: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Vision Audit — EP{self.episode_number:02d} ===",
            f"Shots: {self.passed_shots}/{self.total_shots} passed",
        ]
        if self.regen_required:
            lines.append(f"Regen required: {', '.join(self.regen_required)}")
        for r in self.shot_results:
            status = "PASS" if r.passed else "REGEN"
            f_count = len(r.fatals) if r.fatals else 0
            t_count = len(r.tolerables) if r.tolerables else 0
            grade = f" (F:{f_count} T:{t_count})" if (f_count or t_count) else ""
            clip = f" ({Path(r.clip_path).name})" if r.clip_path else ""
            defects = f" — {'; '.join(r.fatals + r.tolerables)}" if (r.fatals or r.tolerables) else ""
            lines.append(f"  {r.shot_id}: {status}{grade}{clip}{defects}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "episode_number": self.episode_number,
            "total_shots": self.total_shots,
            "passed_shots": self.passed_shots,
            "regen_required": self.regen_required,
            "shot_results": [r.to_dict() for r in self.shot_results],
        }

    def save_to_log(self, series_dir: Path, round_num: int) -> None:
        """Persist this audit report as one round in the series audit log.

        Creates ``{series_dir}/audit_logs/ep{NN}_audit.jsonl`` if it doesn't
        exist, then appends one JSON-Lines record for this round.
        """
        log_dir = series_dir / "audit_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"ep{self.episode_number:02d}_audit.jsonl"
        audit_log = AuditLog(log_path)
        audit_log.append_round(self, round_num)


# ---------------------------------------------------------------------------
# Human preview breakpoint (人工预览断点)
# ---------------------------------------------------------------------------


def preview_and_confirm(
    report: EpisodeAuditReport,
) -> list[str]:
    """Present passed/tolerable shots for optional human review.

    Prints a Rich table of all shots with status and defects.
    Prompts the user to enter scene IDs to force-regen (comma-separated),
    or press Enter to accept all.

    Returns list of additional scene_ids the user wants to regen.
    """
    console = Console()

    table = Table(title=f"EP{report.episode_number:02d} Shot Review")
    table.add_column("Shot ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Fatals", justify="right")
    table.add_column("Tolerables", justify="right")
    table.add_column("Defects", style="dim")

    for r in report.shot_results:
        status = "[green]PASS[/green]" if r.passed else "[red]REGEN[/red]"
        f_count = str(len(r.fatals)) if r.fatals else "0"
        t_count = str(len(r.tolerables)) if r.tolerables else "0"
        defects = "; ".join(r.fatals + r.tolerables) if (r.fatals or r.tolerables) else ""
        table.add_row(r.shot_id, status, f_count, t_count, defects)

    console.print(table)

    user_input = console.input(
        "\n[yellow]Enter scene IDs to force-regen (comma-separated), "
        "or press Enter to accept:[/yellow] "
    )

    if not user_input.strip():
        return []

    return [sid.strip() for sid in user_input.split(",") if sid.strip()]


# ---------------------------------------------------------------------------
# AuditLog — JSON-Lines persistence for audit rounds
# ---------------------------------------------------------------------------

class AuditLog:
    """Append-only JSON-Lines audit log.

    Each line stores one audit round as a JSON object::

        {"round": N, "timestamp": "ISO", "episode": N,
         "results": [...], "summary": {"passed": N, "total": N, "regen_ids": [...]}}

    Parameters
    ----------
    log_path:
        Path to the ``.jsonl`` file.  Created on first write.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def append_round(self, report: EpisodeAuditReport, round_num: int) -> None:
        """Append one audit round record to the log file."""
        record = {
            "round": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode": report.episode_number,
            "results": [r.to_dict() for r in report.shot_results],
            "summary": {
                "passed": report.passed_shots,
                "total": report.total_shots,
                "regen_ids": list(report.regen_required),
            },
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        """Read all rounds from the log file.  Returns ``[]`` if file missing."""
        if not self.log_path.exists():
            return []
        rounds: list[dict] = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rounds.append(json.loads(line))
        return rounds

    def get_frequent_defects(self, min_count: int = 3) -> list[str]:
        """Return defect descriptions that appeared >= *min_count* times across all rounds.

        Scans ``fatals`` and ``tolerables`` from every shot result in every round.
        """
        rounds = self.read_all()
        if not rounds:
            return []
        counter: Counter[str] = Counter()
        for rnd in rounds:
            for result in rnd.get("results", []):
                for defect in result.get("fatals", []):
                    counter[defect] += 1
                for defect in result.get("tolerables", []):
                    counter[defect] += 1
        return [defect for defect, count in counter.items() if count >= min_count]


# ---------------------------------------------------------------------------
# Clip resolution
# ---------------------------------------------------------------------------

def resolve_clip(scene_id: str, clip_dir: Path, video_asset_path: str | None = None) -> Path | None:
    """Find the best matching MP4 clip for *scene_id* in *clip_dir*.

    Resolution order:
    1. ``video_asset_path`` if set and file exists (set by drama runner)
    2. ``{clip_dir}/{scene_id}.mp4`` — exact match
    3. ``{clip_dir}/*_{scene_id}.mp4`` — session-prefixed variants;
       the most recently modified file wins (so session6 trumps session5)

    Returns ``None`` if no file is found.
    """
    if video_asset_path:
        p = Path(video_asset_path)
        if p.exists():
            return p

    direct = clip_dir / f"{scene_id}.mp4"
    if direct.exists():
        return direct

    candidates = sorted(
        clip_dir.glob(f"*_{scene_id}.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# VisionAuditor
# ---------------------------------------------------------------------------

class VisionAuditor:
    """Audits generated video clips using Claude Vision.

    Parameters
    ----------
    llm:
        Optional pre-built LLMClient.  Defaults to ``claude-sonnet-4-6``.
    frame_count:
        Number of keyframes to extract per clip (1–5).  Default 3.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        frame_count: int = 3,
    ) -> None:
        self._llm = llm
        self.frame_count = max(1, min(5, frame_count))

    def _ensure_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(default_model=VISION_MODEL)
        return self._llm

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def extract_keyframes(self, clip_path: Path) -> list[str]:
        """Extract *frame_count* frames from *clip_path*; return base64 JPEGs."""
        frames: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(clip_path)],
                capture_output=True, text=True,
            )
            duration = 5.0
            if probe.returncode == 0 and probe.stdout.strip():
                try:
                    duration = float(probe.stdout.strip())
                except ValueError:
                    pass

            for i, ts in enumerate(self._frame_timestamps(duration)):
                out_file = tmp_path / f"frame_{i:02d}.jpg"
                result = subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(clip_path),
                     "-vframes", "1", "-q:v", "3", "-vf", "scale=720:-1", str(out_file)],
                    capture_output=True,
                )
                if result.returncode == 0 and out_file.exists():
                    frames.append(base64.b64encode(out_file.read_bytes()).decode())
                else:
                    logger.warning("Frame extraction failed at ts=%.2f for %s", ts, clip_path.name)

        logger.debug("Extracted %d frames from %s", len(frames), clip_path.name)
        return frames

    def _frame_timestamps(self, duration: float) -> list[float]:
        if self.frame_count == 1:
            return [duration / 2]
        start = min(0.3, duration * 0.05)
        end = max(start + 0.1, duration - 0.5)
        if self.frame_count == 2:
            return [start, end]
        mid = (start + end) / 2
        return [start, mid, end][: self.frame_count]

    # ------------------------------------------------------------------
    # Per-shot audit — 3-layer pipeline (v2)
    # ------------------------------------------------------------------

    async def audit_shot(
        self,
        scene: DramaScene,
        clip_path: Path,
        *,
        series: DramaSeries | None = None,
    ) -> ShotAuditResult:
        """Audit a single clip against its scene specification.

        Three-layer pipeline:
        - Layer 0: ffprobe metadata rules (duration check).
        - Layer 1: SSIM temporal stability (frame_analyzer).
        - Layer 2: Claude Vision LLM audit.

        Each layer short-circuits on fatal — later layers are skipped.
        """
        if not clip_path.exists():
            return ShotAuditResult.error_result(
                scene.scene_id, f"clip not found: {clip_path}"
            )

        logger.info("Auditing %s (%s)...", scene.scene_id, clip_path.name)

        all_fatals: list[str] = []
        all_tolerables: list[str] = []

        # --- Layer 0: ffprobe metadata rules ---
        try:
            info = await get_video_info(clip_path)
            duration = float(info.get("format", {}).get("duration", 0))
            if duration <= 0.5:
                all_fatals.append(f"clip duration too short ({duration:.2f}s)")
        except Exception as exc:
            logger.warning("Layer 0 ffprobe failed for %s: %s", scene.scene_id, exc)
            all_fatals.append(f"ffprobe error: {exc}")

        if all_fatals:
            result = self._build_verdict(scene.scene_id, all_fatals, all_tolerables, clip_path=str(clip_path))
            logger.info("  %s: Layer 0 short-circuit — %s", scene.scene_id, "; ".join(all_fatals))
            return result

        # --- Layer 1: temporal stability (SSIM) ---
        l1_fatals, l1_tolerables = await self._layer1_temporal(clip_path)
        all_fatals.extend(l1_fatals)
        all_tolerables.extend(l1_tolerables)

        if all_fatals:
            result = self._build_verdict(scene.scene_id, all_fatals, all_tolerables, clip_path=str(clip_path))
            logger.info("  %s: Layer 1 short-circuit — %s", scene.scene_id, "; ".join(all_fatals))
            return result

        # --- Layer 2: Claude Vision LLM ---
        l2_fatals, l2_tolerables = await self._layer2_vision_llm(scene, clip_path)
        all_fatals.extend(l2_fatals)
        all_tolerables.extend(l2_tolerables)

        result = self._build_verdict(scene.scene_id, all_fatals, all_tolerables, clip_path=str(clip_path))

        status = "PASS" if result.passed else "REGEN"
        logger.info("  %s: %s — F:%d T:%d", scene.scene_id, status,
                    len(result.fatals), len(result.tolerables))
        return result

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    async def _layer1_temporal(self, clip_path: Path) -> tuple[list[str], list[str]]:
        """Layer 1: run frame_analyzer SSIM checks, return (fatals, tolerables)."""
        from videoclaw.drama.frame_analyzer import (
            extract_frames_as_arrays,
            detect_temporal_breaks,
        )

        fatals: list[str] = []
        tolerables: list[str] = []
        try:
            frames = extract_frames_as_arrays(clip_path, n=10)
            breaks = detect_temporal_breaks(
                frames,
                fatal_threshold=_FATAL_THRESHOLD,
                tolerable_threshold=_TOLERABLE_THRESHOLD,
            )
            for brk in breaks:
                label = (
                    f"temporal_break_f{brk.frame_pair[0]}_f{brk.frame_pair[1]}"
                    f"_ssim{brk.ssim_score:.2f}"
                )
                if brk.severity == "fatal":
                    fatals.append(label)
                else:
                    tolerables.append(label)
        except Exception as exc:
            logger.warning("Layer 1 frame analysis failed for %s: %s", clip_path.name, exc)
            tolerables.append("layer1_analysis_unavailable")
        return fatals, tolerables

    async def _layer2_vision_llm(
        self,
        scene: DramaScene,
        clip_path: Path,
    ) -> tuple[list[str], list[str]]:
        """Layer 2: Claude Vision audit with V2 pragmatic prompts, return (fatals, tolerables)."""
        fatals: list[str] = []
        tolerables: list[str] = []

        frames = self.extract_keyframes(clip_path)
        if not frames:
            logger.warning("Layer 2: no frames extracted for %s", clip_path.name)
            return fatals, tolerables

        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in frames
        ]
        content.append({"type": "text", "text": _AUDIT_USER_TEMPLATE_V2.format(
            scene_id=scene.scene_id,
            description=scene.description or "",
            time_of_day=scene.time_of_day or "unspecified",
            characters=(
                ", ".join(scene.characters_present)
                if scene.characters_present
                else "none specified"
            ),
            dialogue=(scene.dialogue[:80] + "\u2026" if len(scene.dialogue) > 80 else scene.dialogue),
            shot_role=scene.shot_role or "normal",
            emotion=scene.emotion or "",
            shot_scale=scene.shot_scale.value if scene.shot_scale else "unspecified",
            frame_count=len(frames),
        )})

        messages = [
            {"role": "system", "content": _AUDIT_SYSTEM_V2},
            {"role": "user", "content": content},
        ]

        llm = self._ensure_llm()
        try:
            raw = await llm.chat(messages, model=VISION_MODEL, temperature=0.1, max_tokens=1024)
        except Exception as exc:
            logger.error("Vision API error for %s: %s", scene.scene_id, exc)
            return fatals, tolerables

        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            data = json.loads(text.strip())
            fatals = list(data.get("fatals", []))
            tolerables = list(data.get("tolerables", []))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse Layer 2 audit JSON for %s: %s", scene.scene_id, exc)

        return fatals, tolerables

    def _build_verdict(
        self,
        shot_id: str,
        fatals: list[str],
        tolerables: list[str],
        clip_path: str = "",
    ) -> ShotAuditResult:
        """Decision rule: fatal -> REGEN, <=2 tolerable -> PASS, >=3 tolerable -> REGEN."""
        if fatals:
            return ShotAuditResult(
                shot_id=shot_id,
                passed=False,
                regen_required=True,
                fatals=list(fatals),
                tolerables=list(tolerables),
                issues=fatals + tolerables,
                clip_path=clip_path,
            )
        if len(tolerables) >= 3:
            return ShotAuditResult(
                shot_id=shot_id,
                passed=False,
                regen_required=True,
                fatals=list(fatals),
                tolerables=list(tolerables),
                issues=tolerables,
                clip_path=clip_path,
            )
        return ShotAuditResult(
            shot_id=shot_id,
            passed=True,
            regen_required=False,
            fatals=list(fatals),
            tolerables=list(tolerables),
            issues=tolerables if tolerables else [],
            clip_path=clip_path,
        )

    # ------------------------------------------------------------------
    # Episode-level audit (standalone mode)
    # ------------------------------------------------------------------

    async def audit_clip_dir(
        self,
        scenes: list[DramaScene],
        clip_dir: Path,
        *,
        series: DramaSeries | None = None,
        incremental: bool = False,
    ) -> EpisodeAuditReport:
        """Audit all scenes by discovering clips in *clip_dir*.

        Uses :func:`resolve_clip` for smart path resolution — automatically
        finds ``session{N}_{scene_id}.mp4`` variants and picks the newest.

        Parameters
        ----------
        scenes:
            Ordered list of :class:`DramaScene` for the episode.
        clip_dir:
            Directory to search for MP4 files.
        series:
            Optional series context for richer audit prompts.
        incremental:
            When ``True``, skip scenes whose ``audit_result`` already shows
            ``passed=True`` and ``regen_required=False`` — carry them over
            as PASS without re-auditing.
        """
        report = EpisodeAuditReport(
            series_id=series.series_id if series else "",
            episode_number=1,
            total_shots=len(scenes),
        )

        for scene in scenes:
            # Incremental: skip previously passed scenes
            if incremental and scene.audit_result:
                ar = scene.audit_result
                if ar.get("passed") and not ar.get("regen_required"):
                    result = ShotAuditResult(
                        shot_id=scene.scene_id,
                        passed=True,
                        regen_required=False,
                        fatals=list(ar.get("fatals", [])),
                        tolerables=list(ar.get("tolerables", [])),
                        clip_path=ar.get("clip_path", ""),
                    )
                    report.shot_results.append(result)
                    report.passed_shots += 1
                    logger.info("  %s: incremental skip (already passed)", scene.scene_id)
                    continue

            clip_path = resolve_clip(scene.scene_id, clip_dir, scene.video_asset_path)
            if clip_path is None:
                logger.warning("No clip found for %s in %s — skipping", scene.scene_id, clip_dir)
                result = ShotAuditResult.error_result(
                    scene.scene_id, f"no clip found in {clip_dir}"
                )
            else:
                result = await self.audit_shot(scene, clip_path, series=series)

            report.shot_results.append(result)
            if result.passed:
                report.passed_shots += 1
            if result.regen_required:
                report.regen_required.append(scene.scene_id)

        return report

    # ------------------------------------------------------------------
    # Series-aware mode (integrates with drama manager)
    # ------------------------------------------------------------------

    async def audit_series_episode(
        self,
        series: DramaSeries,
        episode_number: int = 1,
        *,
        clip_dir: Path | None = None,
        drama_manager: DramaManager | None = None,
        persist_results: bool = True,
        incremental: bool = False,
    ) -> EpisodeAuditReport:
        """Audit an episode from a :class:`DramaSeries` managed by :class:`DramaManager`.

        Clip discovery order for each scene:
        1. ``scene.video_asset_path`` — set by the drama runner
        2. *clip_dir* override — if provided
        3. ``{series_dir}/video_clips/`` — standard location relative to series

        When *persist_results* is ``True`` and *drama_manager* is provided,
        the audit result for each scene is written to
        ``DramaScene.audit_result`` and persisted via ``drama_manager.save()``.

        Parameters
        ----------
        series:
            The :class:`DramaSeries` to audit.
        episode_number:
            Episode number to audit (1-based).
        clip_dir:
            Override clip directory.  When ``None``, the auditor infers the
            directory from ``scene.video_asset_path`` or the series path.
        drama_manager:
            If provided, audit results are persisted back to the series state.
        persist_results:
            Whether to write audit results into ``DramaScene.audit_result``.
        incremental:
            When ``True``, skip scenes whose ``audit_result`` shows
            ``passed=True`` and ``regen_required=False``.
        """
        from videoclaw.config import get_config

        episode = next(
            (e for e in series.episodes if e.number == episode_number), None
        )
        if episode is None:
            raise ValueError(
                f"Episode {episode_number} not found in series {series.series_id!r}"
            )

        # Infer clip dir from series path when not overridden
        effective_clip_dir: Path | None = clip_dir
        if effective_clip_dir is None:
            series_dir = get_config().projects_dir / "dramas" / series.series_id
            candidate = series_dir / "video_clips"
            if candidate.is_dir():
                effective_clip_dir = candidate
                logger.info("Using inferred clip dir: %s", effective_clip_dir)

        report = EpisodeAuditReport(
            series_id=series.series_id,
            episode_number=episode_number,
            total_shots=len(episode.scenes),
        )

        for scene in episode.scenes:
            # Incremental: skip already-passed shots
            if incremental and scene.audit_result:
                prev = scene.audit_result
                if prev.get("passed") and not prev.get("regen_required"):
                    logger.info("  %s: incremental skip (already passed)", scene.scene_id)
                    result = ShotAuditResult(
                        shot_id=scene.scene_id,
                        passed=True,
                        fatals=prev.get("fatals", []),
                        tolerables=prev.get("tolerables", []),
                        clip_path=prev.get("clip_path", ""),
                    )
                    report.shot_results.append(result)
                    report.passed_shots += 1
                    continue

            # Resolve clip: video_asset_path > clip_dir > glob
            cp = resolve_clip(
                scene.scene_id,
                effective_clip_dir or Path("."),
                scene.video_asset_path,
            )
            if cp is None:
                logger.warning(
                    "No clip found for %s — skipping (video_asset_path=%r, clip_dir=%s)",
                    scene.scene_id, scene.video_asset_path, effective_clip_dir,
                )
                result = ShotAuditResult.error_result(
                    scene.scene_id, "clip not found"
                )
            else:
                result = await self.audit_shot(scene, cp, series=series)

            report.shot_results.append(result)
            if result.passed:
                report.passed_shots += 1
            if result.regen_required:
                report.regen_required.append(scene.scene_id)

            # Persist audit result back into scene state
            if persist_results:
                scene.audit_result = result.to_dict()

        # Save updated series state
        if persist_results and drama_manager is not None:
            drama_manager.save(series)
            logger.info("Audit results persisted to series %s", series.series_id)

        return report

    # ------------------------------------------------------------------
    # Composition-level audit (成片审计)
    # ------------------------------------------------------------------

    async def audit_composition(
        self,
        video_path: Path,
        episode_number: int,
        total_duration: float | None = None,
        alignment_report: dict | None = None,
    ) -> ShotAuditResult:
        """Audit a final composed video with duration-adaptive strategy.

        For short videos (< 30s): extract 5 frames, full LLM audit.
        For medium videos (30-90s): extract 8 frames, full LLM audit.
        For long videos (> 90s): extract 10 frames, SSIM check only + LLM
            on first/last 2 frames (performance optimization).

        Parameters
        ----------
        alignment_report:
            Optional alignment report dict from pre-compose phase.
            If present, misaligned scenes are surfaced as tolerables.

        Returns a ShotAuditResult with the composition-level verdict.
        """
        from videoclaw.drama.frame_analyzer import (
            detect_temporal_breaks,
            extract_frames_as_arrays,
        )

        shot_id = f"composition_ep{episode_number:02d}"

        # Get duration
        duration = total_duration
        if duration is None:
            duration = await get_video_duration(video_path)

        # Determine frame count by duration bracket
        if duration < 30:
            n_frames = 5
        elif duration <= 90:
            n_frames = 8
        else:
            n_frames = 10

        logger.info(
            "Composition audit: %s — %.0fs, extracting %d frames",
            video_path.name, duration, n_frames,
        )

        # Extract frames
        frames = extract_frames_as_arrays(video_path, n=n_frames)

        # Layer 0: Duration alignment check (from pre-compose alignment report)
        all_fatals: list[str] = []
        all_tolerables: list[str] = []

        if alignment_report and not alignment_report.get("is_aligned", True):
            total_drift = alignment_report.get("total_drift", 0.0)
            misaligned_ids = alignment_report.get("misaligned_scene_ids", [])
            for clip_info in alignment_report.get("clips", []):
                if clip_info.get("drift", 0.0) > 1.0:
                    label = (
                        f"duration_drift_{clip_info['scene_id']}"
                        f"_scripted{clip_info['scripted']:.1f}s"
                        f"_actual{clip_info['actual']:.1f}s"
                    )
                    # Drift > 3s is fatal (severe misalignment), else tolerable
                    if clip_info["drift"] > 3.0:
                        all_fatals.append(label)
                    else:
                        all_tolerables.append(label)
            logger.info(
                "[audit] Duration alignment: %d misaligned scenes, "
                "total drift %.1fs: %s",
                len(misaligned_ids), total_drift,
                ", ".join(misaligned_ids),
            )

        # Layer 1: SSIM temporal check on ALL extracted frames

        breaks = detect_temporal_breaks(
            frames,
            fatal_threshold=_FATAL_THRESHOLD,
            tolerable_threshold=_TOLERABLE_THRESHOLD,
        )
        for brk in breaks:
            label = (
                f"composition_temporal_break_f{brk.frame_pair[0]}_f{brk.frame_pair[1]}"
                f"_ssim{brk.ssim_score:.2f}"
            )
            if brk.severity == "fatal":
                all_fatals.append(label)
            else:
                all_tolerables.append(label)

        # LLM audit — for long videos, only send first 2 + last 2 frames
        if duration > 90:
            llm_frames = frames[:2] + frames[-2:]
        else:
            llm_frames = frames

        l2_fatals, l2_tolerables = await self._composition_vision_llm(
            llm_frames, duration,
        )
        all_fatals.extend(l2_fatals)
        all_tolerables.extend(l2_tolerables)

        result = self._build_verdict(
            shot_id, all_fatals, all_tolerables, clip_path=str(video_path),
        )
        status = "PASS" if result.passed else "REGEN"
        logger.info(
            "Composition %s: %s — F:%d T:%d",
            shot_id, status, len(result.fatals), len(result.tolerables),
        )
        return result

    async def _composition_vision_llm(
        self,
        frames: list,
        duration: float,
    ) -> tuple[list[str], list[str]]:
        """Run LLM audit on composition frames, return (fatals, tolerables)."""
        fatals: list[str] = []
        tolerables: list[str] = []

        # Encode frames to base64 for the vision model
        from PIL import Image

        content: list[dict[str, Any]] = []
        for frame in frames:
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        content.append({"type": "text", "text": _COMPOSITION_AUDIT_PROMPT.format(
            frame_count=len(frames),
            duration=duration,
        )})

        messages = [
            {"role": "system", "content": _AUDIT_SYSTEM_V2},
            {"role": "user", "content": content},
        ]

        llm = self._ensure_llm()
        try:
            raw = await llm.chat(messages, model=VISION_MODEL, temperature=0.1, max_tokens=1024)
        except Exception as exc:
            logger.error("Vision API error for composition audit: %s", exc)
            return fatals, tolerables

        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            data = json.loads(text.strip())
            fatals = list(data.get("fatals", []))
            tolerables = list(data.get("tolerables", []))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse composition audit JSON: %s", exc)

        return fatals, tolerables
