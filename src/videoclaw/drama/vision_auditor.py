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
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from videoclaw.drama.models import DramaScene, DramaSeries
from videoclaw.models.llm.litellm_wrapper import LLMClient

if TYPE_CHECKING:
    from videoclaw.drama.models import DramaManager

logger = logging.getLogger(__name__)

# Vision model — sonnet is multimodal capable and cost-efficient
VISION_MODEL = "claude-sonnet-4-6"

_AUDIT_SYSTEM = """\
You are a professional AI video quality auditor for TikTok-style short dramas.
You will be shown keyframes from a video clip alongside the scene specification.
Evaluate the clip against the spec and return a structured JSON audit result.

Be concise and specific. Focus on production-blocking issues, not minor imperfections.
Return ONLY valid JSON matching the schema given in the user message.
"""

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
            status = "PASS" if r.passed else "FAIL"
            regen = " [REGEN]" if r.regen_required else ""
            issues = f" — {'; '.join(r.issues)}" if r.issues else ""
            clip = f" ({Path(r.clip_path).name})" if r.clip_path else " (clip not found)"
            lines.append(f"  {r.shot_id}: {status}{regen}{clip}{issues}")
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
    # Per-shot audit
    # ------------------------------------------------------------------

    async def audit_shot(
        self,
        scene: DramaScene,
        clip_path: Path,
        *,
        series: DramaSeries | None = None,
    ) -> ShotAuditResult:
        """Audit a single clip against its scene specification."""
        if not clip_path.exists():
            return ShotAuditResult.error_result(
                scene.scene_id, f"clip not found: {clip_path}"
            )

        logger.info("Auditing %s (%s)...", scene.scene_id, clip_path.name)

        frames = self.extract_keyframes(clip_path)
        if not frames:
            return ShotAuditResult.error_result(
                scene.scene_id, "frame extraction produced no frames"
            )

        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in frames
        ]
        content.append({"type": "text", "text": _AUDIT_USER_TEMPLATE.format(
            scene_id=scene.scene_id,
            description=scene.description or "",
            time_of_day=scene.time_of_day or "unspecified",
            characters=(
                ", ".join(scene.characters_present)
                if scene.characters_present
                else "none specified"
            ),
            dialogue=(scene.dialogue[:80] + "…" if len(scene.dialogue) > 80 else scene.dialogue),
            shot_role=scene.shot_role or "normal",
            emotion=scene.emotion or "",
            frame_count=len(frames),
        )})

        messages = [
            {"role": "system", "content": _AUDIT_SYSTEM},
            {"role": "user", "content": content},
        ]

        llm = self._ensure_llm()
        try:
            raw = await llm.chat(messages, model=VISION_MODEL, temperature=0.1, max_tokens=1024)
        except Exception as exc:
            logger.error("Vision API error for %s: %s", scene.scene_id, exc)
            return ShotAuditResult.error_result(scene.scene_id, str(exc))

        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            data = json.loads(text.strip())
            result = ShotAuditResult.from_json(data, scene.scene_id, str(clip_path))
            result.raw_response = raw
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse audit JSON for %s: %s", scene.scene_id, exc)
            result = ShotAuditResult.error_result(scene.scene_id, f"JSON parse error: {exc}")
            result.raw_response = raw

        status = "PASS" if result.passed else "FAIL"
        regen = " [REGEN NEEDED]" if result.regen_required else ""
        logger.info("  %s: %s%s — %s", scene.scene_id, status, regen,
                    "; ".join(result.issues) or "clean")
        return result

    # ------------------------------------------------------------------
    # Episode-level audit (standalone mode)
    # ------------------------------------------------------------------

    async def audit_clip_dir(
        self,
        scenes: list[DramaScene],
        clip_dir: Path,
        *,
        series: DramaSeries | None = None,
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
        """
        report = EpisodeAuditReport(
            series_id=series.series_id if series else "",
            episode_number=1,
            total_shots=len(scenes),
        )

        for scene in scenes:
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
