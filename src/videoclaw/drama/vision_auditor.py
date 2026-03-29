"""Vision auditor — Claude Vision QA for generated video clips.

Extracts keyframes from each generated clip and sends them to Claude Vision
for per-shot quality inspection.  Checks performed:

- ``time_of_day``: lighting consistency with script spec (day vs. night)
- ``characters``: expected characters visible and consistent with turnarounds
- ``subtitle_spelling``: visible on-screen text free of OCR-style errors
- ``artifacts``: obvious generation artifacts (anatomical distortions, blurring)
- ``dramatic_tension``: hook / cliffhanger quality for first / last shots

Usage::

    auditor = VisionAuditor()
    report = await auditor.audit_episode(episode, series, clip_dir)
    for r in report.shot_results:
        print(r.shot_id, "PASS" if r.passed else "FAIL", r.issues)
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from videoclaw.config import get_config
from videoclaw.drama.models import DramaScene, DramaSeries
from videoclaw.models.llm.litellm_wrapper import LLMClient

logger = logging.getLogger(__name__)

# Model to use for vision — sonnet is capable and cheaper than opus
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
    raw_response: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any], shot_id: str) -> ShotAuditResult:
        return cls(
            shot_id=data.get("shot_id", shot_id),
            passed=bool(data.get("passed", True)),
            regen_required=bool(data.get("regen_required", False)),
            regen_note=data.get("regen_note", ""),
            issues=list(data.get("issues", [])),
            checks=dict(data.get("checks", {})),
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
            lines.append(f"  {r.shot_id}: {status}{regen}{issues}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "episode_number": self.episode_number,
            "total_shots": self.total_shots,
            "passed_shots": self.passed_shots,
            "regen_required": self.regen_required,
            "shot_results": [
                {
                    "shot_id": r.shot_id,
                    "passed": r.passed,
                    "regen_required": r.regen_required,
                    "regen_note": r.regen_note,
                    "issues": r.issues,
                    "checks": r.checks,
                }
                for r in self.shot_results
            ],
        }


# ---------------------------------------------------------------------------
# VisionAuditor
# ---------------------------------------------------------------------------

class VisionAuditor:
    """Audits generated video clips using Claude Vision.

    Parameters
    ----------
    llm:
        Optional pre-built LLMClient. Uses ``claude-sonnet-4-6`` by default.
    frame_count:
        Number of keyframes to extract per clip (1-5). Default 3.
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
        """Extract ``frame_count`` frames from *clip_path*, return base64 JPEGs.

        Uses FFmpeg ``select`` filter to pick evenly-spaced frames.
        Falls back to thumbnail filter if select fails.
        """
        frames: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Get video duration first
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    str(clip_path),
                ],
                capture_output=True, text=True,
            )
            duration = 5.0
            if probe.returncode == 0 and probe.stdout.strip():
                try:
                    duration = float(probe.stdout.strip())
                except ValueError:
                    pass

            # Extract frames at 0.1s, mid, and (duration-0.5)s
            timestamps = self._frame_timestamps(duration)

            for i, ts in enumerate(timestamps):
                out_file = tmp_path / f"frame_{i:02d}.jpg"
                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", f"{ts:.2f}",
                        "-i", str(clip_path),
                        "-vframes", "1",
                        "-q:v", "3",
                        "-vf", "scale=720:-1",
                        str(out_file),
                    ],
                    capture_output=True,
                )
                if result.returncode == 0 and out_file.exists():
                    raw = out_file.read_bytes()
                    frames.append(base64.b64encode(raw).decode())
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
        # 3 frames: start, mid, end
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
        """Audit a single clip against its scene spec.

        Returns a :class:`ShotAuditResult`.
        """
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

        # Build multimodal message
        content: list[dict[str, Any]] = []
        for b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        prompt = _AUDIT_USER_TEMPLATE.format(
            scene_id=scene.scene_id,
            description=scene.description or "",
            time_of_day=scene.time_of_day or "unspecified",
            characters=", ".join(scene.characters_present) if scene.characters_present else "none specified",
            dialogue=scene.dialogue[:80] + "…" if len(scene.dialogue) > 80 else scene.dialogue,
            shot_role=scene.shot_role or "normal",
            emotion=scene.emotion or "",
            frame_count=len(frames),
        )
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": _AUDIT_SYSTEM},
            {"role": "user", "content": content},
        ]

        llm = self._ensure_llm()
        try:
            raw = await llm.chat(
                messages,
                model=VISION_MODEL,
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception as exc:
            logger.error("Vision API error for %s: %s", scene.scene_id, exc)
            return ShotAuditResult.error_result(scene.scene_id, str(exc))

        # Parse JSON response
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            data = json.loads(text.strip())
            result = ShotAuditResult.from_json(data, scene.scene_id)
            result.raw_response = raw
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse audit JSON for %s: %s\nRaw: %s", scene.scene_id, exc, raw[:500])
            result = ShotAuditResult.error_result(scene.scene_id, f"JSON parse error: {exc}")
            result.raw_response = raw

        status = "PASS" if result.passed else "FAIL"
        regen = " [REGEN NEEDED]" if result.regen_required else ""
        logger.info("  %s: %s%s — %s", scene.scene_id, status, regen, "; ".join(result.issues) or "clean")
        return result

    # ------------------------------------------------------------------
    # Episode audit
    # ------------------------------------------------------------------

    async def audit_episode(
        self,
        scenes: list[DramaScene],
        clip_dir: Path,
        *,
        series: DramaSeries | None = None,
        clip_prefix: str = "session5_",
        session6_scenes: list[str] | None = None,
    ) -> EpisodeAuditReport:
        """Audit all scenes in an episode sequentially.

        Parameters
        ----------
        scenes:
            Ordered list of :class:`DramaScene` for the episode.
        clip_dir:
            Directory containing the generated MP4 files.
        series:
            Optional series context for richer prompts.
        clip_prefix:
            Filename prefix for clips (e.g. ``"session5_"``).
        session6_scenes:
            Scene IDs that were re-generated in Session 6 (use ``"session6_"`` prefix).
        """
        report = EpisodeAuditReport(
            series_id=series.series_id if series else "",
            episode_number=1,
            total_shots=len(scenes),
        )
        session6_scenes = session6_scenes or []

        for scene in scenes:
            # Resolve clip path: session6 overrides session5 for specific scenes
            if scene.scene_id in session6_scenes:
                clip_path = clip_dir / f"session6_{scene.scene_id}.mp4"
            else:
                clip_path = clip_dir / f"{clip_prefix}{scene.scene_id}.mp4"

            result = await self.audit_shot(scene, clip_path, series=series)
            report.shot_results.append(result)
            if result.passed:
                report.passed_shots += 1
            if result.regen_required:
                report.regen_required.append(scene.scene_id)

        return report
