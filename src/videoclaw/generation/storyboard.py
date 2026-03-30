"""Storyboard generator -- decomposes a script into visual shots.

The :class:`StoryboardGenerator` takes a :class:`Script` and uses an LLM to
produce a list of :class:`Shot` objects, each carrying a generation-ready
prompt optimised for AI video models (rich in camera, lighting, and motion
detail).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from videoclaw.config import get_config
from videoclaw.core.events import TASK_COMPLETED, TASK_STARTED, event_bus
from videoclaw.core.state import Shot, ShotStatus
from videoclaw.generation.script import Script
from videoclaw.models.llm.litellm_wrapper import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

PROMPT_OPTIMIZATION_SYSTEM: str = """\
You are an expert at writing prompts for AI video generation models,
specialised in Seedance 2.0, Kling, Sora, Runway Gen-3, and similar.

Given a script with sections, produce a shot list where each shot has a
generation-ready prompt specifically tailored for text-to-video models.

# Duration constraints (Seedance 2.0)
- Each shot's duration_seconds MUST be between 4 and 15 seconds.
- Never generate shots shorter than 4s or longer than 15s.
- Fast-paced action: use 4-5s per shot. Slow establishing shots: 8-12s.

# Prompt-writing rules
- Start with the primary subject and action.
- Describe the environment / background in detail.
- Specify lighting conditions (golden hour, overcast, neon, studio, natural).
- Include color palette or mood adjectives (warm tones, desaturated, vibrant).
- State the camera angle explicitly: eye-level, low-angle, high-angle,
  bird's-eye, dutch-angle, over-the-shoulder.
- State the camera movement: static, slow pan left, dolly forward, tracking
  shot following subject, smooth crane rising, handheld shake.
  Use ONE motion verb per shot — never stack multiple movements.
- Mention the shot scale: extreme close-up, close-up, medium shot, full shot,
  wide shot, extreme wide shot.
- Add temporal cues: slow motion, time-lapse, normal speed.
- Write a negative prompt listing things to exclude (blur, distortion,
  watermark, text overlay, low quality, deformed hands).
- Keep prompts in English regardless of the script language -- all major video
  models perform best with English prompts.

# Material description rules
- Characters must have a complete appearance description on first appearance
  (age, gender, body type, hair style/color, clothing, accessories, distinctive marks).
- Objects should describe material, color, size, and key visual features.
- Maintain visual consistency: the same character across shots must use
  identical visual keywords.

# Seedance 2.0 Universal Reference (全能参考)
- When character reference images are available, the model uses Universal
  Reference mode for character consistency. Prompts should still contain full
  character appearance keywords to supplement the reference images.

# Background audio considerations
- Note ambient sounds and environmental audio cues for each shot
  (wind, traffic, crowd murmur, rain, silence, etc.).
- Sound effects should serve the narrative rhythm: sharp/sudden sounds for
  tension, soft ambient noise for calm scenes.
- Record sound design notes in the sfx field.

Output JSON schema (array of objects):
[
  {
    "shot_id": "<unique id>",
    "prompt": "<optimised generation prompt in English>",
    "negative_prompt": "<things to exclude>",
    "description": "<brief human-readable description>",
    "duration_seconds": <float, 5-15>,
    "suggested_model": "<seedance-2.0 | kling | runway | sora | auto>",
    "visual_style": "<cinematic | photorealistic | anime | etc.>",
    "camera_movement": "<static | pan_left | dolly_in | etc.>",
    "sfx": "<sound effects for this shot, e.g. footsteps, door slam, thunder>",
    "notes": "<special considerations: character traits, mood, pacing>"
  }
]

Return ONLY the JSON array -- no markdown fences, no commentary.
"""


# ---------------------------------------------------------------------------
# Default negative prompt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Aspect ratio -> resolution mapping (shared with director; duplicated for
# independence)
# ---------------------------------------------------------------------------

_ASPECT_TO_RESOLUTION: dict[str, tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1024, 1024),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "21:9": (1280, 549),
}


# ---------------------------------------------------------------------------
# StoryboardGenerator
# ---------------------------------------------------------------------------

class StoryboardGenerator:
    """Decomposes a :class:`Script` into generation-ready :class:`Shot` objects.

    Parameters
    ----------
    llm:
        An :class:`LLMClient` instance.  When *None*, a default client is
        created on first use.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def _ensure_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(default_model=get_config().default_llm)
        return self._llm

    async def decompose(
        self,
        script: Script,
        style: str = "cinematic",
        aspect_ratio: str = "16:9",
    ) -> list[Shot]:
        """Break *script* into a list of :class:`Shot` objects.

        Parameters
        ----------
        script:
            The structured script to decompose.
        style:
            Overarching visual style applied to all shots.
        aspect_ratio:
            Target aspect ratio (used to select resolution metadata).

        Returns
        -------
        list[Shot]
            Generation-ready shots with optimised prompts.
        """
        await event_bus.emit(
            TASK_STARTED,
            {
                "task": "storyboard.decompose",
                "title": script.title,
                "section_count": len(script.sections),
            },
        )
        logger.info(
            "Decomposing script into storyboard: title=%r style=%s",
            script.title,
            style,
        )

        llm = self._ensure_llm()
        user_message = self._build_user_message(script, style, aspect_ratio)

        raw_response = await llm.chat(
            messages=[
                {"role": "system", "content": PROMPT_OPTIMIZATION_SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )

        shots = self._parse_shots(raw_response, aspect_ratio)

        await event_bus.emit(
            TASK_COMPLETED,
            {
                "task": "storyboard.decompose",
                "shot_count": len(shots),
            },
        )
        logger.info("Storyboard produced %d shots", len(shots))
        return shots

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _build_user_message(
        script: Script,
        style: str,
        aspect_ratio: str,
    ) -> str:
        """Assemble the LLM user prompt from the script data."""
        width, height = _ASPECT_TO_RESOLUTION.get(aspect_ratio, (1280, 720))

        sections_text = "\n".join(
            f"[Section {s.section_id}] ({s.duration_seconds}s)\n"
            f"  Narration: {s.text}\n"
            f"  Visual: {s.visual_description}\n"
            f"  Notes: {s.notes}"
            for s in script.sections
        )

        return (
            f"Video title: {script.title}\n"
            f"Visual style: {style}\n"
            f"Target resolution: {width}x{height} ({aspect_ratio})\n"
            f"Total duration: {script.total_duration}s\n\n"
            f"Script sections:\n{sections_text}\n\n"
            "Produce the shot list as a JSON array."
        )

    @staticmethod
    def _parse_shots(raw_response: str, aspect_ratio: str) -> list[Shot]:
        """Parse LLM output into :class:`Shot` objects."""
        text = raw_response.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data: list[dict[str, Any]] = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse storyboard LLM response: %s", exc)
            logger.debug("Raw response:\n%s", raw_response)
            raise ValueError(
                "Storyboard LLM returned invalid JSON. Retry the request."
            ) from exc

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array from the LLM, got {type(data).__name__}"
            )

        shots: list[Shot] = []
        for entry in data:
            combined_prompt = entry.get("prompt", "")

            # Clamp duration to Seedance 2.0 range (5-15s)
            raw_dur = float(entry.get("duration_seconds", 5.0))
            clamped_dur = max(5.0, min(15.0, raw_dur))

            shot = Shot(
                shot_id=entry.get("shot_id", uuid.uuid4().hex[:12]),
                description=entry.get("description", ""),
                prompt=combined_prompt,
                duration_seconds=clamped_dur,
                model_id=entry.get("suggested_model", "auto"),
                status=ShotStatus.PENDING,
            )
            shots.append(shot)

        return shots
