"""Director Agent -- the creative brain of VideoClaw.

The Director receives raw user intent (a text prompt) and produces a fully
structured :class:`ProjectState` containing a scene-by-scene production plan
ready for downstream generation modules.  It delegates creative decisions to
an LLM while enforcing structural constraints required by the pipeline.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from videoclaw.config import get_config
from videoclaw.core.events import event_bus, TASK_STARTED, TASK_COMPLETED
from videoclaw.core.state import ProjectState, Shot, ShotStatus
from videoclaw.models.llm.litellm_wrapper import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

DIRECTOR_SYSTEM_PROMPT: str = """\
You are an expert video director working inside an AI video generation system.
Your role is to take a user's creative intent and produce a precise, structured
production plan that downstream AI modules will execute.

Guidelines:
- Break the video into discrete scenes, each 3-8 seconds long.
- Every scene must have a clear visual description rich enough for an AI
  video-generation model (include subject, action, environment, lighting,
  color palette, camera angle, and camera movement).
- Choose camera movements that serve the narrative: static, pan, tilt, dolly,
  zoom, tracking, crane, or handheld.
- Write voice-over text that matches the total requested duration when read
  at a natural pace (~150 words per minute for Chinese, ~130 wpm for English).
- Select a music style, tempo (BPM), and mood that complement the content.
- Respect the requested total duration by distributing time across scenes.
- Return ONLY valid JSON -- no markdown fences, no commentary.

Output JSON schema:
{
  "title": "<short video title>",
  "description": "<one-paragraph summary>",
  "scenes": [
    {
      "scene_id": "<unique id, e.g. scene_01>",
      "description": "<detailed visual description for AI generation>",
      "duration": <seconds as float>,
      "visual_style": "<e.g. cinematic, anime, photorealistic, watercolor>",
      "camera_movement": "<static | pan_left | pan_right | tilt_up | tilt_down | dolly_in | dolly_out | zoom_in | zoom_out | tracking | crane_up | crane_down | handheld>"
    }
  ],
  "voice_over": {
    "text": "<full narration script>",
    "tone": "<warm | authoritative | playful | dramatic | calm | energetic>",
    "language": "<zh | en | ja | ...>"
  },
  "music": {
    "style": "<e.g. orchestral, lo-fi, electronic, acoustic>",
    "tempo": <BPM as int>,
    "mood": "<e.g. uplifting, melancholic, tense, serene>"
  }
}
"""

REFINE_SYSTEM_PROMPT: str = """\
You are a senior video director reviewing AI-generated footage.  Given the
original generation prompt and reviewer feedback, produce an improved prompt
that addresses the feedback while preserving the original creative intent.
Return ONLY the improved prompt text -- no explanation, no JSON wrapping.
"""

# ---------------------------------------------------------------------------
# Aspect ratio -> resolution mapping
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
# Director Agent
# ---------------------------------------------------------------------------

class Director:
    """Converts user intent into a complete production plan.

    The Director wraps an LLM call with a carefully crafted system prompt and
    post-processes the structured output into VideoClaw's internal state model.

    Parameters
    ----------
    llm:
        An :class:`LLMClient` instance.  When *None*, a default client is
        created on first use.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm
        self._initialized = False

    # -- lazy init ----------------------------------------------------------

    def _ensure_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(default_model=get_config().default_llm)
        return self._llm

    # -- public API ---------------------------------------------------------

    async def plan(
        self,
        prompt_or_state: str | ProjectState,
        duration: float = 30.0,
        style: str | None = None,
        aspect_ratio: str = "16:9",
        preferred_model: str | None = None,
    ) -> ProjectState:
        """Generate a full production plan from a user prompt.

        Parameters
        ----------
        prompt_or_state:
            Either a free-form description string or an existing
            :class:`ProjectState` (whose ``prompt`` field will be used).
        duration:
            Target video length in seconds.
        style:
            Optional visual style hint (e.g. ``"cinematic"``, ``"anime"``).
        aspect_ratio:
            Target aspect ratio key (see :data:`_ASPECT_TO_RESOLUTION`).
        preferred_model:
            If set, all shots will default to this model adapter id.

        Returns
        -------
        ProjectState
            A fully populated project state with shots ready for generation.
        """
        if isinstance(prompt_or_state, ProjectState):
            existing_state = prompt_or_state
            prompt = existing_state.prompt
        else:
            existing_state = None
            prompt = prompt_or_state

        await event_bus.emit(TASK_STARTED, {"task": "director.plan", "prompt": prompt})
        logger.info("Director planning: prompt=%r duration=%.1fs style=%s", prompt, duration, style)

        llm = self._ensure_llm()

        user_message = self._build_user_message(prompt, duration, style, aspect_ratio)
        raw_response = await llm.chat(
            messages=[
                {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        plan_data = self._parse_plan(raw_response)

        if existing_state is not None:
            # Update existing project state in-place.
            state = existing_state
            self._populate_state(state, plan_data, aspect_ratio, preferred_model)
        else:
            state = self._build_project_state(
                plan_data=plan_data,
                original_prompt=prompt,
                aspect_ratio=aspect_ratio,
                preferred_model=preferred_model,
            )

        await event_bus.emit(
            TASK_COMPLETED,
            {
                "task": "director.plan",
                "project_id": state.project_id,
                "shot_count": len(state.storyboard),
            },
        )
        logger.info(
            "Director produced plan: project_id=%s shots=%d",
            state.project_id,
            len(state.storyboard),
        )
        return state

    async def refine_prompt(self, original_prompt: str, feedback: str) -> str:
        """Improve a generation prompt based on reviewer feedback.

        Parameters
        ----------
        original_prompt:
            The prompt that was used to generate the shot.
        feedback:
            Textual feedback describing what should change.

        Returns
        -------
        str
            An improved generation prompt.
        """
        llm = self._ensure_llm()

        user_message = (
            f"Original prompt:\n{original_prompt}\n\n"
            f"Reviewer feedback:\n{feedback}\n\n"
            "Please produce an improved prompt."
        )

        refined = await llm.chat(
            messages=[
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return refined.strip()

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _build_user_message(
        prompt: str,
        duration: float,
        style: str | None,
        aspect_ratio: str,
    ) -> str:
        """Construct the user message sent to the LLM."""
        parts = [
            f"Create a video production plan for the following concept:\n\n{prompt}",
            f"\nTarget duration: {duration} seconds",
            f"Aspect ratio: {aspect_ratio}",
        ]
        if style:
            parts.append(f"Visual style: {style}")
        parts.append(
            "\nReturn the production plan as a single JSON object following "
            "the schema described in your instructions."
        )
        return "\n".join(parts)

    @staticmethod
    def _parse_plan(raw_response: str) -> dict[str, Any]:
        """Parse the LLM's raw text into a plan dictionary.

        Handles common LLM formatting quirks such as markdown code fences.
        """
        text = raw_response.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Remove opening fence (possibly with language tag)
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()

        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Director LLM response as JSON: %s", exc)
            logger.debug("Raw LLM response:\n%s", raw_response)
            raise ValueError(
                "Director LLM returned invalid JSON. "
                "This is usually a transient issue -- retry the request."
            ) from exc

    @staticmethod
    def _shots_from_plan(
        plan_data: dict[str, Any],
        preferred_model: str | None = None,
    ) -> list[Shot]:
        """Extract Shot objects from the parsed plan data."""
        shots: list[Shot] = []
        for scene in plan_data.get("scenes", []):
            shot = Shot(
                shot_id=scene.get("scene_id", uuid.uuid4().hex[:12]),
                description=scene.get("description", ""),
                prompt=scene.get("description", ""),
                duration_seconds=float(scene.get("duration", 5.0)),
                model_id=preferred_model or "mock",
                status=ShotStatus.PENDING,
            )
            shots.append(shot)
        return shots

    @staticmethod
    def _plan_metadata(
        plan_data: dict[str, Any],
        aspect_ratio: str,
    ) -> dict[str, Any]:
        width, height = _ASPECT_TO_RESOLUTION.get(aspect_ratio, (1280, 720))
        return {
            "title": plan_data.get("title", ""),
            "description": plan_data.get("description", ""),
            "aspect_ratio": aspect_ratio,
            "resolution": {"width": width, "height": height},
            "voice_over": plan_data.get("voice_over", {}),
            "music": plan_data.get("music", {}),
            "scenes_raw": plan_data.get("scenes", []),
        }

    @classmethod
    def _populate_state(
        cls,
        state: ProjectState,
        plan_data: dict[str, Any],
        aspect_ratio: str,
        preferred_model: str | None = None,
    ) -> None:
        """Update an existing ProjectState with plan data in-place."""
        voice_over = plan_data.get("voice_over", {})
        state.script = voice_over.get("text")
        state.storyboard = cls._shots_from_plan(plan_data, preferred_model)
        state.metadata = cls._plan_metadata(plan_data, aspect_ratio)

    @classmethod
    def _build_project_state(
        cls,
        plan_data: dict[str, Any],
        original_prompt: str,
        aspect_ratio: str,
        preferred_model: str | None = None,
    ) -> ProjectState:
        """Convert parsed plan JSON into a :class:`ProjectState`."""
        voice_over = plan_data.get("voice_over", {})
        state = ProjectState(
            prompt=original_prompt,
            script=voice_over.get("text"),
            storyboard=cls._shots_from_plan(plan_data, preferred_model),
            metadata=cls._plan_metadata(plan_data, aspect_ratio),
        )
        return state
