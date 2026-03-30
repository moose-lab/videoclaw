"""Script generator -- produces structured video scripts from a topic prompt.

The :class:`ScriptGenerator` uses an LLM to create a time-segmented script
complete with narration text, per-section visual descriptions, and timing
metadata.  The resulting :class:`Script` feeds directly into the storyboard
decomposition step.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from videoclaw.config import get_config
from videoclaw.core.events import TASK_COMPLETED, TASK_STARTED, event_bus
from videoclaw.models.llm.litellm_wrapper import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ScriptSection:
    """A single timed section within a video script."""

    section_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    text: str = ""
    visual_description: str = ""
    duration_seconds: float = 5.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "text": self.text,
            "visual_description": self.visual_description,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScriptSection:
        return cls(
            section_id=data.get("section_id", uuid.uuid4().hex[:10]),
            text=data.get("text", ""),
            visual_description=data.get("visual_description", ""),
            duration_seconds=float(data.get("duration_seconds", 5.0)),
            notes=data.get("notes", ""),
        )


@dataclass
class Script:
    """A complete video script with titled sections and voice-over text."""

    title: str = ""
    sections: list[ScriptSection] = field(default_factory=list)
    voice_over_text: str = ""
    total_duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "voice_over_text": self.voice_over_text,
            "total_duration": self.total_duration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Script:
        return cls(
            title=data.get("title", ""),
            sections=[
                ScriptSection.from_dict(s)
                for s in data.get("sections", [])
            ],
            voice_over_text=data.get("voice_over_text", ""),
            total_duration=float(data.get("total_duration", 0.0)),
        )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SCRIPT_SYSTEM_PROMPT: str = """\
You are a professional video scriptwriter.  Given a topic, target duration, tone,
and language, produce a structured video script in JSON format.

Guidelines:
- Divide the script into logical sections, each 3-10 seconds of screen time.
- For each section, write the narration text AND a separate visual description
  that tells a director what should be shown on screen.
- The combined section durations must approximate the target duration.
- Write voice-over text that sounds natural when read aloud.
- Adapt your writing style to the requested tone.
- For Chinese (zh), aim for ~4 characters per second of narration.
- For English (en), aim for ~2.2 words per second of narration.
- Return ONLY valid JSON -- no markdown fences, no extra commentary.

Output JSON schema:
{
  "title": "<video title>",
  "sections": [
    {
      "section_id": "<e.g. sec_01>",
      "text": "<narration / voice-over text for this section>",
      "visual_description": "<what the viewer sees on screen>",
      "duration_seconds": <float>,
      "notes": "<optional production notes>"
    }
  ],
  "voice_over_text": "<complete narration script concatenated>",
  "total_duration": <float, sum of section durations>
}
"""

# ---------------------------------------------------------------------------
# ScriptGenerator
# ---------------------------------------------------------------------------


class ScriptGenerator:
    """Generates video scripts by prompting an LLM with structured constraints.

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

    async def generate(
        self,
        topic: str,
        duration: float = 30.0,
        tone: str = "professional",
        language: str = "zh",
    ) -> Script:
        """Generate a structured video script.

        Parameters
        ----------
        topic:
            The subject or theme of the video.
        duration:
            Target total duration in seconds.
        tone:
            Desired narrative tone (e.g. ``"professional"``, ``"playful"``).
        language:
            ISO language code for the narration text.

        Returns
        -------
        Script
            A fully populated script ready for storyboard decomposition.
        """
        await event_bus.emit(
            TASK_STARTED,
            {"task": "script.generate", "topic": topic, "duration": duration},
        )
        logger.info(
            "Generating script: topic=%r duration=%.1fs tone=%s lang=%s",
            topic, duration, tone, language,
        )

        llm = self._ensure_llm()

        user_message = (
            f"Topic: {topic}\n"
            f"Target duration: {duration} seconds\n"
            f"Tone: {tone}\n"
            f"Language: {language}\n\n"
            "Generate the video script as JSON."
        )

        raw_response = await llm.chat(
            messages=[
                {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        script = self._parse_response(raw_response, duration)

        await event_bus.emit(
            TASK_COMPLETED,
            {
                "task": "script.generate",
                "title": script.title,
                "section_count": len(script.sections),
                "total_duration": script.total_duration,
            },
        )
        logger.info(
            "Script generated: title=%r sections=%d duration=%.1fs",
            script.title,
            len(script.sections),
            script.total_duration,
        )
        return script

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _parse_response(raw_response: str, target_duration: float) -> Script:
        """Parse LLM output into a :class:`Script` instance."""
        text = raw_response.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse script LLM response: %s", exc)
            logger.debug("Raw response:\n%s", raw_response)
            raise ValueError(
                "Script LLM returned invalid JSON. Retry the request."
            ) from exc

        script = Script.from_dict(data)

        # Ensure total_duration is populated
        if script.total_duration <= 0.0:
            script.total_duration = sum(
                s.duration_seconds for s in script.sections
            )

        # If voice_over_text was not provided, concatenate section texts
        if not script.voice_over_text:
            script.voice_over_text = "\n".join(
                s.text for s in script.sections if s.text
            )

        return script
