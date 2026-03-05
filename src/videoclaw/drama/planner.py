"""Drama planner — uses LLM to generate series outlines and episode scripts.

The DramaPlanner converts a high-level concept (genre, synopsis, character list)
into a structured multi-episode plan, then generates per-episode scripts that
feed into the existing VideoClaw pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from videoclaw.config import get_config
from videoclaw.drama.models import Character, DramaSeries, DramaStatus, Episode, EpisodeStatus
from videoclaw.models.llm.litellm_wrapper import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SERIES_OUTLINE_PROMPT: str = """\
You are an expert screenwriter for AI-generated short-form drama series
(竖屏短剧 / vertical short drama). Each episode is 30-90 seconds long.

Given a concept, produce a series outline with episode synopses and character
descriptions. The drama should have clear story arcs, emotional hooks, and
cliffhangers between episodes to drive engagement.

Guidelines:
- Each episode should have a self-contained mini-arc while advancing the main plot.
- End each episode with a hook or cliffhanger.
- Characters should have distinct visual descriptions (for AI video consistency).
- Write in the requested language.
- Return ONLY valid JSON — no markdown fences, no commentary.

Output JSON schema:
{
  "title": "<series title>",
  "genre": "<genre>",
  "synopsis": "<overall series synopsis>",
  "characters": [
    {
      "name": "<character name>",
      "description": "<personality and role in story>",
      "visual_prompt": "<detailed visual description for AI generation: age, gender, appearance, clothing, distinctive features>",
      "voice_style": "<warm | authoritative | playful | dramatic | calm>"
    }
  ],
  "episodes": [
    {
      "number": <int>,
      "title": "<episode title>",
      "synopsis": "<2-3 sentence episode synopsis with emotional beats>",
      "duration_seconds": <float>
    }
  ]
}
"""

EPISODE_SCRIPT_PROMPT: str = """\
You are an expert screenwriter for AI-generated short drama.

Given a series context and episode synopsis, produce a detailed shot-by-shot
script for this episode. Each scene should be 3-8 seconds and have a precise
visual description optimized for AI video generation models.

Character consistency rules:
- ALWAYS include the character's full visual description when they appear.
- Use the exact same visual prompt keywords for the same character across scenes.
- Include clothing, hair style, and distinctive features in every scene prompt.

Guidelines:
- Open with a hook (continue from previous episode's cliffhanger if applicable).
- Build tension through the middle.
- End with a cliffhanger or emotional peak.
- Camera movements should serve the narrative.
- Return ONLY valid JSON — no markdown fences.

Output JSON schema:
{
  "episode_title": "<title>",
  "scenes": [
    {
      "scene_id": "<e.g. ep01_s01>",
      "description": "<brief description>",
      "visual_prompt": "<detailed AI video generation prompt in English, include character visual descriptions>",
      "camera_movement": "<static | pan_left | pan_right | dolly_in | tracking | crane_up | handheld>",
      "duration_seconds": <float>,
      "dialogue": "<character dialogue if any, in original language>",
      "narration": "<voice-over narration if any>"
    }
  ],
  "voice_over": {
    "text": "<full narration script>",
    "tone": "<warm | dramatic | tense | playful>",
    "language": "<zh | en>"
  },
  "music": {
    "style": "<orchestral | electronic | acoustic | lo-fi>",
    "mood": "<tense | romantic | mysterious | triumphant>",
    "tempo": <BPM int>
  },
  "cliffhanger": "<description of episode-ending hook>"
}
"""


class DramaPlanner:
    """Plans multi-episode drama series using LLM."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def _ensure_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(default_model=get_config().default_llm)
        return self._llm

    async def plan_series(self, series: DramaSeries) -> DramaSeries:
        """Generate the full series outline from a synopsis.

        Populates the series with characters and episode synopses.
        """
        logger.info("Planning series: %r (%d episodes)", series.title or series.synopsis[:40], series.total_episodes)
        series.status = DramaStatus.PLANNING

        llm = self._ensure_llm()
        user_message = (
            f"Concept: {series.synopsis}\n"
            f"Genre: {series.genre or 'drama'}\n"
            f"Number of episodes: {series.total_episodes}\n"
            f"Episode duration: {series.target_episode_duration} seconds each\n"
            f"Visual style: {series.style}\n"
            f"Language: {series.language}\n"
        )

        raw = await llm.chat(
            messages=[
                {"role": "system", "content": SERIES_OUTLINE_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        plan = self._parse_json(raw)

        series.title = plan.get("title", series.title)
        series.genre = plan.get("genre", series.genre)
        series.synopsis = plan.get("synopsis", series.synopsis)
        series.characters = [
            Character.from_dict(c) for c in plan.get("characters", [])
        ]
        series.episodes = [
            Episode(
                number=ep.get("number", i + 1),
                title=ep.get("title", f"Episode {i + 1}"),
                synopsis=ep.get("synopsis", ""),
                duration_seconds=float(ep.get("duration_seconds", series.target_episode_duration)),
                status=EpisodeStatus.PENDING,
            )
            for i, ep in enumerate(plan.get("episodes", []))
        ]

        series.status = DramaStatus.DRAFT
        logger.info("Series planned: %r with %d characters, %d episodes",
                     series.title, len(series.characters), len(series.episodes))
        return series

    async def script_episode(
        self,
        series: DramaSeries,
        episode: Episode,
        previous_cliffhanger: str | None = None,
    ) -> dict[str, Any]:
        """Generate a detailed shot-by-shot script for one episode.

        Returns the raw script data dict with scenes, voice_over, music, and cliffhanger.
        """
        logger.info("Scripting episode %d: %r", episode.number, episode.title)

        llm = self._ensure_llm()

        characters_text = "\n".join(
            f"  - {c.name}: {c.visual_prompt} ({c.description})"
            for c in series.characters
        )

        prev_episodes_text = ""
        for ep in series.episodes:
            if ep.number < episode.number and ep.synopsis:
                prev_episodes_text += f"  Episode {ep.number} ({ep.title}): {ep.synopsis}\n"

        user_message = (
            f"Series: {series.title}\n"
            f"Genre: {series.genre}\n"
            f"Style: {series.style}\n"
            f"Language: {series.language}\n\n"
            f"Characters:\n{characters_text}\n\n"
        )

        if prev_episodes_text:
            user_message += f"Previous episodes:\n{prev_episodes_text}\n"
        if previous_cliffhanger:
            user_message += f"Previous cliffhanger: {previous_cliffhanger}\n\n"

        user_message += (
            f"Now write Episode {episode.number}: {episode.title}\n"
            f"Synopsis: {episode.synopsis}\n"
            f"Target duration: {episode.duration_seconds} seconds\n"
        )

        raw = await llm.chat(
            messages=[
                {"role": "system", "content": EPISODE_SCRIPT_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        script_data = self._parse_json(raw)
        episode.script = json.dumps(script_data, ensure_ascii=False)
        episode.scene_prompts = script_data.get("scenes", [])
        return script_data

    @staticmethod
    def _parse_json(raw_response: str) -> dict[str, Any]:
        text = raw_response.strip()
        if text.startswith("```"):
            first_nl = text.index("\n")
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM response as JSON: %s", exc)
            raise ValueError("LLM returned invalid JSON — retry the request.") from exc
