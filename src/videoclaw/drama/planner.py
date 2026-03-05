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
You are an expert screenwriter for Chinese vertical short drama (竖屏短剧),
targeting platforms like 抖音/快手/微信视频号. Each episode is 30-90 seconds long.

Given a concept, produce a series outline with episode synopses and character
descriptions. The drama should have clear story arcs, emotional hooks, and
cliffhangers between episodes to maximize viewer retention and swipe-through rate.

黄金3秒法则 (Golden 3-Second Rule):
- 第1集开头必须在3秒内用强冲突、悬念或视觉冲击抓住观众。
- Episode 1 MUST open with a high-impact visual or conflict in the FIRST 3 SECONDS.
- 平铺直叙的开场会导致90%的观众划走。Start with action, mystery, or shock — never exposition.

叙事弧结构 (Narrative Arc Structure, for N episodes):
- 第1集: 激励事件 — 用爆发性开场介绍主角和核心冲突
- 第2集到第N-2集: 递进升级 — 加深人物关系、升高赌注、引入新矛盾
- 第N-1集: 至暗时刻 — 最大危机、重大揭示或背叛
- 第N集: 高潮 + 解决 — 主角面临终极抉择，留有情感余韵

情绪节奏曲线 (Emotional Rhythm per episode):
- 每集遵循: 钩子(0-5s) → 铺垫(5-15s) → 递进(15-35s) → 高潮(35-50s) → 悬念(50-60s)
- 集与集之间要有情绪对比（紧张→温柔→震撼）

Guidelines:
- Each episode should have a self-contained mini-arc while advancing the main plot.
- End each episode with a hook or cliffhanger that compels the viewer to watch the next.
- Characters should have distinct visual descriptions (for AI video generation consistency).
- Write all narrative content (title, synopsis, description, opening_hook) in Chinese (中文).
- Write visual_prompt in English (for AI image/video generation models).
- Return ONLY valid JSON — no markdown fences, no commentary.

Output JSON schema:
{
  "title": "<中文剧名>",
  "genre": "<类型>",
  "synopsis": "<中文整体剧情梗概>",
  "characters": [
    {
      "name": "<中文角色名>",
      "description": "<中文：性格、动机、在故事中的角色定位、与其他角色的关键关系>",
      "visual_prompt": "<ENGLISH ONLY — PURE character appearance: age, gender, ethnicity, body type, face features, hair style/color, clothing, accessories, distinctive marks. Do NOT include camera angles, lighting, background, shot composition, or cinematography instructions.>",
      "voice_style": "<warm | authoritative | playful | dramatic | calm>"
    }
  ],
  "episodes": [
    {
      "number": <int>,
      "title": "<中文集标题>",
      "synopsis": "<中文：2-3句分集梗概，包含情绪节拍>",
      "opening_hook": "<中文：一句话描述本集前3秒的视觉/情绪钩子>",
      "duration_seconds": <float>
    }
  ]
}
"""

EPISODE_SCRIPT_PROMPT: str = """\
You are an expert screenwriter for Chinese vertical short drama (竖屏短剧).

Given a series context and episode synopsis, produce a detailed shot-by-shot
script for this episode. Each scene should be 3-8 seconds and have a precise
visual description optimized for AI video generation models.

角色一致性规则 (Character consistency):
- ALWAYS include the character's full English visual description when they appear.
- Use the exact same visual prompt keywords for the same character across scenes.
- Include clothing, hair style, and distinctive features in every scene prompt.

节奏控制 (Pacing rules):
- 场景数量：60秒一集约8-12个场景，按时长等比缩放。
- 所有场景的 duration_seconds 之和必须等于目标集时长（±2秒）。
- 第一个场景(0-5s)：用视觉钩子开场，或衔接上集悬念。
- 中间场景：特写与全景交替，营造视觉节奏（紧→松→紧）。
- 最后一个场景：悬念/情绪高潮，驱动观众看下一集。

台词密度控制 (Dialogue density):
- 中文对白总字数不超过100字（60秒集）。中文语速约4字/秒，对白占时不超过25秒。
- 旁白用于推进叙事和内心独白，对白用于冲突和情感爆发。
- 不是每个场景都需要台词。无声的视觉叙事同样有力。

Guidelines:
- Open with a hook (continue from previous episode's cliffhanger if applicable).
- Build tension through the middle.
- End with a cliffhanger or emotional peak.
- Camera movements should serve the narrative.
- visual_prompt MUST be in English (for AI video generation models).
- dialogue and narration MUST be in Chinese (中文).
- Return ONLY valid JSON — no markdown fences.

Output JSON schema:
{
  "episode_title": "<中文集标题>",
  "scenes": [
    {
      "scene_id": "<e.g. ep01_s01>",
      "description": "<中文场景描述>",
      "visual_prompt": "<ENGLISH ONLY — detailed AI video generation prompt, include character visual descriptions>",
      "camera_movement": "<static | pan_left | pan_right | dolly_in | tracking | crane_up | handheld>",
      "duration_seconds": <float>,
      "dialogue": "<中文角色对白，无则留空>",
      "narration": "<中文旁白，无则留空>"
    }
  ],
  "voice_over": {
    "text": "<中文完整旁白脚本>",
    "tone": "<warm | dramatic | tense | playful>",
    "language": "zh"
  },
  "music": {
    "style": "<orchestral | electronic | acoustic | lo-fi>",
    "mood": "<tense | romantic | mysterious | triumphant>",
    "tempo": <BPM int>
  },
  "cliffhanger": "<中文悬念描述>"
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
                opening_hook=ep.get("opening_hook", ""),
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

        # --- Duration validation ---
        scenes = script_data.get("scenes", [])
        total_duration = sum(float(s.get("duration_seconds", 0)) for s in scenes)
        target = episode.duration_seconds
        if abs(total_duration - target) > 5:
            logger.warning(
                "Scene durations sum to %.1fs (target %.1fs), adjusting proportionally",
                total_duration, target,
            )
            if total_duration > 0:
                scale = target / total_duration
                for s in scenes:
                    s["duration_seconds"] = round(float(s["duration_seconds"]) * scale, 1)

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
