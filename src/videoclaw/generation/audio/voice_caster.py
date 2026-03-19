"""VoiceCaster — LLM-powered voice casting for short-drama multi-role TTS.

Analyses a drama script to determine genre, assigns voice profiles to each
character via LLM reasoning, and extracts typed dialogue lines (narration,
dialogue, inner monologue) for downstream TTS synthesis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from videoclaw.config import get_config
from videoclaw.drama.models import (
    Character,
    DialogueLine,
    DramaGenre,
    DramaScene,
    DramaSeries,
    Episode,
    LineType,
    NARRATOR_PRESETS,
    VoiceProfile,
)
from videoclaw.models.llm.litellm_wrapper import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

GENRE_ANALYSIS_PROMPT: str = """\
你是一位精通中国竖屏短剧的内容分析专家。
请根据提供的剧本文本判断该短剧的类型。

可选类型:
- sweet_romance: 甜宠/恋爱/霸总甜文
- male_power_fantasy: 男频/逆袭/战神/赘婿
- suspense_thriller: 悬疑/惊悚/推理
- ancient_xianxia: 古装/仙侠/玄幻
- comedy: 喜剧/搞笑/沙雕
- family_drama: 家庭/亲情/伦理
- other: 无法归类的其他类型

仅返回合法 JSON，不要 markdown 围栏或注释。

Output JSON schema:
{
  "genre": "<类型标识符，必须为上述之一>"
}
"""

VOICE_CASTING_PROMPT: str = """\
你是一位专业的短剧配音导演，精通中文语音合成(TTS)的声音选角。
根据剧集的角色设定和剧情类型，为每个角色分配最合适的语音配置。

可用的 voice_id 选项:
- Friendly_Person: 亲和温暖的声音
- Imposing_Manner: 威严沉稳的声音
- Lively_Girl: 活泼俏皮的女声
- Determined_Man: 刚毅坚定的男声
- Calm_Woman: 沉稳优雅的女声

参数说明:
- speed: 语速倍率 (0.5~2.0，默认1.0)
- pitch: 音调偏移 (-5~5，默认0)
- emotion: 情感基调 (neutral/happy/sad/angry/fearful/surprised)
- age_feel: 年龄感 (child/young_adult/middle_aged/elderly)
- energy: 能量水平 (low/medium/high)
- description: 对该角色声音风格的中文描述

仅返回合法 JSON，不要 markdown 围栏或注释。

Output JSON schema:
{
  "characters": [
    {
      "name": "<角色名>",
      "voice_id": "<voice_id>",
      "speed": <float>,
      "pitch": <int>,
      "emotion": "<emotion>",
      "age_feel": "<age_feel>",
      "energy": "<energy>",
      "description": "<中文声音描述>"
    }
  ]
}
"""

DIALOGUE_EXTRACTION_PROMPT: str = """\
你是一位专业的短剧台词分析师。
请将场景文本分类为以下类型之一:

- narration: 旁白叙述
- dialogue: 角色对白
- inner_monologue: 内心独白（标识词: "心想"、"暗道"、"(OS)"、第一人称内心活动）

对于每条台词，标注说话角色、台词类型和情感提示。
如果场景没有对白也没有旁白，则跳过该场景。

仅返回合法 JSON，不要 markdown 围栏或注释。

Output JSON schema:
{
  "lines": [
    {
      "text": "<台词文本>",
      "speaker": "<说话角色或 narrator>",
      "line_type": "<narration | dialogue | inner_monologue>",
      "emotion_hint": "<情感提示>"
    }
  ]
}
"""


class VoiceCaster:
    """LLM-powered voice casting for multi-role TTS in short dramas."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # Lazy LLM init (same pattern as DramaPlanner)
    # ------------------------------------------------------------------

    def _ensure_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(default_model=get_config().default_llm)
        return self._llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_genre(self, script_text: str, language: str = "zh") -> DramaGenre:
        """Classify the drama genre from script text using LLM.

        Returns a :class:`DramaGenre` enum value. Falls back to
        ``DramaGenre.OTHER`` if the LLM returns an unrecognised genre.

        *language* selects the locale (and thus the system prompt language).
        """
        llm = self._ensure_llm()
        from videoclaw.drama.locale import get_locale
        locale = get_locale(language)

        if language == "zh":
            user_message = f"剧本文本:\n{script_text}"
        else:
            user_message = f"Script text:\n{script_text}"

        raw = await llm.chat(
            messages=[
                {"role": "system", "content": locale.genre_analysis_prompt or GENRE_ANALYSIS_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        data = self._parse_json(raw)
        genre_str = data.get("genre", "other")

        try:
            return DramaGenre(genre_str)
        except ValueError:
            logger.warning("Unknown genre %r from LLM, falling back to OTHER", genre_str)
            return DramaGenre.OTHER

    async def cast_voices(
        self,
        series: DramaSeries,
        genre: DramaGenre,
        language: str = "zh",
    ) -> dict[str, VoiceProfile]:
        """Assign a VoiceProfile to each character using LLM analysis.

        Adds a narrator voice from locale narrator_presets keyed by *genre*.
        Returns a dict mapping role name to :class:`VoiceProfile`.

        *language* selects the locale (and thus the system prompt language).
        """
        llm = self._ensure_llm()
        from videoclaw.drama.locale import get_locale
        locale = get_locale(language)

        characters_text = "\n".join(
            f"- {c.name}: {c.description} (voice_style: {c.voice_style})"
            for c in series.characters
        )

        if language == "zh":
            user_message = (
                f"剧情类型: {genre.value}\n"
                f"角色列表:\n{characters_text}\n"
            )
        else:
            user_message = (
                f"Genre: {genre.value}\n"
                f"Characters:\n{characters_text}\n"
            )

        raw = await llm.chat(
            messages=[
                {"role": "system", "content": locale.voice_casting_prompt or VOICE_CASTING_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        data = self._parse_json(raw)

        voice_map: dict[str, VoiceProfile] = {}

        for char_data in data.get("characters", []):
            name = char_data.get("name", "")
            if not name:
                continue
            voice_map[name] = VoiceProfile(
                voice_id=char_data.get("voice_id", "Friendly_Person"),
                speed=float(char_data.get("speed", 1.0)),
                pitch=int(char_data.get("pitch", 0)),
                emotion=char_data.get("emotion", "neutral"),
                role_name=name,
                line_type=LineType.DIALOGUE,
                age_feel=char_data.get("age_feel", "young_adult"),
                energy=char_data.get("energy", "medium"),
                description=char_data.get("description", ""),
            )

        # Add narrator from locale genre presets, falling back to module-level dict
        narrator_presets = locale.narrator_presets if locale.narrator_presets else NARRATOR_PRESETS
        narrator_profile = narrator_presets.get(genre, narrator_presets.get(DramaGenre.OTHER))
        if narrator_profile is None:
            narrator_profile = NARRATOR_PRESETS.get(genre, NARRATOR_PRESETS[DramaGenre.OTHER])
        voice_map["narrator"] = narrator_profile

        logger.info(
            "Voice casting complete: %d characters + narrator for genre %s",
            len(voice_map) - 1,
            genre.value,
        )
        return voice_map

    async def extract_dialogue_lines(
        self,
        episode: Episode,
        voice_map: dict[str, VoiceProfile],
        language: str = "zh",
    ) -> list[DialogueLine]:
        """Extract typed dialogue lines from an episode's scenes.

        Iterates through the episode's scenes, uses LLM to classify each
        scene's text as narration, dialogue, or inner monologue. Skips empty
        scenes (no dialogue and no narration). Returns a list of
        :class:`DialogueLine` with ``scene_id`` for timeline alignment.

        *language* selects the locale (and thus the system prompt language).
        """
        llm = self._ensure_llm()
        from videoclaw.drama.locale import get_locale
        locale = get_locale(language)
        all_lines: list[DialogueLine] = []

        for scene in episode.scenes:
            # Skip empty scenes
            if not scene.dialogue and not scene.narration:
                continue

            scene_text_parts: list[str] = []
            if language == "zh":
                if scene.dialogue:
                    scene_text_parts.append(f"对白: {scene.dialogue}")
                if scene.narration:
                    scene_text_parts.append(f"旁白: {scene.narration}")
                if scene.speaking_character:
                    scene_text_parts.append(f"说话角色: {scene.speaking_character}")
                user_prefix = f"场景 {scene.scene_id}:"
            else:
                if scene.dialogue:
                    scene_text_parts.append(f"Dialogue: {scene.dialogue}")
                if scene.narration:
                    scene_text_parts.append(f"Narration: {scene.narration}")
                if scene.speaking_character:
                    scene_text_parts.append(f"Speaking character: {scene.speaking_character}")
                user_prefix = f"Scene {scene.scene_id}:"

            scene_text = "\n".join(scene_text_parts)

            raw = await llm.chat(
                messages=[
                    {"role": "system", "content": locale.dialogue_extraction_prompt or DIALOGUE_EXTRACTION_PROMPT},
                    {"role": "user", "content": f"{user_prefix}\n{scene_text}"},
                ],
            )

            data = self._parse_json(raw)

            for line_data in data.get("lines", []):
                text = line_data.get("text", "").strip()
                if not text:
                    continue

                speaker = line_data.get("speaker", "narrator")
                line_type_str = line_data.get("line_type", "dialogue")

                try:
                    line_type = LineType(line_type_str)
                except ValueError:
                    line_type = LineType.DIALOGUE

                all_lines.append(
                    DialogueLine(
                        text=text,
                        speaker=speaker,
                        line_type=line_type,
                        scene_id=scene.scene_id,
                        emotion_hint=line_data.get("emotion_hint"),
                    )
                )

        logger.info(
            "Extracted %d dialogue lines from episode %d",
            len(all_lines),
            episode.number,
        )
        return all_lines

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw_response: str) -> dict[str, Any]:
        """Best-effort extraction of a JSON object from raw LLM text.

        Handles common LLM quirks such as markdown code fences.
        """
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
