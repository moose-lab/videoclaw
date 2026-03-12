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
from videoclaw.drama.models import Character, DramaScene, DramaSeries, DramaStatus, Episode, EpisodeStatus
from videoclaw.models.llm.litellm_wrapper import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SERIES_OUTLINE_PROMPT: str = """\
你是一位资深的中国竖屏短剧（竖屏短剧）编剧，精通抖音/快手/微信视频号等平台的内容生态。
每集时长30-90秒。你的作品需要在信息流中脱颖而出，驱动完播率和付费转化。

根据给定的概念，产出一份包含分集梗概和角色设定的剧集大纲。

# 核心创作法则

## 黄金3秒法则
- 第1集开头必须在3秒内用强冲突、悬念或视觉冲击抓住观众
- 平铺直叙 = 90%划走率。必须以动作、悬念或震撼开场，绝不用铺垫
- 经典开场模式：撞见出轨/突然重生/当众打脸/神秘能力觉醒/倒叙高潮

## 高概念提炼（一句话卖点）
- 每部剧必须能用一句话概括核心卖点，例如：
  "重生回到被抛弃前一天，这次她要让所有人付出代价"
  "每天凌晨3点，手机会收到24小时后的新闻"
- 这句话要同时包含：主角身份 + 核心冲突 + 情感驱动力

## 叙事弧结构（N集剧本）
- 第1集：激励事件 — 爆发性开场，3秒内建立主角处境和核心冲突
- 第2~3集：上瘾铺垫 — 快速建立人物关系网，埋设悬念线，制造第一个小高潮
- 第4~N-2集：递进升级 — 加深人物关系、升高赌注、引入新矛盾和反转
- 第N-1集：至暗时刻 — 最大危机、重大揭示或背叛，情绪谷底
- 第N集：高潮 + 解决 — 主角终极抉择，核心矛盾爆发，留有情感余韵

## 爽点密度规则
- 每15秒必须有一个"爽点"：逆袭、打脸、反转、揭秘、深情表白
- 爽点类型交替使用，避免审美疲劳：智商碾压 → 情感爆发 → 身份揭示 → 实力展示
- 第1集爽点密度最高（每10秒一个），后续集可适当放缓建立深度

## 角色人设公式
- 主角必须有"反差人设"：表面身份 vs 隐藏身份（例：保洁阿姨实为集团董事长）
- 每个角色需要"记忆点"：一个标志性动作、口头禅或视觉符号
- 角色关系必须形成"三角张力"：至少存在一组三角关系（情感/利益/权力）
- 反派不能纯粹是坏人，需有可理解的动机

## 情绪节奏曲线
- 每集遵循：钩子(0-5s) → 铺垫(5-15s) → 递进(15-35s) → 高潮(35-50s) → 悬念(50-60s)
- 集与集之间要有情绪对比：紧张→温柔→震撼→甜蜜→虐心
- 情绪强度整体呈上升趋势，第N-1集达到情绪顶峰

## 悬念阶梯
- 每集结尾的悬念必须比上一集更强
- 悬念类型递进：好奇型(这是谁？) → 担忧型(他会怎样？) → 震撼型(不可能！)
- 禁止使用"未解决=悬念"的偷懒方式，每个悬念都要有情绪重量

# 输出规范
- 中文内容：标题、梗概、描述、opening_hook 用中文
- visual_prompt 用英文（供AI视频生成模型使用）
- 仅返回合法 JSON，不要 markdown 围栏或注释

Output JSON schema:
{
  "title": "<中文剧名，2-6字，朗朗上口，暗示核心冲突>",
  "genre": "<类型>",
  "synopsis": "<中文整体剧情梗概，包含高概念一句话卖点>",
  "characters": [
    {
      "name": "<中文角色名>",
      "description": "<中文：表面身份/隐藏身份、性格反差、核心动机、与其他角色的三角关系、记忆点>",
      "visual_prompt": "<ENGLISH ONLY — PURE appearance: age, gender, body type, face features, hair style/color, clothing, accessories, distinctive marks. Do NOT include camera angles, lighting, or background.>",
      "voice_style": "<warm | authoritative | playful | dramatic | calm>"
    }
  ],
  "episodes": [
    {
      "number": <int>,
      "title": "<中文集标题，4-8字，含悬念或情绪关键词>",
      "synopsis": "<中文：2-3句分集梗概，标注每个爽点位置和类型>",
      "opening_hook": "<中文：一句话描述本集前3秒的视觉/情绪钩子>",
      "duration_seconds": <float>
    }
  ]
}
"""

EPISODE_SCRIPT_PROMPT: str = """\
你是一位资深的中国竖屏短剧分镜编剧，精通AI视频生成的视觉提示词写作。
根据剧集上下文和分集梗概，产出一份逐场景的分镜脚本。

# 角色视觉一致性（最高优先级）
- 角色每次出镜时，visual_prompt 必须包含其完整的英文外貌描述
- 同一角色跨场景必须使用完全相同的视觉关键词（衣着/发型/标志性特征）
- 禁止在 visual_prompt 中省略角色外貌，即使是同一场景的连续镜头

# 竖屏构图法则（9:16）
- 人物面部居于画面上1/3，这是竖屏观看的视觉焦点区
- 单人特写效果远优于多人全景 — 竖屏上细节比场面更有冲击力
- 景别分配：close_up + medium_close 占总场景的40-50%（竖屏核心优势）
- wide/extreme_wide 仅用于建立场景或制造反差，不超过总场景的15%
- 避免横向运镜（pan_left/pan_right），竖屏上效果差；优先使用 dolly_in 和 crane_up

# 景别与叙事节拍对应
- establishing（建立镜头）→ wide/extreme_wide：开场/转场/新环境
- reaction（反应镜头）→ close_up：角色情绪爆发、震惊、心碎时刻
- action（动作镜头）→ medium/medium_close：对话、互动、冲突
- detail（细节镜头）→ close_up：关键道具、手部动作、眼神特写
- pov（主观镜头）→ medium_close：代入角色视角，增强沉浸感

# 情绪控制词表
- 情绪关键词必须精确，不要泛化。使用以下标准词汇：
  紧张类: tense, anxious, dread, suspense
  愤怒类: angry, furious, resentful, defiant
  悲伤类: sad, heartbroken, grieving, melancholy
  震惊类: shock, disbelief, stunned, revelation
  温暖类: warm, tender, nostalgic, grateful
  甜蜜类: sweet, flirty, blissful, intimate
  恐惧类: fear, panic, horror, uneasy
  得意类: triumphant, smug, vindicated, proud

# 音效设计
- 每个场景应标注关键音效（门声、脚步、环境音等）
- 音效服务于叙事节奏：紧张场景用尖锐/突发音效，温柔场景用环境白噪音
- 音效与画面同步，标注在 sfx 字段中
- 无音效的镜头留空即可

# 转场策略
- cut（硬切）：默认转场，节奏快，紧张/动作场景
- dissolve（叠化）：时间流逝、回忆、温柔情绪过渡
- fade_in/fade_out：开场/结尾、重大情绪转折
- match_cut（匹配剪辑）：两个视觉相似画面之间的创意过渡
- jump_cut（跳切）：同场景内时间压缩，制造紧迫感

# 节奏控制
- 场景数量：60秒约8-12个场景，按时长等比缩放
- 所有场景的 duration_seconds 之和必须等于目标集时长（±2秒）
- 节奏模板：钩子(0-5s) → 铺垫(5-15s) → 递进(15-35s) → 高潮(35-50s) → 悬念(50-60s)
- 高潮场景用短镜头快切（2-3秒/镜头），铺垫场景可适当拉长（5-8秒）
- 第一个场景必须是视觉钩子或衔接上集悬念，最后一个场景必须制造悬念

# 台词与旁白
- 中文对白总字数不超过100字（60秒集），语速约4字/秒
- 对白要求："说人话" — 短句、口语化、有情绪爆发力，单句不超过15字
- 旁白用于推进叙事和内心独白，对白用于冲突和情感爆发
- 不是每个场景都需要台词，无声的表情特写同样有力
- speaking_character 必须与 characters_present 中的角色一致

# 内心独白（内心OS）
- 内心独白是中国短剧的标志性叙事手法：角色第一人称内心活动
- 标记为 dialogue_line_type: "inner_monologue"
- 触发标志：角色内心活动、自言自语式的思考、"心想"、"暗道"、不可能式的自我反问
- 内心独白用与该角色对白相同的声音，但后期加混响处理，制造"脑内回响"效果
- 内心独白既不是对白（说给其他角色听），也不是旁白（第三人称叙述者视角）

# visual_prompt 写作规范
- 必须用英文，针对AI视频生成模型优化
- 结构：[环境/场景] + [角色完整外貌] + [动作/表情] + [光影/氛围]
- 包含光影描述：soft lighting, dramatic shadows, golden hour, neon lights
- 包含氛围词：cinematic, atmospheric, moody, ethereal
- 禁止使用中文或抽象描述，要具象可视化

# 输出规范
- visual_prompt 必须用英文
- dialogue 和 narration 必须用中文
- 仅返回合法 JSON，不要 markdown 围栏

Output JSON schema:
{
  "episode_title": "<中文集标题>",
  "scenes": [
    {
      "scene_id": "<e.g. ep01_s01>",
      "description": "<中文场景描述：谁在哪里做什么，情绪状态>",
      "visual_prompt": "<ENGLISH ONLY — [setting] + [character full appearance] + [action/expression] + [lighting/mood]. Be specific and visual.>",
      "camera_movement": "<static | pan_left | pan_right | dolly_in | tracking | crane_up | handheld>",
      "duration_seconds": <float>,
      "dialogue": "<中文角色对白，短句口语化，无则留空>",
      "dialogue_line_type": "<dialogue | inner_monologue — 角色间对白为dialogue，内心独白(第一人称OS/心理活动)为inner_monologue，无对白则留空>",
      "narration": "<中文旁白，无则留空>",
      "speaking_character": "<说话角色名，必须在 characters_present 中，无则留空>",
      "shot_scale": "<close_up | medium_close | medium | wide | extreme_wide>",
      "shot_type": "<establishing | reaction | action | detail | pov>",
      "emotion": "<从情绪词表中选择精确词汇>",
      "characters_present": ["<本场景出镜的角色名列表>"],
      "transition": "<cut | dissolve | fade_in | fade_out | wipe | match_cut | jump_cut>",
      "sfx": "<此镜头关键音效，如：门声、脚步声、雷鸣。无则留空>"
    }
  ],
  "voice_over": {
    "text": "<中文完整旁白脚本>",
    "tone": "<warm | dramatic | tense | playful>",
    "language": "zh"
  },
  "music": {
    "style": "<orchestral | electronic | acoustic | lo-fi | chinese_traditional>",
    "mood": "<tense | romantic | mysterious | triumphant | melancholy | epic>",
    "tempo": <BPM int>
  },
  "cliffhanger": "<中文悬念描述：用一句话制造'不看下一集会死'的冲动>"
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
        episode.scenes = [DramaScene.from_dict(s) for s in scenes]
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
