"""Drama planner — uses LLM to generate series outlines and episode scripts.

The DramaPlanner converts a high-level concept (genre, synopsis, character list)
into a structured multi-episode plan, then generates per-episode scripts that
feed into the existing VideoClaw pipeline.

For **imported** (complete) scripts, the planner operates in strict decompose-only
mode: it converts the existing script into shot-by-shot storyboards without any
creative modifications. Any detected gaps require explicit user approval before
being patched.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable

from videoclaw.config import get_config
from videoclaw.drama.models import (
    Character,
    ConsistencyManifest,
    DramaScene,
    DramaSeries,
    DramaStatus,
    Episode,
    EpisodeStatus,
    ScriptModification,
)
from videoclaw.models.llm.litellm_wrapper import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dialogue pacing helpers
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7ff]')

def _min_duration_for_dialogue(dialogue: str, max_cjk_cps: float = 3.5, max_en_wps: float = 2.5) -> float:
    """Return the minimum shot duration (seconds) needed for natural speech pacing.

    Chinese (CJK-dominant) text uses chars-per-second; English uses words-per-second.
    Returns 0.0 if dialogue is empty.
    """
    text = (dialogue or "").strip()
    if not text:
        return 0.0
    cjk_count = len(_CJK_RE.findall(text))
    if cjk_count > len(text) * 0.3:
        return math.ceil(cjk_count / max_cjk_cps)
    return math.ceil(len(text.split()) / max_en_wps)


def _enforce_pacing(scenes: list[dict[str, Any]]) -> None:
    """Mutate scene dicts in-place: raise duration_seconds to meet pacing floor.

    Caps at 15s (Seedance max). Logs every adjustment so the change is visible.
    """
    for s in scenes:
        floor = _min_duration_for_dialogue(s.get("dialogue", ""))
        if floor <= 0:
            continue
        cur = float(s.get("duration_seconds", 5.0))
        target = min(15.0, max(cur, floor))
        if target > cur:
            logger.info(
                "Pacing: %s duration %.1fs → %.1fs (dialogue needs ≥%.1fs)",
                s.get("scene_id", "?"), cur, target, floor,
            )
            s["duration_seconds"] = target


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
  "title": "<中文剧名，不超过20个中文字，朗朗上口，暗示核心冲突>",
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

# 节奏控制（Seedance 2.0 约束：单镜头时长 5～15 秒）
- 每个场景的 duration_seconds 必须在 5～15 秒之间（视频模型硬限制）
- 场景数量：**6-10 个场景**（单集最长 60 秒）。**硬上限：不超过 12 场景**
- 每个 shot = 一次 Seedance 视频生成调用，合并细碎动作为复合镜头
  例如"角色说话 + 对方反应"= 1 个 shot，不是 2 个
- 所有场景的 duration_seconds 之和必须等于目标集时长（±2秒）
- 节奏模板：钩子(0-5s) → 铺垫(5-15s) → 递进(15-35s) → 高潮(35-50s) → 悬念(50-60s)
- 高潮场景用短镜头快切（5-6秒/镜头），铺垫场景可适当拉长（8-12秒）
- 第一个场景必须是视觉钩子或衔接上集悬念，最后一个场景必须制造悬念

# 语速规则 — 硬性约束（系统会自动校正违规，但请先自查）
- 人声自然语速：中文 ≤ 3.5 字/秒，英文 ≤ 2.5 词/秒
- 公式：duration_seconds ≥ ceil(字数 / 3.5)（中文）
         duration_seconds ≥ ceil(词数 / 2.5)（英文）
- 设置时长前必须先数对白字数，再套公式计算最小时长
  示例：对白20字 → ceil(20 / 3.5) = 6s 起；对白35字 → ceil(35 / 3.5) = 10s 起
- 如果计算最小时长超过 15s，设为 15s，保留完整对白
- 禁止为了凑时长缩写对白；应调整时长来适配对白，而非反向操作

# 台词与旁白
- 中文对白总字数不超过100字（60秒集），语速约4字/秒
- 对白要求："说人话" — 短句、口语化、有情绪爆发力，单句不超过15字
- 旁白用于推进叙事和内心独白，对白用于冲突和情感爆发
- 不是每个场景都需要台词，无声的表情特写同样有力
- speaking_character 必须与 characters_present 中的角色一致
- 单个场景的对白/旁白字数应匹配该场景的 duration_seconds（约4字/秒）

# 旁白类型区分（narration_type）
- voiceover: 语音旁白 — 由旁白演员朗读，经过TTS合成，显示为底部字幕
- title_card: 视觉文字卡 — 仅作为画面上的文字叠加，**不**合成语音
  - 典型场景："一个月前"、"三年后"、"第二天清晨"、地点字幕、章节标题
  - title_card 在视频中央大字显示，不走TTS流程
- 默认为 voiceover，只有时间跳转、地点标注等纯视觉文字才用 title_card

# 内心独白（内心OS）
- 内心独白是中国短剧的标志性叙事手法：角色第一人称内心活动
- 标记为 dialogue_line_type: "inner_monologue"
- 触发标志：角色内心活动、自言自语式的思考、"心想"、"暗道"、不可能式的自我反问
- 内心独白用与该角色对白相同的声音，但后期加混响处理，制造"脑内回响"效果
- 内心独白既不是对白（说给其他角色听），也不是旁白（第三人称叙述者视角）

# TikTok 平台受众约束（优先级低于剧本忠实度）
- 优先级顺序：剧本要求 > 平台审美。如果剧本明确描述了角色年龄或外貌，必须忠实于剧本
- 目标受众：18-30 岁。当剧本未指定次要/背景角色的年龄和外貌时，默认为年轻有吸引力
- 宾客/配角/群演（无明确描述时）：20-35 岁，时尚穿搭，当代审美
- 如果剧本提到"宾客"、"人群"、"路人"且无具体描述，默认为年轻时尚群体（20-30 岁）

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
      "duration_seconds": <float, 5-15>,
      "dialogue": "<中文角色对白，短句口语化，无则留空>",
      "dialogue_line_type": "<dialogue | inner_monologue — 角色间对白为dialogue，内心独白(第一人称OS/心理活动)为inner_monologue，无对白则留空>",
      "narration": "<中文旁白，无则留空>",
      "narration_type": "<voiceover | title_card — 语音旁白为voiceover，纯视觉文字(时间跳转/地点标注)为title_card，默认voiceover>",
      "speaking_character": "<说话角色名，必须在 characters_present 中，无则留空>",
      "shot_scale": "<close_up | medium_close | medium | wide | extreme_wide>",
      "shot_type": "<establishing | reaction | action | detail | pov>",
      "emotion": "<从情绪词表中选择精确词汇>",
      "characters_present": ["<本场景出镜的角色名列表>"],
      "transition": "<cut | dissolve | fade_in | fade_out | wipe | match_cut | jump_cut>",
      "sfx": "<此镜头关键音效，如：门声、脚步声、雷鸣。无则留空>",
      "time_of_day": "<morning|day|evening|night|unspecified — 从场景语境推断，保持同场景内一致>",
      "scene_group": "<A|B|C|... — 剧本位置区块标签，同一物理地点的镜头应相同，如 A=泳池边 B=水下 C=走廊>",
      "shot_role": "<hook|normal|cliffhanger — 第一个镜头=hook，最后一个镜头=cliffhanger，其余=normal>"
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


IMPORT_DECOMPOSE_PROMPT: str = """\
You are a professional short-drama storyboard decomposer for TikTok-style vertical \
(9:16) AI-generated short dramas. You are given a COMPLETE, FINALIZED script.

# ABSOLUTE CONSTRAINTS — READ BEFORE ANYTHING ELSE
1. This script is LOCKED AND FINAL. You must NOT add, remove, or modify ANY \
   dialogue, action, character behavior, or plot point.
2. Your ONLY job is to decompose the existing script into shot-by-shot visual \
   prompts for AI video generation (Seedance 2.0).
3. Every shot must map 1:1 to content in the original script. NO creative \
   additions whatsoever.
4. If a scene in the script has no explicit dialogue, leave dialogue empty — \
   do NOT invent lines.
5. Character names, ages, descriptions must match EXACTLY what the script provides.

# Seedance 2.0 technical constraints
- Each shot: 5-15s (hard limit). One shot = one Seedance video generation call.
- Seedance 2.0 co-generates video + audio + dialogue + subtitles in one pass.
- 9:16 vertical format, 720p.
- visual_prompt must be in ENGLISH, even if the script is in another language.
- Dialogue in the shot should be in the SCRIPT'S ORIGINAL LANGUAGE.

# Dialogue pacing — HARD RULE (system auto-corrects violations)
- Natural human speech: English ≤ 2.5 words/s, Chinese ≤ 3.5 chars/s
- FORMULA: duration_seconds >= ceil(word_count / 2.5) for EN
           duration_seconds >= ceil(cjk_char_count / 3.5) for ZH
- Always COUNT words/chars in the dialogue, then compute the minimum duration.
  Example: 26-word EN dialogue → ceil(26 / 2.5) = 11s minimum. Do NOT set 7s.
  Example: 20-char ZH dialogue → ceil(20 / 3.5) = 6s minimum.
- If the minimum exceeds 15s (Seedance max), set 15s and keep the full dialogue.
- DO NOT shorten the script's dialogue to fit a shorter duration.
  Adjust duration to fit the dialogue, not the other way around.

# Shot count constraint (CRITICAL — read before decomposing)
- Target: **6-10 shots** per episode (max 60s). HARD CEILING: 12 shots max.
- ALL duration_seconds MUST NOT exceed the target maximum episode duration.
- MERGE consecutive fine-grained actions into COMPOSITE shots.
  Each shot can contain multiple narrative beats (dialogue + reaction + transition).
  Example: "Ivy pleads → Colton turns coldly → guests gasp" = ONE 10-12s shot, NOT three 4s shots.
- Reaction shots of minor characters (guests, extras) should be part of the main shot,
  NOT separate shots. Only cut away if the reaction is a major dramatic beat.
- Underwater/flashback sequences: merge into 1-2 long shots (10-15s), not 4-5 short ones.
- If your first pass exceeds 12 shots, RE-MERGE until you are within limit.

# Decomposition rules
For each shot:
1. Write an ENGLISH visual_prompt describing the EXACT scene as written.
   Structure: [setting] + [character full appearance] + [action/expression] + [lighting/mood]
2. Assign shot_scale, shot_type, camera_movement based on the scene content.
3. Set duration_seconds: COUNT the dialogue words FIRST, then apply the formula below.
   Action cuts with no dialogue: 5-6s. Complex dialogue shots: use pacing formula.
4. Copy the EXACT dialogue from the script — do NOT paraphrase or summarize.
   Include dialogue as subtitle instruction for Seedance to render in-video.
5. Identify characters_present strictly from who the script says is in the scene.
6. Assign emotion from: tense, anxious, angry, furious, sad, heartbroken, shock, \
   disbelief, warm, tender, sweet, intimate, fear, panic, triumphant, defiant.

# TikTok Platform Audience Constraints (LOWER priority than script fidelity)
- Priority: script requirements > platform aesthetics. If the script explicitly
  describes a character's age or appearance, HONOUR the script exactly.
- Target audience: 18-30 year olds. When the script does NOT specify age/appearance
  for secondary/background characters, default to youthful and attractive.
- Guest/extra characters without explicit description: age 20-35, fashionable, contemporary.
- If the script mentions "guests", "crowd", "partygoers" without detail,
  describe as young (20s-30s), stylish, and diverse in the visual_prompt.

# Gap detection
If you notice any of the following gaps, report them in the "detected_gaps" array:
- A character appears but has no physical description in the script
- A scene location is referenced but never described
- A dialogue line has no clear speaker
- A transition between scenes has a logical discontinuity
- Guest/extra characters that appear older than 35 in the original script (flag for review)

# Output JSON schema
{
  "episodes": [
    {
      "number": <int>,
      "title": "<episode title from script>",
      "scenes": [
        {
          "scene_id": "<ep01_s01>",
          "description": "<scene description from script>",
          "visual_prompt": "<ENGLISH — exact scene visualization for Seedance 2.0>",
          "camera_movement": "<static|dolly_in|tracking|crane_up|handheld|pan_left|pan_right>",
          "duration_seconds": <float, 5-15>,
          "dialogue": "<EXACT dialogue from script, original language>",
          "dialogue_line_type": "<dialogue|inner_monologue>",
          "narration": "<narration text if any, otherwise empty>",
          "narration_type": "<voiceover|title_card — spoken narration is voiceover, visual-only text (time jumps, location cards) is title_card, default voiceover>",
          "speaking_character": "<character name from script>",
          "shot_scale": "<close_up|medium_close|medium|wide|extreme_wide>",
          "shot_type": "<establishing|reaction|action|detail|pov>",
          "emotion": "<from emotion list above>",
          "characters_present": ["<names from script>"],
          "transition": "<cut|dissolve|fade_in|fade_out|match_cut>",
          "sfx": "<sound effects implied by the script>",
          "time_of_day": "<morning|day|evening|night|unspecified — infer from script context>",
          "scene_group": "<A|B|C|... — location block label, e.g. A=poolside B=underwater C=corridor>",
          "shot_role": "<hook|normal|cliffhanger — first shot=hook, last shot=cliffhanger, rest=normal>"
        }
      ]
    }
  ],
  "characters": [
    {
      "name": "<from script>",
      "description": "<from script>",
      "visual_prompt": "<ENGLISH — appearance only, from script descriptions>",
      "voice_style": "<warm|authoritative|playful|dramatic|calm>"
    }
  ],
  "detected_gaps": [
    {
      "scene_id": "<affected scene or empty>",
      "field": "<which field has a gap>",
      "description": "<what is missing or inconsistent>"
    }
  ]
}

Return ONLY valid JSON — no markdown fences, no commentary.
"""


class DramaPlanner:
    """Plans multi-episode drama series using LLM.

    When operating on imported (locked) scripts, the planner uses
    decompose-only mode — no creative modifications are applied.
    """

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

        from videoclaw.drama.locale import get_locale
        locale = get_locale(series.language)
        prompt = locale.series_outline_prompt or SERIES_OUTLINE_PROMPT
        raw = await llm.chat(
            messages=[
                {"role": "system", "content": prompt},
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

        from videoclaw.drama.locale import get_locale
        locale = get_locale(series.language)
        prompt = locale.episode_script_prompt or EPISODE_SCRIPT_PROMPT
        raw = await llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
        )

        script_data = self._parse_json(raw)

        # --- Duration validation ---
        scenes = script_data.get("scenes", [])

        # Step 1: Proportional scaling to match target duration
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

        # Step 2: Clamp individual scene durations to Seedance 2.0 range (5-15s)
        for s in scenes:
            dur = float(s.get("duration_seconds", 5.0))
            clamped = max(5.0, min(15.0, dur))
            if clamped != dur:
                logger.debug(
                    "Scene %s duration %.1fs clamped to %.1fs (Seedance 5-15s range)",
                    s.get("scene_id", "?"), dur, clamped,
                )
            s["duration_seconds"] = clamped

        # Step 3: Enforce dialogue pacing floor (dialogue drives duration, not vice versa)
        _enforce_pacing(scenes)

        episode.script = json.dumps(script_data, ensure_ascii=False)
        episode.scenes = [DramaScene.from_dict(s) for s in scenes]
        return script_data

    # ------------------------------------------------------------------
    # Complete-script import (decompose-only, no creative modification)
    # ------------------------------------------------------------------

    @staticmethod
    def read_script_file(path: str | Path) -> str:
        """Read a complete script from a file (.docx or .txt).

        Returns the raw text content of the script.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Script file not found: {path}")

        if path.suffix.lower() == ".docx":
            try:
                from docx import Document  # type: ignore[import-untyped]
            except ImportError:
                raise ImportError(
                    "python-docx is required to read .docx files. "
                    "Install it with: pip install python-docx"
                )
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            return path.read_text(encoding="utf-8")

    async def import_complete_script(
        self,
        series: DramaSeries,
        script_text: str,
        *,
        confirm_callback: Callable[[list[ScriptModification]], list[ScriptModification]] | None = None,
    ) -> DramaSeries:
        """Import a complete, finalized script and decompose it into shots.

        This method sets ``script_locked=True`` on the series, preventing any
        creative modifications. The LLM is used ONLY for decomposing scenes
        into Seedance 2.0-compatible shot descriptions.

        Parameters
        ----------
        series:
            The target series to populate.
        script_text:
            The complete script text to import.
        confirm_callback:
            Optional callback invoked when gaps are detected. Receives a list
            of :class:`ScriptModification` proposals and must return the
            approved subset. In CLI mode this prompts the user interactively.
            If ``None``, gaps are logged but no automatic fixes are applied.

        Returns
        -------
        DramaSeries
            The series with script_locked=True and all episodes/scenes populated.
        """
        logger.info(
            "Importing complete script for series %r (script_locked=True)",
            series.title or series.series_id,
        )

        series.script_locked = True
        series.script_source = "imported"
        series.status = DramaStatus.PLANNING

        llm = self._ensure_llm()

        # Build the user message with the full script and character info
        characters_text = ""
        if series.characters:
            characters_text = "\nKnown characters:\n" + "\n".join(
                f"  - {c.name}: {c.visual_prompt} ({c.description})"
                for c in series.characters
            )

        max_dur = series.target_episode_duration or 60.0
        user_message = (
            f"Series: {series.title}\n"
            f"Genre: {series.genre or 'drama'}\n"
            f"Language: {series.language}\n"
            f"Target aspect ratio: {series.aspect_ratio}\n"
            f"Maximum episode duration: {max_dur:.0f} seconds (MUST NOT exceed)\n"
            f"Video model: Seedance 2.0 (5-15s per clip, audio co-generation)\n"
            f"HARD CONSTRAINT: 6-10 shots. NEVER exceed 12 shots. "
            f"Sum of durations MUST NOT exceed {max_dur:.0f}s.\n"
            f"{characters_text}\n\n"
            f"=== COMPLETE SCRIPT (DO NOT MODIFY) ===\n\n"
            f"{script_text}\n\n"
            f"=== END OF SCRIPT ===\n\n"
            f"Decompose this script into shot-by-shot storyboard. "
            f"Do NOT add or change any content."
        )

        # Retry up to 3 times for JSON parse failures
        last_error: Exception | None = None
        for attempt in range(3):
            raw = await llm.chat(
                messages=[
                    {"role": "system", "content": IMPORT_DECOMPOSE_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=16384,
            )
            try:
                result = self._parse_json(raw)
                break
            except ValueError as e:
                last_error = e
                logger.warning("JSON parse attempt %d/3 failed: %s", attempt + 1, e)
        else:
            raise last_error  # type: ignore[misc]

        # --- Extract characters ---
        if result.get("characters"):
            series.characters = [
                Character.from_dict(c) for c in result["characters"]
            ]

        # --- Extract episodes and scenes ---
        episodes_data = result.get("episodes", [])
        series.episodes = []
        for ep_data in episodes_data:
            scenes = [DramaScene.from_dict(s) for s in ep_data.get("scenes", [])]

            # Clamp durations to Seedance 2.0 range
            for scene in scenes:
                scene.duration_seconds = max(5.0, min(15.0, scene.duration_seconds))

            # Enforce dialogue pacing floor on raw dicts before DramaScene construction
            raw_scenes = ep_data.get("scenes", [])
            _enforce_pacing(raw_scenes)
            for scene, raw in zip(scenes, raw_scenes):
                scene.duration_seconds = float(raw.get("duration_seconds", scene.duration_seconds))

            episode = Episode(
                number=ep_data.get("number", len(series.episodes) + 1),
                title=ep_data.get("title", f"Episode {len(series.episodes) + 1}"),
                scenes=scenes,
                duration_seconds=sum(s.duration_seconds for s in scenes),
                status=EpisodeStatus.PENDING,
                script=json.dumps(ep_data, ensure_ascii=False),
            )
            series.episodes.append(episode)

        series.total_episodes = len(series.episodes)

        # --- Handle detected gaps ---
        gaps = result.get("detected_gaps", [])
        if gaps:
            logger.warning("Detected %d gaps in imported script:", len(gaps))
            modifications: list[ScriptModification] = []
            for gap in gaps:
                mod = ScriptModification(
                    scene_id=gap.get("scene_id", ""),
                    field_name=gap.get("field", ""),
                    reason=gap.get("description", ""),
                    original_value="",
                    proposed_value="",
                )
                modifications.append(mod)
                logger.warning(
                    "  [%s] %s: %s",
                    mod.scene_id or "global",
                    mod.field_name,
                    mod.reason,
                )

            if confirm_callback is not None:
                approved = confirm_callback(modifications)
                series.pending_modifications = [
                    m for m in modifications if m not in approved
                ]
            else:
                series.pending_modifications = modifications

        # --- Build consistency manifest ---
        series.consistency_manifest = self._build_consistency_manifest(series)

        series.status = DramaStatus.DRAFT
        logger.info(
            "Script imported: %d episodes, %d total scenes, script_locked=True",
            len(series.episodes),
            sum(len(ep.scenes) for ep in series.episodes),
        )
        return series

    @staticmethod
    def _build_consistency_manifest(series: DramaSeries) -> ConsistencyManifest:
        """Build a ConsistencyManifest from the current series state.

        Freezes character visuals and reference image paths so they remain
        identical across all generated clips.
        """
        manifest = ConsistencyManifest(
            style_anchor=series.style,
        )

        for char in series.characters:
            if char.visual_prompt:
                manifest.character_visuals[char.name] = char.visual_prompt
            if char.reference_image:
                manifest.character_references[char.name] = char.reference_image
            if char.reference_images:
                manifest.character_multi_references[char.name] = list(char.reference_images)

        # Extract unique scene settings from all episodes
        for ep in series.episodes:
            for scene in ep.scenes:
                if scene.scene_id and scene.visual_prompt:
                    manifest.scene_settings[scene.scene_id] = scene.visual_prompt

        manifest.verify_references()
        return manifest

    def guard_script_locked(
        self,
        series: DramaSeries,
        operation: str,
        *,
        confirm_callback: Callable[[list[ScriptModification]], list[ScriptModification]] | None = None,
    ) -> bool:
        """Check if a modification is allowed on a locked script.

        When ``series.script_locked`` is True, this method creates a
        :class:`ScriptModification` record and invokes the confirm callback.
        Returns True if the operation is approved, False otherwise.
        """
        if not series.script_locked:
            return True

        mod = ScriptModification(
            reason=f"Attempted operation on locked script: {operation}",
        )

        logger.warning(
            "Script is locked (source=%s). Operation %r requires user approval.",
            series.script_source,
            operation,
        )

        if confirm_callback is not None:
            approved = confirm_callback([mod])
            return len(approved) > 0

        # No callback — reject by default
        series.pending_modifications.append(mod)
        return False

    @staticmethod
    def _parse_json(raw_response: str) -> dict[str, Any]:
        text = raw_response.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            first_nl = text.index("\n")
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try fixing common LLM JSON issues: unescaped quotes in strings
        # e.g., "dialogue": "He said "hello" to her" -> needs escaping
        import re
        fixed = text
        # Fix unescaped newlines in strings
        fixed = re.sub(r'(?<=": ")([^"]*?)\n([^"]*?)(?=")', r'\1\\n\2', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Find the outermost JSON object { ... }
        brace_start = text.find("{")
        if brace_start >= 0:
            # Scan for the last balanced closing brace
            depth = 0
            last_brace = -1
            in_string = False
            escape = False
            for i in range(brace_start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        last_brace = i
            if last_brace > brace_start:
                try:
                    return json.loads(text[brace_start : last_brace + 1])
                except json.JSONDecodeError:
                    pass

            # Truncated JSON — try closing progressively from the end
            candidate = text[brace_start:]
            for i in range(len(candidate) - 1, max(len(candidate) - 500, 0), -1):
                if candidate[i] in ("}", "]"):
                    try:
                        return json.loads(candidate[: i + 1])
                    except json.JSONDecodeError:
                        continue

        logger.error(
            "Failed to parse LLM JSON (%d chars). First 300: %s",
            len(text), text[:300],
        )
        raise ValueError("LLM returned invalid JSON — retry the request.")
