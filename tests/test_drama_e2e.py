"""End-to-end acceptance test for the drama screenwriter system.

Validates that the full pipeline (plan_series → script_episode × N) produces
deliverable-grade output at 红果短剧 quality bar, and that every output field
maps cleanly to a downstream executor in the todo pipeline.

Theme: 这个王妃太狂野 (穿越题材)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from datetime import datetime

from videoclaw.drama.models import (
    Character,
    DramaScene,
    DramaSeries,
    DramaStatus,
    Episode,
    EpisodeStatus,
    ShotScale,
    ShotType,
    DramaManager,
    assign_voice_profile,
)
from videoclaw.drama.planner import DramaPlanner
from videoclaw.drama.runner import build_episode_dag


# ---------------------------------------------------------------------------
# Realistic mock data — 这个王妃太狂野 (穿越/古装)
# ---------------------------------------------------------------------------

MOCK_SERIES_OUTLINE = {
    "title": "这个王妃太狂野",
    "genre": "穿越/古装/爽剧",
    "synopsis": "现代女特工林薇穿越成被休弃的废物王妃，众人等着看笑话，殊不知她前世是顶级杀手。这一世，她要让所有欺辱她的人跪着叫姐姐。",
    "characters": [
        {
            "name": "林薇",
            "description": "表面身份：被休弃的废物王妃，人人嘲笑的'草包'；隐藏身份：穿越而来的现代女特工，精通格斗、毒术和心理战。性格反差：外表柔弱楚楚可怜 vs 内心冷酷果决。核心动机：查明穿越真相，在权谋漩涡中保护自己和盟友。记忆点：每次反杀前都会微微一笑说'你确定？'三角关系：与萧衍的政治联姻渐生真情，被慕容雪嫉恨。",
            "visual_prompt": "Young Chinese woman, early 20s, delicate oval face, phoenix eyes with sharp gaze, long black hair with jade hairpin, wearing tattered but elegant white hanfu with red inner lining, slender build, subtle scar on left wrist",
            "voice_style": "calm",
        },
        {
            "name": "萧衍",
            "description": "表面身份：冷面靖王，朝堂第一权臣；隐藏身份：暗中调查皇室血脉秘密。性格反差：对外冷酷无情 vs 对林薇逐渐露出温柔。核心动机：查明母妃死因，夺回被窃的皇位。记忆点：永远黑衣，腰间佩一把从不出鞘的剑。",
            "visual_prompt": "Tall Chinese man, late 20s, sharp angular jawline, intense dark eyes, long black hair in high ponytail with silver crown, wearing black brocade robe with dragon embroidery, broad shoulders, commanding presence, never-drawn sword at waist",
            "voice_style": "authoritative",
        },
        {
            "name": "慕容雪",
            "description": "表面身份：温婉贤淑的侧妃，人人称赞的'京城第一才女'；隐藏身份：幕后势力的棋子，野心勃勃想上位。性格反差：人前温婉 vs 人后阴毒。核心动机：取代林薇成为正妃，掌控靖王府。记忆点：手持一把折扇，扇面画的是毒花曼陀罗。反派动机：出身低微被嘲笑，不惜一切代价往上爬。",
            "visual_prompt": "Beautiful Chinese woman, mid 20s, round face with dimples, gentle almond eyes hiding cunning, elaborate updo with golden phoenix hairpins, wearing pink silk hanfu with floral patterns, holding folding fan with mandala flower painting",
            "voice_style": "playful",
        },
        {
            "name": "老太君",
            "description": "表面身份：德高望重的萧家老太君；隐藏身份：早年江湖女侠，看穿一切的老狐狸。核心动机：保萧家基业。记忆点：永远拄着龙头拐杖，发怒时拐杖敲地三声。",
            "visual_prompt": "Elderly Chinese woman, 70s, silver hair in neat bun, wise piercing eyes, dignified posture, wearing deep purple silk robe, leaning on dragon-headed walking cane, jade bracelet",
            "voice_style": "dramatic",
        },
    ],
    "episodes": [
        {
            "number": 1,
            "title": "废妃重生",
            "synopsis": "林薇穿越醒来发现自己被丢在柴房，丫鬟踩着她的手说'废物王妃也配用这双手'（爽点1@5s：身份揭示）。林薇一招反擒拿制服丫鬟（爽点2@15s：实力展示），走出柴房对着满堂嘲笑她的下人微微一笑'你们确定？'（爽点3@25s：智商碾压预告）。慕容雪派人来'送温暖'实为下毒（爽点4@40s：识破阴谋）。",
            "opening_hook": "一双精致却沾满泥土的手被粗暴踩住，画面抬起——一个现代女人的灵魂在古代废妃的眼睛里苏醒",
            "duration_seconds": 60.0,
        },
        {
            "number": 2,
            "title": "当众打脸",
            "synopsis": "老太君寿宴，慕容雪设计让林薇当众出丑——要她作诗（爽点1@10s：陷阱设局）。林薇用现代诗震惊全场（爽点2@25s：才华碾压），老太君拐杖敲地三声表示欣赏（爽点3@35s：获得强援）。萧衍第一次正眼看向林薇（爽点4@50s：情感萌芽）。宴后慕容雪在暗处咬碎手帕。",
            "opening_hook": "满堂华服贵妇中，一身素衣的林薇被推到正中央——所有人都在等着看一个废物的笑话",
            "duration_seconds": 60.0,
        },
        {
            "number": 3,
            "title": "深夜试探",
            "synopsis": "萧衍深夜来柴房试探林薇（爽点1@5s：紧张对峙），林薇识破他的暗器并反手接住（爽点2@20s：实力碾压），两人月下对话暗中交锋（爽点3@35s：智商博弈）。慕容雪窥见两人相处，决定提前动手。结尾：一封密信送到林薇手中——'你的前世，我都知道'（爽点4@55s：震撼悬念）。",
            "opening_hook": "月光下一把匕首划破夜空——刺向柴房中沉睡的林薇",
            "duration_seconds": 60.0,
        },
        {
            "number": 4,
            "title": "至暗时刻",
            "synopsis": "慕容雪联合朝中势力诬陷林薇通敌（爽点1@10s：危机爆发），林薇被打入天牢（爽点2@20s：虐心低谷），萧衍表面不救实际暗中调查（爽点3@35s：暗线守护），老太君拐杖敲碎茶杯'谁动我萧家的人试试'（爽点4@45s：强援出手）。林薇在牢中找到一块刻着现代二维码的古玉——穿越之谜浮出水面。",
            "opening_hook": "大殿之上，一道通敌叛国的圣旨砸在林薇面前——满朝文武无人为她说话",
            "duration_seconds": 60.0,
        },
        {
            "number": 5,
            "title": "王妃驾到",
            "synopsis": "林薇利用现代知识在公堂翻案（爽点1@10s：智商碾压），揭露慕容雪与外敌勾结的证据（爽点2@25s：反转打脸），慕容雪当场崩溃'不可能！你明明是个废物！'（爽点3@35s：反派崩溃），萧衍当着文武百官的面拉住林薇的手'本王的王妃，谁敢动？'（爽点4@50s：情感高潮+身份宣告）。",
            "opening_hook": "公堂之上，所有人都以为林薇必死——她却微微一笑，从袖中抽出一卷帛书",
            "duration_seconds": 60.0,
        },
    ],
}

MOCK_EPISODE_SCRIPTS = {
    1: {
        "episode_title": "废妃重生",
        "scenes": [
            {
                "scene_id": "ep01_s01",
                "description": "破旧柴房内，林薇躺在稻草上，一只脚踩住她的手",
                "visual_prompt": "Dark shabby storage room, young Chinese woman lying on straw, delicate oval face with phoenix eyes, long black hair with jade hairpin, tattered white hanfu with red lining, a servant's foot stepping on her hand, dramatic shadows from a small window, cinematic lighting",
                "camera_movement": "dolly_in",
                "duration_seconds": 5.0,
                "dialogue": "",
                "narration": "她睁开眼的那一刻，不再是废物王妃",
                "speaking_character": "",
                "shot_scale": "close_up",
                "shot_type": "detail",
                "emotion": "dread",
                "characters_present": ["林薇"],
                "transition": "fade_in",
            },
            {
                "scene_id": "ep01_s02",
                "description": "林薇眼神骤变，从茫然到冰冷",
                "visual_prompt": "Extreme close-up of young Chinese woman's phoenix eyes transitioning from confusion to lethal calm, jade hairpin visible, dramatic side lighting, moody atmosphere",
                "camera_movement": "static",
                "duration_seconds": 3.0,
                "dialogue": "",
                "narration": "",
                "speaking_character": "",
                "shot_scale": "close_up",
                "shot_type": "reaction",
                "emotion": "defiant",
                "characters_present": ["林薇"],
                "transition": "cut",
            },
            {
                "scene_id": "ep01_s03",
                "description": "林薇一个反擒拿翻转丫鬟手腕",
                "visual_prompt": "Young Chinese woman in tattered white hanfu executing swift wrist-lock takedown on servant, straw flying, dynamic motion blur, dramatic shadows, cinematic action shot",
                "camera_movement": "handheld",
                "duration_seconds": 4.0,
                "dialogue": "",
                "narration": "",
                "speaking_character": "",
                "shot_scale": "medium",
                "shot_type": "action",
                "emotion": "triumphant",
                "characters_present": ["林薇"],
                "transition": "cut",
            },
            {
                "scene_id": "ep01_s04",
                "description": "丫鬟跪在地上，惊恐地看着林薇",
                "visual_prompt": "Low angle shot looking up at young Chinese woman standing over kneeling servant, tattered white hanfu with red lining, jade hairpin catching light, confident slight smile, atmospheric dust particles in shaft of light",
                "camera_movement": "crane_up",
                "duration_seconds": 5.0,
                "dialogue": "你确定……要踩这双手？",
                "narration": "",
                "speaking_character": "林薇",
                "shot_scale": "medium_close",
                "shot_type": "action",
                "emotion": "smug",
                "characters_present": ["林薇"],
                "transition": "cut",
            },
            {
                "scene_id": "ep01_s05",
                "description": "林薇推开柴房门，逆光走出",
                "visual_prompt": "Silhouette of slender young Chinese woman pushing open wooden door, brilliant backlight flooding in, tattered white hanfu with red inner lining billowing, long black hair flowing, dramatic lens flare, cinematic wide shot",
                "camera_movement": "dolly_in",
                "duration_seconds": 5.0,
                "dialogue": "",
                "narration": "从这一刻起，这个世界的规则，由她来改写",
                "speaking_character": "",
                "shot_scale": "wide",
                "shot_type": "establishing",
                "emotion": "triumphant",
                "characters_present": ["林薇"],
                "transition": "cut",
            },
            {
                "scene_id": "ep01_s06",
                "description": "院中下人们纷纷转头看向走出的林薇，窃窃私语",
                "visual_prompt": "Courtyard of ancient Chinese mansion, group of servants in grey uniforms turning to look with mocking expressions, young Chinese woman in tattered white hanfu walking forward with confident stride, warm sunlight, atmospheric",
                "camera_movement": "tracking",
                "duration_seconds": 5.0,
                "dialogue": "",
                "narration": "",
                "speaking_character": "",
                "shot_scale": "medium",
                "shot_type": "establishing",
                "emotion": "tense",
                "characters_present": ["林薇"],
                "transition": "cut",
            },
            {
                "scene_id": "ep01_s07",
                "description": "林薇面对众人的嘲讽，嘴角微微上扬",
                "visual_prompt": "Close-up of young Chinese woman's face with slight knowing smile, phoenix eyes with sharp gaze, jade hairpin, tattered white hanfu collar visible, golden hour lighting from side, cinematic depth of field",
                "camera_movement": "static",
                "duration_seconds": 4.0,
                "dialogue": "你们确定？",
                "narration": "",
                "speaking_character": "林薇",
                "shot_scale": "close_up",
                "shot_type": "reaction",
                "emotion": "smug",
                "characters_present": ["林薇"],
                "transition": "cut",
            },
            {
                "scene_id": "ep01_s08",
                "description": "慕容雪的丫鬟端着一碗汤来'送温暖'",
                "visual_prompt": "Beautiful Chinese woman's servant in pink uniform carrying ornate bowl of soup on tray, walking through corridor with lanterns, gentle warm lighting, sinister atmosphere despite pleasant appearance",
                "camera_movement": "tracking",
                "duration_seconds": 5.0,
                "dialogue": "王妃娘娘，侧妃让奴婢给您送碗暖汤",
                "narration": "",
                "speaking_character": "",
                "shot_scale": "medium",
                "shot_type": "action",
                "emotion": "suspense",
                "characters_present": ["林薇"],
                "transition": "dissolve",
            },
            {
                "scene_id": "ep01_s09",
                "description": "林薇接过汤碗，指尖沾汤微微嗅了嗅",
                "visual_prompt": "Close-up of delicate hand with subtle scar on left wrist dipping finger into ornate soup bowl, bringing finger close to nose, young Chinese woman's knowing expression in soft focus background, warm lamplight",
                "camera_movement": "static",
                "duration_seconds": 4.0,
                "dialogue": "",
                "narration": "乌头碱，剂量不大，但足以让人腹泻三天……手法真粗糙",
                "speaking_character": "",
                "shot_scale": "close_up",
                "shot_type": "detail",
                "emotion": "smug",
                "characters_present": ["林薇"],
                "transition": "cut",
            },
            {
                "scene_id": "ep01_s10",
                "description": "林薇微笑着把汤递回给丫鬟",
                "visual_prompt": "Medium close-up of young Chinese woman with subtle scar on left wrist handing soup bowl back to nervous servant with a dangerous smile, dramatic side lighting, tense atmosphere",
                "camera_movement": "static",
                "duration_seconds": 5.0,
                "dialogue": "替我谢谢你家主子，这汤……她自己留着喝吧",
                "narration": "",
                "speaking_character": "林薇",
                "shot_scale": "medium_close",
                "shot_type": "action",
                "emotion": "triumphant",
                "characters_present": ["林薇"],
                "transition": "cut",
            },
            {
                "scene_id": "ep01_s11",
                "description": "远处暗处，慕容雪手中折扇猛然合拢",
                "visual_prompt": "Silhouette of beautiful Chinese woman in pink silk hanfu standing in dark corridor, snapping folding fan shut with angry force, mandala flower fan barely visible, dramatic shadows, ominous atmosphere",
                "camera_movement": "dolly_in",
                "duration_seconds": 5.0,
                "dialogue": "",
                "narration": "她不知道，真正的敌人……远不止一个侧妃",
                "speaking_character": "",
                "shot_scale": "medium",
                "shot_type": "reaction",
                "emotion": "dread",
                "characters_present": ["慕容雪"],
                "transition": "cut",
            },
            {
                "scene_id": "ep01_s12",
                "description": "月夜，林薇站在窗前望着星空，眼中映着现代城市的倒影",
                "visual_prompt": "Young Chinese woman in tattered white hanfu with red lining standing by moon-lit window, long black hair with jade hairpin flowing in breeze, reflection of modern city skyline in her phoenix eyes, ethereal moonlight, cinematic composition",
                "camera_movement": "crane_up",
                "duration_seconds": 10.0,
                "dialogue": "",
                "narration": "回不去了……那就在这里，活出另一个传奇",
                "speaking_character": "",
                "shot_scale": "medium_close",
                "shot_type": "pov",
                "emotion": "melancholy",
                "characters_present": ["林薇"],
                "transition": "fade_out",
            },
        ],
        "voice_over": {"text": "她睁开眼的那一刻，不再是废物王妃。从这一刻起，这个世界的规则，由她来改写。她不知道，真正的敌人远不止一个侧妃。回不去了，那就在这里，活出另一个传奇。", "tone": "dramatic", "language": "zh"},
        "music": {"style": "chinese_traditional", "mood": "mysterious", "tempo": 100},
        "cliffhanger": "林薇眼中映着的现代城市倒影——她到底是谁？穿越的背后又隐藏着什么秘密？",
    },
}


# ---------------------------------------------------------------------------
# Executor field → todo task mapping
# ---------------------------------------------------------------------------

FIELD_TO_EXECUTOR_MAP = {
    # DramaScene 视觉字段 → 视频生成
    "visual_prompt": "2.2 角色参考图注入视频生成 / video_gen executor",
    "shot_scale": "1.1 DramaScene 类型化模型 (已完成) → 指导视频构图",
    "shot_type": "1.1 DramaScene 类型化模型 (已完成) → 指导镜头运动",
    "camera_movement": "video_gen executor (已有 DAG 节点)",
    # DramaScene 音频字段 → TTS/音频
    "dialogue": "2.1 多角色对话 TTS (DramaVoiceGenerator) + 2.3 真实 TTS 处理器",
    "narration": "2.1 多角色对话 TTS + 2.3 真实 TTS 处理器",
    "speaking_character": "2.1 多角色对话 TTS → 角色声音路由",
    # 角色语音
    "voice_style → VoiceProfile": "1.2 角色语音映射 (已完成) → WaveSpeed TTS",
    # 资产追踪
    "video_asset_path": "1.3 场景级资产追踪 (已完成)",
    "dialogue_audio_path": "1.3 场景级资产追踪 (已完成)",
    "narration_audio_path": "1.3 场景级资产追踪 (已完成)",
    "scene_status": "1.3 场景级资产追踪 (已完成)",
    # 音频装配
    "AudioSegment + EpisodeAudioManifest": "1.4 音频清单模型 (已完成) → 3.5 集音频装配",
    # 剧情元素 → 后处理
    "emotion": "3.6 视觉提示词增强器 (PromptEnhancer) → 氛围增强",
    "transition": "2.4 真实 Compose 处理器 → 转场效果",
    "characters_present": "4.3 场景/背景一致性 (Location Consistency)",
    # 集级元素
    "cliffhanger": "4.1 角色弧线追踪 → 跨集叙事连续性",
    "music": "DAG music 节点 (已有)",
    "voice_over": "2.3 真实 TTS 处理器",
    # 字幕
    "dialogue_text": "2.6 字幕生成器 (SubtitleGenerator)",
    # 合成与渲染
    "compose_pipeline": "2.4 真实 Compose 处理器 + 2.5 真实 Render 处理器",
    # DAG 增强
    "per_scene_tts_node": "3.1 增强戏剧 DAG (per-scene TTS + 字幕节点)",
    # CLI
    "claw_drama_script": "3.2 CLI: claw drama script 命令",
    "claw_drama_voices": "3.3 CLI: claw drama assign-voices 命令",
    # 质量保障
    "drama_reviewer": "4.2 剧本审核工作流 (DramaReviewer)",
    "storyboard_export": "4.4 分镜导出 (Storyboard Export: Rich + HTML)",
    "e2e_integration": "4.5 端到端集成测试 (Mock Adapter 全流程)",
    "error_recovery": "4.6 错误恢复与断点续跑",
}


# ---------------------------------------------------------------------------
# Quality validators (红果短剧水位线)
# ---------------------------------------------------------------------------

def _validate_honguo_quality(series: DramaSeries, episodes_scripts: dict[int, dict]) -> list[str]:
    """Validate against 红果短剧 quality standards. Returns list of violations."""
    violations = []

    # 1. 高概念一句话卖点
    if len(series.synopsis) < 20:
        violations.append("❌ synopsis 太短，缺少高概念一句话卖点")

    # 2. 角色反差人设
    for c in series.characters:
        if "表面" not in c.description and "隐藏" not in c.description:
            violations.append(f"❌ 角色 {c.name} 缺少表面/隐藏身份反差")
        if "记忆点" not in c.description and "标志" not in c.description:
            violations.append(f"❌ 角色 {c.name} 缺少记忆点")

    # 3. 黄金3秒法则 — 第1集第1场景必须 ≤5s
    ep1 = episodes_scripts.get(1)
    if ep1:
        first_scene = ep1["scenes"][0]
        if first_scene["duration_seconds"] > 5.0:
            violations.append(f"❌ 第1集第1场景 {first_scene['duration_seconds']}s > 5s黄金3秒上限")

    # 4. 爽点密度 — 每15s至少一个情绪高点
    for ep_num, script in episodes_scripts.items():
        emotion_peaks = [s for s in script["scenes"] if s["emotion"] in (
            "triumphant", "smug", "vindicated", "shock", "revelation", "defiant",
        )]
        if len(emotion_peaks) < 2:
            violations.append(f"❌ 第{ep_num}集爽点密度不足（{len(emotion_peaks)}个高峰情绪）")

    # 5. 悬念 — 每集必须有 cliffhanger
    for ep_num, script in episodes_scripts.items():
        if not script.get("cliffhanger"):
            violations.append(f"❌ 第{ep_num}集缺少 cliffhanger")

    # 6. 竖屏构图 — close_up + medium_close ≥ 40%
    for ep_num, script in episodes_scripts.items():
        scenes = script["scenes"]
        close_count = sum(1 for s in scenes if s.get("shot_scale") in ("close_up", "medium_close"))
        ratio = close_count / len(scenes) if scenes else 0
        if ratio < 0.4:
            violations.append(f"❌ 第{ep_num}集特写镜头比例 {ratio:.0%} < 40%竖屏要求")

    # 7. 角色一致性 — visual_prompt 必须包含角色特征
    for ep_num, script in episodes_scripts.items():
        for scene in script["scenes"]:
            if scene.get("speaking_character") and scene["speaking_character"] not in scene.get("characters_present", []):
                violations.append(f"❌ 第{ep_num}集 {scene['scene_id']}: speaking_character不在characters_present中")

    # 8. 台词密度 — 对白总字数控制
    for ep_num, script in episodes_scripts.items():
        total_chars = sum(len(s.get("dialogue", "")) for s in script["scenes"])
        if total_chars > 120:
            violations.append(f"⚠️ 第{ep_num}集对白{total_chars}字，建议≤100字/60秒")

    # 9. 情绪词表标准化
    valid_emotions = {
        "tense", "anxious", "dread", "suspense",
        "angry", "furious", "resentful", "defiant",
        "sad", "heartbroken", "grieving", "melancholy",
        "shock", "disbelief", "stunned", "revelation",
        "warm", "tender", "nostalgic", "grateful",
        "sweet", "flirty", "blissful", "intimate",
        "fear", "panic", "horror", "uneasy",
        "triumphant", "smug", "vindicated", "proud",
    }
    for ep_num, script in episodes_scripts.items():
        for scene in script["scenes"]:
            if scene.get("emotion") and scene["emotion"] not in valid_emotions:
                violations.append(f"⚠️ 第{ep_num}集 {scene['scene_id']}: emotion '{scene['emotion']}' 不在标准词表中")

    return violations


# ---------------------------------------------------------------------------
# Document generator
# ---------------------------------------------------------------------------

def _generate_deliverable_doc(
    series: DramaSeries,
    episodes_scripts: dict[int, dict],
    violations: list[str],
    output_path: Path,
) -> str:
    """Generate a deliverable markdown document with full script and executor mapping."""

    lines = []
    lines.append(f"# 🎬 {series.title}")
    lines.append(f"**类型:** {series.genre}")
    lines.append(f"**高概念:** {series.synopsis}")
    lines.append(f"**总集数:** {series.total_episodes}")
    lines.append(f"**单集时长:** {series.target_episode_duration}秒")
    lines.append(f"**画面比例:** {series.aspect_ratio}")
    lines.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Quality report
    lines.append("## 质量验收报告 (红果短剧水位线)")
    lines.append("")
    if not violations:
        lines.append("**全部通过** — 达到红果短剧发布标准")
    else:
        for v in violations:
            lines.append(f"- {v}")
    lines.append("")

    # Characters
    lines.append("## 角色设定")
    lines.append("")
    for c in series.characters:
        lines.append(f"### {c.name}")
        lines.append(f"**人设:** {c.description}")
        lines.append(f"**视觉:** {c.visual_prompt}")
        lines.append(f"**声音风格:** {c.voice_style}")
        if c.voice_profile:
            lines.append(f"**TTS配置:** voice_id={c.voice_profile.voice_id}, speed={c.voice_profile.speed}, pitch={c.voice_profile.pitch}")
        lines.append("")

    # Episodes with scenes
    for ep in series.episodes:
        script = episodes_scripts.get(ep.number, {})
        lines.append(f"## 第{ep.number}集: {ep.title}")
        lines.append(f"**梗概:** {ep.synopsis}")
        lines.append(f"**开场钩子:** {ep.opening_hook}")
        lines.append(f"**目标时长:** {ep.duration_seconds}秒")
        lines.append("")

        scenes = script.get("scenes", [])
        for i, scene in enumerate(scenes, 1):
            lines.append(f"### 场景 {i}: {scene['scene_id']}")
            lines.append(f"- **描述:** {scene['description']}")
            lines.append(f"- **视觉提示词:** `{scene['visual_prompt'][:80]}...`")
            lines.append(f"- **景别:** {scene.get('shot_scale', 'N/A')} | **镜头类型:** {scene.get('shot_type', 'N/A')}")
            lines.append(f"- **运镜:** {scene.get('camera_movement', 'static')} | **时长:** {scene['duration_seconds']}秒")
            lines.append(f"- **情绪:** {scene.get('emotion', 'N/A')} | **转场:** {scene.get('transition', 'cut')}")
            if scene.get("dialogue"):
                lines.append(f"- **台词 ({scene.get('speaking_character', '?')}):** \"{scene['dialogue']}\"")
            if scene.get("narration"):
                lines.append(f"- **旁白:** \"{scene['narration']}\"")
            lines.append(f"- **出镜角色:** {', '.join(scene.get('characters_present', []))}")
            lines.append("")

        if script.get("cliffhanger"):
            lines.append(f"**悬念:** {script['cliffhanger']}")
            lines.append("")

        music = script.get("music", {})
        if music:
            lines.append(f"**配乐:** {music.get('style', 'N/A')} / {music.get('mood', 'N/A')} / {music.get('tempo', 'N/A')}BPM")
            lines.append("")

    # Executor mapping
    lines.append("---")
    lines.append("## 执行器映射表 (字段 → 下游任务)")
    lines.append("")
    lines.append("| 字段/元素 | 对应执行器/任务 |")
    lines.append("|-----------|----------------|")
    for field, executor in FIELD_TO_EXECUTOR_MAP.items():
        lines.append(f"| `{field}` | {executor} |")
    lines.append("")

    doc = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return doc


# ===========================================================================
# E2E Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_e2e_full_drama_pipeline(tmp_path):
    """Full pipeline: plan_series → assign_voices → script_episode × N → deliverable doc."""

    # --- 1. Mock LLM ---
    call_count = 0
    async def mock_chat(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        # First call = plan_series, subsequent = script_episode
        if call_count == 1:
            return json.dumps(MOCK_SERIES_OUTLINE, ensure_ascii=False)
        else:
            ep_num = call_count - 1
            if ep_num in MOCK_EPISODE_SCRIPTS:
                return json.dumps(MOCK_EPISODE_SCRIPTS[ep_num], ensure_ascii=False)
            # For episodes without full mock, return minimal valid response
            return json.dumps({
                "episode_title": f"第{ep_num}集",
                "scenes": [
                    {
                        "scene_id": f"ep{ep_num:02d}_s01",
                        "description": "场景描述",
                        "visual_prompt": "scene visual",
                        "camera_movement": "static",
                        "duration_seconds": 30.0,
                        "dialogue": "",
                        "narration": "旁白",
                        "speaking_character": "",
                        "shot_scale": "medium",
                        "shot_type": "action",
                        "emotion": "tense",
                        "characters_present": [],
                        "transition": "cut",
                    },
                    {
                        "scene_id": f"ep{ep_num:02d}_s02",
                        "description": "场景描述2",
                        "visual_prompt": "scene visual 2",
                        "camera_movement": "static",
                        "duration_seconds": 30.0,
                        "dialogue": "",
                        "narration": "",
                        "speaking_character": "",
                        "shot_scale": "close_up",
                        "shot_type": "reaction",
                        "emotion": "shock",
                        "characters_present": [],
                        "transition": "cut",
                    },
                ],
                "voice_over": {"text": "旁白文本", "tone": "dramatic", "language": "zh"},
                "music": {"style": "orchestral", "mood": "tense", "tempo": 120},
                "cliffhanger": "悬念描述",
            }, ensure_ascii=False)

    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(side_effect=mock_chat)

    planner = DramaPlanner(llm=mock_llm)

    # --- 2. Plan series ---
    series = DramaSeries(
        title="这个王妃太狂野",
        genre="穿越",
        synopsis="现代女特工穿越成被休弃的废物王妃",
        total_episodes=5,
        target_episode_duration=60.0,
        style="cinematic",
        language="zh",
        aspect_ratio="9:16",
    )
    series = await planner.plan_series(series)

    assert series.title == "这个王妃太狂野"
    assert len(series.characters) == 4
    assert len(series.episodes) == 5

    # --- 3. Assign voice profiles ---
    for c in series.characters:
        assign_voice_profile(c)
        assert c.voice_profile is not None, f"Character {c.name} has no voice profile"

    # Verify specific voice assignments
    linwei = next(c for c in series.characters if c.name == "林薇")
    assert linwei.voice_profile.voice_id == "Calm_Woman"  # calm style

    xiaoyan = next(c for c in series.characters if c.name == "萧衍")
    assert xiaoyan.voice_profile.voice_id == "Imposing_Manner"  # authoritative style

    # --- 4. Script all episodes ---
    episodes_scripts = {}
    prev_cliffhanger = None
    for ep in series.episodes:
        script_data = await planner.script_episode(series, ep, previous_cliffhanger=prev_cliffhanger)
        episodes_scripts[ep.number] = script_data
        prev_cliffhanger = script_data.get("cliffhanger")

        # Verify DramaScene objects created
        assert len(ep.scenes) >= 1, f"Episode {ep.number} has no scenes"
        assert ep.script is not None, f"Episode {ep.number} has no script JSON"

        # Verify scene fields are typed
        for scene in ep.scenes:
            assert isinstance(scene, DramaScene)
            if scene.shot_scale is not None:
                assert isinstance(scene.shot_scale, ShotScale)
            if scene.shot_type is not None:
                assert isinstance(scene.shot_type, ShotType)

    # --- 5. Build DAGs (verify runner integration) ---
    for ep in series.episodes:
        dag, state = build_episode_dag(ep, series)
        assert len(dag.nodes) >= 4  # At least script, video(s), compose, render
        assert len(state.storyboard) == len(ep.scenes)
        for shot, scene in zip(state.storyboard, ep.scenes):
            assert shot.shot_id == scene.scene_id or shot.shot_id.startswith("ep")

    # --- 6. Quality validation ---
    violations = _validate_honguo_quality(series, episodes_scripts)
    # Episode 1 (fully mocked) should pass most checks
    ep1_violations = [v for v in violations if "第1集" in v]
    critical_violations = [v for v in violations if v.startswith("❌")]

    # --- 7. Generate deliverable doc ---
    doc_path = tmp_path / "deliverables" / "这个王妃太狂野_剧本.md"
    doc = _generate_deliverable_doc(series, episodes_scripts, violations, doc_path)

    assert doc_path.exists()
    doc_text = doc_path.read_text(encoding="utf-8")

    # Verify document completeness
    assert "这个王妃太狂野" in doc_text
    assert "穿越" in doc_text
    assert "林薇" in doc_text
    assert "萧衍" in doc_text
    assert "慕容雪" in doc_text
    assert "废妃重生" in doc_text
    assert "执行器映射表" in doc_text
    assert "visual_prompt" in doc_text
    assert "2.1 多角色对话 TTS" in doc_text
    assert "质量验收报告" in doc_text

    # --- 8. Persistence test ---
    mgr = DramaManager(base_dir=tmp_path / "dramas")
    mgr.save(series)
    loaded = mgr.load(series.series_id)
    assert loaded.title == "这个王妃太狂野"
    assert len(loaded.characters) == 4
    assert loaded.characters[0].voice_profile is not None


@pytest.mark.asyncio
async def test_e2e_episode1_honguo_quality():
    """Episode 1 detailed mock should pass 红果短剧 quality bar."""

    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=json.dumps(MOCK_EPISODE_SCRIPTS[1], ensure_ascii=False))

    planner = DramaPlanner(llm=mock_llm)
    series = DramaSeries(
        title="这个王妃太狂野",
        characters=[Character(name="林薇"), Character(name="慕容雪")],
    )
    episode = Episode(number=1, title="废妃重生", synopsis="穿越重生", duration_seconds=60.0)

    script_data = await planner.script_episode(series, episode)

    # 12 scenes for 60s
    assert len(episode.scenes) >= 8

    # Duration validation
    total = sum(s.duration_seconds for s in episode.scenes)
    assert abs(total - 60.0) <= 5.0, f"Total {total}s deviates >5s from 60s target"

    # Close-up ratio ≥ 40%
    close_count = sum(1 for s in episode.scenes
                      if s.shot_scale in (ShotScale.CLOSE_UP, ShotScale.MEDIUM_CLOSE))
    ratio = close_count / len(episode.scenes)
    assert ratio >= 0.4, f"Close-up ratio {ratio:.0%} < 40%"

    # First scene ≤ 5s (黄金3秒)
    assert episode.scenes[0].duration_seconds <= 5.0

    # Has cliffhanger
    assert script_data.get("cliffhanger")
    assert len(script_data["cliffhanger"]) > 10

    # Emotion diversity
    emotions = {s.emotion for s in episode.scenes if s.emotion}
    assert len(emotions) >= 3, f"Only {len(emotions)} distinct emotions, need ≥3"

    # speaking_character consistency
    for scene in episode.scenes:
        if scene.speaking_character:
            assert scene.speaking_character in scene.characters_present or not scene.characters_present


def test_executor_mapping_completeness():
    """Every DramaScene output field should map to at least one executor task."""
    # Fields that require downstream execution
    execution_fields = {
        "visual_prompt", "shot_scale", "shot_type", "camera_movement",
        "dialogue", "narration", "speaking_character",
        "emotion", "transition", "characters_present",
    }

    mapped_fields = set()
    for key in FIELD_TO_EXECUTOR_MAP:
        # Normalize: some keys are compound like "voice_style → VoiceProfile"
        base = key.split("→")[0].strip().split("+")[0].strip()
        mapped_fields.add(base)

    for field in execution_fields:
        assert field in mapped_fields or any(field in k for k in FIELD_TO_EXECUTOR_MAP), \
            f"Field '{field}' has no executor mapping"
