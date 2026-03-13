"""Generate episodes 2-5 of 这个王妃太狂野 using Kimi 2.5 Thinking LLM.

Usage:
    .venv/bin/python scripts/generate_wangfei_episodes.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from videoclaw.drama.models import (
    Character,
    DramaSeries,
    Episode,
    EpisodeStatus,
    DramaManager,
    assign_voice_profile,
)
from videoclaw.drama.planner import DramaPlanner
from videoclaw.models.llm.litellm_wrapper import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Pre-built series outline (from mock data, validated in e2e test)
SERIES_DATA = {
    "title": "这个王妃太狂野",
    "genre": "穿越/古装/爽剧",
    "synopsis": "现代女特工林薇穿越成被休弃的废物王妃，众人等着看笑话，殊不知她前世是顶级杀手。这一世，她要让所有欺辱她的人跪着叫姐姐。",
    "style": "cinematic",
    "language": "zh",
    "aspect_ratio": "9:16",
    "target_episode_duration": 60.0,
    "total_episodes": 5,
    "characters": [
        {
            "name": "林薇",
            "description": "表面身份：被休弃的废物王妃；隐藏身份：穿越而来的现代女特工。性格反差：外表柔弱 vs 内心冷酷。记忆点：每次反杀前微微一笑说'你确定？'",
            "visual_prompt": "Young Chinese woman, early 20s, delicate oval face, phoenix eyes with sharp gaze, long black hair with jade hairpin, wearing tattered but elegant white hanfu with red inner lining, slender build, subtle scar on left wrist",
            "voice_style": "calm",
        },
        {
            "name": "萧衍",
            "description": "表面身份：冷面靖王；隐藏身份：暗中调查皇室血脉秘密。性格反差：对外冷酷 vs 对林薇逐渐温柔。记忆点：永远黑衣，腰间佩一把从不出鞘的剑。",
            "visual_prompt": "Tall Chinese man, late 20s, sharp angular jawline, intense dark eyes, long black hair in high ponytail with silver crown, wearing black brocade robe with dragon embroidery, broad shoulders, never-drawn sword at waist",
            "voice_style": "authoritative",
        },
        {
            "name": "慕容雪",
            "description": "表面身份：温婉贤淑的侧妃；隐藏身份：幕后势力的棋子。性格反差：人前温婉 vs 人后阴毒。记忆点：手持折扇，扇面画曼陀罗。",
            "visual_prompt": "Beautiful Chinese woman, mid 20s, round face with dimples, gentle almond eyes hiding cunning, elaborate updo with golden phoenix hairpins, wearing pink silk hanfu with floral patterns, holding folding fan with mandala flower painting",
            "voice_style": "playful",
        },
        {
            "name": "老太君",
            "description": "表面身份：德高望重的萧家老太君；隐藏身份：早年江湖女侠。记忆点：龙头拐杖，发怒时拐杖敲地三声。",
            "visual_prompt": "Elderly Chinese woman, 70s, silver hair in neat bun, wise piercing eyes, dignified posture, wearing deep purple silk robe, leaning on dragon-headed walking cane, jade bracelet",
            "voice_style": "dramatic",
        },
    ],
    "episodes": [
        {"number": 1, "title": "废妃重生", "synopsis": "林薇穿越醒来被丢在柴房，丫鬟踩着她的手。林薇一招反擒拿制服丫鬟，走出柴房微笑'你们确定？'慕容雪派人送毒汤被林薇识破。", "opening_hook": "一双精致却沾满泥土的手被粗暴踩住", "duration_seconds": 60.0},
        {"number": 2, "title": "当众打脸", "synopsis": "老太君寿宴，慕容雪设计让林薇当众出丑作诗。林薇用现代诗震惊全场，老太君欣赏，萧衍第一次正眼看她。", "opening_hook": "满堂华服贵妇中，一身素衣的林薇被推到正中央", "duration_seconds": 60.0},
        {"number": 3, "title": "深夜试探", "synopsis": "萧衍深夜来柴房试探林薇，林薇识破暗器并反手接住。两人月下交锋。慕容雪决定提前动手。密信：'你的前世，我都知道'。", "opening_hook": "月光下一把匕首划破夜空", "duration_seconds": 60.0},
        {"number": 4, "title": "至暗时刻", "synopsis": "慕容雪联合朝中势力诬陷林薇通敌，林薇被打入天牢。萧衍暗中调查，老太君'谁动我萧家的人试试'。林薇在牢中找到刻着现代二维码的古玉。", "opening_hook": "大殿之上，一道通敌叛国的圣旨砸在林薇面前", "duration_seconds": 60.0},
        {"number": 5, "title": "王妃驾到", "synopsis": "林薇利用现代知识公堂翻案，揭露慕容雪与外敌勾结。慕容雪崩溃。萧衍当众拉住林薇的手'本王的王妃，谁敢动？'", "opening_hook": "公堂之上，所有人都以为林薇必死", "duration_seconds": 60.0},
    ],
}

# Episode 1 already has full mock data — load it from e2e test
EP1_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "tests" / "test_drama_e2e.py"


async def main():
    model = "openai/kimi-k2-thinking"
    output_dir = Path("docs/deliverables/这个王妃太狂野")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Using model: %s", model)

    # Build series from pre-defined data
    series = DramaSeries(
        title=SERIES_DATA["title"],
        genre=SERIES_DATA["genre"],
        synopsis=SERIES_DATA["synopsis"],
        style=SERIES_DATA["style"],
        language=SERIES_DATA["language"],
        aspect_ratio=SERIES_DATA["aspect_ratio"],
        target_episode_duration=SERIES_DATA["target_episode_duration"],
        total_episodes=SERIES_DATA["total_episodes"],
    )
    series.characters = [Character.from_dict(c) for c in SERIES_DATA["characters"]]
    for c in series.characters:
        assign_voice_profile(c)
    series.episodes = [
        Episode(
            number=ep["number"],
            title=ep["title"],
            synopsis=ep["synopsis"],
            opening_hook=ep["opening_hook"],
            duration_seconds=ep["duration_seconds"],
        )
        for ep in SERIES_DATA["episodes"]
    ]

    # Save series outline
    outline_path = output_dir / "00_series_outline.json"
    outline_path.write_text(json.dumps(series.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Series outline saved: %s", outline_path)

    # Create real LLM client
    llm = LLMClient(default_model=model)
    planner = DramaPlanner(llm=llm)

    # Episode 1 cliffhanger (from our mock data)
    ep1_cliffhanger = "林薇眼中映着的现代城市倒影——她到底是谁？穿越的背后又隐藏着什么秘密？"

    # Generate episodes 2-5 with real LLM
    prev_cliffhanger = ep1_cliffhanger
    all_scripts = {}

    # Load any previously generated episodes
    for ep in series.episodes:
        ep_path = output_dir / f"ep{ep.number:02d}_{ep.title}_script.json"
        if ep_path.exists():
            existing = json.loads(ep_path.read_text(encoding="utf-8"))
            all_scripts[ep.number] = existing
            prev_cliffhanger = existing.get("cliffhanger", prev_cliffhanger)
            logger.info("Loaded existing episode %d from cache", ep.number)

    for ep in series.episodes:
        if ep.number == 1:
            logger.info("Skipping episode 1 (using existing mock data)")
            continue

        if ep.number in all_scripts:
            logger.info("Skipping episode %d (already generated)", ep.number)
            continue

        logger.info("=" * 60)
        logger.info("Generating episode %d: %s", ep.number, ep.title)
        logger.info("=" * 60)

        try:
            script_data = await planner.script_episode(
                series, ep, previous_cliffhanger=prev_cliffhanger
            )

            # Save raw script JSON
            ep_path = output_dir / f"ep{ep.number:02d}_{ep.title}_script.json"
            ep_path.write_text(
                json.dumps(script_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Save scene summary
            scenes = script_data.get("scenes", [])
            total_dur = sum(float(s.get("duration_seconds", 0)) for s in scenes)
            logger.info(
                "Episode %d: %d scenes, %.1fs total, cliffhanger: %s",
                ep.number,
                len(scenes),
                total_dur,
                script_data.get("cliffhanger", "N/A")[:50],
            )

            # Print scene breakdown
            for s in scenes:
                logger.info(
                    "  %s | %s/%s | %.1fs | %s | %s",
                    s.get("scene_id", "?"),
                    s.get("shot_scale", "?"),
                    s.get("shot_type", "?"),
                    float(s.get("duration_seconds", 0)),
                    s.get("emotion", "?"),
                    s.get("description", "?")[:40],
                )

            all_scripts[ep.number] = script_data
            prev_cliffhanger = script_data.get("cliffhanger", "")
            ep.status = EpisodeStatus.COMPLETED

        except Exception as e:
            logger.error("Episode %d FAILED: %s", ep.number, e)
            ep.status = EpisodeStatus.FAILED
            raise

    # Save final series state
    final_path = output_dir / "final_series.json"
    final_path.write_text(json.dumps(series.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate combined deliverable markdown
    _generate_full_doc(series, all_scripts, output_dir / "这个王妃太狂野_完整剧本.md")

    logger.info("=" * 60)
    logger.info("DONE — All episodes generated")
    logger.info("Token usage: %s", llm.usage)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)


def _generate_full_doc(series: DramaSeries, scripts: dict, output_path: Path):
    """Generate human-readable combined script document."""
    lines = [
        f"# {series.title}",
        f"**类型:** {series.genre}",
        f"**高概念:** {series.synopsis}",
        f"**画面比例:** {series.aspect_ratio} | **单集时长:** {series.target_episode_duration}秒",
        "",
        "## 角色表",
        "",
    ]

    for c in series.characters:
        lines.append(f"### {c.name}")
        lines.append(f"- **人设:** {c.description}")
        lines.append(f"- **视觉:** {c.visual_prompt}")
        if c.voice_profile:
            lines.append(f"- **TTS:** {c.voice_profile.voice_id} (speed={c.voice_profile.speed}, pitch={c.voice_profile.pitch})")
        lines.append("")

    for ep_num, script in sorted(scripts.items()):
        ep = next((e for e in series.episodes if e.number == ep_num), None)
        lines.append(f"---")
        lines.append(f"## 第{ep_num}集: {script.get('episode_title', ep.title if ep else '?')}")
        if ep:
            lines.append(f"**梗概:** {ep.synopsis}")
            lines.append(f"**开场钩子:** {ep.opening_hook}")
        lines.append("")

        for i, scene in enumerate(script.get("scenes", []), 1):
            lines.append(f"### [{scene.get('scene_id', f's{i:02d}')}] {scene.get('description', '')}")
            lines.append(f"- **景别:** {scene.get('shot_scale', '?')} | **类型:** {scene.get('shot_type', '?')} | **运镜:** {scene.get('camera_movement', 'static')} | **{scene.get('duration_seconds', 0)}秒**")
            lines.append(f"- **情绪:** {scene.get('emotion', '?')} | **转场:** {scene.get('transition', 'cut')}")
            lines.append(f"- **visual_prompt:** {scene.get('visual_prompt', '')}")
            if scene.get("dialogue"):
                lines.append(f"- **台词 ({scene.get('speaking_character', '?')}):** \"{scene['dialogue']}\"")
            if scene.get("narration"):
                lines.append(f"- **旁白:** \"{scene['narration']}\"")
            if scene.get("characters_present"):
                lines.append(f"- **出镜:** {', '.join(scene['characters_present'])}")
            lines.append("")

        if script.get("cliffhanger"):
            lines.append(f"**悬念:** {script['cliffhanger']}")
        music = script.get("music", {})
        if music:
            lines.append(f"**配乐:** {music.get('style', '?')} / {music.get('mood', '?')} / {music.get('tempo', '?')}BPM")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Full script doc: %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
