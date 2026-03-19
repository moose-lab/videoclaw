"""Drama quality validation — locale-aware quality gates.

Validates drama output (series outlines, episode scripts) against
market-specific quality standards.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from videoclaw.drama.models import DramaSeries

logger = logging.getLogger(__name__)

# Regex to detect CJK characters in visual prompts (must be English-only)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# Western strong-emotion keywords that count as emotional peaks / payoff moments
_WESTERN_STRONG_EMOTIONS = frozenset({
    "shock", "triumphant", "angry", "fearful", "revelation",
    "horror", "furious", "vindicated", "panic", "disbelief", "stunned",
    "defiant", "outraged",
})

# Chinese strong-emotion keywords (爽点) for 爽点密度 check
_CHINESE_STRONG_EMOTIONS = frozenset({
    "triumphant", "smug", "vindicated", "shock", "revelation", "defiant",
    "shocked", "furious", "outraged",
})

# Duality keywords for English character-duality check
_DUALITY_KEYWORDS = frozenset({"but", "secretly", "hidden", "beneath", "actually"})


# ---------------------------------------------------------------------------
# Western / English quality validator
# ---------------------------------------------------------------------------

def validate_western_quality(
    series: DramaSeries,
    episode_scripts: dict[int, dict[str, Any]],
) -> list[str]:
    """Validate against Western/English short-drama quality standards.

    Returns a list of violation strings.  Empty list = all checks passed.
    """
    violations: list[str] = []

    # ------------------------------------------------------------------
    # 1. Logline check — synopsis is 10-150 words in English
    # ------------------------------------------------------------------
    word_count = len(series.synopsis.split())
    if word_count < 10 or word_count > 150:
        violations.append(
            f"Synopsis word count {word_count} is outside the 10-150 word range for English logline"
        )

    # ------------------------------------------------------------------
    # 2. Character duality — surface and hidden aspects in description
    # ------------------------------------------------------------------
    for char in series.characters:
        desc_lower = char.description.lower()
        if not any(kw in desc_lower for kw in _DUALITY_KEYWORDS):
            violations.append(
                f"Character '{char.name}' lacks duality keywords "
                f"('but', 'secretly', 'hidden', 'beneath', 'actually') in description"
            )

    # ------------------------------------------------------------------
    # 3. Scroll-stopping hook — first scene of each episode ≤ 5 seconds
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        if scenes:
            first = scenes[0]
            if first.get("duration_seconds", 0) > 5.0:
                violations.append(
                    f"Episode {ep_num}: first scene is {first['duration_seconds']}s "
                    f"(must be ≤5s scroll-stopping hook)"
                )

    # ------------------------------------------------------------------
    # 4. Payoff density — ≥1 emotional peak per 15 seconds
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        total_duration = sum(s.get("duration_seconds", 0) for s in scenes)
        peaks = [s for s in scenes if s.get("emotion", "") in _WESTERN_STRONG_EMOTIONS]
        required_peaks = max(1, int(total_duration / 15))
        if len(peaks) < required_peaks:
            violations.append(
                f"Episode {ep_num}: payoff density {len(peaks)} emotional peaks "
                f"for {total_duration:.0f}s (need ≥{required_peaks})"
            )

    # ------------------------------------------------------------------
    # 5. Cliffhanger — every episode must have a non-empty cliffhanger
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        if not script.get("cliffhanger"):
            violations.append(f"Episode {ep_num}: missing cliffhanger field")

    # ------------------------------------------------------------------
    # 6. Vertical framing — close_up + medium_close ≥ 40% of scenes
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        if not scenes:
            continue
        close_count = sum(
            1 for s in scenes if s.get("shot_scale") in ("close_up", "medium_close")
        )
        ratio = close_count / len(scenes)
        if ratio < 0.4:
            violations.append(
                f"Episode {ep_num}: vertical-framing ratio {ratio:.0%} "
                f"(close_up + medium_close) < 40%"
            )

    # ------------------------------------------------------------------
    # 7. Character consistency — speaking_character in characters_present
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        for scene in script.get("scenes", []):
            speaker = scene.get("speaking_character", "")
            present = scene.get("characters_present", [])
            if speaker and present and speaker not in present:
                violations.append(
                    f"Episode {ep_num} scene '{scene.get('scene_id', '?')}': "
                    f"speaking_character '{speaker}' not in characters_present {present}"
                )

    # ------------------------------------------------------------------
    # 8. Dialogue density — ≤100 English dialogue words per 60 seconds
    #    (prompt allows ~150 total spoken words including narration)
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        total_duration = sum(s.get("duration_seconds", 0) for s in scenes) or 60.0
        total_words = sum(
            len(s.get("dialogue", "").split()) for s in scenes
        )
        limit = int(100 * total_duration / 60)
        if total_words > limit:
            violations.append(
                f"Episode {ep_num}: dialogue density {total_words} words "
                f"exceeds limit of {limit} words for {total_duration:.0f}s episode"
            )

    # ------------------------------------------------------------------
    # 9. Emotion vocabulary — every scene has a non-empty emotion field
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        for scene in script.get("scenes", []):
            if not scene.get("emotion"):
                violations.append(
                    f"Episode {ep_num} scene '{scene.get('scene_id', '?')}': "
                    f"missing emotion field"
                )

    # ------------------------------------------------------------------
    # 10. English visual prompts — no CJK characters in visual_prompt fields
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        for scene in script.get("scenes", []):
            vp = scene.get("visual_prompt", "")
            if _CJK_RE.search(vp):
                violations.append(
                    f"Episode {ep_num} scene '{scene.get('scene_id', '?')}': "
                    f"visual_prompt contains CJK characters (must be English)"
                )

    return violations


# ---------------------------------------------------------------------------
# Chinese quality validator
# ---------------------------------------------------------------------------

def validate_chinese_quality(
    series: DramaSeries,
    episode_scripts: dict[int, dict[str, Any]],
) -> list[str]:
    """Validate against 红果短剧 / Chinese short-drama quality standards.

    Returns a list of violation strings.  Empty list = all checks passed.
    """
    violations: list[str] = []

    # ------------------------------------------------------------------
    # 1. 高概念一句话卖点 — synopsis ≥ 20 Chinese characters
    # ------------------------------------------------------------------
    if len(series.synopsis) < 20:
        violations.append("synopsis 太短，缺少高概念一句话卖点（需≥20字）")

    # ------------------------------------------------------------------
    # 2. 角色反差人设 — character has 反差 (dual identity)
    # ------------------------------------------------------------------
    for char in series.characters:
        if "表面" not in char.description and "隐藏" not in char.description:
            violations.append(f"角色 '{char.name}' 缺少表面/隐藏身份反差（反差人设）")

    # ------------------------------------------------------------------
    # 3. 黄金3秒法则 — first scene of each episode ≤ 5 seconds
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        if scenes:
            first = scenes[0]
            if first.get("duration_seconds", 0) > 5.0:
                violations.append(
                    f"第{ep_num}集第1场景 {first['duration_seconds']}s > 5s 黄金3秒上限"
                )

    # ------------------------------------------------------------------
    # 4. 爽点密度 — ≥2 emotional peaks per 15 seconds
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        total_duration = sum(s.get("duration_seconds", 0) for s in scenes)
        peaks = [s for s in scenes if s.get("emotion", "") in _CHINESE_STRONG_EMOTIONS]
        required_peaks = max(2, int(total_duration / 15) * 2)
        if len(peaks) < required_peaks:
            violations.append(
                f"第{ep_num}集爽点密度不足：{len(peaks)}个高峰情绪，"
                f"需≥{required_peaks}个（每15s至少2个爽点）"
            )

    # ------------------------------------------------------------------
    # 5. 悬念 — every episode must have a non-empty cliffhanger
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        if not script.get("cliffhanger"):
            violations.append(f"第{ep_num}集缺少 cliffhanger")

    # ------------------------------------------------------------------
    # 6. 竖屏构图 — close_up + medium_close ≥ 40% of scenes
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        if not scenes:
            continue
        close_count = sum(
            1 for s in scenes if s.get("shot_scale") in ("close_up", "medium_close")
        )
        ratio = close_count / len(scenes)
        if ratio < 0.4:
            violations.append(
                f"第{ep_num}集特写镜头比例 {ratio:.0%} < 40% 竖屏要求"
            )

    # ------------------------------------------------------------------
    # 7. 角色一致性 — speaking_character in characters_present
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        for scene in script.get("scenes", []):
            speaker = scene.get("speaking_character", "")
            present = scene.get("characters_present", [])
            if speaker and present and speaker not in present:
                violations.append(
                    f"第{ep_num}集 {scene.get('scene_id', '?')}: "
                    f"speaking_character '{speaker}' 不在 characters_present 中"
                )

    # ------------------------------------------------------------------
    # 8. 台词密度 — ≤100 Chinese characters per 60 seconds
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        total_duration = sum(s.get("duration_seconds", 0) for s in scenes) or 60.0
        total_chars = sum(len(s.get("dialogue", "")) for s in scenes)
        limit = int(100 * total_duration / 60)
        if total_chars > limit:
            violations.append(
                f"第{ep_num}集对白 {total_chars} 字，超过 {limit} 字上限（≤100字/60秒）"
            )

    # ------------------------------------------------------------------
    # 9. 情绪词表标准化 — every scene has a non-empty emotion field
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        for scene in script.get("scenes", []):
            if not scene.get("emotion"):
                violations.append(
                    f"第{ep_num}集 {scene.get('scene_id', '?')}: 缺少 emotion 字段"
                )

    # ------------------------------------------------------------------
    # 10. 英文视觉提示词 — no CJK in visual_prompt fields
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        for scene in script.get("scenes", []):
            vp = scene.get("visual_prompt", "")
            if _CJK_RE.search(vp):
                violations.append(
                    f"第{ep_num}集 {scene.get('scene_id', '?')}: "
                    f"visual_prompt 含有中文字符（应为英文）"
                )

    return violations


# ---------------------------------------------------------------------------
# Locale-aware dispatcher
# ---------------------------------------------------------------------------

class DramaQualityValidator:
    """Validates drama output against market-specific quality standards."""

    def validate(
        self,
        series: DramaSeries,
        episode_scripts: dict[int, dict[str, Any]],
    ) -> list[str]:
        """Run quality validation, returning a list of violation strings.

        Empty list = passed all checks.
        """
        from videoclaw.drama.locale import get_locale

        locale = get_locale(series.language)
        if locale.quality_validator is not None:
            return locale.quality_validator(series, episode_scripts)
        return []
