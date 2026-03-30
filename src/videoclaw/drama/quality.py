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
    # 6. Vertical framing — close_up + medium_close ≥ 50% of scenes
    #    (real data benchmark: 59% in professional TikTok short dramas)
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        if not scenes:
            continue
        close_count = sum(
            1 for s in scenes if s.get("shot_scale") in ("close_up", "medium_close")
        )
        ratio = close_count / len(scenes)
        if ratio < 0.5:
            violations.append(
                f"Episode {ep_num}: vertical-framing ratio {ratio:.0%} "
                f"(close_up + medium_close) < 50%"
            )

    # ------------------------------------------------------------------
    # 6b. Shot density — avg shot duration ≤ 5s (benchmark: 3.7s)
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        if not scenes:
            continue
        total_duration = sum(s.get("duration_seconds", 0) for s in scenes)
        avg_shot = total_duration / len(scenes) if scenes else 0
        if avg_shot > 5.0:
            violations.append(
                f"Episode {ep_num}: avg shot duration {avg_shot:.1f}s > 5.0s "
                f"(benchmark: 3.7s for TikTok pacing)"
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

    # ------------------------------------------------------------------
    # 11. V.O. ratio — narration ≤ 20% of total spoken words
    #     (real data benchmark: 19% in professional TikTok short dramas)
    # ------------------------------------------------------------------
    for ep_num, script in episode_scripts.items():
        scenes = script.get("scenes", [])
        dialogue_words = sum(
            len(s.get("dialogue", "").split()) for s in scenes
        )
        narration_words = sum(
            len(s.get("narration", "").split()) for s in scenes
        )
        total_spoken = dialogue_words + narration_words
        if total_spoken > 0:
            ratio = narration_words / total_spoken
            if ratio > 0.20:
                violations.append(
                    f"Episode {ep_num}: V.O. ratio {ratio:.0%} exceeds "
                    f"20% limit (benchmark: 19%)"
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
# Dialogue pacing validator (language-agnostic, called from both validators)
# ---------------------------------------------------------------------------

def validate_dialogue_pacing(
    scenes: list[dict[str, Any]],
    episode_num: int,
    *,
    max_cjk_cps: float = 3.5,
    max_en_wps: float = 2.5,
    seedance_max_duration: float = 15.0,
) -> list[str]:
    """Warn when dialogue is too dense for comfortable speech at natural pace.

    Rules:
    - CJK (Chinese/Japanese/Korean): ≤3.5 characters/second
    - English / other: ≤2.5 words/second
    - Hard warning when dialogue requires more than Seedance's max duration
      (15s) even at the maximum comfortable pacing.

    Returns warning strings (not hard violations — never blocks generation).
    """
    warnings: list[str] = []
    for scene in scenes:
        dialogue = (scene.get("dialogue") or "").strip()
        if not dialogue:
            continue
        dur = float(scene.get("duration_seconds") or 5.0)
        if dur <= 0:
            continue
        sid = scene.get("scene_id", "?")

        if _CJK_RE.search(dialogue):
            char_count = len(dialogue)
            min_duration = char_count / max_cjk_cps
            ratio = char_count / dur
            if min_duration > seedance_max_duration:
                warnings.append(
                    f"⚠️ Episode {episode_num} {sid}: "
                    f"dialogue {char_count}字 needs {min_duration:.1f}s at {max_cjk_cps}字/s "
                    f"but Seedance max is {seedance_max_duration}s — "
                    f"dialogue WILL be rushed (consider splitting shot or trimming)"
                )
            elif ratio > max_cjk_cps:
                warnings.append(
                    f"Episode {episode_num} {sid}: "
                    f"dialogue {char_count}字 / {dur}s = {ratio:.1f}字/s "
                    f"(max {max_cjk_cps}) — consider shorter dialogue or longer shot"
                )
        else:
            word_count = len(dialogue.split())
            min_duration = word_count / max_en_wps
            ratio = word_count / dur
            if min_duration > seedance_max_duration:
                warnings.append(
                    f"⚠️ Episode {episode_num} {sid}: "
                    f"dialogue {word_count} words needs {min_duration:.1f}s at {max_en_wps}w/s "
                    f"but Seedance max is {seedance_max_duration}s — "
                    f"dialogue WILL be rushed (consider splitting shot or trimming)"
                )
            elif ratio > max_en_wps:
                warnings.append(
                    f"Episode {episode_num} {sid}: "
                    f"dialogue {word_count} words / {dur}s = {ratio:.1f}w/s "
                    f"(max {max_en_wps}) — consider shorter dialogue or longer shot"
                )
    return warnings


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
        violations: list[str] = []
        if locale.quality_validator is not None:
            violations.extend(locale.quality_validator(series, episode_scripts))
        # Dialogue pacing check is language-agnostic — always run
        for ep_num, script in episode_scripts.items():
            pacing_warnings = validate_dialogue_pacing(
                script.get("scenes", []), ep_num
            )
            if pacing_warnings:
                logger.warning("Dialogue pacing warnings:\n  %s", "\n  ".join(pacing_warnings))
                violations.extend(pacing_warnings)
        return violations
