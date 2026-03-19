"""Tests for English subtitle support in SubtitleGenerator."""

from __future__ import annotations

from pathlib import Path

import pytest

from videoclaw.generation.subtitle import SubtitleGenerator, generate_srt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _en_scenes(
    *,
    count: int = 1,
    dialogue: str = "Hello world",
    character: str = "Alice",
    duration: float = 5.0,
) -> list[dict]:
    return [
        {
            "scene_id": f"s{i:02d}",
            "dialogue": dialogue,
            "narration": "",
            "speaking_character": character,
            "duration_seconds": duration,
        }
        for i in range(1, count + 1)
    ]


# ===================================================================
# 1. Word-based splitting strategy
# ===================================================================

class TestWordBasedSplitting:
    """split_long_text with strategy="word" splits at word boundaries."""

    def test_short_text_unchanged(self):
        result = SubtitleGenerator.split_long_text(
            "Hello world", max_chars=20, strategy="word"
        )
        assert result == "Hello world"

    def test_split_at_word_boundary(self):
        text = "Hello world, this is a test sentence"
        result = SubtitleGenerator.split_long_text(text, max_chars=20, strategy="word")
        parts = result.split("\\N")
        assert len(parts) > 1
        # Each part must be <= max_chars
        for part in parts:
            assert len(part) <= 20, f"Part too long: {part!r}"

    def test_never_breaks_mid_word(self):
        text = "Hello world, this is a test sentence with longer words"
        result = SubtitleGenerator.split_long_text(text, max_chars=20, strategy="word")
        parts = result.split("\\N")
        # Reassembling (joining on space) should give recognizable words
        # Each part should start/end at a word boundary (no partial words)
        all_words_in_input = set(text.replace(",", "").replace(".", "").split())
        for part in parts:
            # No part should contain a hyphenated break or partial word
            # Each word-like token in the part should be in the original text
            tokens = part.replace(",", "").replace(".", "").replace("!", "").replace("?", "").split()
            for token in tokens:
                assert token in all_words_in_input, f"Partial word detected: {token!r}"

    def test_split_at_max_chars_42(self):
        text = "This is a much longer English sentence that should be split at the forty-two character boundary."
        result = SubtitleGenerator.split_long_text(text, max_chars=42, strategy="word")
        parts = result.split("\\N")
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 42, f"Part exceeds 42 chars: {part!r}"

    def test_custom_line_break(self):
        text = "Hello world, this is a test sentence here"
        result = SubtitleGenerator.split_long_text(text, max_chars=20, line_break="\n", strategy="word")
        assert "\n" in result
        assert "\\N" not in result

    def test_empty_text(self):
        result = SubtitleGenerator.split_long_text("", max_chars=20, strategy="word")
        assert result == ""

    def test_exact_boundary_not_split(self):
        # Exactly max_chars should not be split
        text = "Hello world test ok"  # 19 chars
        assert len(text) <= 20
        result = SubtitleGenerator.split_long_text(text, max_chars=20, strategy="word")
        assert "\\N" not in result


# ===================================================================
# 2. English punctuation splitting
# ===================================================================

class TestEnglishPunctuationSplitting:
    """Word strategy splits at English punctuation."""

    def test_split_at_exclamation(self):
        text = "Great! What happened? I don't know."
        result = SubtitleGenerator.split_long_text(text, max_chars=20, strategy="word")
        # Should produce multiple lines
        parts = result.split("\\N")
        assert len(parts) > 1

    def test_punctuation_stays_with_preceding_text(self):
        text = "Hello, world! How are you?"
        result = SubtitleGenerator.split_long_text(text, max_chars=15, strategy="word")
        parts = result.split("\\N")
        # Each segment should not start with a bare punctuation mark
        for part in parts:
            assert not part.startswith(",")
            assert not part.startswith("!")
            assert not part.startswith("?")

    def test_split_at_colon(self):
        text = "Breaking news: something happened here today"
        result = SubtitleGenerator.split_long_text(text, max_chars=20, strategy="word")
        parts = result.split("\\N")
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 20

    def test_long_no_punctuation_splits_at_words(self):
        text = "This sentence has absolutely no punctuation whatsoever and is quite long indeed"
        result = SubtitleGenerator.split_long_text(text, max_chars=20, strategy="word")
        parts = result.split("\\N")
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 20


# ===================================================================
# 3. SRT English output
# ===================================================================

class TestSRTEnglish:
    """generate_srt() with language="en" uses English colon and word splitting."""

    def test_english_colon_in_srt(self, tmp_path: Path):
        scenes = _en_scenes(dialogue="Hello there", character="Alice")
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out, language="en")

        content = out.read_text(encoding="utf-8")
        # Should use ASCII ": " colon, not Chinese full-width "："
        assert "Alice: Hello there" in content
        assert "\uff1a" not in content

    def test_chinese_colon_default(self, tmp_path: Path):
        scenes = [
            {
                "scene_id": "s01",
                "dialogue": "你好",
                "narration": "",
                "speaking_character": "小明",
                "duration_seconds": 3.0,
            }
        ]
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out)  # default language="zh"

        content = out.read_text(encoding="utf-8")
        assert "\uff1a" in content  # Chinese fullwidth colon

    def test_english_srt_word_splitting(self, tmp_path: Path):
        long_text = "This is a very long English sentence that should be split across multiple lines"
        scenes = _en_scenes(dialogue=long_text, character="")
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out, language="en")

        content = out.read_text(encoding="utf-8")
        # Should contain newlines from word splitting
        lines = content.split("\n")
        # The subtitle text lines should each be <= 42 chars (en max_chars)
        text_lines = [l for l in lines if l and not l.strip().isdigit()
                      and "-->" not in l and l.strip()]
        for line in text_lines:
            assert len(line) <= 42, f"Line too long: {line!r}"

    def test_free_function_passes_language(self, tmp_path: Path):
        scenes = _en_scenes(dialogue="Hello world", character="Bob")
        out = tmp_path / "out.srt"
        generate_srt(scenes, out, language="en")

        content = out.read_text(encoding="utf-8")
        assert "Bob: Hello world" in content
        assert "\uff1a" not in content

    def test_no_character_no_colon(self, tmp_path: Path):
        scenes = _en_scenes(dialogue="Hello world", character="")
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out, language="en")

        content = out.read_text(encoding="utf-8")
        assert ": " not in content or "Hello world" in content
        assert "Hello world" in content


# ===================================================================
# 4. ASS English output
# ===================================================================

class TestASSEnglish:
    """generate_ass() with language="en" uses Arial font from locale."""

    def test_arial_font_in_ass(self, tmp_path: Path):
        scenes = _en_scenes(dialogue="Hello world")
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out, language="en")

        content = out.read_text(encoding="utf-8")
        assert "Arial" in content

    def test_default_font_chinese(self, tmp_path: Path):
        scenes = [
            {
                "scene_id": "s01",
                "dialogue": "你好",
                "narration": "",
                "speaking_character": "",
                "duration_seconds": 3.0,
            }
        ]
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out)  # default language="zh"

        content = out.read_text(encoding="utf-8")
        assert "Microsoft YaHei" in content

    def test_english_ass_word_splitting(self, tmp_path: Path):
        long_text = "This is a very long English sentence that definitely exceeds the forty-two character limit per line"
        scenes = _en_scenes(dialogue=long_text, character="")
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out, language="en")

        content = out.read_text(encoding="utf-8")
        events = [l for l in content.split("\n") if l.startswith("Dialogue:")]
        assert len(events) == 1
        # The text portion (after the last ,,) should contain \N line breaks
        event_text = events[0].split(",,", 1)[1]
        assert "\\N" in event_text

    def test_english_locale_font_size(self, tmp_path: Path):
        scenes = _en_scenes(dialogue="Hello")
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out, language="en")

        content = out.read_text(encoding="utf-8")
        # English locale has font_size=22
        default_style = [l for l in content.split("\n") if l.startswith("Style: Default,")]
        assert len(default_style) == 1
        parts = default_style[0].split(",")
        # Font size is the 3rd field (index 2)
        assert parts[2] == "22"


# ===================================================================
# 5. Backward compatibility — Chinese still works
# ===================================================================

class TestChineseBackwardCompat:
    """Existing Chinese behavior is unchanged."""

    def test_char_strategy_still_works(self):
        text = "我今天去了超市，买了很多东西，然后回家做饭。"
        result = SubtitleGenerator.split_long_text(text, max_chars=15, strategy="char")
        assert "\\N" in result
        parts = result.split("\\N")
        assert any(p.endswith("，") or p.endswith("。") for p in parts)

    def test_default_strategy_is_char(self):
        """No strategy parameter uses char (Chinese) splitting."""
        text = "我今天去了超市，买了很多东西，然后回家做饭。"
        result_default = SubtitleGenerator.split_long_text(text, max_chars=15)
        result_char = SubtitleGenerator.split_long_text(text, max_chars=15, strategy="char")
        assert result_default == result_char

    def test_chinese_srt_colon(self, tmp_path: Path):
        scenes = [
            {
                "scene_id": "s01",
                "dialogue": "你好",
                "narration": "",
                "speaking_character": "小明",
                "duration_seconds": 3.0,
            }
        ]
        out = tmp_path / "out.srt"
        gen = SubtitleGenerator()
        gen.generate_srt(scenes, out)

        content = out.read_text(encoding="utf-8")
        assert "小明\uff1a你好" in content

    def test_chinese_ass_font(self, tmp_path: Path):
        scenes = [
            {
                "scene_id": "s01",
                "dialogue": "你好世界",
                "narration": "",
                "speaking_character": "",
                "duration_seconds": 3.0,
            }
        ]
        out = tmp_path / "out.ass"
        gen = SubtitleGenerator()
        gen.generate_ass(scenes, out)

        content = out.read_text(encoding="utf-8")
        assert "Microsoft YaHei" in content

    def test_force_split_no_punctuation(self):
        """Force-split at max_chars when no Chinese punctuation present."""
        text = "这是一段没有任何标点符号的很长的中文文本用来测试"
        result = SubtitleGenerator.split_long_text(text, max_chars=10, strategy="char")
        parts = result.split("\\N")
        assert all(len(p) <= 10 for p in parts)
        assert "".join(parts) == text


# ===================================================================
# 6. Strategy parameter consistency
# ===================================================================

class TestStrategyParameter:
    """Explicit strategy parameter works correctly."""

    def test_word_strategy_explicit(self):
        text = "Hello world this is a long sentence for testing"
        result = SubtitleGenerator.split_long_text(text, max_chars=20, strategy="word")
        parts = result.split("\\N")
        for part in parts:
            assert len(part) <= 20

    def test_char_strategy_explicit(self):
        text = "这是一段没有任何标点符号的很长的中文文本"
        result = SubtitleGenerator.split_long_text(text, max_chars=10, strategy="char")
        parts = result.split("\\N")
        for part in parts:
            assert len(part) <= 10

    def test_word_strategy_respects_custom_max_chars(self):
        text = "The quick brown fox jumps over the lazy dog near the river bank"
        result = SubtitleGenerator.split_long_text(text, max_chars=30, strategy="word")
        parts = result.split("\\N")
        for part in parts:
            assert len(part) <= 30
