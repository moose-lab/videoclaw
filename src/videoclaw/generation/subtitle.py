"""Subtitle generator -- produces SRT and ASS files from drama scene dialogue and timing."""

from __future__ import annotations

import json as _json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Chinese punctuation marks suitable for line-break points
_ZH_PUNCTUATION = re.compile(r"([，。！？、；])")

# English punctuation marks suitable for line-break points
_EN_PUNCTUATION = re.compile(r"([,\.!?;:\-\u2014])")


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format (H:MM:SS.cc)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _rgb_to_ass_color(hex_color: str) -> str:
    """Convert an RGB hex color (e.g. '#FF0000' or 'FF0000') to ASS &H00BBGGRR format."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "&H00FFFFFF"
    r = hex_color[0:2]
    g = hex_color[2:4]
    b = hex_color[4:6]
    return f"&H00{b}{g}{r}".upper()


def _get_scene_duration_from_manifest(
    scene_id: str,
    audio_manifest: dict[str, Any] | None,
) -> float | None:
    """Look up total duration for a scene_id in an audio manifest dict.

    Returns the sum of all segment durations matching *scene_id*, or ``None``
    if the manifest is unavailable or has no matching segments.
    """
    if not audio_manifest:
        return None

    segments = audio_manifest.get("segments", [])
    total = 0.0
    found = False
    for seg in segments:
        if seg.get("scene_id") == scene_id:
            total += seg.get("duration_seconds", 0.0)
            found = True

    return total if found else None


class SubtitleGenerator:
    """Generates SRT and ASS subtitle files from drama scene data."""

    # ------------------------------------------------------------------
    # SRT generation
    # ------------------------------------------------------------------

    def generate_srt(
        self,
        scenes: list[dict[str, Any]],
        output_path: Path | str,
        *,
        include_narration: bool = False,
        audio_manifest: dict[str, Any] | None = None,
        language: str = "zh",
    ) -> Path:
        """Generate SRT subtitles.

        When *audio_manifest* is provided, use its segment durations for
        accurate timing instead of scene ``duration_seconds``.
        """
        output_path = Path(output_path)
        entries: list[str] = []
        current_time = 0.0
        index = 1

        # Resolve locale-specific settings for non-Chinese languages
        split_strategy = "char"
        max_chars = 20
        if language != "zh":
            from videoclaw.drama.locale import get_locale
            locale = get_locale(language)
            split_strategy = locale.subtitle_config.line_break_strategy
            max_chars = locale.subtitle_config.max_chars_per_line

        for scene in scenes:
            dialogue = scene.get("dialogue", "").strip()
            narration = scene.get("narration", "").strip()
            scene_id = scene.get("scene_id", "")
            character = scene.get("speaking_character", "")

            # Resolve duration: prefer manifest, fall back to scene field
            manifest_dur = _get_scene_duration_from_manifest(scene_id, audio_manifest)
            duration = manifest_dur if manifest_dur is not None else float(scene.get("duration_seconds", 5.0))

            text = dialogue
            if not text and include_narration:
                text = narration

            if text:
                start = current_time
                end = current_time + duration

                colon = "\uff1a" if language == "zh" else ": "
                display_text = f"{character}{colon}{text}" if character else text
                # Apply line splitting for SRT (use \n)
                display_text = self.split_long_text(
                    display_text, max_chars=max_chars, line_break="\n", strategy=split_strategy
                )

                entries.append(
                    f"{index}\n"
                    f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n"
                    f"{display_text}\n"
                )
                index += 1

            current_time += duration

        srt_content = "\n".join(entries)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(srt_content, encoding="utf-8")

        logger.info("Generated SRT with %d entries -> %s", index - 1, output_path)
        return output_path

    # ------------------------------------------------------------------
    # ASS generation
    # ------------------------------------------------------------------

    def generate_ass(
        self,
        scenes: list[dict[str, Any]],
        output_path: Path | str,
        *,
        include_narration: bool = False,
        audio_manifest: dict[str, Any] | None = None,
        character_colors: dict[str, str] | None = None,
        font_name: str = "Microsoft YaHei",
        font_size: int = 20,
        title: str = "Untitled",
        language: str = "zh",
    ) -> Path:
        """Generate ASS (Advanced SubStation Alpha) subtitles.

        Features:
        - Per-character color coding via *character_colors* (name -> hex RGB)
        - Configurable font
        - Bottom-center positioning for dialogue, top for narration
        - Auto line-break for text longer than ~20 Chinese chars per line
        """
        output_path = Path(output_path)
        character_colors = character_colors or {}

        # Resolve locale-specific settings for non-Chinese languages
        split_strategy = "char"
        max_chars = 20
        if language != "zh":
            from videoclaw.drama.locale import get_locale
            locale = get_locale(language)
            font_name = locale.subtitle_config.font_name
            font_size = locale.subtitle_config.font_size
            split_strategy = locale.subtitle_config.line_break_strategy
            max_chars = locale.subtitle_config.max_chars_per_line

        # Build dynamic per-character styles
        char_styles: dict[str, str] = {}
        char_style_lines: list[str] = []
        for char_name, hex_color in character_colors.items():
            style_name = f"Char_{char_name}"
            ass_color = _rgb_to_ass_color(hex_color)
            char_styles[char_name] = style_name
            char_style_lines.append(
                f"Style: {style_name},{font_name},{font_size},{ass_color},"
                f"&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1"
            )

        # ASS header
        header = (
            f"[Script Info]\n"
            f"Title: {title}\n"
            f"ScriptType: v4.00+\n"
            f"PlayResX: 1080\n"
            f"PlayResY: 1920\n"
            f"\n"
            f"[V4+ Styles]\n"
            f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            f"Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            f"Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            f"0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1\n"
            f"Style: Narration,{font_name},{font_size - 2},&H00CCCCCC,&H000000FF,&H00000000,&H80000000,"
            f"0,1,0,0,100,100,0,0,1,2,1,8,10,10,30,1\n"
            f"Style: TitleCard,{font_name},{font_size + 12},&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,"
            f"1,0,0,0,100,100,2,0,1,3,2,5,10,10,10,1\n"
        )

        if char_style_lines:
            header += "\n".join(char_style_lines) + "\n"

        header += (
            f"\n"
            f"[Events]\n"
            f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        # Build dialogue events
        events: list[str] = []
        current_time = 0.0

        for scene in scenes:
            dialogue = scene.get("dialogue", "").strip()
            narration = scene.get("narration", "").strip()
            scene_id = scene.get("scene_id", "")
            character = scene.get("speaking_character", "")

            # Resolve duration
            manifest_dur = _get_scene_duration_from_manifest(scene_id, audio_manifest)
            duration = manifest_dur if manifest_dur is not None else float(scene.get("duration_seconds", 5.0))

            start = current_time
            end = current_time + duration

            start_ts = _format_ass_time(start)
            end_ts = _format_ass_time(end)

            # Dialogue line
            if dialogue:
                style = char_styles.get(character, "Default")
                text = self.split_long_text(dialogue, max_chars=max_chars, line_break="\\N", strategy=split_strategy)
                events.append(
                    f"Dialogue: 0,{start_ts},{end_ts},{style},{character},0,0,0,,{text}"
                )

            # Narration line
            # title_card: always shown as centered large text (visual overlay, no TTS)
            # voiceover: shown as top narration subtitle (with TTS audio)
            if narration:
                narration_type = scene.get("narration_type", "voiceover")
                if narration_type == "title_card":
                    # Title card: always render — centered, large, bold
                    text = self.split_long_text(narration, max_chars=max_chars, line_break="\\N", strategy=split_strategy)
                    events.append(
                        f"Dialogue: 0,{start_ts},{end_ts},TitleCard,,0,0,0,,{text}"
                    )
                elif include_narration or not dialogue:
                    text = self.split_long_text(narration, max_chars=max_chars, line_break="\\N", strategy=split_strategy)
                    events.append(
                        f"Dialogue: 0,{start_ts},{end_ts},Narration,,0,0,0,,{text}"
                    )

            current_time += duration

        content = header + "\n".join(events) + "\n"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        logger.info("Generated ASS with %d events -> %s", len(events), output_path)
        return output_path

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------

    @staticmethod
    def split_long_text(
        text: str,
        max_chars: int = 20,
        *,
        line_break: str = "\\N",
        strategy: str = "char",  # "char" for Chinese, "word" for English
    ) -> str:
        """Split long text into multiple lines at natural break points.

        For Chinese text (strategy="char"), split at punctuation marks or at
        *max_chars* boundary.  For English text (strategy="word"), split at
        word boundaries and English punctuation, never breaking mid-word.
        Returns text with the given *line_break* separator
        (``\\N`` for ASS, ``\\n`` for SRT).
        """
        if len(text) <= max_chars:
            return text

        if strategy == "word":
            return SubtitleGenerator._split_word_strategy(text, max_chars, line_break)

        # --- char strategy (Chinese, default) ---
        # Try splitting at Chinese punctuation first
        parts = _ZH_PUNCTUATION.split(text)
        # Re-attach punctuation to the preceding segment
        segments: list[str] = []
        i = 0
        while i < len(parts):
            seg = parts[i]
            # If the next part is a punctuation match, attach it
            if i + 1 < len(parts) and _ZH_PUNCTUATION.fullmatch(parts[i + 1]):
                seg += parts[i + 1]
                i += 2
            else:
                i += 1
            if seg:
                segments.append(seg)

        if not segments:
            return text

        # Merge segments into lines respecting max_chars
        lines: list[str] = []
        current_line = ""
        for seg in segments:
            if current_line and len(current_line) + len(seg) > max_chars:
                lines.append(current_line)
                current_line = seg
            else:
                current_line += seg

        if current_line:
            lines.append(current_line)

        # If punctuation splitting didn't help (single long segment), force-split
        final_lines: list[str] = []
        for line in lines:
            if len(line) <= max_chars:
                final_lines.append(line)
            else:
                # Force split at max_chars
                for start in range(0, len(line), max_chars):
                    final_lines.append(line[start : start + max_chars])

        return line_break.join(final_lines)

    @staticmethod
    def _split_word_strategy(text: str, max_chars: int, line_break: str) -> str:
        """Word-based splitting for English text.

        1. Split at English punctuation first, keeping punctuation with preceding text.
        2. Merge segments into lines respecting max_chars.
        3. If a segment is still too long, split at word boundaries (spaces).
        4. Never break mid-word.
        """
        # Step 1: split at English punctuation, re-attaching punctuation to preceding text
        parts = _EN_PUNCTUATION.split(text)
        segments: list[str] = []
        i = 0
        while i < len(parts):
            seg = parts[i]
            if i + 1 < len(parts) and _EN_PUNCTUATION.fullmatch(parts[i + 1]):
                seg += parts[i + 1]
                i += 2
            else:
                i += 1
            seg = seg.strip()
            if seg:
                segments.append(seg)

        if not segments:
            segments = [text]

        # Step 2: merge segments into lines respecting max_chars
        lines: list[str] = []
        current_line = ""
        for seg in segments:
            if not current_line:
                current_line = seg
            elif len(current_line) + 1 + len(seg) <= max_chars:
                current_line += " " + seg
            else:
                lines.append(current_line)
                current_line = seg

        if current_line:
            lines.append(current_line)

        # Step 3: if any line is still too long, split at word boundaries
        final_lines: list[str] = []
        for line in lines:
            if len(line) <= max_chars:
                final_lines.append(line)
            else:
                # Split at word boundaries
                words = line.split(" ")
                current = ""
                for word in words:
                    if not current:
                        current = word
                    elif len(current) + 1 + len(word) <= max_chars:
                        current += " " + word
                    else:
                        if current:
                            final_lines.append(current)
                        current = word
                if current:
                    final_lines.append(current)

        return line_break.join(final_lines) if final_lines else text


# ------------------------------------------------------------------
# Backward-compatible free function
# ------------------------------------------------------------------

def generate_srt(
    scenes: list[dict[str, Any]],
    output_path: Path,
    *,
    include_narration: bool = False,
    audio_manifest: dict[str, Any] | None = None,
    language: str = "zh",
) -> Path:
    """Generate an SRT subtitle file from drama scene data.

    This is a backward-compatible wrapper around
    :meth:`SubtitleGenerator.generate_srt`.

    Parameters
    ----------
    scenes:
        List of scene dicts, each with at least ``dialogue``,
        ``duration_seconds``, and optionally ``narration`` and
        ``speaking_character``.
    output_path:
        Where to write the SRT file.
    include_narration:
        If *True*, narration text is included as subtitles when
        there is no dialogue for a scene.
    audio_manifest:
        Optional EpisodeAudioManifest dict for accurate timing.
    language:
        Language code (e.g. ``"zh"`` or ``"en"``).  Controls colon
        character, font, and line-breaking strategy.

    Returns
    -------
    Path
        The *output_path* on success.
    """
    return SubtitleGenerator().generate_srt(
        scenes,
        output_path,
        include_narration=include_narration,
        audio_manifest=audio_manifest,
        language=language,
    )
