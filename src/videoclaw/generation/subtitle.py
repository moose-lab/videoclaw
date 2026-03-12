"""Subtitle generator -- produces SRT files from drama scene dialogue and timing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(
    scenes: list[dict[str, Any]],
    output_path: Path,
    *,
    include_narration: bool = False,
) -> Path:
    """Generate an SRT subtitle file from drama scene data.

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

    Returns
    -------
    Path
        The *output_path* on success.
    """
    entries: list[str] = []
    current_time = 0.0
    index = 1

    for scene in scenes:
        dialogue = scene.get("dialogue", "").strip()
        narration = scene.get("narration", "").strip()
        duration = float(scene.get("duration_seconds", 5.0))
        character = scene.get("speaking_character", "")

        text = dialogue
        if not text and include_narration:
            text = narration

        if text:
            start = current_time
            end = current_time + duration

            # Prefix with character name if available (drama convention)
            display_text = f"{character}：{text}" if character else text

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
