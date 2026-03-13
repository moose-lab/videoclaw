"""FFmpeg-based audio post-processing for short drama TTS output.

Applies filter chains (echo, EQ, loudness normalisation) based on the
:class:`~videoclaw.drama.models.LineType` of each spoken line so that
inner-monologue, narration, and dialogue tracks each have a distinct sonic
character before they are mixed together.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from videoclaw.drama.models import LineType
from videoclaw.utils.ffmpeg import run_ffmpeg

logger = logging.getLogger(__name__)


class AudioPostProcessor:
    """Apply line-type-aware FFmpeg filters to TTS audio files."""

    # ------------------------------------------------------------------
    # Filter definitions
    # ------------------------------------------------------------------

    @staticmethod
    def inner_monologue_filter() -> str:
        """Return the FFmpeg filter chain for inner-monologue lines.

        Adds a subtle echo, rolls off highs, and slightly reduces volume to
        create a "thinking voice" effect.
        """
        return "aecho=0.8:0.88:60:0.4,lowpass=f=3000,volume=0.9"

    @staticmethod
    def narration_filter() -> str:
        """Return the FFmpeg filter chain for narration lines.

        Applies EBU R128 loudness normalisation so narration sits at a
        consistent level across episodes.
        """
        return "loudnorm=I=-16:TP=-1.5:LRA=11"

    @staticmethod
    def get_filter_for(line_type: LineType) -> str:
        """Dispatch to the correct filter chain for *line_type*.

        Returns an empty string for :attr:`LineType.DIALOGUE` (no processing
        needed).
        """
        if line_type == LineType.INNER_MONOLOGUE:
            return AudioPostProcessor.inner_monologue_filter()
        if line_type == LineType.NARRATION:
            return AudioPostProcessor.narration_filter()
        return ""

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    async def process(
        self,
        input_path: Path,
        output_path: Path,
        line_type: LineType,
    ) -> Path:
        """Apply the appropriate filter chain and write to *output_path*.

        If *line_type* is :attr:`LineType.DIALOGUE` (no filter), the input
        file is copied verbatim so that callers always get a file at
        *output_path*.

        Parameters
        ----------
        input_path:
            Source audio file (e.g. WAV or MP3 from the TTS engine).
        output_path:
            Destination path for the processed audio.
        line_type:
            Determines which filter chain to apply.

        Returns
        -------
        Path
            *output_path* on success.

        Raises
        ------
        FileNotFoundError
            If *input_path* does not exist.
        RuntimeError
            If FFmpeg exits with a non-zero code.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_filter = self.get_filter_for(line_type)

        if not audio_filter:
            # No processing needed -- straight copy.
            shutil.copy2(input_path, output_path)
            logger.debug(
                "No filter for %s; copied %s -> %s",
                line_type.value,
                input_path,
                output_path,
            )
            return output_path

        args = [
            "-y",
            "-i", str(input_path),
            "-af", audio_filter,
            str(output_path),
        ]
        await run_ffmpeg(args)
        logger.info(
            "Post-processed %s (%s) -> %s",
            input_path.name,
            line_type.value,
            output_path.name,
        )
        return output_path

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    @staticmethod
    async def get_audio_duration(path: Path) -> float:
        """Return the duration of *path* in seconds using ``ffprobe``.

        Parameters
        ----------
        path:
            Path to an audio file.

        Returns
        -------
        float
            Duration in seconds.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        RuntimeError
            If ``ffprobe`` fails or the duration cannot be parsed.
        """
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed for {path} (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )

        try:
            data = json.loads(stdout.decode())
            duration_str = data["format"]["duration"]
            return float(duration_str)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Could not parse duration from ffprobe output for {path}"
            ) from exc
