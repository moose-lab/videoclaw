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
    # EQ / Reverb / Silence filter builders (Task 3.1.3)
    # ------------------------------------------------------------------

    @staticmethod
    def build_eq_filter(line_type: LineType) -> str:
        """Return an FFmpeg EQ filter string tailored to *line_type*.

        - **DIALOGUE**: slight presence boost at 3kHz so speech cuts through
          background music / SFX.
        - **NARRATION**: warmth boost at 300Hz with a gentle high-shelf
          roll-off to keep the narrator full but not boomy.
        - **INNER_MONOLOGUE**: high-pass at 200Hz to thin the voice and
          create a detached, "in-the-head" feel.
        """
        if line_type == LineType.DIALOGUE:
            # Presence boost: +4dB at 3kHz, bandwidth 1.5 octaves
            return "equalizer=f=3000:t=h:w=1.5:g=4"
        if line_type == LineType.NARRATION:
            # Warmth boost: +3dB at 300Hz, gentle shelf at 8kHz -2dB
            return (
                "equalizer=f=300:t=h:w=2.0:g=3,"
                "equalizer=f=8000:t=h:w=1.0:g=-2"
            )
        if line_type == LineType.INNER_MONOLOGUE:
            # Cut lows for a thin, ethereal quality
            return "highpass=f=200:p=2"
        return ""

    @staticmethod
    def build_reverb_filter(room_type: str) -> str:
        """Return an FFmpeg reverb filter for the given *room_type*.

        Since FFmpeg lacks a native reverb plugin, we approximate spatial
        reverb using ``aecho`` with parameters tuned per environment:

        - **palace**: long, bright reverb (large marble hall).
        - **cave**: dark, resonant reverb with heavy decay.
        - **outdoor**: short, diffuse reflections (open air).
        - **chamber**: tight, intimate room reverb.
        - **none**: no reverb (returns empty string).
        """
        presets: dict[str, str] = {
            "palace": "aecho=0.8:0.9:120|180:0.4|0.25",
            "cave": "aecho=0.8:0.85:200|300|400:0.5|0.35|0.2",
            "outdoor": "aecho=0.8:0.7:40|80:0.2|0.1",
            "chamber": "aecho=0.8:0.88:60|100:0.3|0.15",
        }
        return presets.get(room_type, "")

    @staticmethod
    def build_silence(duration_ms: int) -> str:
        """Return an FFmpeg filter expression that generates silence.

        Uses ``anullsrc`` to create a silent audio segment of the
        requested duration (for inserting ``pause_before_ms`` gaps
        between dialogue lines).

        Returns an empty string when *duration_ms* is zero or negative.
        """
        if duration_ms <= 0:
            return ""
        duration_s = duration_ms / 1000.0
        return (
            f"anullsrc=r=44100:cl=mono,"
            f"atrim=0:{duration_s:.3f},"
            f"asetpts=N/SR/TB"
        )

    def build_filter_chain(
        self,
        line_type: LineType,
        room_type: str = "none",
        pause_before_ms: int = 0,
    ) -> str:
        """Combine EQ + reverb + silence into a single FFmpeg filter chain.

        The three components are concatenated with ``,`` (FFmpeg filter
        separator) in order: EQ, reverb, silence.  Empty components are
        omitted.  If all three are empty the method returns ``""``.

        Parameters
        ----------
        line_type:
            Determines the EQ preset.
        room_type:
            Determines the reverb preset (``"none"`` skips reverb).
        pause_before_ms:
            Duration of leading silence in milliseconds (0 skips).
        """
        parts: list[str] = []

        eq = self.build_eq_filter(line_type)
        if eq:
            parts.append(eq)

        reverb = self.build_reverb_filter(room_type)
        if reverb:
            parts.append(reverb)

        silence = self.build_silence(pause_before_ms)
        if silence:
            parts.append(silence)

        return ",".join(parts)

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
