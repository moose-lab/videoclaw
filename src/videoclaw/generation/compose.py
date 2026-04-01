"""Video composer -- stitches shots together and mixes audio using FFmpeg.

The :class:`VideoComposer` provides a high-level async API over FFmpeg
subprocesses.  It handles video concatenation with transitions, audio mixing,
subtitle overlay, and a one-call ``render_final`` pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from videoclaw.utils.ffmpeg import check_ffmpeg, get_video_duration, run_ffmpeg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AudioType(StrEnum):
    """Classification of an audio track."""
    VOICE = "voice"
    MUSIC = "music"
    SFX = "sfx"


@dataclass
class AudioTrack:
    """An audio source to be mixed into the final video."""

    path: Path
    type: AudioType = AudioType.MUSIC
    volume: float = 1.0  # 0.0 - 1.0
    start_time: float = 0.0  # seconds offset in the final timeline

    def __post_init__(self) -> None:
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError(f"Volume must be 0.0-1.0, got {self.volume}")


# ---------------------------------------------------------------------------
# Transition presets
# ---------------------------------------------------------------------------

_SUPPORTED_TRANSITIONS: set[str] = {
    "dissolve",
    "fade",
    "wipeleft",
    "wiperight",
    "wipeup",
    "wipedown",
    "slideleft",
    "slideright",
}


# ---------------------------------------------------------------------------
# VideoComposer
# ---------------------------------------------------------------------------

class VideoComposer:
    """Composites video clips, audio, and subtitles into a final render.

    All operations are async and shell out to FFmpeg via
    :func:`videoclaw.utils.ffmpeg.run_ffmpeg`.
    """

    async def compose(
        self,
        video_paths: list[Path],
        output_path: Path,
        transition: str = "dissolve",
        transition_duration: float = 0.5,
        transitions: list[str] | None = None,
        clip_durations: list[float] | None = None,
    ) -> Path:
        """Concatenate *video_paths* with transitions into *output_path*.

        Parameters
        ----------
        video_paths:
            Ordered list of video clips to stitch together.
        output_path:
            Destination file path for the composed video.
        transition:
            Default transition type between clips (see :data:`_SUPPORTED_TRANSITIONS`).
            Used when *transitions* is ``None`` or when a per-boundary entry is empty.
        transition_duration:
            Duration of each transition in seconds.
        transitions:
            Per-boundary transition list (length = ``len(video_paths) - 1``).
            Each entry specifies the transition for that clip boundary.
            Empty strings fall back to *transition*.  When ``None``, the single
            *transition* parameter is used for all boundaries.
        clip_durations:
            Duration of each clip in seconds.  When ``None``, durations are
            probed via ``ffprobe``.  Must have the same length as *video_paths*.

        Returns
        -------
        Path
            The *output_path* on success.
        """
        await self._ensure_ffmpeg()

        if len(video_paths) == 0:
            raise ValueError("At least one video path is required")

        if len(video_paths) == 1:
            # No transition needed -- simple remux
            cmd = self._build_single_copy_cmd(video_paths[0], output_path)
        else:
            # Build the resolved per-boundary transition list
            n_boundaries = len(video_paths) - 1
            if transitions is not None:
                resolved = [
                    (t if t and t in _SUPPORTED_TRANSITIONS else transition)
                    for t in transitions[:n_boundaries]
                ]
                # Pad if transitions list is shorter than n_boundaries
                while len(resolved) < n_boundaries:
                    resolved.append(transition)
            else:
                resolved = [transition] * n_boundaries

            # Resolve clip durations -- probe if not provided
            if clip_durations is None:
                durations = []
                for vp in video_paths:
                    dur = await get_video_duration(vp)
                    durations.append(dur)
                clip_durations = durations

            cmd = self._build_concat_cmd(
                video_paths, output_path, resolved, transition_duration,
                clip_durations,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_ffmpeg(cmd)
        logger.info("Composed %d clips -> %s", len(video_paths), output_path)
        return output_path

    async def add_audio(
        self,
        video_path: Path,
        audio_tracks: list[AudioTrack],
        output_path: Path,
    ) -> Path:
        """Mix *audio_tracks* onto *video_path*.

        Parameters
        ----------
        video_path:
            Source video file.
        audio_tracks:
            Audio tracks to mix; each can have independent volume and offset.
        output_path:
            Destination file path.

        Returns
        -------
        Path
            The *output_path* on success.
        """
        await self._ensure_ffmpeg()

        if not audio_tracks:
            # Nothing to add -- copy through
            cmd = self._build_single_copy_cmd(video_path, output_path)
        else:
            cmd = self._build_audio_mix_cmd(video_path, audio_tracks, output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_ffmpeg(cmd)
        logger.info(
            "Mixed %d audio tracks onto %s -> %s",
            len(audio_tracks),
            video_path,
            output_path,
        )
        return output_path

    async def add_subtitles(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
    ) -> Path:
        """Burn subtitles from *subtitle_path* into *video_path*.

        Parameters
        ----------
        video_path:
            Source video.
        subtitle_path:
            SRT / ASS subtitle file.
        output_path:
            Destination file.

        Returns
        -------
        Path
            The *output_path* on success.
        """
        await self._ensure_ffmpeg()

        cmd = self._build_subtitle_cmd(video_path, subtitle_path, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_ffmpeg(cmd)
        logger.info("Subtitles burned: %s -> %s", subtitle_path, output_path)
        return output_path

    async def render_final(
        self,
        video_path: Path,
        audio_tracks: list[AudioTrack],
        subtitle_path: Path | None,
        output_path: Path,
    ) -> Path:
        """Full render pipeline: audio mix + subtitles in a single pass.

        When *subtitle_path* is ``None``, subtitles are skipped.

        Parameters
        ----------
        video_path:
            The composed video (output of :meth:`compose`).
        audio_tracks:
            Audio tracks to overlay.
        subtitle_path:
            Optional subtitle file to burn in.
        output_path:
            Final output file path.

        Returns
        -------
        Path
            The *output_path* on success.
        """
        await self._ensure_ffmpeg()

        cmd = self._build_final_render_cmd(
            video_path, audio_tracks, subtitle_path, output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_ffmpeg(cmd)
        logger.info("Final render complete: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # FFmpeg command builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_single_copy_cmd(source: Path, dest: Path) -> list[str]:
        """Copy a single file without re-encoding."""
        return [
            "ffmpeg", "-y",
            "-i", str(source),
            "-c", "copy",
            str(dest),
        ]

    @staticmethod
    def _build_concat_cmd(
        video_paths: list[Path],
        output_path: Path,
        transitions: list[str],
        transition_duration: float,
        clip_durations: list[float],
    ) -> list[str]:
        """Build an FFmpeg xfade filter-chain for concatenation with transitions.

        This constructs a chain of ``xfade`` filters that progressively merges
        each pair of adjacent clips.

        Parameters
        ----------
        transitions:
            A list of transition types, one per clip boundary
            (length = ``len(video_paths) - 1``).  Each entry must already be a
            valid supported transition string.
        clip_durations:
            Duration of each clip in seconds (same length as *video_paths*).
            Used to compute the correct ``offset`` for each xfade filter.
        """
        n = len(video_paths)

        # Compute xfade offsets from clip durations.
        # offset_0 = clip_durations[0] - transition_duration
        # offset_i = offset_{i-1} + clip_durations[i] - transition_duration
        offsets: list[float] = []
        for i in range(n - 1):
            if i == 0:
                offset = clip_durations[0] - transition_duration
            else:
                offset = offsets[i - 1] + clip_durations[i] - transition_duration
            offsets.append(max(0.0, offset))

        # Input arguments
        cmd: list[str] = ["ffmpeg", "-y"]
        for vp in video_paths:
            cmd.extend(["-i", str(vp)])

        if n == 2:
            trans = transitions[0] if transitions[0] in _SUPPORTED_TRANSITIONS else "dissolve"
            filter_str = (
                f"[0:v][1:v]xfade=transition={trans}"
                f":duration={transition_duration}:offset={offsets[0]}[outv]"
            )
            cmd.extend([
                "-filter_complex", filter_str,
                "-map", "[outv]",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                str(output_path),
            ])
            return cmd

        # General case: chain xfade filters for n > 2 clips
        filters: list[str] = []
        prev_label = "0:v"
        for i in range(1, n):
            raw_trans = transitions[i - 1]
            trans = raw_trans if raw_trans in _SUPPORTED_TRANSITIONS else "dissolve"
            out_label = f"v{i}" if i < n - 1 else "outv"
            filters.append(
                f"[{prev_label}][{i}:v]xfade=transition={trans}"
                f":duration={transition_duration}:offset={offsets[i - 1]}[{out_label}]"
            )
            prev_label = out_label

        filter_complex = ";".join(filters)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            str(output_path),
        ])
        return cmd

    @staticmethod
    def _build_audio_mix_cmd(
        video_path: Path,
        audio_tracks: list[AudioTrack],
        output_path: Path,
    ) -> list[str]:
        """Build FFmpeg command to mix audio tracks onto a video."""
        cmd: list[str] = ["ffmpeg", "-y", "-i", str(video_path)]

        # Add each audio track as an input
        for track in audio_tracks:
            cmd.extend(["-i", str(track.path)])

        # Build amix filter
        # Each audio input gets volume adjustment and delay
        filters: list[str] = []
        mix_inputs: list[str] = []

        for i, track in enumerate(audio_tracks):
            input_idx = i + 1  # 0 is the video
            label = f"a{i}"
            parts: list[str] = []

            if track.start_time > 0:
                delay_ms = int(track.start_time * 1000)
                parts.append(f"adelay={delay_ms}|{delay_ms}")

            if track.volume != 1.0:
                parts.append(f"volume={track.volume}")

            if parts:
                filter_chain = ",".join(parts)
                filters.append(f"[{input_idx}:a]{filter_chain}[{label}]")
                mix_inputs.append(f"[{label}]")
            else:
                mix_inputs.append(f"[{input_idx}:a]")

        # Mix all audio tracks together
        n_audio = len(audio_tracks)
        mix_label = "mixed"
        mix_str = "".join(mix_inputs)
        filters.append(
            f"{mix_str}amix=inputs={n_audio}:duration=longest"
            f":dropout_transition=2[{mix_label}]"
        )

        filter_complex = ";".join(filters)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", f"[{mix_label}]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ])
        return cmd

    @staticmethod
    def _build_subtitle_cmd(
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
    ) -> list[str]:
        """Build FFmpeg command to burn subtitles into a video."""
        # Escape special characters in path for FFmpeg subtitle filter.
        # FFmpeg filter syntax requires escaping: \ : ' [ ] ;
        sub_path_escaped = str(subtitle_path)
        for ch in ("\\", ":", "'", "[", "]", ";"):
            sub_path_escaped = sub_path_escaped.replace(ch, f"\\{ch}")
        return [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles={sub_path_escaped}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path),
        ]

    @staticmethod
    def _build_final_render_cmd(
        video_path: Path,
        audio_tracks: list[AudioTrack],
        subtitle_path: Path | None,
        output_path: Path,
    ) -> list[str]:
        """Build a single FFmpeg command for audio + subtitles in one pass."""
        cmd: list[str] = ["ffmpeg", "-y", "-i", str(video_path)]

        filter_parts: list[str] = []

        # -- Audio inputs and filters --
        if audio_tracks:
            for track in audio_tracks:
                cmd.extend(["-i", str(track.path)])

            mix_inputs: list[str] = []
            for i, track in enumerate(audio_tracks):
                input_idx = i + 1
                label = f"a{i}"
                parts: list[str] = []

                if track.start_time > 0:
                    delay_ms = int(track.start_time * 1000)
                    parts.append(f"adelay={delay_ms}|{delay_ms}")
                if track.volume != 1.0:
                    parts.append(f"volume={track.volume}")

                if parts:
                    chain = ",".join(parts)
                    filter_parts.append(f"[{input_idx}:a]{chain}[{label}]")
                    mix_inputs.append(f"[{label}]")
                else:
                    mix_inputs.append(f"[{input_idx}:a]")

            n_audio = len(audio_tracks)
            mix_str = "".join(mix_inputs)
            filter_parts.append(
                f"{mix_str}amix=inputs={n_audio}:duration=longest"
                f":dropout_transition=2[outa]"
            )

        # -- Subtitle filter on video stream --
        video_filters: list[str] = []
        if subtitle_path is not None:
            sub_escaped = str(subtitle_path)
            for ch in ("\\", ":", "'", "[", "]", ";"):
                sub_escaped = sub_escaped.replace(ch, f"\\{ch}")
            video_filters.append(f"subtitles={sub_escaped}")

        if video_filters:
            vf_chain = ",".join(video_filters)
            filter_parts.append(f"[0:v]{vf_chain}[outv]")

        # -- Assemble command --
        if filter_parts:
            cmd.extend(["-filter_complex", ";".join(filter_parts)])

        # Map outputs
        if video_filters:
            cmd.extend(["-map", "[outv]"])
        else:
            cmd.extend(["-map", "0:v"])

        if audio_tracks:
            cmd.extend(["-map", "[outa]"])

        # Encoding settings
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
        ])
        if audio_tracks:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])

        cmd.append(str(output_path))
        return cmd

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _ensure_ffmpeg() -> None:
        """Verify FFmpeg is available, raising if not."""
        available = await check_ffmpeg()
        if not available:
            raise RuntimeError(
                "FFmpeg is not installed or not found on PATH. "
                "Install it via: brew install ffmpeg (macOS) / "
                "apt install ffmpeg (Ubuntu)"
            )

    @staticmethod
    async def _run_ffmpeg(cmd: list[str]) -> None:
        """Run an FFmpeg command, raising on failure."""
        logger.debug("FFmpeg command: %s", " ".join(cmd))
        await run_ffmpeg(cmd[1:])  # run_ffmpeg prepends 'ffmpeg' or takes args
        # run_ffmpeg from utils already raises on error
