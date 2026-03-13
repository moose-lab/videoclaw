"""Video renderer -- final FFmpeg encode with resolution, codec, metadata, and watermark.

The :class:`VideoRenderer` takes a composed video and produces the final
deliverable with configurable encoding parameters, metadata injection,
and optional watermark overlay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from videoclaw.utils.ffmpeg import run_ffmpeg, check_ffmpeg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Render profile
# ---------------------------------------------------------------------------


@dataclass
class RenderProfile:
    """Encoding parameters for the final render pass."""

    resolution: tuple[int, int] | None = None  # (width, height), e.g. (1080, 1920) for 9:16
    bitrate: str = "8M"
    audio_bitrate: str = "192k"
    codec: str = "libx264"
    preset: str = "medium"
    crf: int | None = 23
    metadata: dict[str, str] = field(default_factory=dict)
    watermark_path: Path | None = None


# Aspect ratio to resolution mapping for final render
_ASPECT_TO_RENDER_RESOLUTION: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:3": (1440, 1080),
    "3:4": (1080, 1440),
}


# ---------------------------------------------------------------------------
# VideoRenderer
# ---------------------------------------------------------------------------


class VideoRenderer:
    """Produces the final deliverable video via FFmpeg.

    Uses the same async subprocess pattern as
    :class:`videoclaw.generation.compose.VideoComposer`.
    """

    async def render(
        self,
        input_path: Path,
        output_path: Path,
        *,
        resolution: tuple[int, int] | None = None,
        bitrate: str = "8M",
        audio_bitrate: str = "192k",
        codec: str = "libx264",
        preset: str = "medium",
        crf: int | None = 23,
        metadata: dict[str, str] | None = None,
        watermark_path: Path | None = None,
    ) -> Path:
        """Render *input_path* to *output_path* with the given encoding settings.

        Parameters
        ----------
        input_path:
            Source video (typically the composed output).
        output_path:
            Destination for the final encoded video.
        resolution:
            Target (width, height). When set, a scale filter is applied.
        bitrate:
            Video bitrate (e.g. ``"8M"``).
        audio_bitrate:
            Audio bitrate (e.g. ``"192k"``).
        codec:
            Video codec (e.g. ``"libx264"``, ``"libx265"``).
        preset:
            Encoding speed/quality preset (e.g. ``"medium"``).
        crf:
            Constant rate factor. ``None`` disables CRF (bitrate-only mode).
        metadata:
            Key-value pairs injected as FFmpeg metadata tags.
        watermark_path:
            Path to a watermark image to overlay in the top-right corner.

        Returns
        -------
        Path
            The *output_path* on success.
        """
        await self._ensure_ffmpeg()

        cmd = self.build_cmd(
            input_path=input_path,
            output_path=output_path,
            resolution=resolution,
            bitrate=bitrate,
            audio_bitrate=audio_bitrate,
            codec=codec,
            preset=preset,
            crf=crf,
            metadata=metadata,
            watermark_path=watermark_path,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run_ffmpeg(cmd)
        logger.info("Render complete: %s", output_path)
        return output_path

    @staticmethod
    def build_cmd(
        *,
        input_path: Path,
        output_path: Path,
        resolution: tuple[int, int] | None = None,
        bitrate: str = "8M",
        audio_bitrate: str = "192k",
        codec: str = "libx264",
        preset: str = "medium",
        crf: int | None = 23,
        metadata: dict[str, str] | None = None,
        watermark_path: Path | None = None,
    ) -> list[str]:
        """Build the FFmpeg command list for the render pass.

        This is exposed as a static method so tests can inspect the
        generated command without running FFmpeg.
        """
        cmd: list[str] = ["ffmpeg", "-y", "-i", str(input_path)]

        # Watermark overlay input
        if watermark_path is not None:
            cmd.extend(["-i", str(watermark_path)])

        # --- Video filters ---
        video_filters: list[str] = []

        if resolution is not None:
            w, h = resolution
            video_filters.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")

        if watermark_path is not None:
            # Overlay watermark in top-right with 10px padding
            video_filters.append("overlay=W-w-10:10")

        if video_filters:
            # When watermark is present, we need filter_complex instead of -vf
            if watermark_path is not None:
                # Build filter_complex for watermark overlay
                filter_str = ""
                if resolution is not None:
                    w, h = resolution
                    filter_str += f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2[scaled];[scaled][1:v]overlay=W-w-10:10[outv]"
                else:
                    filter_str += "[0:v][1:v]overlay=W-w-10:10[outv]"
                cmd.extend(["-filter_complex", filter_str, "-map", "[outv]", "-map", "0:a?"])
            else:
                cmd.extend(["-vf", ";".join(video_filters) if len(video_filters) == 1 else ",".join(video_filters)])
        # No video filters and no watermark: just pass through

        # --- Codec and encoding ---
        cmd.extend(["-c:v", codec])
        cmd.extend(["-preset", preset])

        if crf is not None:
            cmd.extend(["-crf", str(crf)])

        cmd.extend(["-b:v", bitrate])
        cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])

        # --- Metadata ---
        if metadata:
            for key, value in metadata.items():
                cmd.extend(["-metadata", f"{key}={value}"])

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
        result = await run_ffmpeg(cmd[1:])  # run_ffmpeg prepends 'ffmpeg'
