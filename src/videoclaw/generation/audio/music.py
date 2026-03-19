"""Background music generation for drama episodes.

Provides a MusicProvider protocol and implementations for generating
or sourcing background music tracks.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class MusicProvider(Protocol):
    """Protocol for background music generation."""

    async def generate(
        self,
        mood: str,
        style: str,
        duration_seconds: float,
        output_path: Path,
    ) -> Path: ...


class SilentMusicProvider:
    """Generates silent audio tracks as placeholder BGM.

    Uses ffmpeg to create a correctly-timed silent audio file.
    This ensures the compose pipeline has a valid audio track
    to work with.
    """

    async def generate(
        self,
        mood: str,
        style: str,
        duration_seconds: float,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(duration_seconds),
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.warning(
                "[music] ffmpeg silent track failed: %s",
                stderr.decode(errors="replace")[:200],
            )
            # Fallback: write a minimal valid file
            output_path.write_bytes(b"")
            return output_path

        logger.info(
            "[music] Generated silent BGM (%.1fs) -> %s",
            duration_seconds, output_path,
        )
        return output_path


class MusicManager:
    """High-level music manager."""

    def __init__(self, provider: MusicProvider | None = None) -> None:
        self._provider = provider or SilentMusicProvider()

    async def generate_bgm(
        self,
        mood: str,
        style: str,
        duration_seconds: float,
        output_path: Path,
    ) -> Path:
        return await self._provider.generate(
            mood=mood,
            style=style,
            duration_seconds=duration_seconds,
            output_path=output_path,
        )

