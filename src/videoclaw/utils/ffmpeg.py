"""FFmpeg utility helpers -- thin async wrappers around the FFmpeg CLI.

These functions handle process management, timeout enforcement, and metadata
probing so that higher-level modules never construct ``subprocess`` calls
directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def check_ffmpeg() -> bool:
    """Return ``True`` if ``ffmpeg`` and ``ffprobe`` are available on PATH.

    This performs a lightweight ``-version`` call to verify that the binaries
    are installed and executable.
    """
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            logger.warning("%s not found on PATH", binary)
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                binary, "-version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10)
            if proc.returncode != 0:
                logger.warning("%s returned exit code %d", binary, proc.returncode)
                return False
        except (TimeoutError, OSError) as exc:
            logger.warning("Failed to run %s: %s", binary, exc)
            return False

    return True


async def get_video_duration(path: Path) -> float:
    """Probe the duration of *path* in seconds using ``ffprobe``.

    Parameters
    ----------
    path:
        Path to a video file.

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
        raise FileNotFoundError(f"Video file not found: {path}")

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


async def get_video_info(path: Path) -> dict:
    """Probe full video metadata as a dictionary.

    The returned dict contains top-level keys ``"format"`` and ``"streams"``
    as reported by ``ffprobe -show_format -show_streams``.

    Parameters
    ----------
    path:
        Path to a video file.

    Returns
    -------
    dict
        Parsed ffprobe JSON output.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If ``ffprobe`` fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
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
        return json.loads(stdout.decode())  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse ffprobe JSON output for {path}"
        ) from exc


async def run_ffmpeg(
    args: list[str],
    timeout: int = 300,
) -> asyncio.subprocess.Process:
    """Run ``ffmpeg`` with *args* and raise on failure.

    Parameters
    ----------
    args:
        Arguments passed to ``ffmpeg`` (do **not** include the ``ffmpeg``
        binary name -- it is prepended automatically).
    timeout:
        Maximum execution time in seconds.

    Returns
    -------
    asyncio.subprocess.Process
        The completed process object.

    Raises
    ------
    RuntimeError
        If FFmpeg exits with a non-zero code or the timeout is exceeded.
    """
    cmd = ["ffmpeg", *args]
    logger.debug("Running: %s", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(
            f"FFmpeg timed out after {timeout}s. Command: {' '.join(cmd)}"
        )

    if proc.returncode != 0:
        stderr_text = stderr.decode(errors="replace").strip()
        # Extract the last few lines for a concise error message
        error_lines = stderr_text.splitlines()[-10:]
        error_summary = "\n".join(error_lines)
        raise RuntimeError(
            f"FFmpeg exited with code {proc.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr (last 10 lines):\n{error_summary}"
        )

    logger.debug(
        "FFmpeg completed successfully: %s",
        " ".join(cmd[:6]) + ("..." if len(cmd) > 6 else ""),
    )
    return proc
