"""Frame-level temporal stability analyzer using simplified SSIM.

Layer 1 of the pragmatic audit system — algorithm-based, cheap, no external CV
libraries required.  Detects abrupt visual discontinuities between adjacent
frames by computing a center-cropped SSIM score and classifying each detected
break as either ``fatal`` (hard scene cut or severe artifact) or ``tolerable``
(minor flicker or gradual luminance shift).

Intended to be imported by ``vision_auditor.py`` as a fast pre-filter before
the more expensive Claude Vision call.
"""

from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TemporalBreak:
    """A detected discontinuity between two adjacent frames."""

    frame_pair: tuple[int, int]
    ssim_score: float
    severity: str  # "fatal" | "tolerable"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# SSIM stabilisation constants (standard luminance range 0-255)
_C1 = (0.01 * 255) ** 2
_C2 = (0.03 * 255) ** 2


def _crop_center(img: np.ndarray, ratio: float = 0.6) -> np.ndarray:
    """Return the center ``ratio`` fraction of *img* (HxWxC)."""
    h, w = img.shape[:2]
    crop_h = int(h * ratio)
    crop_w = int(w * ratio)
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    return img[y0 : y0 + crop_h, x0 : x0 + crop_w]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_center_ssim(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    center_ratio: float = 0.6,
) -> float:
    """Compute simplified (no-Gaussian) SSIM on the center region of two frames.

    Parameters
    ----------
    frame_a, frame_b:
        HxWxC uint8 numpy arrays representing RGB frames.
    center_ratio:
        Fraction of each dimension to keep (default 0.6 → center 60%).

    Returns
    -------
    float
        SSIM score in roughly [-1, 1]; identical images ≈ 1.0.
    """
    a_crop = _crop_center(frame_a, center_ratio).astype(np.float64)
    b_crop = _crop_center(frame_b, center_ratio).astype(np.float64)

    # Convert to grayscale via luminance weights
    if a_crop.ndim == 3:
        a = 0.2989 * a_crop[..., 0] + 0.5870 * a_crop[..., 1] + 0.1140 * a_crop[..., 2]
    else:
        a = a_crop
    if b_crop.ndim == 3:
        b = 0.2989 * b_crop[..., 0] + 0.5870 * b_crop[..., 1] + 0.1140 * b_crop[..., 2]
    else:
        b = b_crop

    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))

    sigma_a_sq = float(np.var(a))
    sigma_b_sq = float(np.var(b))

    # Degenerate case: one or both frames are near-uniform — the SSIM
    # structure component cannot be estimated reliably when variance is
    # near zero.  Fall back to an RMSE-based similarity on the raw RGB
    # crop so that mild uniform-image perturbations (e.g. flickering) are
    # scored proportionally to their absolute pixel difference.
    if sigma_a_sq < _C2 or sigma_b_sq < _C2:
        rmse = float(np.sqrt(np.mean((a_crop - b_crop) ** 2)))
        return max(0.0, 1.0 - rmse / 128.0)

    sigma_ab = float(np.mean((a - mu_a) * (b - mu_b)))

    numerator = (2 * mu_a * mu_b + _C1) * (2 * sigma_ab + _C2)
    denominator = (mu_a**2 + mu_b**2 + _C1) * (sigma_a_sq + sigma_b_sq + _C2)

    return float(numerator / denominator)


def detect_temporal_breaks(
    frames: list[np.ndarray],
    *,
    fatal_threshold: float = 0.75,
    tolerable_threshold: float = 0.85,
) -> list[TemporalBreak]:
    """Compare adjacent frames and return detected temporal discontinuities.

    Parameters
    ----------
    frames:
        Ordered list of HxWxC uint8 frames (e.g. from
        :func:`extract_frames_as_arrays`).
    fatal_threshold:
        SSIM below this value → severity ``"fatal"``.
    tolerable_threshold:
        SSIM in [fatal_threshold, tolerable_threshold) → severity ``"tolerable"``.
        SSIM >= tolerable_threshold → no break recorded.

    Returns
    -------
    list[TemporalBreak]
        Detected breaks in frame-index order.
    """
    if len(frames) < 2:
        return []

    breaks: list[TemporalBreak] = []
    for i in range(len(frames) - 1):
        score = compute_center_ssim(frames[i], frames[i + 1])
        if score < fatal_threshold:
            breaks.append(
                TemporalBreak(
                    frame_pair=(i, i + 1),
                    ssim_score=score,
                    severity="fatal",
                )
            )
        elif score < tolerable_threshold:
            breaks.append(
                TemporalBreak(
                    frame_pair=(i, i + 1),
                    ssim_score=score,
                    severity="tolerable",
                )
            )
    return breaks


def extract_frames_as_arrays(clip_path: Path, n: int = 10) -> list[np.ndarray]:
    """Extract *n* equally-spaced frames from *clip_path* as numpy arrays.

    Uses ``ffprobe`` to determine duration then ``ffmpeg`` to seek to each
    timestamp and capture a single PNG frame, decoded via PIL.

    Parameters
    ----------
    clip_path:
        Path to any video file supported by ffmpeg.
    n:
        Number of frames to extract (default 10).

    Returns
    -------
    list[np.ndarray]
        HxWxC uint8 arrays in presentation order.

    Raises
    ------
    RuntimeError
        If ffprobe/ffmpeg fails or the clip has zero duration.
    """
    clip_path = Path(clip_path)

    # --- get duration via ffprobe ---
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(clip_path),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse duration from ffprobe output: {result.stdout!r}"
        ) from exc
    if duration <= 0:
        raise RuntimeError(f"Clip has non-positive duration: {duration}")

    # --- extract n equally-spaced frames ---
    frames: list[np.ndarray] = []
    for i in range(n):
        # Space timestamps evenly across the clip duration
        timestamp = duration * (i + 0.5) / n
        ffmpeg_cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", str(clip_path),
            "-frames:v", "1",
            "-f", "image2",
            "-vcodec", "png",
            "pipe:1",
        ]
        frame_result = subprocess.run(
            ffmpeg_cmd, capture_output=True
        )
        if frame_result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed at timestamp {timestamp}: "
                f"{frame_result.stderr.decode(errors='replace').strip()}"
            )
        img = Image.open(io.BytesIO(frame_result.stdout)).convert("RGB")
        frames.append(np.array(img, dtype=np.uint8))

    return frames
