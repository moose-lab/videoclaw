import numpy as np
import pytest
from PIL import Image

from videoclaw.drama.frame_analyzer import (
    compute_center_ssim,
    detect_temporal_breaks,
    TemporalBreak,
)


class TestComputeCenterSSIM:
    def test_identical_frames_return_1(self):
        frame = np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)
        assert compute_center_ssim(frame, frame) == pytest.approx(1.0, abs=0.01)

    def test_completely_different_frames_return_low(self):
        white = np.full((1280, 720, 3), 255, dtype=np.uint8)
        black = np.zeros((1280, 720, 3), dtype=np.uint8)
        score = compute_center_ssim(white, black)
        assert score < 0.2

    def test_uses_center_60_percent(self):
        frame_a = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame_b = frame_a.copy()
        frame_b[:20, :, :] = 0
        frame_b[80:, :, :] = 0
        frame_b[:, :20, :] = 0
        frame_b[:, 80:, :] = 0
        score = compute_center_ssim(frame_a, frame_b)
        assert score > 0.95


class TestDetectTemporalBreaks:
    def test_stable_video_returns_no_breaks(self):
        frame = np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)
        frames = [frame] * 10
        breaks = detect_temporal_breaks(frames)
        assert breaks == []

    def test_sudden_scene_change_detected(self):
        frames_a = [np.full((1280, 720, 3), 128, dtype=np.uint8)] * 5
        frames_b = [np.full((1280, 720, 3), 0, dtype=np.uint8)] * 5
        frames = frames_a + frames_b
        breaks = detect_temporal_breaks(frames, fatal_threshold=0.75)
        assert len(breaks) >= 1
        assert breaks[0].frame_pair == (4, 5)
        assert breaks[0].severity == "fatal"

    def test_minor_flicker_detected_as_tolerable(self):
        base = np.full((1280, 720, 3), 128, dtype=np.uint8)
        noisy = base.copy()
        noise = np.random.randint(-40, 40, base.shape, dtype=np.int16)
        noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames = [base, base, base, noisy, base, base]
        breaks = detect_temporal_breaks(
            frames, fatal_threshold=0.75, tolerable_threshold=0.85
        )
        tolerable = [b for b in breaks if b.severity == "tolerable"]
        assert len(tolerable) >= 1

    def test_returns_temporal_break_dataclass(self):
        frame = np.full((1280, 720, 3), 128, dtype=np.uint8)
        different = np.zeros((1280, 720, 3), dtype=np.uint8)
        frames = [frame, different]
        breaks = detect_temporal_breaks(frames, fatal_threshold=0.75)
        assert len(breaks) == 1
        b = breaks[0]
        assert isinstance(b, TemporalBreak)
        assert isinstance(b.ssim_score, float)
        assert b.frame_pair == (0, 1)


class TestEmptyAndEdgeCases:
    def test_empty_frames_returns_empty(self):
        assert detect_temporal_breaks([]) == []

    def test_single_frame_returns_empty(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detect_temporal_breaks([frame]) == []
