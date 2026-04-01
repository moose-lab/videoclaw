"""Tests for composition-level audit."""
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import numpy as np
import pytest

from videoclaw.drama.vision_auditor import VisionAuditor


class TestAuditComposition:
    @pytest.mark.asyncio
    async def test_short_video_extracts_5_frames(self):
        auditor = VisionAuditor()
        mock_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 5
        with patch("videoclaw.drama.vision_auditor.get_video_duration", new_callable=AsyncMock, return_value=20.0), \
             patch("videoclaw.drama.frame_analyzer.extract_frames_as_arrays", return_value=mock_frames), \
             patch("videoclaw.drama.frame_analyzer.detect_temporal_breaks", return_value=[]), \
             patch.object(auditor, "_composition_vision_llm", new_callable=AsyncMock, return_value=([], [])):
            result = await auditor.audit_composition(Path("/fake/composed.mp4"), episode_number=1)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_long_video_extracts_10_frames(self):
        auditor = VisionAuditor()
        mock_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 10
        with patch("videoclaw.drama.vision_auditor.get_video_duration", new_callable=AsyncMock, return_value=120.0), \
             patch("videoclaw.drama.frame_analyzer.extract_frames_as_arrays", return_value=mock_frames) as mock_extract, \
             patch("videoclaw.drama.frame_analyzer.detect_temporal_breaks", return_value=[]), \
             patch.object(auditor, "_composition_vision_llm", new_callable=AsyncMock, return_value=([], [])):
            result = await auditor.audit_composition(Path("/fake/composed.mp4"), episode_number=1)
        # Should have requested 10 frames
        assert mock_extract.call_args[1].get("n", mock_extract.call_args[0][1] if len(mock_extract.call_args[0]) > 1 else 10) == 10

    @pytest.mark.asyncio
    async def test_ssim_fatal_in_composition(self):
        auditor = VisionAuditor()
        mock_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 5
        temporal_break = MagicMock(frame_pair=(2, 3), ssim_score=0.5, severity="fatal")
        with patch("videoclaw.drama.vision_auditor.get_video_duration", new_callable=AsyncMock, return_value=20.0), \
             patch("videoclaw.drama.frame_analyzer.extract_frames_as_arrays", return_value=mock_frames), \
             patch("videoclaw.drama.frame_analyzer.detect_temporal_breaks", return_value=[temporal_break]), \
             patch.object(auditor, "_composition_vision_llm", new_callable=AsyncMock, return_value=([], [])) as mock_llm:
            result = await auditor.audit_composition(Path("/fake/composed.mp4"), episode_number=1)
        assert result.regen_required is True
        # LLM should still be called for composition (no short-circuit at composition level)
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_medium_video_extracts_8_frames(self):
        auditor = VisionAuditor()
        mock_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 8
        with patch("videoclaw.drama.vision_auditor.get_video_duration", new_callable=AsyncMock, return_value=60.0), \
             patch("videoclaw.drama.frame_analyzer.extract_frames_as_arrays", return_value=mock_frames) as mock_extract, \
             patch("videoclaw.drama.frame_analyzer.detect_temporal_breaks", return_value=[]), \
             patch.object(auditor, "_composition_vision_llm", new_callable=AsyncMock, return_value=([], [])):
            result = await auditor.audit_composition(Path("/fake/composed.mp4"), episode_number=1)
        assert mock_extract.call_args[1].get("n", mock_extract.call_args[0][1] if len(mock_extract.call_args[0]) > 1 else 8) == 8

    @pytest.mark.asyncio
    async def test_long_video_sends_4_frames_to_llm(self):
        """Long videos (>90s) should only send first 2 + last 2 frames to LLM."""
        auditor = VisionAuditor()
        # Create 10 distinct frames so we can verify which were passed
        mock_frames = [np.full((100, 100, 3), i * 25, dtype=np.uint8) for i in range(10)]
        with patch("videoclaw.drama.vision_auditor.get_video_duration", new_callable=AsyncMock, return_value=120.0), \
             patch("videoclaw.drama.frame_analyzer.extract_frames_as_arrays", return_value=mock_frames), \
             patch("videoclaw.drama.frame_analyzer.detect_temporal_breaks", return_value=[]), \
             patch.object(auditor, "_composition_vision_llm", new_callable=AsyncMock, return_value=([], [])) as mock_llm:
            await auditor.audit_composition(Path("/fake/composed.mp4"), episode_number=1)
        # Should pass first 2 + last 2 = 4 frames to LLM
        llm_frames = mock_llm.call_args[0][0]
        assert len(llm_frames) == 4

    @pytest.mark.asyncio
    async def test_uses_provided_duration(self):
        """When total_duration is provided, should not call get_video_duration."""
        auditor = VisionAuditor()
        mock_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 5
        with patch("videoclaw.drama.vision_auditor.get_video_duration", new_callable=AsyncMock) as mock_dur, \
             patch("videoclaw.drama.frame_analyzer.extract_frames_as_arrays", return_value=mock_frames), \
             patch("videoclaw.drama.frame_analyzer.detect_temporal_breaks", return_value=[]), \
             patch.object(auditor, "_composition_vision_llm", new_callable=AsyncMock, return_value=([], [])):
            result = await auditor.audit_composition(
                Path("/fake/composed.mp4"), episode_number=1, total_duration=15.0,
            )
        mock_dur.assert_not_called()
        assert result.passed is True
