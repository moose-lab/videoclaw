"""Tests for AudioPostProcessor — FFmpeg-based audio post-processing."""

from videoclaw.drama.models import LineType
from videoclaw.generation.audio.audio_post import AudioPostProcessor


def test_inner_monologue_filter_string():
    proc = AudioPostProcessor()
    filt = proc.inner_monologue_filter()
    assert "aecho" in filt
    assert "lowpass" in filt


def test_narration_filter_string():
    proc = AudioPostProcessor()
    filt = proc.narration_filter()
    assert "loudnorm" in filt


def test_get_filter_for_line_type():
    proc = AudioPostProcessor()
    assert "aecho" in proc.get_filter_for(LineType.INNER_MONOLOGUE)
    assert "loudnorm" in proc.get_filter_for(LineType.NARRATION)
    assert proc.get_filter_for(LineType.DIALOGUE) == ""
