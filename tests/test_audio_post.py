"""Tests for AudioPostProcessor — FFmpeg-based audio post-processing."""

from videoclaw.drama.models import LineType
from videoclaw.generation.audio.audio_post import AudioPostProcessor


# ---------------------------------------------------------------------------
# Original tests (pre-existing)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 3.1.3 — build_eq_filter
# ---------------------------------------------------------------------------


class TestBuildEqFilter:
    """build_eq_filter returns distinct FFmpeg equalizer strings per LineType."""

    def test_dialogue_presence_boost(self):
        proc = AudioPostProcessor()
        filt = proc.build_eq_filter(LineType.DIALOGUE)
        # Dialogue gets a 2-4kHz presence boost via the equalizer filter
        assert "equalizer" in filt
        assert filt != ""

    def test_narration_warmth_boost(self):
        proc = AudioPostProcessor()
        filt = proc.build_eq_filter(LineType.NARRATION)
        # Narration boosts low-mids for warmth (200-400Hz range)
        assert "equalizer" in filt
        assert filt != ""

    def test_inner_monologue_highpass(self):
        proc = AudioPostProcessor()
        filt = proc.build_eq_filter(LineType.INNER_MONOLOGUE)
        # Inner monologue cuts lows via highpass
        assert "highpass" in filt
        assert filt != ""

    def test_each_line_type_different(self):
        proc = AudioPostProcessor()
        filters = {lt: proc.build_eq_filter(lt) for lt in LineType}
        # All three must be distinct
        values = list(filters.values())
        assert len(set(values)) == len(values), "Each LineType must produce a unique EQ filter"


# ---------------------------------------------------------------------------
# Task 3.1.3 — build_reverb_filter
# ---------------------------------------------------------------------------


class TestBuildReverbFilter:
    """build_reverb_filter returns valid aecho-based filter strings per room type."""

    def test_palace_reverb(self):
        proc = AudioPostProcessor()
        filt = proc.build_reverb_filter("palace")
        assert "aecho" in filt
        assert filt != ""

    def test_cave_reverb(self):
        proc = AudioPostProcessor()
        filt = proc.build_reverb_filter("cave")
        assert "aecho" in filt
        assert filt != ""

    def test_outdoor_reverb(self):
        proc = AudioPostProcessor()
        filt = proc.build_reverb_filter("outdoor")
        assert "aecho" in filt
        assert filt != ""

    def test_chamber_reverb(self):
        proc = AudioPostProcessor()
        filt = proc.build_reverb_filter("chamber")
        assert "aecho" in filt
        assert filt != ""

    def test_none_returns_empty(self):
        proc = AudioPostProcessor()
        filt = proc.build_reverb_filter("none")
        assert filt == ""

    def test_each_room_type_different(self):
        proc = AudioPostProcessor()
        room_types = ["palace", "cave", "outdoor", "chamber"]
        filters = {rt: proc.build_reverb_filter(rt) for rt in room_types}
        values = list(filters.values())
        assert len(set(values)) == len(values), "Each room type must produce a unique reverb filter"


# ---------------------------------------------------------------------------
# Task 3.1.3 — build_silence
# ---------------------------------------------------------------------------


class TestBuildSilence:
    """build_silence generates anullsrc-based silence for pause_before_ms gaps."""

    def test_500ms_silence(self):
        proc = AudioPostProcessor()
        filt = proc.build_silence(500)
        assert "anullsrc" in filt
        assert filt != ""

    def test_zero_returns_empty(self):
        proc = AudioPostProcessor()
        filt = proc.build_silence(0)
        assert filt == ""

    def test_positive_duration_contains_duration_info(self):
        proc = AudioPostProcessor()
        filt = proc.build_silence(1000)
        # The filter should encode the duration somehow (1.0s or 1000ms)
        assert "anullsrc" in filt
        # Must reference the duration value
        assert "1" in filt  # at least the leading digit of the duration


# ---------------------------------------------------------------------------
# Task 3.1.3 — build_filter_chain
# ---------------------------------------------------------------------------


class TestBuildFilterChain:
    """build_filter_chain combines EQ + reverb + silence into one string."""

    def test_all_effects_combined(self):
        proc = AudioPostProcessor()
        chain = proc.build_filter_chain(
            line_type=LineType.DIALOGUE,
            room_type="palace",
            pause_before_ms=500,
        )
        # Must contain EQ (equalizer), reverb (aecho), and silence (anullsrc)
        assert "equalizer" in chain
        assert "aecho" in chain
        assert "anullsrc" in chain

    def test_no_reverb_no_pause(self):
        proc = AudioPostProcessor()
        chain = proc.build_filter_chain(
            line_type=LineType.NARRATION,
            room_type="none",
            pause_before_ms=0,
        )
        # Only EQ, no reverb or silence
        assert "equalizer" in chain
        assert "aecho" not in chain
        assert "anullsrc" not in chain

    def test_no_effects_returns_empty(self):
        """When room_type='none' and pause_before_ms=0 and line_type produces
        only an EQ filter, the chain is NOT empty (EQ is always present).
        But we want a test that verifies the chain is empty when ALL parts
        are empty — we can't easily make EQ empty since all LineTypes produce
        one. So this test verifies that with defaults, at minimum EQ is present."""
        proc = AudioPostProcessor()
        chain = proc.build_filter_chain(
            line_type=LineType.DIALOGUE,
            room_type="none",
            pause_before_ms=0,
        )
        # EQ is always present for any LineType, so chain is non-empty
        assert chain != ""
        # But no reverb or silence parts
        assert "aecho" not in chain
        assert "anullsrc" not in chain

    def test_with_pause_only(self):
        proc = AudioPostProcessor()
        chain = proc.build_filter_chain(
            line_type=LineType.INNER_MONOLOGUE,
            room_type="none",
            pause_before_ms=300,
        )
        # EQ (highpass) + silence, no reverb
        assert "highpass" in chain
        assert "anullsrc" in chain
        assert "aecho" not in chain

    def test_with_reverb_only(self):
        proc = AudioPostProcessor()
        chain = proc.build_filter_chain(
            line_type=LineType.NARRATION,
            room_type="cave",
            pause_before_ms=0,
        )
        # EQ + reverb, no silence
        assert "equalizer" in chain
        assert "aecho" in chain
        assert "anullsrc" not in chain
