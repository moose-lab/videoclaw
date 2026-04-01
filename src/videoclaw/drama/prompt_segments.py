"""Prompt segment data structures, slot allocator, segmenter, and content builder.

Provides building blocks for constructing Seedance 2.0 content[] arrays with
interleaved text and reference-image entries from annotated prompt text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from videoclaw.drama.models import ShotScale

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REFERENCE_IMAGES = 9

# Budget: (chars, scenes, props) keyed by ShotScale value
_SLOT_BUDGET: dict[str | None, tuple[int, int, int]] = {
    ShotScale.CLOSE_UP:      (6, 2, 1),
    ShotScale.MEDIUM_CLOSE:  (5, 3, 1),
    ShotScale.MEDIUM:        (4, 3, 2),
    ShotScale.WIDE:          (3, 4, 2),
    ShotScale.EXTREME_WIDE:  (2, 5, 2),
    None:                    (4, 3, 2),
}

# Regex for [ref:key] markers — leading/trailing whitespace consumed
_REF_PATTERN = re.compile(r"\s*\[ref:([a-zA-Z0-9_]+)\]\s*")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReferenceMedia:
    """A single reference image asset for a shot."""

    ref_type: str  # "character" | "scene" | "prop"
    key: str       # character name, location key, or prop name
    url: str | None = None   # HTTPS URL (preferred for Seedance API)
    path: str | None = None  # local file path (fallback)


@dataclass(slots=True)
class PromptSegment:
    """One chunk of prompt text paired with an optional reference image."""

    text: str
    reference: ReferenceMedia | None = None


# ---------------------------------------------------------------------------
# Slot allocator
# ---------------------------------------------------------------------------


def allocate_reference_slots(
    shot_scale: ShotScale | None,
    available: dict[str, dict[str, str]],
    *,
    speaking_character: str | None = None,
) -> list[ReferenceMedia]:
    """Dynamically allocate up to MAX_REFERENCE_IMAGES reference slots.

    Parameters
    ----------
    shot_scale:
        The ShotScale of the current shot — determines per-category budget.
    available:
        Dict with keys ``"characters"``, ``"scenes"``, ``"props"``.
        Each value maps ``name -> URL-or-path`` string.
    speaking_character:
        If given, this character is placed first in the character slot list.

    Returns
    -------
    list[ReferenceMedia]
        Up to MAX_REFERENCE_IMAGES entries in priority order.
    """
    char_budget, scene_budget, prop_budget = _SLOT_BUDGET.get(
        shot_scale, _SLOT_BUDGET[None]
    )

    chars_map: dict[str, str] = available.get("characters", {})
    scenes_map: dict[str, str] = available.get("scenes", {})
    props_map: dict[str, str] = available.get("props", {})

    result: list[ReferenceMedia] = []

    # --- Characters ---
    char_keys: list[str] = []
    if speaking_character and speaking_character in chars_map:
        char_keys.append(speaking_character)
    for k in chars_map:
        if k not in char_keys:
            char_keys.append(k)
    for key in char_keys[:char_budget]:
        ref = _make_ref("character", key, chars_map[key])
        result.append(ref)

    # --- Scenes ---
    for key, value in list(scenes_map.items())[:scene_budget]:
        result.append(_make_ref("scene", key, value))

    # --- Props ---
    for key, value in list(props_map.items())[:prop_budget]:
        result.append(_make_ref("prop", key, value))

    return result[:MAX_REFERENCE_IMAGES]


def _make_ref(ref_type: str, key: str, value: str) -> ReferenceMedia:
    """Build a ReferenceMedia from a value that may be a URL or local path."""
    if value.startswith("http"):
        return ReferenceMedia(ref_type=ref_type, key=key, url=value)
    return ReferenceMedia(ref_type=ref_type, key=key, path=value)


# ---------------------------------------------------------------------------
# PromptSegmenter
# ---------------------------------------------------------------------------


class PromptSegmenter:
    """Parse [ref:key] markers from enhanced prompt text into PromptSegments."""

    @staticmethod
    def parse(
        text: str,
        ref_map: dict[str, ReferenceMedia],
    ) -> list[PromptSegment]:
        """Split text at [ref:key] markers into structured segments.

        - Text before each marker → segment paired with the ref (if key known).
        - Unknown keys: marker stripped, text merged into surrounding segment.
        - Trailing text after last marker → segment with no reference.
        - No markers → single segment with full text.
        """
        parts = _REF_PATTERN.split(text)
        # split() with one capture group gives:
        #   [text0, key1, text1, key2, text2, ...]
        # parts[0]       — text before first marker (may be "")
        # parts[1]       — first key
        # parts[2]       — text between first and second marker
        # ...

        if len(parts) == 1:
            # No markers found at all
            return [PromptSegment(text=text.strip())]

        segments: list[PromptSegment] = []
        # Accumulated text for segments whose ref key was unknown
        pending_text = parts[0].strip()

        i = 1
        while i < len(parts):
            key = parts[i]
            following_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            i += 2

            ref = ref_map.get(key)
            if ref is None:
                # Unknown key — strip the marker, merge text into pending
                if pending_text and following_text:
                    pending_text = pending_text + " " + following_text
                elif following_text:
                    pending_text = following_text
                # (if both empty, pending_text stays as-is)
                continue

            # Known ref — emit a segment for pending_text+key, then continue
            segment_text = pending_text
            segments.append(PromptSegment(text=segment_text, reference=ref))
            pending_text = following_text

        # Any remaining pending text becomes a trailing plain segment
        if pending_text:
            segments.append(PromptSegment(text=pending_text))
        elif not segments:
            # Edge case: all markers were unknown and text was empty
            segments.append(PromptSegment(text=""))

        return segments

    @staticmethod
    def strip_markers(text: str) -> str:
        """Remove all [ref:key] markers and collapse extra spaces."""
        cleaned = _REF_PATTERN.sub(" ", text)
        # Collapse multiple spaces into one and strip edges
        return re.sub(r" {2,}", " ", cleaned).strip()


# ---------------------------------------------------------------------------
# ContentBuilder
# ---------------------------------------------------------------------------


class ContentBuilder:
    """Convert PromptSegments into an interleaved Seedance 2.0 content array."""

    @staticmethod
    def build(segments: list[PromptSegment]) -> list[dict[str, Any]]:
        """Build an interleaved content array suitable for the Seedance API.

        Rules:
        - Each segment's text produces ``{"type": "text", "text": ...}`` if non-empty.
        - If the segment has a ref with an HTTPS URL and image count < 9, add an
          ``{"type": "image_url", ...}`` entry immediately after the text.
        - Path-only refs are NOT added here (handled by the adapter).
        - Maximum 9 image entries enforced.
        """
        content: list[dict[str, Any]] = []
        image_count = 0

        for seg in segments:
            if seg.text:
                content.append({"type": "text", "text": seg.text})

            if (
                seg.reference is not None
                and seg.reference.url is not None
                and seg.reference.url.startswith("http")
                and image_count < MAX_REFERENCE_IMAGES
            ):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": seg.reference.url},
                        "role": "reference_image",
                    }
                )
                image_count += 1

        return content

    @staticmethod
    def collect_path_refs(segments: list[PromptSegment]) -> list[ReferenceMedia]:
        """Return refs that only have a local path (no URL) for adapter handling."""
        return [
            seg.reference
            for seg in segments
            if seg.reference is not None
            and seg.reference.path is not None
            and seg.reference.url is None
        ]
