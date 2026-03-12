"""Data models for drama series production."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Character:
    """A character in a drama series."""

    name: str = ""
    description: str = ""
    reference_image: str = ""  # file path to character reference PNG


@dataclass
class DramaScene:
    """A single scene within a drama episode."""

    scene_id: str = ""
    description: str = ""
    prompt: str = ""
    duration_seconds: float = 5.0
    characters_present: list[str] = field(default_factory=list)
    speaking_character: str = ""
    model_id: str = ""


@dataclass
class DramaEpisode:
    """An episode consisting of multiple scenes."""

    episode_id: str = ""
    title: str = ""
    scenes: list[DramaScene] = field(default_factory=list)


@dataclass
class DramaSeries:
    """A drama series with characters and episodes."""

    series_id: str = ""
    title: str = ""
    aspect_ratio: str = "16:9"
    characters: list[Character] = field(default_factory=list)
    episodes: list[DramaEpisode] = field(default_factory=list)
