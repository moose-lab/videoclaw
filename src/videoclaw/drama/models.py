"""Data models for AI short drama series."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from videoclaw.config import get_config


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DramaStatus(StrEnum):
    DRAFT = "draft"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class EpisodeStatus(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ShotScale(StrEnum):
    CLOSE_UP = "close_up"
    MEDIUM_CLOSE = "medium_close"
    MEDIUM = "medium"
    WIDE = "wide"
    EXTREME_WIDE = "extreme_wide"


class ShotType(StrEnum):
    ESTABLISHING = "establishing"
    REACTION = "reaction"
    ACTION = "action"
    DETAIL = "detail"
    POV = "pov"


class AudioType(StrEnum):
    DIALOGUE = "dialogue"
    NARRATION = "narration"
    INNER_MONOLOGUE = "inner_monologue"
    SFX = "sfx"
    MUSIC = "music"


class LineType(StrEnum):
    NARRATION = "narration"
    DIALOGUE = "dialogue"
    INNER_MONOLOGUE = "inner_monologue"


class NarrationType(StrEnum):
    """Distinguishes spoken narration from visual-only title cards.

    - ``VOICEOVER``: spoken by a narrator voice — sent to TTS, shown as
      bottom subtitle.
    - ``TITLE_CARD``: visual text overlay only (e.g. "One Month Earlier")
      — **not** sent to TTS, rendered as centered large text.
    """

    VOICEOVER = "voiceover"
    TITLE_CARD = "title_card"


class DramaGenre(StrEnum):
    # Chinese-market genres
    SWEET_ROMANCE = "sweet_romance"
    MALE_POWER_FANTASY = "male_power_fantasy"
    SUSPENSE_THRILLER = "suspense_thriller"
    ANCIENT_XIANXIA = "ancient_xianxia"
    COMEDY = "comedy"
    FAMILY_DRAMA = "family_drama"
    # Western-market genres
    ROMANCE = "romance"
    ACTION_THRILLER = "action_thriller"
    MYSTERY = "mystery"
    SUPERNATURAL = "supernatural"
    DRAMA = "drama"
    SCI_FI = "sci_fi"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Character:
    """A recurring character in the drama series."""

    name: str
    description: str = ""
    visual_prompt: str = ""
    voice_style: str = ""
    reference_image: str | None = None
    reference_images: list[str] = field(default_factory=list)
    reference_image_url: str | None = None  # HTTPS URL for Seedance API
    voice_profile: VoiceProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.voice_profile is not None:
            d["voice_profile"] = self.voice_profile.to_dict()
        else:
            d.pop("voice_profile", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Character:
        data = dict(data)
        vp = data.pop("voice_profile", None)
        # Filter to known fields
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        c = cls(**data)
        if vp is not None:
            c.voice_profile = VoiceProfile.from_dict(vp)
        return c


@dataclass
class DramaScene:
    """A single scene within an episode."""

    scene_id: str = ""
    description: str = ""
    visual_prompt: str = ""
    camera_movement: str = "static"
    duration_seconds: float = 5.0
    dialogue: str = ""
    dialogue_line_type: str = "dialogue"  # "dialogue" | "inner_monologue"
    narration: str = ""
    narration_type: str = "voiceover"  # "voiceover" | "title_card"
    shot_scale: ShotScale | None = None
    shot_type: ShotType | None = None
    speaking_character: str = ""
    emotion: str = ""
    characters_present: list[str] = field(default_factory=list)
    transition: str = ""
    sfx: str = ""
    video_asset_path: str | None = None
    dialogue_audio_path: str | None = None
    narration_audio_path: str | None = None
    scene_status: str = "pending"
    # Structural metadata (added Session 6)
    time_of_day: str = ""       # morning / day / evening / night / unspecified
    scene_group: str = ""       # A / B / C — script location block
    shot_role: str = "normal"   # hook / normal / cliffhanger
    # Vision audit result (written by VisionAuditor, persisted via DramaManager)
    audit_result: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.shot_scale is not None:
            d["shot_scale"] = self.shot_scale.value
        if self.shot_type is not None:
            d["shot_type"] = self.shot_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DramaScene:
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        if data.get("shot_scale") is not None:
            try:
                data["shot_scale"] = ShotScale(data["shot_scale"])
            except ValueError:
                data["shot_scale"] = None
        if data.get("shot_type") is not None:
            try:
                data["shot_type"] = ShotType(data["shot_type"])
            except ValueError:
                data["shot_type"] = None
        return cls(**data)


@dataclass
class VoiceProfile:
    """TTS voice configuration mapped from character personality."""

    voice_id: str = "Friendly_Person"
    speed: float = 1.0
    pitch: int = 0
    emotion: str = "neutral"
    volume: float = 1.0
    role_name: str = ""
    line_type: LineType = LineType.DIALOGUE
    age_feel: str = "young_adult"
    energy: str = "medium"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["line_type"] = self.line_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceProfile:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        if "line_type" in filtered:
            try:
                filtered["line_type"] = LineType(filtered["line_type"])
            except ValueError:
                filtered["line_type"] = LineType.DIALOGUE
        return cls(**filtered)


@dataclass
class DialogueLine:
    """A single line of dialogue, narration, or inner monologue."""

    text: str
    speaker: str
    line_type: LineType = LineType.DIALOGUE
    scene_id: str = ""
    emotion_hint: str | None = None
    duration_seconds: float = 0.0
    asset_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["line_type"] = self.line_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DialogueLine:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        if "line_type" in filtered:
            try:
                filtered["line_type"] = LineType(filtered["line_type"])
            except ValueError:
                filtered["line_type"] = LineType.DIALOGUE
        return cls(**filtered)


@dataclass
class AudioSegment:
    """A single audio segment within an episode timeline."""

    segment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    scene_id: str = ""
    audio_type: AudioType = AudioType.DIALOGUE
    line_type: LineType = LineType.DIALOGUE
    text: str = ""
    character_name: str = ""
    audio_path: str | None = None
    start_time: float = 0.0
    duration_seconds: float = 0.0
    volume: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["audio_type"] = self.audio_type.value
        d["line_type"] = self.line_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioSegment:
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        if data.get("audio_type") is not None:
            data["audio_type"] = AudioType(data["audio_type"])
        if data.get("line_type") is not None:
            try:
                data["line_type"] = LineType(data["line_type"])
            except ValueError:
                data["line_type"] = LineType.DIALOGUE
        return cls(**data)


@dataclass
class EpisodeAudioManifest:
    """Audio manifest describing all audio segments for an episode."""

    episode_id: str = ""
    segments: list[AudioSegment] = field(default_factory=list)
    total_duration: float = 0.0
    mixed_audio_path: str | None = None
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "segments": [s.to_dict() for s in self.segments],
            "total_duration": self.total_duration,
            "mixed_audio_path": self.mixed_audio_path,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeAudioManifest:
        data = dict(data)
        data["segments"] = [
            AudioSegment.from_dict(s) for s in data.get("segments", [])
        ]
        return cls(**data)


# MiniMax speech-02-hd voice profiles keyed by Character.voice_style
VOICE_PROFILES: dict[str, VoiceProfile] = {
    "warm": VoiceProfile(voice_id="Friendly_Person", speed=0.95, emotion="happy"),
    "authoritative": VoiceProfile(voice_id="Imposing_Manner", speed=0.90, pitch=-2),
    "playful": VoiceProfile(voice_id="Lively_Girl", speed=1.10, pitch=2, emotion="happy"),
    "dramatic": VoiceProfile(voice_id="Determined_Man", speed=0.90, pitch=-1),
    "calm": VoiceProfile(voice_id="Calm_Woman", speed=0.90),
    # Period / xianxia drama character voice styles
    "ethereal": VoiceProfile(
        voice_id="Calm_Woman", speed=0.90, pitch=3, emotion="neutral",
        age_feel="young_adult", energy="low", description="飘逸仙气",
    ),
    "commanding": VoiceProfile(
        voice_id="Imposing_Manner", speed=0.85, pitch=-3, emotion="neutral",
        age_feel="middle_aged", energy="high", description="威严帝王",
    ),
    "scheming": VoiceProfile(
        voice_id="Determined_Man", speed=0.95, pitch=-1, emotion="neutral",
        age_feel="middle_aged", energy="medium", description="阴险谋士",
    ),
    "innocent": VoiceProfile(
        voice_id="Lively_Girl", speed=1.05, pitch=3, emotion="happy",
        age_feel="young_adult", energy="medium", description="天真少女",
    ),
    "weathered": VoiceProfile(
        voice_id="Calm_Woman", speed=0.85, pitch=-2, emotion="sad",
        age_feel="middle_aged", energy="low", description="沧桑老者",
    ),
}


# Genre-aware narrator voice presets
NARRATOR_PRESETS: dict[DramaGenre, VoiceProfile] = {
    DramaGenre.SWEET_ROMANCE: VoiceProfile(
        voice_id="Friendly_Person", speed=1.0, pitch=1, emotion="happy",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="medium", description="warm female",
    ),
    DramaGenre.MALE_POWER_FANTASY: VoiceProfile(
        voice_id="Imposing_Manner", speed=0.95, pitch=-2, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="middle_aged", energy="medium", description="mature male",
    ),
    DramaGenre.SUSPENSE_THRILLER: VoiceProfile(
        voice_id="Determined_Man", speed=0.9, pitch=-3, emotion="fearful",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="middle_aged", energy="low", description="deep mysterious",
    ),
    DramaGenre.ANCIENT_XIANXIA: VoiceProfile(
        voice_id="Calm_Woman", speed=0.95, pitch=2, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="low", description="ethereal female",
    ),
    DramaGenre.COMEDY: VoiceProfile(
        voice_id="Lively_Girl", speed=1.1, pitch=1, emotion="happy",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="high", description="energetic",
    ),
    DramaGenre.FAMILY_DRAMA: VoiceProfile(
        voice_id="Calm_Woman", speed=0.95, pitch=0, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="middle_aged", energy="medium", description="warm mature",
    ),
    DramaGenre.OTHER: VoiceProfile(
        voice_id="Friendly_Person", speed=1.0, pitch=0, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="medium", description="default",
    ),
    # Western genre narrator presets (placeholders — overridden by English locale)
    DramaGenre.ROMANCE: VoiceProfile(
        voice_id="Friendly_Person", speed=1.0, pitch=1, emotion="happy",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="medium", description="warm romantic",
    ),
    DramaGenre.ACTION_THRILLER: VoiceProfile(
        voice_id="Determined_Man", speed=0.95, pitch=-2, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="middle_aged", energy="high", description="intense action",
    ),
    DramaGenre.MYSTERY: VoiceProfile(
        voice_id="Calm_Woman", speed=0.9, pitch=-1, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="middle_aged", energy="low", description="mysterious calm",
    ),
    DramaGenre.SUPERNATURAL: VoiceProfile(
        voice_id="Calm_Woman", speed=0.9, pitch=2, emotion="fearful",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="low", description="eerie ethereal",
    ),
    DramaGenre.DRAMA: VoiceProfile(
        voice_id="Friendly_Person", speed=0.95, pitch=0, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="middle_aged", energy="medium", description="neutral dramatic",
    ),
    DramaGenre.SCI_FI: VoiceProfile(
        voice_id="Determined_Man", speed=0.95, pitch=-1, emotion="neutral",
        role_name="narrator", line_type=LineType.NARRATION,
        age_feel="young_adult", energy="medium", description="futuristic",
    ),
}


# Genre-to-archetype voice style recommendations
GENRE_VOICE_RECOMMENDATIONS: dict[DramaGenre, dict[str, str]] = {
    DramaGenre.ANCIENT_XIANXIA: {
        "hero": "ethereal",
        "villain": "scheming",
        "mentor": "weathered",
        "love_interest": "innocent",
        "ruler": "commanding",
        "default": "dramatic",
    },
    DramaGenre.SWEET_ROMANCE: {
        "hero": "warm",
        "villain": "dramatic",
        "love_interest": "playful",
        "default": "warm",
    },
    DramaGenre.SUSPENSE_THRILLER: {
        "hero": "calm",
        "villain": "scheming",
        "detective": "authoritative",
        "default": "dramatic",
    },
    DramaGenre.MALE_POWER_FANTASY: {
        "hero": "commanding",
        "villain": "scheming",
        "love_interest": "playful",
        "default": "authoritative",
    },
    DramaGenre.COMEDY: {
        "hero": "playful",
        "villain": "dramatic",
        "sidekick": "innocent",
        "default": "playful",
    },
    DramaGenre.FAMILY_DRAMA: {
        "hero": "warm",
        "villain": "commanding",
        "elder": "weathered",
        "default": "calm",
    },
    DramaGenre.OTHER: {
        "hero": "warm",
        "villain": "dramatic",
        "default": "warm",
    },
    # Western genre voice recommendations (placeholders — overridden by English locale)
    DramaGenre.ROMANCE: {
        "hero": "warm",
        "villain": "dramatic",
        "love_interest": "playful",
        "default": "warm",
    },
    DramaGenre.ACTION_THRILLER: {
        "hero": "commanding",
        "villain": "scheming",
        "sidekick": "playful",
        "default": "authoritative",
    },
    DramaGenre.MYSTERY: {
        "hero": "calm",
        "villain": "scheming",
        "detective": "authoritative",
        "default": "calm",
    },
    DramaGenre.SUPERNATURAL: {
        "hero": "calm",
        "villain": "scheming",
        "medium": "weathered",
        "default": "dramatic",
    },
    DramaGenre.DRAMA: {
        "hero": "warm",
        "villain": "commanding",
        "default": "warm",
    },
    DramaGenre.SCI_FI: {
        "hero": "calm",
        "villain": "commanding",
        "scientist": "authoritative",
        "default": "authoritative",
    },
}


def recommend_voice_style(genre: DramaGenre | str, archetype: str = "default") -> str:
    """Recommend a voice style for a character archetype within a genre.

    Returns a key from VOICE_PROFILES.
    Falls back to genre default, then to "warm".
    """
    # Resolve string genre to DramaGenre enum
    if isinstance(genre, str):
        try:
            genre = DramaGenre(genre)
        except ValueError:
            return "warm"

    mapping = GENRE_VOICE_RECOMMENDATIONS.get(genre)
    if mapping is None:
        return "warm"

    return mapping.get(archetype, mapping.get("default", "warm"))


def assign_voice_profile(character: Character, language: str = "zh") -> Character:
    """Auto-assign a VoiceProfile based on character.voice_style.

    Skips characters that already have a voice_profile set.
    Falls back to 'warm' profile if voice_style is unknown.

    When *language* is not ``"zh"``, voice profiles are looked up from the
    corresponding locale registry instead of the module-level VOICE_PROFILES
    dict.
    """
    if character.voice_profile is not None:
        return character

    if language != "zh":
        from videoclaw.drama.locale import get_locale  # local import to avoid circular
        locale = get_locale(language)
        profiles = locale.voice_profiles if locale.voice_profiles else VOICE_PROFILES
    else:
        profiles = VOICE_PROFILES

    template = profiles.get(character.voice_style, profiles.get("warm", VOICE_PROFILES["warm"]))
    character.voice_profile = VoiceProfile(
        voice_id=template.voice_id,
        speed=template.speed,
        pitch=template.pitch,
        emotion=template.emotion,
        volume=template.volume,
    )
    return character


@dataclass
class Episode:
    """A single episode in a drama series."""

    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    number: int = 1
    title: str = ""
    synopsis: str = ""
    opening_hook: str = ""
    status: EpisodeStatus = EpisodeStatus.PENDING
    project_id: str | None = None
    duration_seconds: float = 60.0
    script: str | None = None
    scenes: list[DramaScene] = field(default_factory=list)
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["scenes"] = [s.to_dict() for s in self.scenes]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        data = dict(data)
        data["status"] = EpisodeStatus(data.get("status", "pending"))
        data["scenes"] = [DramaScene.from_dict(s) for s in data.get("scenes", [])]
        return cls(**data)


@dataclass
class ScriptModification:
    """A proposed modification to a locked script that requires user approval.

    When ``script_locked`` is ``True`` on a series, any detected gap or
    inconsistency is wrapped in a ScriptModification and presented to the
    user for explicit confirmation before being applied.
    """

    modification_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    scene_id: str = ""
    field_name: str = ""
    reason: str = ""
    original_value: str = ""
    proposed_value: str = ""
    approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScriptModification:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ConsistencyManifest:
    """Pre-generation consistency constraints for multi-clip coherence.

    Built once before video generation starts and injected into every
    Seedance 2.0 generation call to prevent character/scene "崩坏".
    """

    character_visuals: dict[str, str] = field(default_factory=dict)
    """name → frozen visual_prompt (identical across all clips)."""
    character_references: dict[str, str] = field(default_factory=dict)
    """name → primary reference image path (verified to exist)."""
    character_multi_references: dict[str, list[str]] = field(default_factory=dict)
    """name → [front, three_quarter, full_body] reference image paths."""
    scene_settings: dict[str, str] = field(default_factory=dict)
    """scene_id → frozen setting description for location continuity."""
    style_anchor: str = ""
    """Frozen style prompt appended to all generations."""
    verified: bool = False
    """True after all reference paths are validated to exist on disk."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsistencyManifest:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def verify_references(self) -> list[str]:
        """Check that all reference image paths exist on disk.

        Returns a list of missing paths. Sets ``verified = True`` when
        all paths are valid.
        """
        missing: list[str] = []
        for name, path in self.character_references.items():
            if not Path(path).exists():
                missing.append(f"{name}: {path}")
        for name, paths in self.character_multi_references.items():
            for p in paths:
                if not Path(p).exists():
                    missing.append(f"{name}: {p}")
        self.verified = len(missing) == 0
        return missing


@dataclass
class DramaSeries:
    """A complete short drama series with episodes and characters."""

    series_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    title: str = ""
    genre: str = ""
    synopsis: str = ""
    style: str = "cinematic"
    language: str = "zh"
    aspect_ratio: str = "9:16"
    target_episode_duration: float = 60.0  # Maximum seconds per episode (not fixed)
    total_episodes: int = 5
    status: DramaStatus = DramaStatus.DRAFT
    characters: list[Character] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    model_id: str = "seedance-2.0"
    metadata: dict[str, Any] = field(default_factory=dict)
    # --- Script lock & import fields ---
    script_locked: bool = False
    """When True, the script content is frozen — no creative modifications allowed."""
    script_source: str = "generated"
    """Origin of the script: 'generated' (LLM-created) or 'imported' (user-provided complete script)."""
    consistency_manifest: ConsistencyManifest | None = None
    """Pre-built consistency constraints for multi-clip coherence."""
    pending_modifications: list[ScriptModification] = field(default_factory=list)
    """Proposed modifications awaiting user approval (only used when script_locked=True)."""

    def touch(self) -> None:
        self.updated_at = _now_iso()

    @property
    def cost_total(self) -> float:
        return sum(ep.cost for ep in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "series_id": self.series_id,
            "title": self.title,
            "genre": self.genre,
            "synopsis": self.synopsis,
            "style": self.style,
            "language": self.language,
            "aspect_ratio": self.aspect_ratio,
            "target_episode_duration": self.target_episode_duration,
            "total_episodes": self.total_episodes,
            "status": self.status.value,
            "characters": [c.to_dict() for c in self.characters],
            "episodes": [e.to_dict() for e in self.episodes],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model_id": self.model_id,
            "metadata": self.metadata,
            "script_locked": self.script_locked,
            "script_source": self.script_source,
        }
        if self.consistency_manifest is not None:
            d["consistency_manifest"] = self.consistency_manifest.to_dict()
        if self.pending_modifications:
            d["pending_modifications"] = [m.to_dict() for m in self.pending_modifications]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DramaSeries:
        data = dict(data)
        data["status"] = DramaStatus(data.get("status", "draft"))
        data["characters"] = [Character.from_dict(c) for c in data.get("characters", [])]
        data["episodes"] = [Episode.from_dict(e) for e in data.get("episodes", [])]
        cm = data.pop("consistency_manifest", None)
        pm = data.pop("pending_modifications", None)
        # Filter to known fields
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        series = cls(**data)
        if cm is not None:
            series.consistency_manifest = ConsistencyManifest.from_dict(cm)
        if pm is not None:
            series.pending_modifications = [ScriptModification.from_dict(m) for m in pm]
        return series


# ---------------------------------------------------------------------------
# Drama state manager
# ---------------------------------------------------------------------------

class DramaManager:
    """Persists DramaSeries as JSON files on disk.

    Layout::

        {projects_dir}/dramas/{series_id}/series.json
        {projects_dir}/dramas/{series_id}/episodes/{episode_id}/
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = (base_dir or get_config().projects_dir) / "dramas"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _series_path(self, series_id: str) -> Path:
        return self.base_dir / series_id / "series.json"

    def create(self, **kwargs: Any) -> DramaSeries:
        series = DramaSeries(**kwargs)
        self.save(series)
        return series

    def save(self, series: DramaSeries) -> Path:
        series.touch()
        path = self._series_path(series.series_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(series.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, series_id: str) -> DramaSeries:
        path = self._series_path(series_id)
        if not path.exists():
            raise FileNotFoundError(f"Drama series {series_id!r} not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        return DramaSeries.from_dict(data)

    def list_series(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return [
            p.name
            for p in self.base_dir.iterdir()
            if p.is_dir() and (p / "series.json").exists()
        ]

    def delete(self, series_id: str) -> None:
        import shutil

        series_dir = self.base_dir / series_id
        if series_dir.exists():
            shutil.rmtree(series_dir)
