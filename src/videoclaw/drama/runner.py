"""Drama runner — executes episode pipelines using the existing VideoClaw engine.

Converts each episode's script into a ClawFlow-compatible DAG and runs it
through the standard DAGExecutor pipeline.

Includes automatic URL freshness validation for character reference images.
When URLs are expired (HTTP 403/404), the runner auto-refreshes them via
:class:`CharacterDesigner` before proceeding with generation.
"""

from __future__ import annotations

import logging
from typing import Any

from videoclaw.core.events import event_bus
from videoclaw.core.executor import DAGExecutor
from videoclaw.core.planner import DAG, TaskNode, TaskType
from videoclaw.core.state import ProjectState, Shot, ShotStatus, StateManager
from videoclaw.cost.tracker import CostTracker
from videoclaw.drama.models import (
    DramaManager,
    DramaScene,
    DramaSeries,
    DramaStatus,
    Episode,
    EpisodeStatus,
)
from videoclaw.drama.prompt_enhancer import PromptEnhancer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL freshness validation
# ---------------------------------------------------------------------------

async def _check_url_alive(url: str, timeout: float = 10.0) -> bool:
    """Return True if *url* responds with HTTP 2xx to a HEAD request."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.head(url, follow_redirects=True)
            return 200 <= resp.status_code < 400
    except Exception:
        return False


async def ensure_fresh_urls(
    series: DramaSeries,
    drama_manager: DramaManager | None = None,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Validate character reference image URLs; refresh expired ones.

    Performs an HTTP HEAD check on each character's ``reference_image_url``.
    If the URL is missing, empty, or returns non-2xx, re-generates the
    turnaround sheet to obtain a fresh URL.

    Parameters
    ----------
    series:
        The drama series whose characters need URL validation.
    drama_manager:
        If provided, used to persist refreshed URLs back to disk.
    force:
        When ``True``, skip validation and refresh all URLs unconditionally.

    Returns
    -------
    dict[str, str]
        Mapping of character name → current (possibly refreshed) HTTPS URL.
    """
    from videoclaw.drama.character_designer import CharacterDesigner

    chars_needing_refresh: list[str] = []

    if force:
        chars_needing_refresh = [c.name for c in series.characters]
    else:
        for char in series.characters:
            url = char.reference_image_url
            if not url:
                chars_needing_refresh.append(char.name)
                continue
            alive = await _check_url_alive(url)
            if not alive:
                logger.warning(
                    "Character %s URL expired/unreachable: %s...",
                    char.name, url[:60],
                )
                # Clear the stale URL so refresh_urls() will regenerate
                char.reference_image_url = None
                chars_needing_refresh.append(char.name)

    if not chars_needing_refresh:
        logger.info("All character URLs are fresh — no refresh needed")
        return {c.name: c.reference_image_url or "" for c in series.characters}

    logger.info(
        "Refreshing URLs for %d character(s): %s",
        len(chars_needing_refresh),
        ", ".join(chars_needing_refresh),
    )

    mgr = drama_manager or DramaManager()
    designer = CharacterDesigner(drama_manager=mgr)
    refreshed = await designer.refresh_urls(series, force=False)
    return refreshed


def _ensure_consistency_manifest(series: DramaSeries) -> None:
    """Build or verify the ConsistencyManifest before generation.

    When a manifest already exists, verifies that all reference images still
    exist on disk. When missing, builds one from the current series state.
    Logs warnings for any missing reference images.
    """
    from videoclaw.drama.planner import DramaPlanner

    if series.consistency_manifest is None:
        series.consistency_manifest = DramaPlanner._build_consistency_manifest(series)
        logger.info("Built consistency manifest: %d characters, %d scenes",
                     len(series.consistency_manifest.character_visuals),
                     len(series.consistency_manifest.scene_settings))

    missing = series.consistency_manifest.verify_references()
    if missing:
        logger.warning(
            "Consistency manifest: %d reference images missing — "
            "character consistency may degrade: %s",
            len(missing), missing,
        )
    else:
        logger.info("Consistency manifest verified: all reference images present")


def build_episode_dag(
    episode: Episode,
    series: DramaSeries,
    *,
    max_shots: int | None = None,
) -> tuple[DAG, ProjectState]:
    """Convert an episode's scene prompts into a DAG + ProjectState.

    Returns a (dag, project_state) tuple ready for DAGExecutor.

    Unlike the generic ``build_dag()``, this builds a drama-specific DAG
    with richer node params so handlers can access scene dialogue,
    character voices, and subtitle data.

    When ``series.script_locked`` is True, the enhancer is limited to
    visual prompt optimization — no scene content is modified.

    Parameters
    ----------
    max_shots : int | None
        If set, only the first *max_shots* scenes will have video/tts DAG
        nodes created. Useful for test runs (e.g. ``--max-shots 5``).
    """
    # --- Pre-generation consistency enforcement ---
    _ensure_consistency_manifest(series)

    # Build character reference image lookup tables (single + multi-angle)
    # Prefer consistency manifest (frozen) over live character data
    manifest = series.consistency_manifest
    char_ref_map: dict[str, str] = {}
    char_multi_ref_map: dict[str, list[str]] = {}

    if manifest and manifest.verified:
        char_ref_map = dict(manifest.character_references)
        char_multi_ref_map = {k: list(v) for k, v in manifest.character_multi_references.items()}
    else:
        char_ref_map = {
            c.name: c.reference_image
            for c in series.characters
            if c.reference_image
        }
        char_multi_ref_map = {
            c.name: c.reference_images
            for c in series.characters
            if c.reference_images
        }
    # Build character HTTPS URL map (for Seedance API — avoids base64 rejection)
    # Map both full name ("Ivy Angel") and first name ("Ivy") for flexible matching
    char_url_map: dict[str, str] = {}
    for c in series.characters:
        url = getattr(c, "reference_image_url", None)
        if url:
            char_url_map[c.name] = url
            first_name = c.name.split()[0]
            if first_name != c.name:
                char_url_map[first_name] = url

    # Sync scene_blocks → scenes (inject block-level time_of_day, scene_group, etc.)
    if episode.scene_blocks:
        episode.sync_scenes_from_blocks()
        logger.info("Synced %d scenes from %d scene_blocks",
                     len(episode.scenes), len(episode.scene_blocks))

    # Build scene and prop reference maps from ConsistencyManifest
    scene_ref_map: dict[str, str] = {}
    prop_ref_map: dict[str, str] = {}
    if manifest and manifest.verified:
        scene_ref_map = dict(manifest.scene_references)
        prop_ref_map = dict(manifest.prop_references)

    # Build available_refs for PromptEnhancer ref marker injection
    available_refs: dict[str, dict[str, str]] = {
        "characters": char_url_map if char_url_map else dict(char_ref_map),
        "scenes": scene_ref_map,
        "props": prop_ref_map,
    }

    # Enhance visual prompts before building shots
    # For locked scripts: enhancer only optimizes visual_prompt for Seedance,
    # does NOT modify dialogue, narration, or scene structure.
    enhancer = PromptEnhancer()

    # Experience feedback (经验反哺): inject frequent audit defects as constraints
    from videoclaw.config import get_config
    series_dir = get_config().projects_dir / "dramas" / series.series_id
    audit_log_dir = series_dir / "audit_logs"
    if audit_log_dir.is_dir():
        learned = enhancer.load_audit_constraints(series_dir)
        if learned:
            logger.info(
                "Loaded %d learned constraints from audit logs: %s",
                len(learned), learned,
            )

    enhancer.enhance_all_scenes(episode, series, available_refs=available_refs)

    # Build shots from typed DramaScene objects with reference images injected
    shots: list[Shot] = []
    for idx, scene in enumerate(episode.scenes):
        # Primary reference image per character (front view or single)
        ref_images = {
            name: char_ref_map[name]
            for name in scene.characters_present
            if name in char_ref_map
        }
        # Multi-angle references for Seedance Universal Reference (全能参考)
        multi_refs = {
            name: char_multi_ref_map[name]
            for name in scene.characters_present
            if name in char_multi_ref_map
        }
        # HTTPS URLs for Seedance API (vectorspace.cn rejects base64)
        ref_urls = {
            name: char_url_map[name]
            for name in scene.characters_present
            if name in char_url_map
        }
        shots.append(Shot(
            shot_id=scene.scene_id or f"ep{episode.number:02d}_s{idx+1:02d}",
            description=scene.description,
            prompt=scene.effective_prompt,
            duration_seconds=scene.duration_seconds,
            model_id=series.model_id,
            status=ShotStatus.PENDING,
            reference_images=ref_images,
            multi_reference_images=multi_refs,
            reference_image_urls=ref_urls,
            scene_reference_urls=scene_ref_map,
            prop_reference_urls=prop_ref_map,
        ))

    # Create project state for this episode
    meta: dict[str, Any] = {
        "series_id": series.series_id,
        "episode_id": episode.episode_id,
        "episode_number": episode.number,
        "style": series.style,
        "aspect_ratio": series.aspect_ratio,
        "language": series.language,
        "script_locked": series.script_locked,
        "script_source": series.script_source,
        "voice_map": {
            c.name: c.voice_profile.to_dict()
            for c in series.characters
            if c.voice_profile
        },
    }
    if manifest and manifest.verified:
        meta["consistency_manifest"] = manifest.to_dict()

    state = ProjectState(
        prompt=f"[{series.title}] Episode {episode.number}: {episode.title}",
        script=episode.script,
        storyboard=shots,
        metadata=meta,
    )
    episode.project_id = state.project_id

    # ---- Quality gate: validate before committing to expensive generation ----
    from videoclaw.drama.quality import DramaQualityValidator

    validator = DramaQualityValidator()
    episode_scripts = {
        episode.number: {
            "scenes": [s.to_dict() for s in episode.scenes],
            "cliffhanger": episode.synopsis,  # best-effort cliffhanger field
        },
    }
    violations = validator.validate(series, episode_scripts)
    if violations:
        logger.warning(
            "Quality validation found %d violations for episode %d: %s",
            len(violations), episode.number, violations,
        )
        # Log but don't block — violations are warnings for now
        # TODO: make configurable (strict mode raises, lenient mode warns)

    # Build drama-specific DAG with enriched params
    dag = _build_drama_dag(state, episode, series, max_shots=max_shots)

    return dag, state


def _build_drama_dag(
    state: ProjectState,
    episode: Episode,
    series: DramaSeries,
    *,
    max_shots: int | None = None,
) -> DAG:
    """Build a drama-specific DAG with per-scene TTS and subtitle nodes.

    Pipeline shape::

        script_gen
            |
        storyboard
            |
        +---+---+---+   [per_scene_tts × N]   music
        | video shots |          |                |
        +---+---+---+          |                |
            |              subtitle_gen           |
            |                    |                |
            +--------+-----------+----------------+
                     |
                  compose
                     |
                   render
    """
    dag = DAG()

    # -- 1. Script generation (already done by DramaPlanner) --
    script_node = TaskNode(
        node_id="script_gen",
        task_type=TaskType.SCRIPT_GEN,
        params={"prompt": state.prompt},
    )
    dag.add_node(script_node)

    # -- 2. Storyboard (already populated from scenes) --
    storyboard_node = TaskNode(
        node_id="storyboard",
        task_type=TaskType.STORYBOARD,
        depends_on=["script_gen"],
        params={"prompt": state.prompt},
    )
    dag.add_node(storyboard_node)

    # -- 2b. Scene validation gate (场景先行 — validate before generation) --
    scenes_for_validate = [s.to_dict() for s in episode.scenes] if episode.scenes else []
    validate_node = TaskNode(
        node_id="scene_validate",
        task_type=TaskType.SCENE_VALIDATE,
        depends_on=["storyboard"],
        params={
            "scenes": scenes_for_validate,
            "language": series.language,
        },
    )
    dag.add_node(validate_node)

    # -- 3. Parallel video generation per shot (depends on scene_validate) --
    # When max_shots is set, only create video/tts nodes for the first N scenes.
    gen_pairs = list(zip(state.storyboard, episode.scenes))
    if max_shots is not None and max_shots > 0:
        gen_pairs = gen_pairs[:max_shots]
        logger.info("max_shots=%d — limiting video/tts generation to first %d scenes",
                     max_shots, len(gen_pairs))
    video_node_ids: list[str] = []
    for shot, scene in gen_pairs:
        vid_id = f"video_{shot.shot_id}"
        dag.add_node(TaskNode(
            node_id=vid_id,
            task_type=TaskType.VIDEO_GEN,
            depends_on=["scene_validate"],
            params={
                "shot_id": shot.shot_id,
                "prompt": shot.prompt,
                "duration": shot.duration_seconds,
                "model_id": shot.model_id,
                "aspect_ratio": series.aspect_ratio,
                "reference_images": shot.reference_images,
                "multi_reference_images": shot.multi_reference_images,
                "reference_image_urls": shot.reference_image_urls,
                "scene_reference_urls": shot.scene_reference_urls,
                "prop_reference_urls": shot.prop_reference_urls,
                "speaking_character": scene.speaking_character,
                "shot_scale": scene.shot_scale.value if scene.shot_scale else None,
            },
        ))
        video_node_ids.append(vid_id)

    # -- 4. Per-scene TTS nodes (parallel, one per scene) --
    character_voices: dict[str, dict] = {}
    for char in series.characters:
        if char.voice_profile:
            character_voices[char.name] = char.voice_profile.to_dict()

    scenes_data: list[dict] = []
    tts_node_ids: list[str] = []
    tts_scenes = [scene for _, scene in gen_pairs]  # same subset as video
    for scene in tts_scenes:
        voice = None
        if scene.speaking_character and scene.speaking_character in character_voices:
            voice = character_voices[scene.speaking_character].get("voice_id")

        scene_dict = {
            "scene_id": scene.scene_id,
            "dialogue": scene.dialogue,
            "dialogue_line_type": getattr(scene, "dialogue_line_type", "dialogue"),
            "narration": scene.narration,
            "narration_type": getattr(scene, "narration_type", "voiceover"),
            "speaking_character": scene.speaking_character,
            "emotion": scene.emotion,
            "duration_seconds": scene.duration_seconds,
            "voice": voice,
            "transition": scene.transition,
        }
        scenes_data.append(scene_dict)

        tts_id = f"tts_{scene.scene_id}"
        dag.add_node(TaskNode(
            node_id=tts_id,
            task_type=TaskType.PER_SCENE_TTS,
            depends_on=["scene_validate"],
            params={
                "scene": scene_dict,
                "language": series.language,
                "voice_map": character_voices,
            },
        ))
        tts_node_ids.append(tts_id)

    # -- 5. Subtitle generation (depends on all per-scene TTS for accurate timing) --
    subtitle_node = TaskNode(
        node_id="subtitle_gen",
        task_type=TaskType.SUBTITLE_GEN,
        depends_on=tts_node_ids,
        params={
            "scenes": scenes_data,
        },
    )
    dag.add_node(subtitle_node)

    # -- 6. Music (placeholder -- no API yet) --
    music_node = TaskNode(
        node_id="music",
        task_type=TaskType.MUSIC,
        depends_on=["scene_validate"],
        params={},
    )
    dag.add_node(music_node)

    # -- 7. Compose: waits for all video clips + subtitle + music --
    compose_deps = [*video_node_ids, "subtitle_gen", "music"]
    compose_node = TaskNode(
        node_id="compose",
        task_type=TaskType.COMPOSE,
        depends_on=compose_deps,
        params={
            "transition": "dissolve",
            "scenes": scenes_data,
        },
    )
    dag.add_node(compose_node)

    # -- 8. Final render --
    render_node = TaskNode(
        node_id="render",
        task_type=TaskType.RENDER,
        depends_on=["compose"],
        params={
            "codec": "libx264",
            "preset": "medium",
            "crf": 23,
            "audio_bitrate": "192k",
        },
    )
    dag.add_node(render_node)

    return dag


def build_scene_regen_dag(
    episode: Episode,
    series: DramaSeries,
    scene_id: str,
    state: ProjectState,
    recompose: bool = False,
) -> DAG:
    """Build a mini-DAG to regenerate a single scene's assets.

    The mini-DAG contains only the target scene's ``video_gen`` and
    ``per_scene_tts`` nodes (with no dependencies so they run immediately).
    When *recompose* is True, ``subtitle_gen``, ``compose``, and ``render``
    nodes are appended so the full episode is re-assembled.

    Raises:
        ValueError: if *scene_id* is not found in the episode's scenes.
    """
    # Find the target scene
    scene_idx: int | None = None
    target_scene: DramaScene | None = None
    for idx, scene in enumerate(episode.scenes):
        if scene.scene_id == scene_id:
            scene_idx = idx
            target_scene = scene
            break

    if target_scene is None or scene_idx is None:
        raise ValueError(
            f"Scene {scene_id!r} not found in episode {episode.number} "
            f"(available: {[s.scene_id for s in episode.scenes]})"
        )

    # Build character reference image lookup (single + multi-angle + HTTPS URLs)
    char_ref_map: dict[str, str] = {
        c.name: c.reference_image
        for c in series.characters
        if c.reference_image
    }
    char_multi_ref_map: dict[str, list[str]] = {
        c.name: c.reference_images
        for c in series.characters
        if c.reference_images
    }
    char_url_map: dict[str, str] = {}
    for c in series.characters:
        url = getattr(c, "reference_image_url", None)
        if url:
            char_url_map[c.name] = url
            first_name = c.name.split()[0]
            if first_name != c.name:
                char_url_map[first_name] = url

    # Build scene and prop reference maps from ConsistencyManifest
    regen_scene_ref_map: dict[str, str] = {}
    regen_prop_ref_map: dict[str, str] = {}
    manifest = series.consistency_manifest
    if manifest and manifest.verified:
        regen_scene_ref_map = dict(manifest.scene_references)
        regen_prop_ref_map = dict(manifest.prop_references)

    # Build character voice lookup
    character_voices: dict[str, dict] = {}
    for char in series.characters:
        if char.voice_profile:
            character_voices[char.name] = char.voice_profile.to_dict()

    # Find the corresponding shot in the storyboard
    shot = next((s for s in state.storyboard if s.shot_id == scene_id), None)

    # Enhance the target scene's visual prompt (with available refs for [ref:key] markers)
    regen_available_refs: dict[str, dict[str, str]] = {
        "characters": char_url_map if char_url_map else dict(char_ref_map),
        "scenes": regen_scene_ref_map,
        "props": regen_prop_ref_map,
    }
    enhancer = PromptEnhancer()
    enhanced = enhancer.enhance_scene_prompt(
        target_scene, series, available_refs=regen_available_refs,
    )
    target_scene.enhanced_visual_prompt = enhanced

    dag = DAG()

    # -- video_gen node (no dependencies) --
    ref_images = {
        name: char_ref_map[name]
        for name in target_scene.characters_present
        if name in char_ref_map
    }
    # Also include speaking character's reference
    if target_scene.speaking_character and target_scene.speaking_character in char_ref_map:
        ref_images[target_scene.speaking_character] = char_ref_map[target_scene.speaking_character]
    # Multi-angle references for Universal Reference (全能参考)
    multi_refs = {
        name: char_multi_ref_map[name]
        for name in target_scene.characters_present
        if name in char_multi_ref_map
    }
    # HTTPS URLs for Seedance API (vectorspace.cn rejects base64)
    ref_urls = {
        name: char_url_map[name]
        for name in target_scene.characters_present
        if name in char_url_map
    }

    vid_id = f"video_{scene_id}"
    dag.add_node(TaskNode(
        node_id=vid_id,
        task_type=TaskType.VIDEO_GEN,
        depends_on=[],
        params={
            "shot_id": scene_id,
            "prompt": target_scene.effective_prompt,
            "duration": target_scene.duration_seconds,
            "model_id": shot.model_id if shot else series.model_id,
            "aspect_ratio": series.aspect_ratio,
            "reference_images": ref_images,
            "multi_reference_images": multi_refs,
            "reference_image_urls": ref_urls,
            "scene_reference_urls": regen_scene_ref_map,
            "prop_reference_urls": regen_prop_ref_map,
            "speaking_character": target_scene.speaking_character,
            "shot_scale": target_scene.shot_scale.value if target_scene.shot_scale else None,
        },
    ))

    # -- per_scene_tts node (no dependencies) --
    voice = None
    if target_scene.speaking_character and target_scene.speaking_character in character_voices:
        voice = character_voices[target_scene.speaking_character].get("voice_id")

    scene_dict = {
        "scene_id": target_scene.scene_id,
        "dialogue": target_scene.dialogue,
        "dialogue_line_type": getattr(target_scene, "dialogue_line_type", "dialogue"),
        "narration": target_scene.narration,
        "narration_type": getattr(target_scene, "narration_type", "voiceover"),
        "speaking_character": target_scene.speaking_character,
        "emotion": target_scene.emotion,
        "duration_seconds": target_scene.duration_seconds,
        "voice": voice,
        "transition": target_scene.transition,
    }

    tts_id = f"tts_{scene_id}"
    dag.add_node(TaskNode(
        node_id=tts_id,
        task_type=TaskType.PER_SCENE_TTS,
        depends_on=[],
        params={
            "scene": scene_dict,
            "language": series.language,
            "voice_map": character_voices,
        },
    ))

    # -- Optional recompose pipeline --
    if recompose:
        # Build scenes_data for all scenes (subtitle_gen and compose need full context)
        scenes_data: list[dict] = []
        for sc in episode.scenes:
            sc_voice = None
            if sc.speaking_character and sc.speaking_character in character_voices:
                sc_voice = character_voices[sc.speaking_character].get("voice_id")
            scenes_data.append({
                "scene_id": sc.scene_id,
                "dialogue": sc.dialogue,
                "dialogue_line_type": getattr(sc, "dialogue_line_type", "dialogue"),
                "narration": sc.narration,
                "speaking_character": sc.speaking_character,
                "emotion": sc.emotion,
                "duration_seconds": sc.duration_seconds,
                "voice": sc_voice,
                "transition": sc.transition,
            })

        subtitle_node = TaskNode(
            node_id="subtitle_gen",
            task_type=TaskType.SUBTITLE_GEN,
            depends_on=[tts_id],
            params={"scenes": scenes_data},
        )
        dag.add_node(subtitle_node)

        compose_deps = [vid_id, "subtitle_gen"]
        compose_node = TaskNode(
            node_id="compose",
            task_type=TaskType.COMPOSE,
            depends_on=compose_deps,
            params={
                "transition": "dissolve",
                "scenes": scenes_data,
            },
        )
        dag.add_node(compose_node)

        render_node = TaskNode(
            node_id="render",
            task_type=TaskType.RENDER,
            depends_on=["compose"],
            params={
                "codec": "libx264",
                "preset": "medium",
                "crf": 23,
                "audio_bitrate": "192k",
            },
        )
        dag.add_node(render_node)

    return dag


class DramaRunner:
    """Runs drama episodes through the VideoClaw pipeline sequentially.

    Before generation, automatically validates character reference image URLs
    and refreshes expired ones to ensure Seedance 2.0 can access them.
    """

    def __init__(
        self,
        drama_manager: DramaManager | None = None,
        state_manager: StateManager | None = None,
        max_concurrency: int = 4,
        auto_refresh_urls: bool = True,
        budget_usd: float | None = None,
    ) -> None:
        self.drama_mgr = drama_manager or DramaManager()
        self.state_mgr = state_manager or StateManager()
        self.max_concurrency = max_concurrency
        self.auto_refresh_urls = auto_refresh_urls
        self.budget_usd = budget_usd

    async def run_episode(
        self,
        series: DramaSeries,
        episode: Episode,
        *,
        max_shots: int | None = None,
    ) -> ProjectState:
        """Execute a single episode through the full generation pipeline.

        When ``auto_refresh_urls`` is ``True`` (default), validates all
        character reference image URLs before starting generation. Expired
        URLs are automatically refreshed via :func:`ensure_fresh_urls`.

        Parameters
        ----------
        max_shots : int | None
            Limit video/tts generation to the first *max_shots* scenes.
        """
        logger.info("Running episode %d: %r", episode.number, episode.title)
        episode.status = EpisodeStatus.GENERATING

        # --- Pre-generation: validate & refresh character reference URLs ---
        if self.auto_refresh_urls:
            await ensure_fresh_urls(series, drama_manager=self.drama_mgr)

        dag, state = build_episode_dag(episode, series, max_shots=max_shots)
        self.state_mgr.save(state)

        tracker = CostTracker(
            project_id=state.project_id,
            budget_usd=self.budget_usd,
        )

        executor = DAGExecutor(
            dag=dag,
            state=state,
            state_manager=self.state_mgr,
            bus=event_bus,
            max_concurrency=self.max_concurrency,
            cost_tracker=tracker,
        )

        state = await executor.run()

        # Update episode status based on pipeline result
        if state.status.value == "completed":
            episode.status = EpisodeStatus.COMPLETED
            episode.cost = state.cost_total
        else:
            episode.status = EpisodeStatus.FAILED

        self.drama_mgr.save(series)
        return state

    async def run_series(
        self,
        series: DramaSeries,
        start_episode: int = 1,
        end_episode: int | None = None,
        *,
        max_shots: int | None = None,
    ) -> DramaSeries:
        """Run a range of episodes sequentially.

        Episodes are run in order because each may depend on the previous
        episode's cliffhanger for narrative continuity.
        """
        import json as _json

        series.status = DramaStatus.GENERATING
        self.drama_mgr.save(series)

        end = end_episode or len(series.episodes)
        episodes_to_run = [
            ep for ep in series.episodes
            if start_episode <= ep.number <= end
        ]

        # Retrieve cliffhanger from the episode before the first one we run
        prev_cliffhanger: str | None = None
        if episodes_to_run:
            prev_num = episodes_to_run[0].number - 1
            for ep in series.episodes:
                if ep.number == prev_num and ep.script:
                    try:
                        prev_cliffhanger = _json.loads(ep.script).get("cliffhanger")
                    except (TypeError, _json.JSONDecodeError):
                        pass
                    break

        for episode in episodes_to_run:
            if episode.status == EpisodeStatus.COMPLETED:
                logger.info("Skipping completed episode %d", episode.number)
                # Still extract cliffhanger for next episode
                if episode.script:
                    try:
                        prev_cliffhanger = _json.loads(episode.script).get("cliffhanger")
                    except (TypeError, _json.JSONDecodeError):
                        pass
                continue

            # Script the episode if not already scripted (with cliffhanger threading)
            if not episode.scenes:
                if series.script_locked:
                    logger.error(
                        "Episode %d has no scenes but script is locked "
                        "(source=%s). Cannot auto-generate script for "
                        "locked series — import the complete script first.",
                        episode.number, series.script_source,
                    )
                    raise RuntimeError(
                        f"Episode {episode.number} has no scenes and "
                        f"script_locked=True. Use 'claw drama import' to "
                        f"provide the complete script."
                    )

                from videoclaw.drama.planner import DramaPlanner

                planner = DramaPlanner()
                script_data = await planner.script_episode(series, episode, prev_cliffhanger)
                prev_cliffhanger = script_data.get("cliffhanger")
                self.drama_mgr.save(series)

            try:
                await self.run_episode(series, episode, max_shots=max_shots)
                logger.info("Episode %d completed (cost=$%.4f)", episode.number, episode.cost)
                # Extract cliffhanger for next episode
                if episode.script:
                    try:
                        prev_cliffhanger = _json.loads(episode.script).get("cliffhanger")
                    except (TypeError, _json.JSONDecodeError):
                        pass
            except Exception:
                logger.exception("Episode %d failed", episode.number)
                episode.status = EpisodeStatus.FAILED
                series.status = DramaStatus.FAILED
                self.drama_mgr.save(series)
                raise

        # Check if all episodes are done
        if all(ep.status == EpisodeStatus.COMPLETED for ep in series.episodes):
            series.status = DramaStatus.COMPLETED
        elif any(ep.status == EpisodeStatus.FAILED for ep in series.episodes):
            series.status = DramaStatus.FAILED

        self.drama_mgr.save(series)
        logger.info("Series %r finished: status=%s cost=$%.4f",
                     series.title, series.status, series.cost_total)
        return series

    async def regenerate_scene(
        self,
        series: DramaSeries,
        episode: Episode,
        scene_id: str,
        recompose: bool = False,
    ) -> ProjectState:
        """Regenerate a single scene's video and audio assets.

        Resets the target scene's status and asset paths, builds a mini-DAG
        containing only that scene's ``video_gen`` and ``per_scene_tts`` nodes,
        and executes it through the standard :class:`DAGExecutor`.

        When *recompose* is ``True``, the full episode is re-composed and
        re-rendered after the scene assets are regenerated.

        Parameters
        ----------
        series:
            The parent drama series (for character refs, voice profiles, etc.).
        episode:
            The episode containing the scene.
        scene_id:
            The ``scene_id`` of the scene to regenerate.
        recompose:
            If ``True``, run subtitle_gen → compose → render after regen.

        Returns
        -------
        ProjectState
            The updated project state.

        Raises
        ------
        ValueError
            If *scene_id* is not found in the episode's scenes.
        """
        logger.info(
            "Regenerating scene %s in episode %d (recompose=%s)",
            scene_id, episode.number, recompose,
        )

        # 1. Find and reset the target scene
        target_scene: DramaScene | None = None
        for scene in episode.scenes:
            if scene.scene_id == scene_id:
                target_scene = scene
                break
        if target_scene is None:
            raise ValueError(
                f"Scene {scene_id!r} not found in episode {episode.number}"
            )

        target_scene.video_asset_path = None
        target_scene.dialogue_audio_path = None
        target_scene.narration_audio_path = None
        target_scene.scene_status = "pending"

        # 2. Load or create ProjectState
        state: ProjectState
        if episode.project_id:
            try:
                state = self.state_mgr.load(episode.project_id)
            except FileNotFoundError:
                _, state = build_episode_dag(episode, series)
        else:
            _, state = build_episode_dag(episode, series)

        # 3. Reset the corresponding shot in the storyboard
        for shot in state.storyboard:
            if shot.shot_id == scene_id:
                shot.status = ShotStatus.PENDING
                shot.asset_path = None
                break

        self.state_mgr.save(state)

        # 4. Build and execute mini-DAG
        dag = build_scene_regen_dag(episode, series, scene_id, state, recompose)

        tracker = CostTracker(
            project_id=state.project_id,
            budget_usd=self.budget_usd,
        )

        executor = DAGExecutor(
            dag=dag,
            state=state,
            state_manager=self.state_mgr,
            bus=event_bus,
            max_concurrency=self.max_concurrency,
            cost_tracker=tracker,
        )

        state = await executor.run()
        self.drama_mgr.save(series)

        logger.info(
            "Scene %s regeneration %s",
            scene_id,
            "completed" if state.status.value == "completed" else "failed",
        )
        return state
