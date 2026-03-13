"""Drama runner — executes episode pipelines using the existing VideoClaw engine.

Converts each episode's script into a ClawFlow-compatible DAG and runs it
through the standard DAGExecutor pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from videoclaw.config import get_config
from videoclaw.core.events import event_bus
from videoclaw.core.executor import DAGExecutor
from videoclaw.core.planner import DAG, TaskNode, TaskType
from videoclaw.core.state import ProjectState, Shot, ShotStatus, StateManager
from videoclaw.drama.models import DramaManager, DramaSeries, DramaStatus, Episode, EpisodeStatus

logger = logging.getLogger(__name__)


def build_episode_dag(episode: Episode, series: DramaSeries) -> tuple[DAG, ProjectState]:
    """Convert an episode's scene prompts into a DAG + ProjectState.

    Returns a (dag, project_state) tuple ready for DAGExecutor.

    Unlike the generic ``build_dag()``, this builds a drama-specific DAG
    with richer node params so handlers can access scene dialogue,
    character voices, and subtitle data.
    """
    # Build character reference image lookup table
    char_ref_map: dict[str, str] = {
        c.name: c.reference_image
        for c in series.characters
        if c.reference_image
    }

    # Build shots from typed DramaScene objects with reference images injected
    shots: list[Shot] = []
    for idx, scene in enumerate(episode.scenes):
        ref_images = {
            name: char_ref_map[name]
            for name in scene.characters_present
            if name in char_ref_map
        }
        shots.append(Shot(
            shot_id=scene.scene_id or f"ep{episode.number:02d}_s{idx+1:02d}",
            description=scene.description,
            prompt=scene.visual_prompt,
            duration_seconds=scene.duration_seconds,
            model_id=series.model_id,
            status=ShotStatus.PENDING,
            reference_images=ref_images,
        ))

    # Create project state for this episode
    state = ProjectState(
        prompt=f"[{series.title}] Episode {episode.number}: {episode.title}",
        script=episode.script,
        storyboard=shots,
        metadata={
            "series_id": series.series_id,
            "episode_id": episode.episode_id,
            "episode_number": episode.number,
            "style": series.style,
            "aspect_ratio": series.aspect_ratio,
            "language": series.language,
            "voice_map": {
                c.name: c.voice_profile.to_dict()
                for c in series.characters
                if c.voice_profile
            },
        },
    )
    episode.project_id = state.project_id

    # Build drama-specific DAG with enriched params
    dag = _build_drama_dag(state, episode, series)

    return dag, state


def _build_drama_dag(
    state: ProjectState,
    episode: Episode,
    series: DramaSeries,
) -> DAG:
    """Build a drama-specific DAG with enriched handler params.

    Pipeline shape::

        script_gen
            |
        storyboard
            |
        +---+---+---+      tts      music
        | video shots |      |        |
        +---+---+---+      |        |
            |               |        |
            +-------+-------+--------+
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

    # -- 3. Parallel video generation per shot --
    video_node_ids: list[str] = []
    for shot, scene in zip(state.storyboard, episode.scenes):
        vid_id = f"video_{shot.shot_id}"
        dag.add_node(TaskNode(
            node_id=vid_id,
            task_type=TaskType.VIDEO_GEN,
            depends_on=["storyboard"],
            params={
                "shot_id": shot.shot_id,
                "prompt": shot.prompt,
                "duration": shot.duration_seconds,
                "model_id": shot.model_id,
                "aspect_ratio": series.aspect_ratio,
                "reference_images": shot.reference_images,
                "speaking_character": scene.speaking_character,
            },
        ))
        video_node_ids.append(vid_id)

    # -- 4. TTS with per-scene dialogue/narration data --
    # Build voice lookup from series characters
    character_voices: dict[str, dict] = {}
    for char in series.characters:
        if char.voice_profile:
            character_voices[char.name] = char.voice_profile.to_dict()

    scenes_data = []
    for scene in episode.scenes:
        voice = None
        if scene.speaking_character and scene.speaking_character in character_voices:
            voice = character_voices[scene.speaking_character].get("voice_id")
        scenes_data.append({
            "scene_id": scene.scene_id,
            "dialogue": scene.dialogue,
            "dialogue_line_type": getattr(scene, "dialogue_line_type", "dialogue"),
            "narration": scene.narration,
            "speaking_character": scene.speaking_character,
            "emotion": scene.emotion,
            "duration_seconds": scene.duration_seconds,
            "voice": voice,
            "transition": scene.transition,
        })

    tts_node = TaskNode(
        node_id="tts",
        task_type=TaskType.TTS,
        depends_on=["storyboard"],
        params={
            "scenes": scenes_data,
            "language": series.language,
        },
    )
    dag.add_node(tts_node)

    # -- 5. Music (placeholder -- no API yet) --
    music_node = TaskNode(
        node_id="music",
        task_type=TaskType.MUSIC,
        depends_on=["storyboard"],
        params={},
    )
    dag.add_node(music_node)

    # -- 6. Compose: waits for all video clips + audio tracks --
    compose_deps = video_node_ids + ["tts", "music"]
    compose_node = TaskNode(
        node_id="compose",
        task_type=TaskType.COMPOSE,
        depends_on=compose_deps,
        params={
            "transition": "dissolve",
            "scenes": scenes_data,  # needed for subtitle generation
        },
    )
    dag.add_node(compose_node)

    # -- 7. Final render --
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
    """Runs drama episodes through the VideoClaw pipeline sequentially."""

    def __init__(
        self,
        drama_manager: DramaManager | None = None,
        state_manager: StateManager | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self.drama_mgr = drama_manager or DramaManager()
        self.state_mgr = state_manager or StateManager()
        self.max_concurrency = max_concurrency

    async def run_episode(self, series: DramaSeries, episode: Episode) -> ProjectState:
        """Execute a single episode through the full generation pipeline."""
        logger.info("Running episode %d: %r", episode.number, episode.title)
        episode.status = EpisodeStatus.GENERATING

        dag, state = build_episode_dag(episode, series)
        self.state_mgr.save(state)

        executor = DAGExecutor(
            dag=dag,
            state=state,
            state_manager=self.state_mgr,
            bus=event_bus,
            max_concurrency=self.max_concurrency,
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
    ) -> DramaSeries:
        """Run a range of episodes sequentially.

        Episodes are run in order because each may depend on the previous
        episode's cliffhanger for narrative continuity.
        """
        series.status = DramaStatus.GENERATING
        self.drama_mgr.save(series)

        end = end_episode or len(series.episodes)
        episodes_to_run = [
            ep for ep in series.episodes
            if start_episode <= ep.number <= end
        ]

        for episode in episodes_to_run:
            if episode.status == EpisodeStatus.COMPLETED:
                logger.info("Skipping completed episode %d", episode.number)
                continue

            try:
                await self.run_episode(series, episode)
                logger.info("Episode %d completed (cost=$%.4f)", episode.number, episode.cost)
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
