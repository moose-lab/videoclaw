"""Drama episode DAG builder -- converts DramaSeries episodes into executable DAGs.

The :func:`build_episode_dag` function takes a :class:`DramaSeries` and a target
episode, converts each :class:`DramaScene` into a :class:`Shot`, injects character
reference images, and produces a DAG that the :class:`DAGExecutor` can run.
"""

from __future__ import annotations

import logging
import uuid

from videoclaw.core.planner import DAG, TaskNode, TaskType
from videoclaw.core.state import ProjectState, Shot
from videoclaw.drama.models import DramaEpisode, DramaSeries

logger = logging.getLogger(__name__)


def build_episode_dag(
    series: DramaSeries,
    episode: DramaEpisode,
    state: ProjectState,
) -> DAG:
    """Build a complete DAG for a single drama episode.

    This is the main entry point for drama production.  It:

    1. Converts each :class:`DramaScene` into a :class:`Shot`, injecting
       character reference image paths from the series' character roster.
    2. Populates ``state.storyboard`` with the resulting shots.
    3. Calls :func:`_build_drama_dag` to wire up the execution graph.

    Parameters
    ----------
    series:
        The drama series (provides characters and aspect_ratio).
    episode:
        The target episode whose scenes will be converted to shots.
    state:
        The project state to populate with the storyboard.

    Returns
    -------
    DAG
        A ready-to-execute DAG with VIDEO_GEN, TTS, COMPOSE, and RENDER nodes.
    """
    # Build character reference image lookup table
    char_ref_map: dict[str, str] = {
        c.name: c.reference_image
        for c in series.characters
        if c.reference_image
    }

    # Convert DramaScene → Shot with reference images injected
    shots: list[Shot] = []
    for idx, scene in enumerate(episode.scenes):
        ref_images = {
            name: char_ref_map[name]
            for name in scene.characters_present
            if name in char_ref_map
        }
        shot = Shot(
            shot_id=scene.scene_id or uuid.uuid4().hex[:12],
            description=scene.description,
            prompt=scene.prompt,
            duration_seconds=scene.duration_seconds,
            model_id=scene.model_id or "mock",
            reference_images=ref_images,
        )
        shots.append(shot)

    state.storyboard = shots
    logger.info(
        "Built storyboard with %d shots for episode %r",
        len(shots),
        episode.episode_id,
    )

    return _build_drama_dag(state, series, episode)


def _build_drama_dag(
    state: ProjectState,
    series: DramaSeries,
    episode: DramaEpisode,
) -> DAG:
    """Wire up the execution DAG for a drama episode.

    Unlike the generic :func:`build_dag` in :mod:`videoclaw.core.planner`,
    this variant injects ``reference_images`` and ``speaking_character`` into
    each VIDEO_GEN node's params so that the executor can load reference files
    without understanding the character system.
    """
    dag = DAG()

    # Storyboard node (already populated, acts as a sync point)
    dag.add_node(TaskNode(
        node_id="storyboard",
        task_type=TaskType.STORYBOARD,
        params={"episode_id": episode.episode_id},
    ))

    # Parallel video generation per shot
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

    # TTS
    dag.add_node(TaskNode(
        node_id="tts",
        task_type=TaskType.TTS,
        depends_on=["storyboard"],
        params={},
    ))

    # Music
    dag.add_node(TaskNode(
        node_id="music",
        task_type=TaskType.MUSIC,
        depends_on=["storyboard"],
        params={},
    ))

    # Compose -- waits for all video clips + audio
    compose_deps = video_node_ids + ["tts", "music"]
    dag.add_node(TaskNode(
        node_id="compose",
        task_type=TaskType.COMPOSE,
        depends_on=compose_deps,
        params={},
    ))

    # Final render
    dag.add_node(TaskNode(
        node_id="render",
        task_type=TaskType.RENDER,
        depends_on=["compose"],
        params={},
    ))

    return dag
