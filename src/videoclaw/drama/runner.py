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
    """
    # Build shots from typed DramaScene objects
    shots: list[Shot] = []
    for idx, scene in enumerate(episode.scenes):
        shots.append(Shot(
            shot_id=scene.scene_id or f"ep{episode.number:02d}_s{idx+1:02d}",
            description=scene.description,
            prompt=scene.visual_prompt,
            duration_seconds=scene.duration_seconds,
            model_id=series.model_id,
            status=ShotStatus.PENDING,
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
        },
    )
    episode.project_id = state.project_id

    # Build the DAG using the standard pipeline builder
    from videoclaw.core.planner import build_dag
    dag = build_dag(state)

    return dag, state


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
