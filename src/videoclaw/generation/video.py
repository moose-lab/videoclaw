"""Video generation dispatcher -- coordinates shot-level video generation.

The :class:`VideoGenerator` takes :class:`Shot` objects and dispatches them to
the appropriate video-model adapter via the :class:`ModelRouter`.  It supports
concurrent generation with configurable parallelism and emits lifecycle events
for downstream consumers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from videoclaw.core.events import event_bus, SHOT_GENERATED, TASK_STARTED, TASK_COMPLETED
from videoclaw.core.state import Shot, ShotStatus
from videoclaw.models.protocol import GenerationRequest, GenerationResult
from videoclaw.models.router import ModelRouter, RoutingStrategy

logger = logging.getLogger(__name__)


class VideoGenerator:
    """Dispatches shots to video-model adapters via the routing layer.

    Parameters
    ----------
    router:
        A :class:`ModelRouter` instance.  When *None*, a default router is
        created on first use.
    """

    def __init__(self, router: ModelRouter | None = None) -> None:
        self._router = router

    def _ensure_router(self) -> ModelRouter:
        if self._router is None:
            self._router = ModelRouter()
        return self._router

    async def generate_shot(
        self,
        shot: Shot,
        strategy: RoutingStrategy = RoutingStrategy.AUTO,
    ) -> GenerationResult:
        """Generate video for a single shot.

        Parameters
        ----------
        shot:
            The shot to generate, containing prompt and metadata.
        strategy:
            The routing strategy used to select a model adapter.

        Returns
        -------
        GenerationResult
            The result including video data, cost, and metadata.
        """
        router = self._ensure_router()

        logger.info(
            "Generating shot %s (%.1fs) with strategy=%s",
            shot.shot_id,
            shot.duration_seconds,
            strategy.value if hasattr(strategy, "value") else strategy,
        )

        # Build generation request from shot data
        request = GenerationRequest(
            prompt=shot.prompt,
            duration_seconds=shot.duration_seconds,
        )

        # Select the adapter via the router
        adapter = await router.select(
            request=request,
            strategy=strategy,
            preferred_model=shot.model_id if shot.model_id != "auto" else None,
        )

        logger.info(
            "Router selected model %r for shot %s",
            adapter.model_id,
            shot.shot_id,
        )

        # Execute generation
        result = await adapter.generate(request)
        result.model_id = adapter.model_id

        return result

    async def generate_all_shots(
        self,
        shots: list[Shot],
        strategy: RoutingStrategy = RoutingStrategy.AUTO,
        max_concurrency: int = 4,
    ) -> list[GenerationResult]:
        """Generate video for multiple shots with concurrency control.

        Shots are dispatched concurrently up to *max_concurrency* at a time.
        Events are emitted as each shot completes or fails.

        Parameters
        ----------
        shots:
            The list of shots to generate.
        strategy:
            The routing strategy applied to all shots.
        max_concurrency:
            Maximum number of shots generated in parallel.

        Returns
        -------
        list[GenerationResult]
            Results in the same order as the input *shots* list.  Failed shots
            produce a :class:`GenerationResult` with empty ``video_data`` and
            an ``"error"`` key in ``metadata``.
        """
        if not shots:
            return []

        await event_bus.emit(
            TASK_STARTED,
            {"task": "video.generate_all", "shot_count": len(shots)},
        )
        logger.info(
            "Generating %d shots with max_concurrency=%d",
            len(shots),
            max_concurrency,
        )

        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[GenerationResult | None] = [None] * len(shots)

        async def _generate_one(index: int, shot: Shot) -> None:
            async with semaphore:
                try:
                    result = await self.generate_shot(shot, strategy)
                    results[index] = result

                    await event_bus.emit(
                        SHOT_GENERATED,
                        {
                            "shot_id": shot.shot_id,
                            "model_id": result.model_id,
                            "cost_usd": result.cost_usd,
                            "duration": result.duration_seconds,
                        },
                    )
                    logger.info(
                        "Shot %s generated successfully (model=%s cost=$%.4f)",
                        shot.shot_id,
                        result.model_id,
                        result.cost_usd,
                    )
                except Exception as exc:
                    logger.error(
                        "Shot %s generation failed: %s",
                        shot.shot_id,
                        exc,
                        exc_info=True,
                    )
                    # Produce a sentinel result so callers can detect failures
                    results[index] = GenerationResult(
                        video_data=b"",
                        duration_seconds=0.0,
                        metadata={"error": str(exc), "shot_id": shot.shot_id},
                    )

        tasks = [
            asyncio.create_task(_generate_one(i, shot))
            for i, shot in enumerate(shots)
        ]
        await asyncio.gather(*tasks)

        await event_bus.emit(
            TASK_COMPLETED,
            {
                "task": "video.generate_all",
                "shot_count": len(shots),
                "success_count": sum(
                    1 for r in results if r and r.video_data
                ),
            },
        )

        # At this point every slot is filled (either real result or sentinel)
        return results  # type: ignore[return-value]
