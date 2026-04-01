"""Smart model router — selects the best adapter for a given request.

The router scores every *capable and healthy* adapter against the chosen
:class:`RoutingStrategy` and returns the winner.  Callers that already know
which model they want can pass ``preferred_model`` to short-circuit scoring.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

from videoclaw.models.protocol import (
    GenerationRequest,
    ModelCapability,
    VideoModelAdapter,
)
from videoclaw.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class RoutingStrategy(enum.Enum):
    """High-level routing intent."""

    QUALITY_FIRST = "quality_first"
    COST_FIRST = "cost_first"
    SPEED_FIRST = "speed_first"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelScore:
    """Composite score used to rank candidate adapters."""

    model_id: str
    quality_score: float  # 0.0 .. 1.0
    speed_score: float  # 0.0 .. 1.0
    cost_score: float  # 0.0 .. 1.0 (higher = cheaper)
    available: bool
    weighted_score: float = 0.0


# Hardcoded profiles until we collect runtime telemetry.
# Keys are model_id strings; values carry rough quality / speed / cost
# characteristics on a 0-1 normalised scale (higher is better).
# ``price_usd_per_sec`` is the authoritative USD-per-second-of-video rate
# used by cost tracking, adapters, and optimisation hints.
MODEL_PROFILES: dict[str, dict[str, float]] = {
    "sora": {"quality": 0.95, "speed": 0.5, "cost": 0.3, "price_usd_per_sec": 0.10},
    "runway-gen4": {"quality": 0.90, "speed": 0.6, "cost": 0.4, "price_usd_per_sec": 0.08},
    "kling-1.6": {"quality": 0.85, "speed": 0.65, "cost": 0.5, "price_usd_per_sec": 0.03},
    "pika-2.2": {"quality": 0.80, "speed": 0.7, "cost": 0.6, "price_usd_per_sec": 0.04},
    # MiniMax models (海螺AI) - free tier available
    "minimax-hailuo-2.3": {"quality": 0.82, "speed": 0.75, "cost": 0.85, "price_usd_per_sec": 0.02},
    "minimax-hailuo-2.3-fast": {
        "quality": 0.78, "speed": 0.85, "cost": 0.85, "price_usd_per_sec": 0.02,
    },
    "minimax-hailuo-02": {"quality": 0.75, "speed": 0.75, "cost": 0.85, "price_usd_per_sec": 0.02},
    "minimax-s2v-01": {"quality": 0.80, "speed": 0.70, "cost": 0.85, "price_usd_per_sec": 0.02},
    # ZhipuAI models (智谱清影/CogVideoX) - free tier available
    "cogvideox-flash": {"quality": 0.70, "speed": 0.85, "cost": 0.95, "price_usd_per_sec": 0.015},
    "cogvideox": {"quality": 0.75, "speed": 0.60, "cost": 0.95, "price_usd_per_sec": 0.015},
    # ByteDance Seedance (豆包) via Volcengine Ark
    "seedance-2.0": {"quality": 0.92, "speed": 0.65, "cost": 0.80, "price_usd_per_sec": 0.05},
    "seedance-1.5-pro": {"quality": 0.88, "speed": 0.60, "cost": 0.75, "price_usd_per_sec": 0.04},
    "seedance-1.0": {"quality": 0.85, "speed": 0.70, "cost": 0.85, "price_usd_per_sec": 0.03},
    "mock": {"quality": 0.10, "speed": 1.0, "cost": 1.0, "price_usd_per_sec": 0.0},
}

_DEFAULT_PROFILE: dict[str, float] = {
    "quality": 0.5, "speed": 0.5, "cost": 0.5, "price_usd_per_sec": 0.02,
}


def get_price_usd_per_sec(model_id: str) -> float:
    """Return the authoritative USD-per-second pricing for *model_id*."""
    profile = MODEL_PROFILES.get(model_id, _DEFAULT_PROFILE)
    return profile.get("price_usd_per_sec", 0.02)

# Strategy -> (quality_weight, speed_weight, cost_weight)
_STRATEGY_WEIGHTS: dict[RoutingStrategy, tuple[float, float, float]] = {
    RoutingStrategy.QUALITY_FIRST: (0.7, 0.15, 0.15),
    RoutingStrategy.COST_FIRST: (0.15, 0.15, 0.7),
    RoutingStrategy.SPEED_FIRST: (0.15, 0.7, 0.15),
    RoutingStrategy.AUTO: (0.45, 0.30, 0.25),
}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ModelRouter:
    """Selects the best :class:`VideoModelAdapter` from a :class:`ModelRegistry`."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def select(
        self,
        request: GenerationRequest,
        strategy: RoutingStrategy = RoutingStrategy.AUTO,
        preferred_model: str | None = None,
    ) -> VideoModelAdapter:
        """Pick the best adapter for *request* under *strategy*.

        Parameters
        ----------
        request:
            The generation request used to determine required capabilities.
        strategy:
            Ranking preference (quality / cost / speed / auto).
        preferred_model:
            If set **and** the model is healthy, it is returned immediately
            without scoring.

        Returns
        -------
        VideoModelAdapter
            The winning adapter.

        Raises
        ------
        RuntimeError
            If no suitable adapter is available.
        """
        # Fast path: honour explicit preference when healthy.
        if preferred_model is not None:
            try:
                adapter = self._registry.get(preferred_model)
                if await adapter.health_check():
                    logger.debug(
                        "Using preferred model %r (healthy)", preferred_model
                    )
                    return adapter
                logger.warning(
                    "Preferred model %r is unhealthy; falling back to scoring",
                    preferred_model,
                )
            except KeyError:
                logger.warning(
                    "Preferred model %r not found; falling back to scoring",
                    preferred_model,
                )

        # Determine the required capability from the request.
        required = self._infer_capability(request)

        # Score every capable + healthy adapter.
        candidates: list[ModelScore] = []
        for info in self._registry.list_models():
            mid: str = info["model_id"]
            caps: list[str] = info["capabilities"]

            if required is not None and required.value not in caps:
                continue

            adapter = self._registry.get(mid)
            score = await self._score_model(adapter, request, strategy)
            if score.available:
                candidates.append(score)

        if not candidates:
            raise RuntimeError(
                f"No healthy adapter found for capability={required!r} "
                f"with strategy={strategy.value!r}"
            )

        best = max(candidates, key=lambda s: s.weighted_score)
        logger.info(
            "Router selected %r (score=%.3f, strategy=%s)",
            best.model_id,
            best.weighted_score,
            strategy.value,
        )
        return self._registry.get(best.model_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_capability(request: GenerationRequest) -> ModelCapability | None:
        """Best-effort inference of the required capability from *request*."""
        if request.reference_video is not None:
            return ModelCapability.VIDEO_TO_VIDEO
        if request.reference_image is not None:
            return ModelCapability.IMAGE_TO_VIDEO
        if request.prompt:
            return ModelCapability.TEXT_TO_VIDEO
        return None

    async def _score_model(
        self,
        adapter: VideoModelAdapter,
        request: GenerationRequest,
        strategy: RoutingStrategy,
    ) -> ModelScore:
        """Build a :class:`ModelScore` for *adapter*."""
        profile = MODEL_PROFILES.get(adapter.model_id, _DEFAULT_PROFILE)
        quality = profile["quality"]
        speed = profile["speed"]
        cost = profile["cost"]

        # Health check gates availability.
        try:
            available = await adapter.health_check()
        except Exception:
            available = False

        wq, ws, wc = _STRATEGY_WEIGHTS[strategy]
        weighted = (quality * wq) + (speed * ws) + (cost * wc)

        return ModelScore(
            model_id=adapter.model_id,
            quality_score=quality,
            speed_score=speed,
            cost_score=cost,
            available=available,
            weighted_score=weighted,
        )
