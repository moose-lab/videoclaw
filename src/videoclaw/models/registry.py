"""Model registry — discovers, registers, and manages video-model adapters.

The registry is the single source of truth for which adapters are available at
runtime.  Third-party packages can register adapters via the
``videoclaw.adapters`` entry-point group so they are discovered automatically.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from importlib.metadata import entry_points
from typing import Any

from videoclaw.models.protocol import (
    VideoModelAdapter,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central catalogue of :class:`VideoModelAdapter` instances."""

    def __init__(self) -> None:
        self._adapters: dict[str, VideoModelAdapter] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, adapter: VideoModelAdapter) -> None:
        """Add *adapter* to the registry, keyed by its ``model_id``.

        Raises :class:`ValueError` if an adapter with the same ``model_id``
        is already registered.
        """
        mid = adapter.model_id
        if mid in self._adapters:
            raise ValueError(
                f"Adapter with model_id={mid!r} is already registered"
            )
        self._adapters[mid] = adapter
        logger.info("Registered model adapter %r", mid)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> VideoModelAdapter:
        """Return the adapter for *model_id* or raise :class:`KeyError`."""
        try:
            return self._adapters[model_id]
        except KeyError:
            available = ", ".join(sorted(self._adapters)) or "(none)"
            raise KeyError(
                f"No adapter registered for model_id={model_id!r}. "
                f"Available: {available}"
            ) from None

    def list_models(self) -> list[dict[str, Any]]:
        """Return a summary of every registered adapter."""
        return [
            {
                "model_id": adapter.model_id,
                "capabilities": [c.value for c in adapter.capabilities],
                "execution_mode": adapter.execution_mode.value,
            }
            for adapter in self._adapters.values()
        ]

    # ------------------------------------------------------------------
    # Discovery via entry points
    # ------------------------------------------------------------------

    def discover(self) -> None:
        """Auto-discover adapters exposed via the ``videoclaw.adapters``
        entry-point group.

        Each entry point must resolve to a callable that returns a
        :class:`VideoModelAdapter` instance (a factory or class).
        """
        eps = entry_points()
        # Python 3.12+: entry_points() returns SelectableGroups; use .select()
        adapter_eps = eps.select(group="videoclaw.adapters")

        for ep in adapter_eps:
            try:
                factory = ep.load()
                adapter = factory()
                if adapter.model_id in self._adapters:
                    continue  # already registered
                self.register(adapter)
                logger.info("Discovered adapter %r from entry point %r", adapter.model_id, ep.name)
            except Exception:
                logger.exception(
                    "Failed to load adapter from entry point %r", ep.name
                )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check_all(self) -> dict[str, bool]:
        """Run :meth:`~VideoModelAdapter.health_check` on every adapter
        concurrently and return a mapping of ``model_id -> healthy``."""

        async def _check(adapter: VideoModelAdapter) -> tuple[str, bool]:
            try:
                healthy = await adapter.health_check()
            except Exception:
                logger.warning(
                    "Health check raised for %r", adapter.model_id, exc_info=True
                )
                healthy = False
            return adapter.model_id, healthy

        results = await asyncio.gather(
            *(_check(a) for a in self._adapters.values())
        )
        return dict(results)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._adapters

    def __repr__(self) -> str:
        ids = ", ".join(sorted(self._adapters))
        return f"<ModelRegistry adapters=[{ids}]>"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    """Return the global :class:`ModelRegistry` singleton.

    On first call the registry is created empty.  Call
    :meth:`ModelRegistry.discover` to load entry-point adapters, or
    :meth:`ModelRegistry.register` to add adapters manually.
    """
    return ModelRegistry()
