"""Shared utilities for VideoClaw."""

from __future__ import annotations

import os
from datetime import datetime, timezone


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def resolve_credential(
    *,
    explicit: str | None = None,
    env_vars: list[str] | str | None = None,
    config_attr: str | None = None,
) -> str | None:
    """Resolve an API key / credential from multiple sources.

    Priority order: explicit value → environment variables → config attribute.

    Parameters
    ----------
    explicit:
        Value passed directly to the constructor.
    env_vars:
        One or more environment variable names to check (in order).
    config_attr:
        Attribute name on :class:`~videoclaw.config.VideoClawConfig`.

    Returns
    -------
    The first non-empty value found, or ``None``.
    """
    if explicit:
        return explicit

    if env_vars:
        if isinstance(env_vars, str):
            env_vars = [env_vars]
        for var in env_vars:
            if val := os.environ.get(var):
                return val

    if config_attr:
        from videoclaw.config import get_config
        cfg = get_config()
        if val := getattr(cfg, config_attr, None):
            return val

    return None
