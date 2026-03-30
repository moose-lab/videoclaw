"""Shared utilities for VideoClaw."""

from datetime import datetime, timezone


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
